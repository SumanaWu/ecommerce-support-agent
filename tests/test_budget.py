"""Online regression grouped by responsibility; no rule-model fallback."""
from tests.online_support import install_plan, plan, task
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
from fastapi.testclient import TestClient
import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
import pytest
from sqlalchemy import select
import api
from eval import run_eval
from src import budget as budget_module, llm, orchestrator, tools
from src.agents import base
from src.auth import Principal
from src.backend.db import ExecutionBudgetRecord, SchemaMigration, Ticket, get_engine, get_session_factory, init_db
from src.backend.seed import seed
from src.budget import ExecutionBudgetExceeded, RequestBudget, check_execution_budget, invoke_model
from src.errors import ModelConfigurationError, invoke_with_retry
from src.runtime_context import active_budget, active_request
from src.service import SupportService


A = Principal('customer_a')
MULTI = '帮我申请订单1001的退款，并帮我付款订单1003'


class QuotaError(Exception):
    status_code = 429


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name, value in [('SUPPORT_REQUEST_MAX_MODEL_CALLS', '24'), ('SUPPORT_REQUEST_MAX_TOOL_CALLS', '24'),
                        ('SUPPORT_REQUEST_TIMEOUT_SECONDS', '120'), ('LLM_FALLBACKS', '[]')]:
        monkeypatch.setenv(name, value)
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(api, 'service', service)
    yield service, engine, tmp_path
    engine.dispose()


def rows(engine, model):
    with get_session_factory(engine)() as db:
        return db.scalars(select(model)).all()


def graph_state(service, result):
    run = service.store.get_run(A.customer_id, result['run_id'])
    with SqliteSaver.from_conn_string(run['checkpoint_path']) as saver:
        return orchestrator.build_graph(saver).get_state({'configurable': {'thread_id': result['run_id']}}).values


@contextmanager
def execution_context():
    context, _, _ = tools.request_journal().begin_request('customer_a', 'online', uuid4().hex, 'unit', 'unit')
    budget = RequestBudget(context, tools._SessionLocal)
    tokens = active_request.set(context), active_budget.set(budget)
    try:
        yield budget, context
    finally:
        active_budget.reset(tokens[1])
        active_request.reset(tokens[0])


def test_final_network_failure_after_deadline_returns_partial_without_escalation(setup, monkeypatch):
    service, engine, _ = setup
    original_time, attempts, offset = budget_module.time.time, [], [0]
    monkeypatch.setattr(budget_module.time, 'time', lambda: original_time() + offset[0])
    monkeypatch.setattr('src.errors.time.sleep', lambda _: None)
    class Failing:
        def invoke(self, _):
            attempts.append(1)
            if len(attempts) == 2:
                offset[0] = 1000
            raise TimeoutError('simulated network failure')
    import src.understanding as understanding
    from src.budget import check_execution_budget
    class StructuredFailing(Failing):
        def invoke_structured(self, *args, **kwargs):
            check_execution_budget('model',consume=True)
            return self.invoke(None)
    monkeypatch.setattr(understanding, 'get_llm', StructuredFailing)
    result = service.chat(A, 'one', '查询订单1001')
    assert result['status'] == 'partial' and result['stop_reason'] == 'budget_exhausted'
    assert result['execution_budget']['real_model_calls'] == len(attempts) == 2
    assert result['memory_saved'] and not rows(engine, Ticket)


def test_sdk_retry_consumes_budget_once_and_stops_before_next_attempt(setup, monkeypatch):
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '1')
    attempts = []
    class Client:
        def invoke(self, _, **kwargs):
            attempts.append(kwargs['timeout'])
            raise QuotaError()
    model = llm.OnlineChatModel(Client(), 'deepseek', 'test-model')
    with execution_context() as (budget, _):
        with pytest.raises(ExecutionBudgetExceeded):
            invoke_with_retry(lambda: invoke_model(model, 'test'), backoff_seconds=0)
        assert budget.snapshot()['real_model_calls'] == 1
    assert len(attempts) == 1 and all(0 < timeout <= 60 for timeout in attempts)


def test_rate_limit_retries_the_same_client_without_pool_rotation(setup):
    attempts = []
    class Client:
        def invoke(self, _, **kwargs):
            attempts.append('same-client')
            if len(attempts) == 1:
                raise QuotaError()
            return AIMessage(content='ok')
    model = llm.OnlineChatModel(Client(), 'deepseek', 'test-model')
    with execution_context() as (budget, _):
        assert invoke_with_retry(lambda: invoke_model(model, 'test'), backoff_seconds=0).content == 'ok'
        assert budget.snapshot()['real_model_calls'] == 2
    assert attempts == ['same-client', 'same-client']


def test_persisted_budget_limits_cannot_be_reset_by_recovery_or_new_env(setup, monkeypatch):
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '2')
    with execution_context() as (budget, context):
        check_execution_budget('model', consume=True)
        monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '99')
        restored = RequestBudget(context, tools._SessionLocal)
        assert restored.snapshot()['model_calls'] == 1 and restored.snapshot()['max_model_calls'] == 2
        restored.check('model', consume=True)
        with pytest.raises(ExecutionBudgetExceeded):
            restored.check('model', consume=True)


def test_concurrent_budget_claims_do_not_exceed_limit(setup, monkeypatch):
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '2')
    with execution_context() as (budget, _):
        def claim(_):
            try:
                budget.check('model', consume=True)
                return True
            except ExecutionBudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert sum(pool.map(claim, range(10))) == 2


@pytest.mark.parametrize('name,value', [('SUPPORT_REQUEST_MAX_MODEL_CALLS', '0'), ('SUPPORT_REQUEST_MAX_TOOL_CALLS', '1.5'),
                                       ('SUPPORT_REQUEST_TIMEOUT_SECONDS', 'NaN')])
def test_invalid_budget_configuration_fails_before_execution(setup, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        with execution_context():
            pytest.fail('invalid budget was accepted')


def test_multi_tool_batch_budget_stops_remaining_calls_without_orphan_messages(setup, monkeypatch):
    service, _, _ = setup
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_TOOL_CALLS', '1')
    install_plan(monkeypatch, plan(task('order_status',[1001])))
    class Batch:
        def bind_tools(self, _):
            return self
        def invoke(self, _):
            return AIMessage(content='', tool_calls=[{'name': name, 'args': {'order_id': 1001}, 'id': str(index), 'type': 'tool_call'}
                for index, name in enumerate(['get_order_status', 'track_shipment', 'get_order_details'])])
    monkeypatch.setattr(base, 'get_llm', Batch)
    result = service.chat(A, 'one', '查询订单1001')
    assert result['execution_budget']['tool_calls'] == 1
    assert result['trace'][0]['result']['status'] == 'delivered'
    assert all(entry['result'].get('executed') is False for entry in result['trace'][1:])
    assert [m.tool_call_id for m in graph_state(service, result)['messages'] if isinstance(m, ToolMessage)] == ['0', '1', '2']


@pytest.mark.parametrize('batch,rounds,reason', [(7, 4, 'batch_limit'), (1, 1, 'round_limit')])
def test_loop_limits_return_partial_without_executing_unpaired_calls(monkeypatch, batch, rounds, reason):
    @tool
    def echo(value: str):
        """Echo."""
        return value
    class Endless:
        counter = 0
        def bind_tools(self, _):
            return self
        def invoke(self, _):
            self.counter += 1
            return AIMessage(content='全部完成。', tool_calls=[{'name': 'echo', 'args': {'value': str(self.counter) + str(index)},
                'id': str(self.counter) + str(index), 'type': 'tool_call'} for index in range(batch)])
    monkeypatch.setattr(base, 'get_llm', Endless)
    result = base.run_tool_agent('test', 'test', [echo], 'test', max_rounds=rounds)
    assert result['status'] == 'partial' and result['stop_reason'] == reason
    assert '全部完成' not in result['reply'] and len(result['trace']) == (1 if batch == 1 else 0)


def test_explicit_selection_ignores_other_keys_and_validates_fallback_list(setup, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('ZHIPU_MODEL', 'glm-test')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-zhipu')
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-gemini')
    monkeypatch.setenv('GROQ_API_KEY', 'unit-groq')
    specs = []
    monkeypatch.setattr(llm, '_provider_client', lambda provider, model, temperature: specs.append(provider) or object())
    assert llm.create_chat_model().provider == 'zhipu' and specs == ['zhipu']
    specs.clear()
    monkeypatch.setenv('LLM_FALLBACKS', '[{"provider":"groq","model":"test-model"}]')
    with pytest.raises(ModelConfigurationError, match='只支持 deepseek、zhipu'):
        llm.get_llm()
    assert specs == []


@pytest.mark.parametrize('config', ['{}', '[{"provider":"unknown","model":"x"}]', '[{"provider":"groq","model":"x","api_key":"not-allowed"}]', 'bad'])
def test_bad_fallback_specs_are_safe_configuration_errors(setup, monkeypatch, config):
    monkeypatch.setenv('LLM_FALLBACKS', config)
    with execution_context():
        with pytest.raises(ModelConfigurationError) as error:
            llm.create_chat_model()
        assert 'not-allowed' not in str(error.value)


def test_real_sdk_uses_remaining_timeout_and_is_counted_once(setup, monkeypatch):
    import langchain_openai
    original = langchain_openai.ChatOpenAI
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('ZHIPU_MODEL', 'glm-test')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-key')
    monkeypatch.setenv('ZHIPU_BASE_URL', 'https://unit.invalid/v4/')
    timeouts = []
    def handle(request):
        timeouts.append(request.extensions['timeout']['read'])
        return httpx.Response(200, json={'id': 'one', 'object': 'chat.completion', 'created': 1, 'model': 'unit',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: original(**kwargs, http_client=client))
        with execution_context() as (budget, _):
            assert invoke_model(llm.get_llm(), 'test').content == 'ok'
            assert budget.snapshot()['real_model_calls'] == 1
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 60


@pytest.mark.parametrize('provider', ['gemini', 'groq'])
def test_other_provider_real_sdks_preserve_tool_contract_with_mock_http(setup, monkeypatch, provider):
    from langchain_core.messages import HumanMessage
    @tool
    def get_order_status(order_id: int):
        """Get local order status."""
        return {'order_id': order_id, 'status': 'delivered'}
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body.get('tools')
        if provider == 'groq':
            if len(requests) == 1:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
                    'function': {'name': 'get_order_status', 'arguments': '{"order_id":1001}'}}]}
            else:
                assert body['messages'][-1]['tool_call_id'] == 'call-1'
                message = {'role': 'assistant', 'content': '已送达。'}
            return httpx.Response(200, json={'id': 'one', 'object': 'chat.completion', 'created': 1, 'model': 'unit',
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop'}],
                'usage': {'prompt_tokens': 30, 'completion_tokens': 5, 'total_tokens': 35}})
        if len(requests) == 1:
            parts = [{'functionCall': {'name': 'get_order_status', 'args': {'order_id': 1001}, 'id': 'call-1'}}]
        else:
            assert body['contents'][-1]['parts'][0]['functionResponse']['name'] == 'get_order_status'
            parts = [{'text': '已送达。'}]
        return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': parts}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 30, 'candidatesTokenCount': 5, 'totalTokenCount': 35}})
    monkeypatch.setenv('LLM_PROVIDER', provider)
    monkeypatch.setenv(provider.upper() + '_MODEL', 'gemini-2.5-flash' if provider == 'gemini' else 'unit-groq-model')
    monkeypatch.setenv('GEMINI_API_KEYS', '')
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-gemini')
    monkeypatch.setenv('GROQ_API_KEY', 'unit-groq')
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if provider == 'groq':
            import langchain_groq
            original = langchain_groq.ChatGroq
            monkeypatch.setattr(langchain_groq, 'ChatGroq', lambda **kwargs: original(**kwargs, http_client=client))
        else:
            import langchain_google_genai
            original = langchain_google_genai.ChatGoogleGenerativeAI
            monkeypatch.setattr(langchain_google_genai, 'ChatGoogleGenerativeAI',
                lambda **kwargs: original(**kwargs, client_args={'transport': httpx.MockTransport(handle)}))
        with execution_context() as (budget, _):
            model = llm.get_llm().bind_tools([get_order_status])
            first = invoke_model(model, [HumanMessage(content='查订单1001')])
            assert first.tool_calls[0]['args'] == {'order_id': 1001}
            second = invoke_model(model, [HumanMessage(content='查订单1001'), first,
                ToolMessage(content='{"order_id":1001,"status":"delivered"}', tool_call_id=first.tool_calls[0]['id'])])
            assert '已送达' in base.as_text(second.content) and llm.extract_usage(second)['total_tokens'] == 35
            assert budget.snapshot()['real_model_calls'] == 2
    assert len(requests) == 2


def test_v4_additive_upgrade_backs_up_and_preserves_v3_business_data(setup):
    _, engine, directory = setup
    before = run_eval.business_snapshot(engine)
    with engine.begin() as db:
        ExecutionBudgetRecord.__table__.drop(db)
        db.execute(SchemaMigration.__table__.delete().where(SchemaMigration.version == 4))
    init_db(engine)
    init_db(engine)
    assert before == run_eval.business_snapshot(engine)
    backups = list(directory.glob('business.sqlite.before-v4-*.bak'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='execution_budgets'").fetchone()


def test_api_exposes_tasks_and_preserves_identity_confirmation_boundary(setup, monkeypatch):
    from tests.online_support import call, install_script
    install_script(monkeypatch, [plan(task('propose_refund', [1001]), task('propose_payment', [1003], task_id='task_2')),
        call('check_refund_eligibility', {'order_id': 1001}, call_id='read-refund'),
        call('initiate_refund', {'order_id': 1001, 'reason': '用户申请'}, call_id='refund'), '请确认退款。',
        call('get_payment_status', {'order_id': 1003}, call_id='read-payment'),
        call('process_payment', {'order_id': 1003, 'method': 'card'}, call_id='payment'), '请确认付款。'])
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    response = client.post('/chat', json={'message': MULTI, 'request_id': 'api-multi'})
    assert response.status_code == 200
    result = response.json()
    assert len(result['tasks']) == len(result['pending_actions']) == 2 and 'execution_budget' in result
    assert client.post('/chat', json={'message': MULTI, 'tasks': []}).status_code == 422
    other = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    assert other.post('/actions/' + result['pending_actions'][0]['action_id'] + '/confirm').status_code == 404


def test_page_displays_task_results_and_independent_buttons(setup, monkeypatch):
    from tests.online_support import call, install_script
    install_script(monkeypatch, [plan(task('propose_refund', [1001]), task('propose_payment', [1003], task_id='task_2')),
        call('check_refund_eligibility', {'order_id': 1001}, call_id='read-refund'),
        call('initiate_refund', {'order_id': 1001, 'reason': '用户申请'}, call_id='refund'), '请确认退款。',
        call('get_payment_status', {'order_id': 1003}, call_id='read-payment'),
        call('process_payment', {'order_id': 1003, 'method': 'card'}, call_id='payment'), '请确认付款。'])
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    _, _, directory = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(directory))
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
        next(item for item in app.text_input if item.label == '测试账户令牌').set_value('test-token-a').run()
        app.chat_input[0].set_value(MULTI).run()
        assert not app.exception and any(item.label == '逐项任务结果' for item in app.expander)
        next(item for item in app.button if item.label == '确认申请').click().run()
        assert not app.exception and any(item.label == '确认付款' for item in app.button)
        assert tools.get_payment_status(1003)['status'] == 'pending'
    finally:
        st.cache_resource.clear()


