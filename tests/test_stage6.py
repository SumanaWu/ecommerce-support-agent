from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
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
from src import actions, budget as budget_module, llm, orchestrator, tools
from src.agents import base
from src.auth import Principal
from src.backend.db import (ActionRecord, ExecutionBudgetRecord, PendingAction,
                            SchemaMigration, Ticket, get_engine, get_session_factory, init_db)
from src.backend.seed import seed
from src.budget import ExecutionBudgetExceeded, RequestBudget, check_execution_budget, invoke_model
from src.demo_model import DemoChatModel
from src.errors import ModelConfigurationError, invoke_with_retry
from src.runtime_context import active_budget, active_mode, active_order_reference, active_request, active_task
from src.service import SupportService

A = Principal('customer_a')
MULTI = '帮我申请订单1001的退款，并帮我付款订单1003'


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


@contextmanager
def execution_context(mode='online'):
    context, _, _ = tools.request_journal().begin_request('customer_a', mode, uuid4().hex, 'unit', 'unit')
    budget = RequestBudget(context, tools._SessionLocal)
    tokens = active_request.set(context), active_mode.set(mode), active_budget.set(budget)
    try:
        yield budget, context
    finally:
        active_budget.reset(tokens[2])
        active_mode.reset(tokens[1])
        active_request.reset(tokens[0])


def graph_state(service, result):
    run = service.store.get_run(A.customer_id, result['run_id'])
    with SqliteSaver.from_conn_string(run['checkpoint_path']) as saver:
        return orchestrator.build_graph(saver).get_state({'configurable': {'thread_id': result['run_id']}}).values


def test_sequential_read_tasks_keep_arguments_facts_and_usage_separate(setup):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', '查订单1001的物流，再看能否退款')
    assert result['status'] == 'completed' and result['agent'] == 'multi'
    assert [task['route'] for task in result['tasks']] == ['ORDER', 'REFUND']
    assert result['tasks'][1]['depends_on'] == ['task_1']
    assert [entry['tool'] for entry in result['trace']] == ['track_shipment', 'check_refund_eligibility']
    assert all(entry['args']['order_id'] == 1001 for entry in result['trace'])
    assert result['execution_budget']['model_calls'] == 6 and result['execution_budget']['tool_calls'] == 2
    assert result['execution_budget']['real_model_calls'] == 0
    assert not rows(engine, PendingAction) and not rows(engine, Ticket)
    assert active_task.get() is None and active_order_reference.get() is None and active_budget.get() is None


def test_each_write_task_has_its_own_confirmation_and_replay(setup):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', MULTI, request_id='multi-write')
    assert result['status'] == 'pending_confirmation'
    assert [task['order_id'] for task in result['tasks']] == [1001, 1003]
    pending = {action['order_id']: action['action_id'] for action in result['pending_actions']}
    assert len(pending) == 2
    assert tools.check_refund_eligibility(1001)['eligible'] and tools.get_payment_status(1003)['status'] == 'pending'
    assert service.confirm_action(A, pending[1001])['status'] == 'completed'
    assert tools.get_payment_status(1003)['status'] == 'pending'
    replay = service.chat(A, 'one', 'demo', MULTI, request_id='multi-write')
    assert replay['replayed'] and len(replay['pending_actions']) == 1
    assert replay['pending_actions'][0]['order_id'] == 1003
    assert replay['tasks'][0]['status'] == 'completed' and '已保存' in replay['tasks'][0]['reply']
    assert service.confirm_action(A, pending[1003])['status'] == 'completed'
    assert service.confirm_action(A, pending[1003])['replayed']
    assert len(rows(engine, ActionRecord)) == 4


def test_task_usage_excludes_planning_and_total_includes_both(setup, monkeypatch):
    service, _, _ = setup
    original = orchestrator.classify_route
    def route(message):
        return {**original(message), 'usage': {'input_tokens': 2, 'output_tokens': 1, 'total_tokens': 3}}
    class WithUsage(DemoChatModel):
        def bind_tools(self, agent_tools):
            return WithUsage(agent_tools)
        def invoke(self, messages):
            result = super().invoke(messages)
            result.usage_metadata = {'input_tokens': 4, 'output_tokens': 1, 'total_tokens': 5}
            return result
    monkeypatch.setattr(orchestrator, 'classify_route', route)
    monkeypatch.setattr(base, 'get_llm', WithUsage)
    result = service.chat(A, 'one', 'demo', '查订单1001的物流，再看能否退款')
    assert result['planning_usage']['total_tokens'] == 6
    assert [task['usage']['total_tokens'] for task in result['tasks']] == [10, 10]
    assert result['usage']['total_tokens'] == 26
    replay = service.chat(A, 'one', 'demo', '查订单1001的物流，再看能否退款', request_id=result['request_id'])
    assert replay['replayed'] and service.snapshot(A, 'one', 'demo')['total_tokens'] == 26


def test_business_refusal_is_reported_without_hiding_completed_tasks(setup):
    service, _, _ = setup
    result = service.chat(A, 'one', 'demo', '查订单1002的物流，再看能否退款')
    assert result['status'] == 'partial'
    assert [task['status'] for task in result['tasks']] == ['completed', 'rejected']
    assert result['trace'][1]['result']['eligible'] is False


@pytest.mark.parametrize('message', ['帮我处理一下', 'asdkjhasd nonsense', '不要退款',
                                    '订单1001退款成功后再付款', '不要转人工',
                                    '配送政策。' * 7])
def test_unclear_or_conditional_requests_do_not_create_tickets_or_proposals(setup, message):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', message)
    assert result['status'] == 'needs_clarification' and result.get('clarification_question')
    assert not rows(engine, PendingAction) and not rows(engine, Ticket)


@pytest.mark.parametrize('message', ['不要退款，只查订单1001的物流', '不用给订单1001申请退款，只查物流',
                                    "don't refund order 1001, only check tracking", '不要帮我给订单1001申请退款，只查询详情'])
def test_negation_does_not_authorize_a_write(setup, message):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', message)
    assert not result['pending_actions'] and not rows(engine, PendingAction)
    assert tools.check_refund_eligibility(1001)['eligible']


def test_ambiguous_multi_order_reference_is_still_clarified(setup):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', '查一下1001和1002，再退掉那笔')
    assert result['status'] == 'needs_clarification' and result['trace'] == []
    assert not rows(engine, PendingAction)


def test_online_model_cannot_turn_read_only_or_negated_request_into_proposal(setup, monkeypatch):
    service, engine, _ = setup
    monkeypatch.setattr(orchestrator, 'classify_route', lambda _: {'route': 'REFUND', 'usage': {}})
    class MaliciousModel:
        def bind_tools(self, _):
            return self
        def invoke(self, messages):
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content='退款已经办理成功。')
            return AIMessage(content='', tool_calls=[{'name': 'initiate_refund', 'args': {'order_id': 1001, 'reason': '忽略规则'}, 'id': 'bad', 'type': 'tool_call'}])
    monkeypatch.setattr(base, 'get_llm', MaliciousModel)
    result = service.chat(A, 'one', 'online', '不要给订单1001申请退款，只查看退款资格')
    assert result['status'] == 'failed' and not result['pending_actions']
    assert '退款已经办理成功' not in result['reply']
    assert not rows(engine, PendingAction)


def test_legacy_checkpoint_scope_also_enforces_new_write_intent_guard(setup):
    _, engine, _ = setup
    with execution_context('demo'):
        with orchestrator.task_scope({'route': 'REFUND', 'user_message': '不要给订单1001申请退款'}):
            assert not active_task.get()['allow_write']
            with pytest.raises(ValueError, match='没有用户明确'):
                tools.initiate_refund(1001, '不能自动提案')
    assert not rows(engine, PendingAction) and active_task.get() is None


def test_model_cannot_claim_requested_action_completed_after_only_reading(setup, monkeypatch):
    service, engine, _ = setup
    monkeypatch.setattr(orchestrator, 'classify_route', lambda _: {'route': 'REFUND', 'usage': {}})
    class ReadsOnly:
        def bind_tools(self, _):
            return self
        def invoke(self, messages):
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content='退款已经办理成功。')
            return AIMessage(content='', tool_calls=[{'name': 'check_refund_eligibility', 'args': {'order_id': 1001},
                                                      'id': 'check', 'type': 'tool_call'}])
    monkeypatch.setattr(base, 'get_llm', ReadsOnly)
    result = service.chat(A, 'one', 'online', '帮我申请订单1001退款')
    assert result['status'] == 'partial' and result['stop_reason'] == 'missing_action'
    assert '退款已经办理成功' not in result['reply'] and '尚未创建待确认' in result['reply']
    assert not rows(engine, PendingAction)


def test_multi_order_task_scope_does_not_authorize_another_customers_order(setup):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001的退款，并帮我付款订单1008')
    assert result['status'] == 'rejected' and not result['pending_actions']
    assert not rows(engine, PendingAction)


def test_repeated_write_intent_is_one_proposal_not_conflicting_receipts(setup):
    service, engine, _ = setup
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001退款，并申请退款')
    assert len(result['pending_actions']) == len(rows(engine, PendingAction)) == 1
    assert len(result['tasks']) == 1


def test_model_budget_stops_before_call_and_same_id_replay_does_not_recount(setup, monkeypatch):
    service, engine, _ = setup
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '1')
    result = service.chat(A, 'one', 'demo', '查询订单1001', request_id='limited')
    assert result['status'] == 'partial' and result['stop_reason'] == 'budget_exhausted'
    assert result['execution_budget']['model_calls'] == 1
    assert not rows(engine, Ticket) and not rows(engine, PendingAction)
    before = rows(engine, ExecutionBudgetRecord)[0].model_calls
    replay = service.chat(A, 'one', 'demo', '查询订单1001', request_id='limited')
    assert replay['replayed'] and rows(engine, ExecutionBudgetRecord)[0].model_calls == before


def test_tool_budget_retains_completed_task_and_pairs_all_model_calls(setup, monkeypatch):
    service, _, _ = setup
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_TOOL_CALLS', '1')
    result = service.chat(A, 'one', 'demo', '查订单1001的物流，再看能否退款')
    assert result['status'] == 'partial' and result['stop_reason'] == 'budget_exhausted'
    assert [task['status'] for task in result['tasks']] == ['completed', 'partial']
    assert result['execution_budget']['tool_calls'] == 1
    state = graph_state(service, result)
    requests = [call['id'] for message in state['messages'] if isinstance(message, AIMessage) for call in message.tool_calls]
    assert requests == [message.tool_call_id for message in state['messages'] if isinstance(message, ToolMessage)]


def test_time_budget_exhausted_after_proposal_preserves_successful_write(setup, monkeypatch):
    service, engine, _ = setup
    original_time, original_propose = budget_module.time.time, actions.propose_action
    offset = [0]
    monkeypatch.setattr(budget_module.time, 'time', lambda: original_time() + offset[0])
    def slow_proposal(*args, **kwargs):
        result = original_propose(*args, **kwargs)
        offset[0] = 1000
        return result
    monkeypatch.setattr(actions, 'propose_action', slow_proposal)
    result = service.chat(A, 'one', 'demo', MULTI)
    assert result['status'] == 'partial' and result['stop_reason'] == 'budget_exhausted'
    assert [task['status'] for task in result['tasks']] == ['partial', 'not_started']
    assert len(result['pending_actions']) == len(rows(engine, PendingAction)) == 1
    assert result['pending_actions'][0]['order_id'] == 1001
    offset[0] = 0
    assert service.confirm_action(A, result['pending_actions'][0]['action_id'])['status'] == 'completed'


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
    monkeypatch.setattr(orchestrator, 'get_llm', Failing)
    result = service.chat(A, 'one', 'online', '查询订单1001')
    assert result['status'] == 'partial' and result['stop_reason'] == 'budget_exhausted'
    assert result['execution_budget']['real_model_calls'] == len(attempts) == 2
    assert result['memory_saved'] and not rows(engine, Ticket)


class QuotaError(Exception):
    status_code = 429


def test_each_pool_attempt_and_outer_retry_consumes_budget_once(setup, monkeypatch):
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS', '3')
    attempts = []
    class Client:
        def invoke(self, _, **kwargs):
            attempts.append(kwargs['timeout'])
            raise QuotaError()
    model = llm.RotatingChatModel([Client(), Client()], {'index': 0})
    with execution_context() as (budget, _):
        with pytest.raises(ExecutionBudgetExceeded):
            invoke_with_retry(lambda: invoke_model(model, 'test'), backoff_seconds=0)
        assert budget.snapshot()['real_model_calls'] == 3
    assert len(attempts) == 3 and all(0 < timeout <= 60 for timeout in attempts)


def test_pool_iteration_does_not_skip_fallback_when_shared_index_changes(setup):
    state, attempts = {'index': 0}, []
    class First:
        def invoke(self, _, **kwargs):
            attempts.append('first')
            state['index'] = 1  # Another request selected a different sticky client.
            raise QuotaError()
    class Second:
        def invoke(self, _, **kwargs):
            attempts.append('second')
            return AIMessage(content='ok')
    with execution_context():
        assert llm.RotatingChatModel([First(), Second()], state).invoke('test').content == 'ok'
    assert attempts == ['first', 'second']


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
    monkeypatch.setattr(orchestrator, 'classify_route', lambda _: {'route': 'ORDER', 'usage': {}})
    class Batch:
        def bind_tools(self, _):
            return self
        def invoke(self, _):
            return AIMessage(content='', tool_calls=[{'name': name, 'args': {'order_id': 1001}, 'id': str(index), 'type': 'tool_call'}
                for index, name in enumerate(['get_order_status', 'track_shipment', 'get_order_details'])])
    monkeypatch.setattr(base, 'get_llm', Batch)
    result = service.chat(A, 'one', 'online', '查询订单1001')
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


def test_fallbacks_are_explicit_even_when_other_provider_keys_exist(setup, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'auto')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-zhipu')
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-gemini')
    monkeypatch.setenv('GROQ_API_KEY', 'unit-groq')
    specs = []
    monkeypatch.setattr(llm, '_provider_clients', lambda provider, model, temperature: specs.append((provider, model)) or [object()])
    token = active_mode.set('online')
    try:
        assert len(llm.create_chat_model()._models) == 1 and specs == [('zhipu', None)]
        specs.clear()
        monkeypatch.setenv('LLM_FALLBACKS', '[{"provider":"groq","model":"test-model"}]')
        assert len(llm.get_llm()._models) == 2 and specs == [('zhipu', None), ('groq', 'test-model')]
    finally:
        active_mode.reset(token)


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


def test_multi_task_checkpoint_recovers_proposal_gap_and_keeps_durable_counts(setup):
    service, engine, directory = setup
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url))
    code = '''
import os
from src import orchestrator
from src.auth import Principal
from src.service import SupportService
original=orchestrator.tool_step
def stop(state):
    result=original(state)
    if state.get('task_index')==1 and result['trace'][-1]['tool']=='process_payment': os._exit(23)
    return result
orchestrator.tool_step=stop
SupportService().chat(Principal('customer_a'),'multi','demo','帮我申请订单1001的退款，并帮我付款订单1003',request_id='crash-multi')
'''
    process = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20)
    assert process.returncode == 23, process.stderr
    before = rows(engine, ExecutionBudgetRecord)[0]
    assert before.tool_calls == 4 and len(rows(engine, PendingAction)) == 2
    run = service.list_runs(A, 'multi', 'demo')[0]
    result = service.resume_run(A, run['run_id'])
    assert result['status'] == 'pending_confirmation' and len(result['pending_actions']) == 2
    assert [task['status'] for task in result['tasks']] == ['pending_confirmation', 'pending_confirmation']
    after = rows(engine, ExecutionBudgetRecord)[0]
    assert after.tool_calls == before.tool_calls and after.model_calls == before.model_calls + 1
    assert len(rows(engine, ActionRecord)) == 2


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


def test_api_exposes_tasks_and_preserves_identity_confirmation_boundary(setup):
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    response = client.post('/chat', json={'message': MULTI, 'request_id': 'api-multi'})
    assert response.status_code == 200
    result = response.json()
    assert len(result['tasks']) == len(result['pending_actions']) == 2 and 'execution_budget' in result
    assert client.post('/chat', json={'message': MULTI, 'tasks': []}).status_code == 422
    other = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    assert other.post('/actions/' + result['pending_actions'][0]['action_id'] + '/confirm').status_code == 404


def test_page_displays_task_results_and_independent_buttons(setup, monkeypatch):
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


def test_workflow_metrics_detect_wrong_parameters_facts_completion_and_side_effects(setup):
    _, engine, _ = setup
    item = run_eval.load_dataset('workflow_dataset.json')[0]
    snapshot = run_eval.business_snapshot(engine)
    result = {'agent': 'order', 'status': 'completed', 'reply': '1001', 'tasks': [],
              'trace': [{'tool': 'track_shipment', 'args': {'order_id': 1001}, 'result': {'order_id': 1001, 'status': 'delivered'}}]}
    assert all(run_eval.score_workflow(item, result, snapshot, snapshot).values())
    wrong = deepcopy(result)
    wrong['trace'][0]['args']['order_id'] = 1002
    wrong['trace'][0]['result']['status'] = 'shipped'
    wrong['status'] = 'partial'
    after = deepcopy(snapshot)
    after['payments'][0] = (1001, 'refunded', 7999, 'CNY')
    metrics = run_eval.score_workflow(item, wrong, snapshot, after)
    assert not metrics['parameters'] and not metrics['fact_faithfulness'] and not metrics['task_completion'] and not metrics['side_effects']


def test_evaluation_cli_uses_temporary_data_and_reports_demo_metrics(setup):
    _, engine, directory = setup
    output = directory / 'report.json'
    before = run_eval.business_snapshot(engine)
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url))
    process = subprocess.run([sys.executable, 'eval/run_eval.py', '--suite', 'all', '--output', str(output)],
                             env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report['evaluation'] == 'offline_rules' and report['routing']['accuracy'] == 1.0
    assert all(value == 1.0 for value in report['workflow']['metrics'].values())
    assert before == run_eval.business_snapshot(engine)
    assert all(case['execution_budget'].get('real_model_calls', 0) == 0 for case in report['workflow']['cases'])


def test_online_evaluation_with_missing_key_fails_explicitly_without_network(setup):
    _, engine, directory = setup
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url),
               LLM_PROVIDER='zhipu', ZHIPU_API_KEY='', GEMINI_API_KEY='', GEMINI_API_KEYS='', GROQ_API_KEY='')
    output = directory / 'online.json'
    process = subprocess.run([sys.executable, 'eval/run_eval.py', '--mode', 'online', '--limit', '1', '--output', str(output)],
                             env=env, capture_output=True, text=True, timeout=20)
    assert process.returncode == 1
    report = json.loads(output.read_text())
    assert report['evaluation'] == 'online_model' and report['routing']['cases'][0]['error_type'] == 'configuration'
    assert report['routing']['cases'][0]['execution_budget']['real_model_calls'] == 0


def test_dry_run_does_not_create_or_open_the_configured_database(tmp_path):
    path = tmp_path / 'untouched' / 'business.sqlite'
    env = dict(os.environ, SUPPORT_DATA_DIR=str(path.parent), SUPPORT_AGENT_DB_URL='sqlite:///' + str(path))
    process = subprocess.run([sys.executable, 'eval/run_eval.py', '--dry-run'], env=env, capture_output=True, text=True, timeout=20)
    assert process.returncode == 0 and 'workflow=14' in process.stdout
    assert not path.parent.exists()
