"""Real SDK contracts against MockTransport; no live model or normal business data."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict
import pytest

from src import budget as budget_module, errors, llm
from src.backend.db import get_engine, get_session_factory, init_db
from src.budget import ExecutionBudgetExceeded, RequestBudget, invoke_model
from src.errors import ModelConfigurationError, ModelOutputError, classify_error, invoke_with_retry
from src.requests import RequestContext
from src.runtime_context import active_budget


class Selection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expert: str
    order_id: int


EXAMPLE = {'expert': 'payment', 'order_id': 1006}


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('LLM_FALLBACKS', '[]')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'unit-deepseek-secret')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-flash')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://unit.invalid')
    yield


@pytest.fixture
def sdk(monkeypatch):
    import langchain_openai
    original = langchain_openai.ChatOpenAI
    @contextmanager
    def install(handle):
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with monkeypatch.context() as patch:
                patch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: original(**kwargs, http_client=client))
                yield
    return install


@pytest.fixture
def budget(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path / 'budget.sqlite'))
    init_db(engine)
    context = RequestContext('customer_a', 'online', uuid4().hex, 'test')
    budget = RequestBudget(context, get_session_factory(engine))
    token = active_budget.set(budget)
    try:
        yield budget
    finally:
        active_budget.reset(token)
        engine.dispose()


def completion(content='完成', *, finish='stop', usage=True, calls=None):
    message = {'role': 'assistant', 'content': content}
    if calls is not None:
        message['tool_calls'] = calls
    data = {'id': 'completion-1', 'object': 'chat.completion', 'created': 1, 'model': 'deepseek-flash',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]}
    if usage:
        data['usage'] = {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}
    return httpx.Response(200, json=data)


@tool
def get_payment_status(order_id: int) -> dict:
    """Get the current payment status for an order."""
    return {'order_id': order_id, 'status': 'paid'}


def test_explicit_provider_real_sdk_tools_messages_and_usage(sdk, budget, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'must-not-be-used')
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == 'https://unit.invalid/chat/completions'
        assert request.headers['Authorization'] == 'Bearer unit-deepseek-secret'
        assert body['model'] == 'deepseek-flash' and body['thinking'] == {'type': 'disabled'}
        assert 'parallel_tool_calls' not in body
        assert body['tools'][0]['function']['name'] == 'get_payment_status'
        assert 0 < request.extensions['timeout']['read'] <= 60
        if len(requests) == 1:
            assert body['tool_choice'] == 'required'
            return completion(None, finish='tool_calls', calls=[{'id': 'model-call-1', 'type': 'function',
                'function': {'name': 'get_payment_status', 'arguments': '{"order_id":1006}'}}])
        assert body['tool_choice'] == 'auto'
        assert body['messages'][-2]['tool_calls'][0]['id'] == 'model-call-1'
        assert body['messages'][-1]['tool_call_id'] == 'model-call-1'
        return completion('订单1006已付款。')
    with sdk(handle):
        model = llm.create_chat_model()
        first = invoke_model(model.bind_tools([get_payment_status], tool_choice='required'), [HumanMessage('1006付了吗')])
        assert first.tool_calls[0]['args'] == {'order_id': 1006}
        second = invoke_model(model.bind_tools([get_payment_status], tool_choice='auto'),
            [HumanMessage('1006付了吗'), first, ToolMessage(content='{"order_id":1006,"status":"paid"}', tool_call_id='model-call-1')])
    assert second.content == '订单1006已付款。'
    assert llm.extract_usage(second)['total_tokens'] == 20
    assert budget.snapshot()['real_model_calls'] == len(requests) == 2


def test_json_output_preserves_raw_usage_and_pydantic_result(sdk, budget):
    requests = []
    def handle(request):
        body = json.loads(request.content); requests.append(body)
        assert body['response_format'] == {'type': 'json_object'}
        assert body['max_tokens'] == 8192 and 'max_completion_tokens' not in body
        assert body['thinking'] == {'type': 'enabled'}
        prompt = body['messages'][0]['content']
        assert 'json' in prompt and 'JSON Schema' in prompt and '1006' in prompt
        assert body['messages'][-1]['content'] == '展示付款情况'
        return completion(json.dumps(EXAMPLE))
    with sdk(handle):
        result = llm.create_chat_model().invoke_structured('展示付款情况', Selection, example=EXAMPLE, max_tokens=256)
    assert result['parsed'] == Selection(**EXAMPLE) and result['parsing_error'] is None
    assert json.loads(result['raw'].content) == EXAMPLE
    assert result['usage'] == llm.extract_usage(result['raw'])
    assert result['usage']['total_tokens'] == 20
    assert budget.snapshot()['real_model_calls'] == len(requests) == 1


@pytest.mark.parametrize('requested,expected', [(64, 8192), (8192, 8192), (10000, 10000)])
def test_structured_reasoning_keeps_usage_context_and_does_not_enable_tool_reasoning(sdk, budget, requested, expected):
    bodies = []
    def handle(request):
        body = json.loads(request.content); bodies.append(body)
        if len(bodies) == 1:
            assert body['thinking'] == {'type': 'enabled'} and body['max_tokens'] == expected
            assert body['response_format'] == {'type': 'json_object'}
            data = json.loads(completion(json.dumps(EXAMPLE)).content)
            data['choices'][0]['message']['reasoning_content'] = 'internal planning text'
            data['usage']['completion_tokens_details'] = {'reasoning_tokens': 5}
            return httpx.Response(200, json=data)
        assert body['thinking'] == {'type': 'disabled'} and 'response_format' not in body
        assert body['tools'][0]['function']['name'] == 'get_payment_status'
        return completion('工具路径仍关闭推理。')
    with sdk(handle):
        model = llm.get_llm()
        structured = model.invoke_structured('query', Selection, example=EXAMPLE, max_tokens=requested)
        ordinary = invoke_model(model.bind_tools([get_payment_status]), [HumanMessage('1006付了吗')])
    assert structured['parsed'] == Selection(**EXAMPLE) and structured['parsing_error'] is None
    assert structured['raw'].text == json.dumps(EXAMPLE) and structured['raw'].response_metadata['finish_reason'] == 'stop'
    assert structured['usage']['total_tokens'] == 20 and isinstance(structured['context_info'], dict)
    assert ordinary.content == '工具路径仍关闭推理。'
    assert budget.snapshot()['model_calls'] == len(bodies) == 2


@pytest.mark.parametrize('content,finish', [('', 'stop'), ('not json', 'stop'), ('{"expert":"payment"}', 'stop'),
    ('{"expert":"payment","order_id":"1006"}', 'stop'),
    ('{"expert":"payment","order_id":1006,"customer_id":"other"}', 'stop'),
    ('{"expert":"payment","order_id":1006', 'stop'), (json.dumps(EXAMPLE), 'length'), (json.dumps(EXAMPLE), 'content_filter')])
def test_invalid_json_is_technical_failure_with_raw_usage_not_transport_retry(sdk, budget, content, finish):
    attempts = []
    def handle(request):
        attempts.append(1)
        return completion(content, finish=finish)
    with sdk(handle):
        result = invoke_with_retry(lambda: llm.create_chat_model().invoke_structured('query', Selection, example=EXAMPLE))
    assert result['parsed'] is None and isinstance(result['parsing_error'], ModelOutputError)
    assert classify_error(result['parsing_error']) == 'model_output'
    assert result['raw'].content == content and result['usage']['total_tokens'] == 20
    assert budget.snapshot()['model_calls'] == len(attempts) == 1


def test_missing_usage_stays_unknown(sdk):
    with sdk(lambda _: completion(json.dumps(EXAMPLE), usage=False)):
        result = llm.create_chat_model().invoke_structured('query', Selection, example=EXAMPLE)
    assert result['parsed'] and result['usage'] == {}


@pytest.mark.parametrize('content,finish,reason,path,issue_type', [
    ('', 'stop', 'empty_output', None, None),
    ('not-json-private-value', 'stop', 'invalid_json', [], 'json_invalid'),
    ('{"expert":"payment","order_id":"private-value"}', 'stop', 'schema_validation', ['order_id'], 'int_type'),
    (json.dumps(EXAMPLE), 'length', 'incomplete_output', None, None),
])
def test_structured_failure_diagnostics_locate_errors_without_logging_values(sdk, monkeypatch, content, finish, reason, path, issue_type):
    events = []
    monkeypatch.setattr(llm, 'record_event', lambda name, **fields: events.append((name, fields)))
    with sdk(lambda _: completion(content, finish=finish)):
        result = llm.get_llm().invoke_structured('query', Selection, example=EXAMPLE)
    diagnostics = result['parsing_error'].diagnostics
    assert diagnostics['reason'] == reason and diagnostics['finish_reason'] == finish
    assert diagnostics['output_chars'] == len(content)
    if path is not None:
        assert {'path': path, 'type': issue_type} in diagnostics['issues']
    invalid = [fields for name, fields in events if name == 'structured_output_invalid']
    assert len(invalid) == 1 and invalid[0]['diagnostics'] == diagnostics
    assert 'private-value' not in json.dumps(invalid) and 'unit-deepseek-secret' not in json.dumps(invalid)


@pytest.mark.parametrize('temperature', [0.0, 0.2])
def test_zhipu_honors_requested_temperature(sdk, monkeypatch, temperature):
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-zhipu')
    monkeypatch.setenv('ZHIPU_MODEL', 'glm-4-flash-250414')
    monkeypatch.setenv('ZHIPU_BASE_URL', 'https://unit.invalid/v4/')
    def handle(request):
        body = json.loads(request.content)
        assert body['temperature'] == temperature
        return completion('完成')
    with sdk(handle):
        llm.create_chat_model(temperature=temperature).invoke('query')


@pytest.mark.parametrize('status,kind,count', [(400,'validation',1), (401,'authentication',1), (402,'balance',1),
    (403,'authentication',1), (404,'configuration',1), (422,'validation',1), (429,'rate_limit',2),
    (500,'service',2), (503,'service',2)])
def test_http_errors_only_retry_recoverable_same_service(sdk, budget, monkeypatch, status, kind, count):
    attempts, events, sleeps = [], [], []
    monkeypatch.setattr(errors.time, 'sleep', sleeps.append)
    monkeypatch.setattr(llm, 'record_event', lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(errors, 'record_event', lambda event, **fields: events.append((event, fields)))
    def handle(request):
        attempts.append(json.loads(request.content)['model'])
        return httpx.Response(status, json={'error': {'message': 'unit-deepseek-secret must not be logged'}})
    with sdk(handle):
        model = llm.create_chat_model()
        assert model.client.max_retries == 0 and model.client.use_responses_api is False
        with pytest.raises(Exception) as error:
            invoke_with_retry(lambda: invoke_model(model, 'query'))
    assert classify_error(error.value) == kind
    assert len(attempts) == budget.snapshot()['model_calls'] == count
    assert set(attempts) == {'deepseek-flash'}
    assert sleeps == ([1] if count == 2 else [])
    assert 'unit-deepseek-secret' not in json.dumps(events) + errors.error_message(kind)


def test_sdk_timeout_retries_without_hidden_attempts(sdk, budget, monkeypatch):
    attempts = []
    def handle(request):
        attempts.append(1)
        raise httpx.ReadTimeout('timeout', request=request)
    with sdk(handle):
        with pytest.raises(Exception) as error:
            invoke_with_retry(lambda: llm.get_llm().invoke('query'), backoff_seconds=0)
    assert classify_error(error.value) == 'network'
    assert len(attempts) == budget.snapshot()['real_model_calls'] == 2


@pytest.mark.parametrize('header,total', [('2.5',2.5), ('0',0), ('-1',0), ('invalid',1), ('NaN',1)])
def test_retry_after_seconds_and_invalid_fallback(sdk, budget, monkeypatch, header, total):
    attempts, sleeps = [], []
    monkeypatch.setattr(errors.time, 'sleep', sleeps.append)
    def handle(request):
        attempts.append(1)
        return httpx.Response(429, headers={'Retry-After': header}, json={}) if len(attempts)==1 else completion()
    with sdk(handle):
        assert invoke_with_retry(lambda: llm.get_llm().invoke('query')).content == '完成'
    assert sum(sleeps) == total and len(attempts) == budget.snapshot()['model_calls'] == 2


def test_retry_after_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    response = httpx.Response(503, headers={'Retry-After': format_datetime(future, usegmt=True)})
    exc = type('ServiceError', (Exception,), {'response': response})()
    assert 8 < errors._retry_delay(exc, 1) <= 10


def test_retry_after_longer_than_budget_stops_without_sleep_or_second_call(sdk, budget, monkeypatch):
    attempts = []
    monkeypatch.setattr(errors.time, 'sleep', lambda _: pytest.fail('must not wait past deadline'))
    def handle(request):
        attempts.append(1)
        return httpx.Response(429, headers={'Retry-After': '10000'}, json={})
    with sdk(handle):
        with pytest.raises(ExecutionBudgetExceeded):
            invoke_with_retry(lambda: llm.get_llm().invoke('query'))
    assert len(attempts) == budget.snapshot()['model_calls'] == 1


def test_structured_retry_and_repair_attempts_share_budget(sdk, budget, monkeypatch):
    calls = []
    monkeypatch.setattr(errors.time, 'sleep', lambda _: None)
    def handle(request):
        calls.append(1)
        return httpx.Response(503, json={}) if len(calls)==1 else completion('')
    with sdk(handle):
        model = llm.get_llm()
        first = invoke_with_retry(lambda: model.invoke_structured('query', Selection, example=EXAMPLE))
        assert first['parsing_error']
        # The future understanding layer can request a correction; it is another counted attempt.
        model.invoke_structured('请纠正 JSON', Selection, example=EXAMPLE)
    assert len(calls) == budget.snapshot()['model_calls'] == 3


def test_remaining_timeout_reaches_sdk_and_expired_budget_prevents_network(sdk, budget, monkeypatch):
    original_time = budget_module.time.time
    offset, timeouts = [110], []
    monkeypatch.setattr(budget_module.time, 'time', lambda: original_time() + offset[0])
    def handle(request):
        timeouts.append(request.extensions['timeout']['read'])
        return completion()
    with sdk(handle):
        model = llm.get_llm()
        model.invoke('query')
        offset[0] = 130
        with pytest.raises(ExecutionBudgetExceeded):
            model.invoke('query')
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 10


@pytest.mark.parametrize('endpoint', ['https://unit.invalid/chat/completions', 'https://unit.invalid?key=secret',
    'https://user:password@unit.invalid', 'not-a-url', 'https://unit.invalid/white space'])
def test_invalid_endpoints_rejected_before_network(monkeypatch, endpoint):
    monkeypatch.setenv('DEEPSEEK_BASE_URL', endpoint)
    with pytest.raises(ModelConfigurationError):
        llm.get_llm()


def test_missing_deepseek_key_does_not_choose_other_configured_service(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', '')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unused')
    monkeypatch.setenv('GROQ_API_KEY', 'unused')
    with pytest.raises(ModelConfigurationError, match='DEEPSEEK_API_KEY'):
        llm.get_llm()


def test_model_setting_read_on_each_construction(sdk, monkeypatch):
    with sdk(lambda _: completion()):
        assert llm.get_llm().model_name == 'deepseek-flash'
        monkeypatch.setenv('DEEPSEEK_MODEL', 'explicit-new-model')
        assert llm.get_llm().model_name == 'explicit-new-model'


@pytest.mark.parametrize('value', [None, '', ' '])
def test_missing_provider_does_not_default_to_a_configured_service(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('LLM_PROVIDER')
    else:
        monkeypatch.setenv('LLM_PROVIDER', value)
    with pytest.raises(ModelConfigurationError, match='LLM_PROVIDER'):
        llm.get_llm()


@pytest.mark.parametrize('provider', ['deepseek', 'zhipu', 'gemini', 'groq'])
@pytest.mark.parametrize('value', [None, ' '])
def test_selected_model_must_be_configured(monkeypatch, provider, value):
    monkeypatch.setenv('LLM_PROVIDER', provider)
    variable = provider.upper() + '_MODEL'
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, value)
    with pytest.raises(ModelConfigurationError, match=variable):
        llm.get_llm()


def test_provider_switch_uses_only_selected_environment_model(monkeypatch):
    selected = []
    monkeypatch.setattr(llm, '_provider_client',
        lambda provider, model, temperature: selected.append((provider, model)) or object())
    for provider in ('deepseek', 'zhipu', 'gemini', 'groq'):
        monkeypatch.setenv(provider.upper() + '_MODEL', provider + '-configured-model')
    for provider in ('zhipu', 'gemini', 'groq', 'deepseek'):
        monkeypatch.setenv('LLM_PROVIDER', provider)
        model = llm.get_llm()
        assert (model.provider, model.model_name) == (provider, provider + '-configured-model')
        assert model.supports_required_tool_choice == (provider == 'deepseek')
    assert selected == [(provider, provider + '-configured-model') for provider in ('zhipu', 'gemini', 'groq', 'deepseek')]


@pytest.mark.parametrize('provider', ['unknown'])
def test_provider_must_be_explicitly_supported(monkeypatch, provider):
    monkeypatch.setenv('LLM_PROVIDER', provider)
    with pytest.raises(ModelConfigurationError):
        llm.get_llm()


def test_gemini_multiple_keys_fail_instead_of_rotating(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-2.5-flash')
    monkeypatch.setenv('GEMINI_API_KEYS', 'first,second')
    with pytest.raises(ModelConfigurationError, match='不再自动轮换'):
        llm.get_llm()


@pytest.mark.parametrize('provider', ['gemini','groq'])
def test_other_native_sdks_return_same_structured_contract(monkeypatch, provider):
    monkeypatch.setenv('LLM_PROVIDER', provider)
    monkeypatch.setenv(provider.upper() + '_MODEL', 'gemini-2.5-flash' if provider == 'gemini' else 'unit-groq-model')
    monkeypatch.setenv('GEMINI_API_KEYS', '')
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-google')
    monkeypatch.setenv('GROQ_API_KEY', 'unit-groq')
    calls = []
    def handle(request):
        body = json.loads(request.content); calls.append(body)
        if provider == 'gemini':
            assert body['generationConfig']['responseMimeType'] == 'application/json'
            assert body['generationConfig']['maxOutputTokens'] == 256
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(EXAMPLE)}]}, 'finishReason': 'STOP'}],
                'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 8, 'totalTokenCount': 20}})
        assert body['response_format'] == {'type': 'json_object'} and body['max_tokens'] == 256
        return completion(json.dumps(EXAMPLE))
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if provider == 'gemini':
            import langchain_google_genai
            original = langchain_google_genai.ChatGoogleGenerativeAI
            monkeypatch.setattr(langchain_google_genai, 'ChatGoogleGenerativeAI',
                lambda **kwargs: original(**kwargs, client_args={'transport': httpx.MockTransport(handle)}))
        else:
            import langchain_groq
            original = langchain_groq.ChatGroq
            monkeypatch.setattr(langchain_groq, 'ChatGroq', lambda **kwargs: original(**kwargs, http_client=client))
        result = llm.create_chat_model().invoke_structured('query', Selection, example=EXAMPLE, max_tokens=256)
    assert result['parsed'] == Selection(**EXAMPLE) and result['usage']['total_tokens'] == 20
    assert len(calls) == 1


def test_unsupported_adapter_capability_is_explicit():
    model = llm.OnlineChatModel(object(), 'deepseek', 'test')
    with pytest.raises(ModelConfigurationError, match='工具调用'):
        model.bind_tools([get_payment_status])
    with pytest.raises(ModelConfigurationError, match='JSON'):
        model.invoke_structured('query', Selection, example=EXAMPLE)


def test_automatic_test_transport_blocks_live_requests():
    with httpx.Client() as client:
        with pytest.raises(AssertionError, match='禁止真实 HTTP'):
            client.get('https://must-not-contact.invalid')


def test_deepseek_runs_existing_online_graph_and_replays_without_more_model_calls(sdk, tmp_path):
    from src import tools
    from src.auth import Principal
    from src.backend.db import Payment
    from src.backend.seed import seed
    from src.service import SupportService
    calls = []
    def handle(request):
        body = json.loads(request.content); calls.append(body)
        if not body.get('tools'):
            from tests.online_support import plan, task
            return completion(json.dumps(plan(task('payment_status', [1006]))))
        if body['messages'][-1]['role'] == 'tool':
            return completion('订单1006已付款，金额 CNY 59.99。')
        return completion(None, finish='tool_calls', calls=[{'id': 'payment-read-1', 'type': 'function',
            'function': {'name': 'get_payment_status', 'arguments': '{"order_id":1006}'}}])
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    try:
        tools.configure_db(engine)
        with get_session_factory(engine)() as db:
            seed(db)
        service = SupportService(tmp_path / 'memory.sqlite')
        with sdk(handle):
            result = service.chat(Principal('customer_a'), 'test', '查询订单1006的付款状态', request_id='deepseek-read')
            replay = service.chat(Principal('customer_a'), 'test', '查询订单1006的付款状态', request_id='deepseek-read')
        assert result['status'] == 'completed' and '59.99' in result['reply']
        assert result['trace'][0]['tool'] == 'get_payment_status'
        assert 'source' not in result['trace'][0]  # Real model call, no synthesized read.
        assert not result['pending_actions'] and replay['replayed']
        assert result['execution_budget']['real_model_calls'] == len(calls) == 3
        with get_session_factory(engine)() as db:
            assert db.get(Payment, 6).status == 'paid'
    finally:
        engine.dispose()


def test_zhipu_supports_same_json_contract(sdk, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-zhipu')
    monkeypatch.setenv('ZHIPU_MODEL', 'glm-test')
    monkeypatch.setenv('ZHIPU_BASE_URL', 'https://unit.invalid/v4/')
    def handle(request):
        body = json.loads(request.content)
        assert str(request.url) == 'https://unit.invalid/v4/chat/completions'
        assert body['model'] == 'glm-test' and body['response_format'] == {'type': 'json_object'}
        assert body['max_tokens'] == 2048 and body['thinking'] == {'type': 'disabled'}
        return completion(json.dumps(EXAMPLE))
    with sdk(handle):
        result = llm.create_chat_model().invoke_structured('query', Selection, example=EXAMPLE)
    assert result['parsed'] == Selection(**EXAMPLE) and result['usage']['total_tokens'] == 20


def test_json_prompt_keeps_system_instruction_and_trims_only_complete_history(sdk, monkeypatch):
    monkeypatch.setenv('SUPPORT_CONTEXT_TOKEN_BUDGET', '700')
    messages = [{'role':'system', 'content':'只处理授权订单'},
                {'role':'user', 'content':'旧问题' * 1000}, {'role':'assistant', 'content':'旧回答' * 1000},
                {'role':'user', 'content':'1006付款状态'}]
    def handle(request):
        sent = json.loads(request.content)['messages']
        assert [message['role'] for message in sent] == ['system', 'user']
        assert '只处理授权订单' in sent[0]['content'] and 'JSON Schema' in sent[0]['content']
        assert sent[-1]['content'] == '1006付款状态'
        return completion(json.dumps(EXAMPLE))
    with sdk(handle):
        result = llm.get_llm().invoke_structured(messages, Selection, example=EXAMPLE)
    assert result['parsed'] and result['context_info']['removed_turns'] == 1
    assert len(messages) == 4 and messages[0]['content'] == '只处理授权订单'
