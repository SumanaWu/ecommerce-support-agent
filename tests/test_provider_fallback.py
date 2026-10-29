"""Provider failover preserves the real SDK contracts and request-wide attempt limits."""
from contextlib import contextmanager
import json
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
import pytest

from src import llm
from src.backend.db import ExecutionBudgetRecord, get_engine, get_session_factory, init_db
from src.budget import ExecutionBudgetExceeded, RequestBudget, invoke_model
from src.errors import ModelConfigurationError, ModelOutputError, classify_error, invoke_with_retry
from src.requests import RequestContext
from src.runtime_context import active_budget
from tests.test_deepseek import completion, get_payment_status


class Selection(BaseModel):
    order_id: int


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'auto')
    monkeypatch.setenv('LLM_FALLBACKS', '[]')
    for provider in ('zhipu', 'deepseek'):
        monkeypatch.setenv(provider.upper() + '_API_KEY', 'unit-' + provider + '-secret')
        monkeypatch.setenv(provider.upper() + '_MODEL', provider + '-configured-model')
        monkeypatch.setenv(provider.upper() + '_BASE_URL', 'https://' + provider + '.invalid/v4/')


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


@pytest.mark.parametrize('primary', ['zhipu', 'deepseek'])
def test_both_directions_keep_tools_history_usage_and_working_provider(sdk, budget, monkeypatch, primary):
    backup = 'deepseek' if primary == 'zhipu' else 'zhipu'
    monkeypatch.setenv('LLM_PROVIDER', primary)
    monkeypatch.setenv('LLM_FALLBACKS', json.dumps([backup]))
    calls, events = [], []
    monkeypatch.setattr(llm, 'record_event', lambda name, **fields: events.append((name, fields)))
    def handle(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        assert body['tools'][0]['function']['name'] == 'get_payment_status'
        assert body['tool_choice'] == 'auto' and 'parallel_tool_calls' not in body
        if len(calls) == 1:
            assert request.url.host == primary + '.invalid'
            return httpx.Response(429, json={'error': {'message': 'unit-' + primary + '-secret'}})
        assert request.url.host == backup + '.invalid'
        assert body['model'] == backup + '-configured-model'
        if len(calls) == 2:
            return completion(None, finish='tool_calls', calls=[{'id': 'read-1', 'type': 'function',
                'function': {'name': 'get_payment_status', 'arguments': '{"order_id":1006}'}}])
        assert body['messages'][-1]['tool_call_id'] == 'read-1'
        return completion('1006已付款。')
    with sdk(handle):
        model = llm.get_llm()
        assert not model.supports_required_tool_choice
        bound = model.bind_tools([get_payment_status], tool_choice='auto')
        first = invoke_model(bound, [HumanMessage('1006付了吗')])
        second = invoke_model(model.bind_tools([get_payment_status], tool_choice='auto'),
            [HumanMessage('1006付了吗'), first, ToolMessage(content='{"status":"paid"}', tool_call_id='read-1')])
        assert model.provider == backup and model.model_name == backup + '-configured-model'
    assert first.tool_calls[0]['args'] == {'order_id': 1006} and second.content == '1006已付款。'
    assert llm.extract_usage(second)['total_tokens'] == 20
    assert budget.snapshot()['model_calls'] == len(calls) == 3
    fallback_events = [fields for name, fields in events if name == 'model_provider_fallback']
    assert len(fallback_events) == 1 and fallback_events[0]['error_type'] == 'rate_limit'
    assert '-secret' not in json.dumps(events)


@pytest.mark.parametrize('primary', ['zhipu', 'deepseek'])
def test_structured_fallback_preserves_json_contract_and_counts_each_http_attempt(sdk, budget, monkeypatch, primary):
    backup = 'deepseek' if primary == 'zhipu' else 'zhipu'
    monkeypatch.setenv('LLM_PROVIDER', primary)
    monkeypatch.setenv('LLM_FALLBACKS', json.dumps([{'provider': backup, 'model': backup + '-override'}]))
    calls = []
    def handle(request):
        body = json.loads(request.content); calls.append(body)
        deepseek = request.url.host == 'deepseek.invalid'
        assert body['response_format'] == {'type': 'json_object'} and body['max_tokens'] == (8192 if deepseek else 256)
        assert body['thinking'] == {'type': 'enabled' if deepseek else 'disabled'}
        if len(calls) == 1:
            return httpx.Response(503, json={})
        assert body['model'] == backup + '-override'
        return completion('{"order_id":1006}')
    with sdk(handle):
        result = llm.get_llm().invoke_structured('查询1006', Selection, example={'order_id': 1006}, max_tokens=256)
    assert result['parsed'] == Selection(order_id=1006) and result['parsing_error'] is None
    assert result['usage']['total_tokens'] == 20 and result['raw'].content == '{"order_id":1006}'
    assert budget.snapshot()['model_calls'] == len(calls) == 2


@pytest.mark.parametrize('status,kind', [(401, 'authentication'), (402, 'balance'), (403, 'authentication'),
    (404, 'configuration'), (429, 'rate_limit'), (500, 'service'), (503, 'service')])
def test_provider_availability_failures_can_fallback(sdk, budget, monkeypatch, status, kind):
    calls, events = [], []
    monkeypatch.setattr(llm, 'record_event', lambda name, **fields: events.append((name, fields)))
    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(status, json={}) if len(calls) == 1 else completion()
    with sdk(handle):
        assert llm.get_llm().invoke('hello').content == '完成'
    assert calls == ['deepseek.invalid', 'zhipu.invalid'] and budget.snapshot()['model_calls'] == 2
    assert [fields['error_type'] for name, fields in events if name == 'model_provider_fallback'] == [kind]


def test_network_failure_can_fallback(sdk, budget):
    calls = []
    def handle(request):
        calls.append(request.url.host)
        if len(calls) == 1:
            raise httpx.ReadTimeout('unit timeout', request=request)
        return completion()
    with sdk(handle):
        assert llm.get_llm().invoke('hello').content == '完成'
    assert budget.snapshot()['model_calls'] == len(calls) == 2


def test_working_provider_is_shared_across_new_models_in_same_request(sdk, budget):
    calls = []
    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(429, json={}) if len(calls) in {1, 3} else completion()
    with sdk(handle):
        assert llm.get_llm().invoke('第一阶段').content == '完成'
        next_stage = llm.get_llm()
        assert next_stage.provider == 'zhipu'
        assert next_stage.invoke('第二阶段').content == '完成'
        assert llm.get_llm().provider == 'deepseek'
    assert calls == ['deepseek.invalid', 'zhipu.invalid', 'zhipu.invalid', 'deepseek.invalid']
    assert budget.snapshot()['model_calls'] == 4


@pytest.mark.parametrize('status', [400, 422])
def test_validation_errors_do_not_fallback_or_retry(sdk, budget, status):
    calls = []
    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(status, json={})
    with sdk(handle):
        model = llm.get_llm()
        with pytest.raises(Exception) as failure:
            invoke_with_retry(lambda: model.invoke('hello'), backoff_seconds=0)
    assert classify_error(failure.value) == 'validation'
    assert calls == ['deepseek.invalid'] and budget.snapshot()['model_calls'] == 1


def test_invalid_json_is_repaired_by_caller_without_provider_switch(sdk, budget):
    calls = []
    def handle(request):
        calls.append(request.url.host)
        return completion('invalid-json')
    with sdk(handle):
        result = llm.get_llm().invoke_structured('hello', Selection, example={'order_id': 1006})
    assert isinstance(result['parsing_error'], ModelOutputError) and result['parsed'] is None
    assert calls == ['deepseek.invalid'] and budget.snapshot()['model_calls'] == 1


def test_shared_budget_can_stop_fallback_before_second_network_call(sdk, budget):
    calls = []
    with budget.sessions.begin() as db:
        db.get(ExecutionBudgetRecord, budget.context.key).max_model_calls = 1
    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(429, json={})
    with sdk(handle):
        with pytest.raises(ExecutionBudgetExceeded):
            invoke_model(llm.get_llm(), 'hello')
    assert calls == ['deepseek.invalid'] and budget.snapshot()['model_calls'] == 1


def test_all_services_failing_keep_original_classification_and_bounded_retries(sdk, budget):
    calls = []
    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(503, json={})
    with sdk(handle):
        model = llm.get_llm()
        with pytest.raises(Exception) as failure:
            invoke_with_retry(lambda: model.invoke('hello'), backoff_seconds=0)
    assert classify_error(failure.value) == 'service'
    assert calls == ['deepseek.invalid', 'zhipu.invalid'] * 2
    assert budget.snapshot()['model_calls'] == 4


def test_program_errors_do_not_fallback(budget):
    class BrokenClient:
        def invoke(self, *args, **kwargs):
            raise RuntimeError('unit program failure')
    class UnusedClient:
        def invoke(self, *args, **kwargs):
            pytest.fail('program error must not select another provider')
    model = llm.ProviderFallbackModel([llm.OnlineChatModel(BrokenClient(), 'zhipu', 'unit'),
                                       llm.OnlineChatModel(UnusedClient(), 'deepseek', 'unit')])
    with pytest.raises(RuntimeError):
        model.invoke('hello')
    assert budget.snapshot()['model_calls'] == 1


def test_auto_uses_only_complete_pair_configuration(monkeypatch):
    specs = []
    monkeypatch.setattr(llm, '_provider_client', lambda provider, model, temperature: specs.append((provider, model)) or object())
    model = llm.get_llm()
    assert isinstance(model, llm.ProviderFallbackModel) and model.provider == 'deepseek'
    assert [provider for provider, _ in specs] == ['deepseek', 'zhipu']
    monkeypatch.setenv('DEEPSEEK_API_KEY', '')
    specs.clear()
    assert llm.get_llm().provider == 'zhipu' and specs == [('zhipu', 'zhipu-configured-model')]
    monkeypatch.setenv('ZHIPU_MODEL', '')
    with pytest.raises(ModelConfigurationError, match='至少需要完整配置'):
        llm.get_llm()


@pytest.mark.parametrize('fallbacks', ['["unknown"]', '["zhipu"]', '[{"provider":"deepseek","model":""}]',
    '[{"provider":"deepseek","api_key":"private-value"}]', '{}', 'invalid-json'])
def test_invalid_fallback_configuration_does_not_echo_values(monkeypatch, fallbacks):
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('LLM_FALLBACKS', fallbacks)
    with pytest.raises(ModelConfigurationError) as failure:
        llm.get_llm()
    assert 'private-value' not in str(failure.value)


def test_auto_rejects_ambiguous_model_or_fallback_override():
    with pytest.raises(ModelConfigurationError, match='auto 使用各服务'):
        llm.get_llm('shared-model-name')
