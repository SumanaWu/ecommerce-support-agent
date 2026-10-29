import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from src import llm
from src.errors import ModelConfigurationError, classify_error, invoke_with_retry


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'zhipu')
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-test-key')
    monkeypatch.setenv('ZHIPU_MODEL', 'glm-4.7-flash')
    monkeypatch.setenv('ZHIPU_BASE_URL', 'https://unit.invalid/api/paas/v4/')
    yield


@tool
def check_refund_eligibility(order_id: int) -> dict:
    """Check local refund eligibility."""
    return {'order_id': order_id, 'eligible': False}


def test_real_sdk_zhipu_chat_tools_result_and_usage(online, monkeypatch):
    import langchain_openai
    original = langchain_openai.ChatOpenAI
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == 'https://unit.invalid/api/paas/v4/chat/completions'
        assert request.headers['Authorization'] == 'Bearer unit-test-key'
        assert body['model'] == 'glm-4.7-flash'
        assert body['thinking'] == {'type': 'disabled'}
        assert 'parallel_tool_calls' not in body
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'check_refund_eligibility', 'arguments': '{"order_id":1002}'}}]}
            finish = 'tool_calls'
        else:
            assert body['messages'][-1]['tool_call_id'] == 'call-1'
            assert json.loads(body['messages'][-1]['content'])['eligible'] is False
            message, finish = {'role': 'assistant', 'content': '未送达，不能退款。'}, 'stop'
        return httpx.Response(200, json={
            'id': 'completion-1', 'object': 'chat.completion', 'created': 1, 'model': 'glm-4.7-flash',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: original(**kwargs, http_client=client))
        model = llm.get_llm().bind_tools([check_refund_eligibility])
        first = model.invoke([HumanMessage('查订单 1002 退款资格')])
        assert first.tool_calls[0]['args'] == {'order_id': 1002}
        second = model.invoke([HumanMessage('查订单 1002 退款资格'), first,
                               ToolMessage(content='{"order_id":1002,"eligible":false}', tool_call_id='call-1')])
    assert second.content == '未送达，不能退款。'
    assert llm.extract_usage(second) == {'input_tokens': 30, 'output_tokens': 10, 'total_tokens': 40}
    assert len(requests) == 2


@pytest.mark.parametrize('code,count,kind', [(401, 1, 'authentication'), (429, 2, 'rate_limit'), (503, 2, 'service')])
def test_zhipu_http_errors_have_no_sdk_or_graph_retry(online, monkeypatch, code, count, kind):
    import langchain_openai
    original = langchain_openai.ChatOpenAI
    requests = []

    def handle(request):
        requests.append(1)
        return httpx.Response(code, json={'error': {'message': 'unit-test-key must not leak', 'type': 'api_error'}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: original(**kwargs, http_client=client))
        model = llm.get_llm()
        with pytest.raises(Exception) as error:
            invoke_with_retry(lambda: model.invoke('test'), backoff_seconds=0)
    assert len(requests) == count
    assert classify_error(error.value) == kind


def test_missing_key_and_bad_endpoint_are_configuration_errors(online, monkeypatch):
    monkeypatch.setenv('ZHIPU_API_KEY', '')
    with pytest.raises(ModelConfigurationError):
        llm.get_llm()
    monkeypatch.setenv('ZHIPU_API_KEY', 'unit-test-key')
    monkeypatch.setenv('ZHIPU_BASE_URL', 'https://unit.invalid/chat/completions')
    with pytest.raises(ModelConfigurationError):
        llm.get_llm()


def test_usage_can_be_read_from_compatible_response_metadata():
    message = SimpleNamespace(response_metadata={'token_usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}})
    assert llm.extract_usage(message)['total_tokens'] == 5


def test_single_zhipu_client_does_not_silently_fallback(online, monkeypatch):
    import langchain_openai
    monkeypatch.setenv('GEMINI_API_KEY', 'unused')
    monkeypatch.setenv('GROQ_API_KEY', 'unused')
    captured = []
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: captured.append(kwargs) or object())
    model = llm.get_llm()
    assert model.provider == 'zhipu'
    assert captured[0]['max_retries'] == 0 and captured[0]['use_responses_api'] is False
