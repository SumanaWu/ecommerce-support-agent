import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from src import llm


def test_online_model_uses_deepseek_configuration(monkeypatch):
    import langchain_openai

    options = {}
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.delenv('DEEPSEEK_BASE_URL', raising=False)
    monkeypatch.delenv('DEEPSEEK_MODEL', raising=False)
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: options.update(kwargs))
    llm.get_llm()
    assert options['model'] == 'deepseek-flash'
    assert options['base_url'] == 'https://api.deepseek.com'
    assert options['api_key'] == 'test-key'
    assert options['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert options['use_responses_api'] is False
    assert options['max_retries'] == 0
    assert options['timeout'] == 60


def test_online_model_requires_deepseek_key(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DEEPSEEK_API_KEY'):
        llm.get_llm()


def test_deepseek_tool_protocol(monkeypatch):
    import langchain_openai

    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'call_1', 'type': 'function',
                                'function': {'name': 'lookup', 'arguments': '{"order_id":1001}'}}],
            }
            finish_reason = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': '已查询。'}
            finish_reason = 'stop'
        return httpx.Response(200, json={
            'id': 'completion_1', 'object': 'chat.completion', 'created': 1,
            'model': 'deepseek-flash', 'choices': [{'index': 0, 'message': message,
                                                     'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15},
        })

    @tool
    def lookup(order_id: int) -> str:
        """Look up an order."""
        return str(order_id)

    real_chat = langchain_openai.ChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI',
                        lambda **kwargs: real_chat(http_client=client, **kwargs))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    try:
        model = llm.get_llm().bind_tools([lookup])
        first = model.invoke([HumanMessage(content='查询订单 1001')])
        assert first.tool_calls[0]['args'] == {'order_id': 1001}
        assert llm.extract_usage(first)['total_tokens'] == 15
        result = model.invoke([HumanMessage(content='查询订单 1001'), first,
                               ToolMessage(content='delivered', tool_call_id='call_1')])
        assert result.content == '已查询。'
    finally:
        client.close()
    assert requests[0]['model'] == 'deepseek-flash'
    assert requests[0]['thinking'] == {'type': 'disabled'}
    assert requests[0]['tools'][0]['function']['name'] == 'lookup'
    assert requests[1]['messages'][-1]['tool_call_id'] == 'call_1'
