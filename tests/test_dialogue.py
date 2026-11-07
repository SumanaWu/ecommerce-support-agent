import json
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, AIMessage as AssistantMessage
from sqlalchemy import select
from src import dialogue, orchestrator, tools
from src.backend.db import Ticket
from tests.test_customer_refunds import svc
from tests.test_actions import business


class Model:
    def __init__(self, result):
        self.result = result
        self.seen = []
    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=json.dumps(self.result))


@pytest.mark.parametrize('action', ['DIRECT', 'CLARIFY'])
def test_main_reply_does_not_create_ticket(svc, monkeypatch, action):
    model = Model({'action': action, 'answer': '你好 😊' if action == 'DIRECT' else '你想了解什么？'})
    model.result['tasks'] = []
    monkeypatch.setattr(orchestrator, 'make_plan', lambda *a: (model.result, {}))
    result = svc.chat('customer_a', 'one', '你好')
    assert result['agent'] == 'assistant' and result['trace'] == []
    with tools._session() as session:
        assert not session.scalars(select(Ticket)).all()


def test_history_and_policy_resolution(monkeypatch):
    from src.planning import make_plan
    model = Model({'action': 'PLAN', 'tasks': [{'task_id': 't1', 'operation': 'faq.policies', 'source_text': '都告诉我', 'topics': ['refund']}]})
    history = [HumanMessage(content='介绍其他政策'), AssistantMessage(content='配送、退货、退款、取消、联系')]
    result, _ = make_plan('都告诉我', history, model=model)
    assert result['tasks'][0]['topics'] == ['refund']
    assert history[0] in model.seen[0] and model.seen[0][-1].content == '都告诉我'


def test_invalid_routing_never_escalates(svc, monkeypatch):
    from src.planning import make_plan
    model = Model({'action': 'ESCALATE', 'escalation_reason': 'unclear'})
    monkeypatch.setattr(orchestrator, 'make_plan', lambda *a: make_plan('查配送', model=model))
    with pytest.raises(RuntimeError):
        svc.chat('customer_a', 'one', '查配送')
    assert len(model.seen) == 4
    with tools._session() as session:
        assert not session.scalars(select(Ticket)).all()


def test_explicit_escalation_keeps_existing_tool(svc):
    result = svc.chat('customer_a', 'one', '我要投诉')
    assert result['agent'] == 'escalate'
    assert result['trace'][0]['result']['ticket_id']


def test_search_sources_and_request_replay(svc, monkeypatch):
    monkeypatch.setattr(dialogue, 'get_llm', lambda: Model({'answer': '北京天气见来源'}))
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        calls[-1]['url'] = url
        return httpx.Response(200, request=httpx.Request('POST', url), json={'results': [{'title': '天气', 'url': 'https://example.com/weather', 'summary': '今日预报', 'date': '2026-11-06'}]})
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    monkeypatch.setattr(dialogue.httpx, 'post', post)
    first = svc.chat('customer_a', 'one', '北京今天的天气', request_id='weather')
    assert first['agent'] == 'assistant' and first['sources'][0]['published_at'] == '2026-11-06'
    assert svc.chat('customer_a', 'one', '北京今天的天气', request_id='weather') == first
    assert len(calls) == 1 and calls[0]['url'] == 'https://api.deepseek.com/v1/tools/search'
    assert calls[0]['headers']['Authorization'] == 'Bearer test'
    assert calls[0]['json'] == {'query': '北京今日天气', 'count': 5, 'need_content': False}
    assert first['sources'][0]['content'] == '今日预报'
    assert 'customer_a' not in calls[0]['json']['query']


def test_search_missing_key(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    assert dialogue.search_web('北京天气')['sources'] == []


@pytest.mark.parametrize('status,attempts', [(401, 1), (429, 2), (500, 2)])
def test_search_error_retries(monkeypatch, status, attempts):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        return httpx.Response(status, request=httpx.Request('POST', url))
    monkeypatch.setattr(dialogue.httpx, 'post', post)
    assert not dialogue.search_web('北京天气')['sources']
    assert len(calls) == attempts


def test_policy_retrieval_always_grounded(monkeypatch):
    from src.agents.faq_agent import run_faq_agent, get_policies
    model = Model({'text': 'policy summary'})
    monkeypatch.setattr(dialogue, 'get_llm', lambda: model)
    assert len(get_policies()) == 5
    result = run_faq_agent('全部政策')
    assert len(result['trace'][0]['result']) == 5
    assert '仅依据' in model.seen[0][0].content
    assert 'shipping' in model.seen[0][-1].content


def test_ticket_claim_without_receipt_is_suppressed():
    assert '工单 #' not in dialogue.checked_reply('已生成工单 #123')


def test_confirmation_failure_does_not_create_ticket(svc, monkeypatch):
    monkeypatch.setattr(svc.graph, 'invoke', lambda *a: (_ for _ in ()).throw(RuntimeError('failed')))
    with pytest.raises(RuntimeError):
        svc._invoke({}, svc._config('customer_a', 'one', 'failed'))
    with tools._session() as session:
        assert not session.scalars(select(Ticket)).all()


def test_ui_main_reply_and_sources(svc, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr(dialogue, 'search_web', lambda query: {'sources': [{'title': '天气来源', 'url': 'https://example.com/weather', 'content': '预报', 'retrieved_at': '2026-11-06T12:00:00Z'}]})
    monkeypatch.setattr(dialogue, 'compose', lambda *a: ('查询到了天气预报 ☀️', {}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('北京天气').run()
    assert not app.exception and not app.chat_input[0].disabled
    assert any(c.value == '主对话层' for c in app.caption)
    assert any('[天气来源](https://example.com/weather)' in m.value for m in app.markdown), [m.value for m in app.markdown]


@pytest.mark.parametrize('base', ['https://api.deepseek.com', 'https://api.deepseek.com/v1/'])
def test_search_uses_configured_base_and_snippet(monkeypatch, base):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', base)
    def post(url, **kwargs):
        assert url == 'https://api.deepseek.com/v1/tools/search'
        return httpx.Response(200, request=httpx.Request('POST', url), json={'results': [
            {'title': '来源', 'url': 'https://example.com', 'snippet': '简要结果', 'date': None}]})
    monkeypatch.setattr(dialogue.httpx, 'post', post)
    source = dialogue.search_web('北京天气')['sources'][0]
    assert source['content'] == '简要结果' and source['published_at'] is None
    assert source['retrieved_at']


def test_search_rejects_overlong_query_without_request(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    monkeypatch.setattr(dialogue.httpx, 'post', lambda *a, **k: pytest.fail('unexpected request'))
    assert '128' in dialogue.search_web('天' * 129)['error']
