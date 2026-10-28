import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import pytest
from sqlalchemy import select

import api
from src import tools
from src.agents import base
from src.auth import AccessError, Principal
from src.backend.db import ActionRecord, PendingAction, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.context import ContextBudgetExceeded, build_context, estimate_context_tokens, fit_model_context
from src.demo_model import DemoChatModel
from src.memory import empty_issue, extract_order_ids
from src.runtime_context import active_order_reference
from src.service import SupportService
from src.storage import ProfileConflict, SessionConflict, SupportStore

A, B = Principal('customer_a'), Principal('customer_b')


@pytest.fixture
def setup(tmp_path, monkeypatch):
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


@pytest.mark.parametrize('text,expected', [
    ('查一下1001和1002，再退掉那笔', [1001, 1002]),
    ('订单1001退款1002元', [1001]), ('退款1001元', []),
    ('订单 1003，金额 34.50', [1003]), ('order 1001 amount 1002', [1001]),
    ('$1001', []), ('1001 dollars', []), ('订单1001.50', []),
    ('不是1002，是1001', [1001]), ('不是订单1002，改为订单1001', [1001]),
    ('order #1001', [1001]), ('订单号：9999', [9999]), ('订单42', [42]),
])
def test_order_ids_do_not_come_from_amounts_or_negated_references(text, expected):
    assert extract_order_ids(text) == expected


def test_multiple_orders_require_selection_before_any_proposal(setup, monkeypatch):
    service, engine, _ = setup
    monkeypatch.setattr(base, 'get_llm', lambda: pytest.fail('ambiguous request reached model'))
    first = service.chat(A, 'one', 'demo', '查一下1001和1002，再退掉那笔', request_id='multi')
    assert first['status'] == 'needs_clarification' and not first['pending_actions']
    issue = service.snapshot(A, 'one', 'demo')['issue']
    assert issue['candidate_order_ids'] == [1001, 1002] and issue['active_order_id'] is None
    second = service.chat(A, 'one', 'demo', '帮我申请退款')
    assert second['status'] == 'needs_clarification'
    assert not rows(engine, ActionRecord) and not rows(engine, PendingAction)
    assert service.chat(A, 'one', 'demo', '查一下1001和1002，再退掉那笔', request_id='multi')['replayed']
    assert len(service.snapshot(A, 'one', 'demo')['turns']) == 2


def test_selecting_an_order_resolves_pronouns_and_corrections_survive_restart(setup):
    service, _, _ = setup
    service.chat(A, 'one', 'demo', '查询1001和1002')
    current = service.snapshot(A, 'one', 'demo')
    selected = service.set_active_order(A, 'one', 'demo', 1001, current['version'])
    assert selected['issue']['selection_source'] == 'user_selection'
    restored = SupportService(service.store.path)
    result = restored.chat(A, 'one', 'demo', '再退掉那笔')
    assert result['status'] == 'pending_confirmation' and result['pending_actions'][0]['order_id'] == 1001
    assert restored.chat(A, 'one', 'demo', '不是1001，是1002')['status'] == 'completed'
    refused = restored.chat(A, 'one', 'demo', '帮我申请退款')
    assert refused['status'] == 'rejected' and refused['trace'][0]['args']['order_id'] == 1002
    assert restored.snapshot(A, 'new', 'demo')['issue'] == empty_issue()
    assert restored.snapshot(A, 'one', 'online')['issue'] == empty_issue()
    assert restored.snapshot(B, 'one', 'demo')['issue'] == empty_issue()


def test_invalid_reference_does_not_reuse_a_previous_order(setup):
    service, engine, _ = setup
    service.chat(A, 'one', 'demo', '查询订单1001')
    assert service.chat(A, 'one', 'demo', '帮我申请订单9999的退款')['status'] == 'rejected'
    assert service.chat(A, 'one', 'demo', '帮我申请退款')['status'] == 'needs_clarification'
    assert not rows(engine, PendingAction)
    assert service.snapshot(A, 'one', 'demo')['last_order_id'] is None
    for malformed in ['帮我申请订单ABC1001的退款', '帮我申请订单1001.50的退款']:
        service.chat(A, 'one', 'demo', '查询订单1006')
        assert service.chat(A, 'one', 'demo', malformed)['status'] == 'rejected'
        assert service.chat(A, 'one', 'demo', '帮我申请退款')['status'] == 'needs_clarification'
        assert not rows(engine, PendingAction)
    for number in [9999, 1008]:
        with pytest.raises(AccessError) as exc:
            service.set_active_order(A, 'one', 'demo', number, service.snapshot(A, 'one', 'demo')['version'])
        assert str(exc.value) == '未找到可访问的订单。'


def test_amount_is_not_used_as_a_new_order_in_demo(setup):
    service, _, _ = setup
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001的退款，金额1002元')
    assert result['pending_actions'][0]['order_id'] == 1001
    assert result['pending_actions'][0]['amount'] == 79.99
    assert service.chat(A, 'new', 'demo', '帮我退款1001元')['status'] == 'needs_clarification'


def test_model_cannot_propose_a_different_order_than_the_server_resolved(setup, monkeypatch):
    from .test_agent_base import FakeLLM, FakeToolCallMessage
    service, engine, _ = setup
    model = FakeLLM([
        FakeToolCallMessage(tool_calls=[{'name': 'initiate_refund', 'args': {'order_id': 1006, 'reason': 'guess'}, 'id': 'guess'}]),
        FakeToolCallMessage(content='done'),
    ])
    monkeypatch.setattr(base, 'get_llm', lambda: model)
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001的退款')
    assert result['status'] == 'failed' and 'error' in result['trace'][0]['result']
    assert not rows(engine, PendingAction)
    assert active_order_reference.get() is None


def test_low_level_proposal_also_rejects_ambiguous_context(setup):
    service, _, _ = setup
    context, _, _ = tools.request_journal().begin_request('customer_a', 'demo', 'direct', 'one', 'test')
    from src.runtime_context import active_request
    request_token = active_request.set(context)
    reference_token = active_order_reference.set({'status': 'ambiguous', 'order_id': None})
    try:
        with pytest.raises(AccessError):
            tools.initiate_refund(1001, 'model guessed')
    finally:
        active_order_reference.reset(reference_token)
        active_request.reset(request_token)


def test_issue_edit_uses_session_version_and_clear_does_not_leak_into_new_sessions(setup):
    service, _, _ = setup
    first = service.set_active_order(A, 'one', 'demo', 1001, 0)
    with pytest.raises(SessionConflict):
        service.set_active_order(A, 'one', 'demo', 1002, 0)
    cleared = service.set_active_order(A, 'one', 'demo', None, first['version'])
    assert not cleared['issue']['candidate_order_ids'] and cleared['issue']['active_order_id'] is None
    assert service.chat(A, 'one', 'demo', '帮我付款')['status'] == 'needs_clarification'


def test_profile_versions_sources_deletion_and_account_isolation(setup):
    service, _, _ = setup
    saved = service.update_profile(A, {'tone': '仅使用简短句子'}, 0)
    assert saved['version'] == 1 and saved['fields']['tone']['source'] == 'user_form'
    assert saved['fields']['tone']['updated_at']
    with pytest.raises(ProfileConflict):
        service.update_profile(A, {'tone': 'stale'}, 0)
    deleted = service.delete_profile_field(A, 'tone', 1)
    assert deleted['values'] == {} and deleted['fields']['tone']['deleted']
    assert deleted['fields']['tone']['version'] == 2
    assert service.profile(B) == {}
    with pytest.raises(ValueError):
        service.update_profile(A, {'customer_id': 'customer_b'}, 2)
    with pytest.raises(ValueError):
        service.update_profile(A, {'tone': 'x' * 51}, 2)
    assert SupportService(service.store.path).profile_details(A) == deleted


def test_chat_does_not_write_profile_and_deleted_values_are_not_model_facts(setup, monkeypatch):
    service, _, _ = setup
    captured = []
    class CapturingModel(DemoChatModel):
        def bind_tools(self, agent_tools):
            self.tools = {tool.name for tool in agent_tools}
            return self
        def invoke(self, messages):
            captured.append(messages[0].content)
            return super().invoke(messages)
    service.update_profile(A, {'tone': 'unit-unique-preference'}, 0)
    service.delete_profile_field(A, 'tone', 1)
    monkeypatch.setattr(base, 'get_llm', CapturingModel)
    service.chat(A, 'one', 'demo', '查询订单1001。我喜欢长回复')
    assert service.profile(A) == {}
    assert all('unit-unique-preference' not in content for content in captured)


def test_related_older_turns_are_selected_in_pairs_within_budget():
    old = {'user': '订单1001退款时不要换成别的订单', 'reply': '记录本轮约束',
           'trace': [{'args': {'order_id': 1001}}]}
    turns = [old] + [{'user': '闲聊' * 150, 'reply': '回复' * 150, 'trace': []} for _ in range(12)]
    issue = {**empty_issue(), 'candidate_order_ids': [1001], 'active_order_id': 1001}
    selected, messages, info = build_context(turns, '查询订单1001退款', issue, {}, budget=1500)
    assert selected == [old] and len(messages) == 2
    assert isinstance(messages[0], HumanMessage) and isinstance(messages[1], AIMessage)
    assert info['selected_turns'] == 1 and info['is_estimate']


def test_large_tool_output_is_shortened_without_losing_pairing_or_facts():
    messages = [SystemMessage('system'), HumanMessage('query'),
                AIMessage(content='', tool_calls=[{'name': 'read', 'args': {}, 'id': 'call'}]),
                ToolMessage(content=json.dumps({'order_id': 1001, 'eligible': True, 'item': 'long' * 10000}), tool_call_id='call')]
    fitted, info = fit_model_context(messages, [], budget=600)
    assert len(fitted) == 4 and fitted[-1].tool_call_id == fitted[-2].tool_calls[0]['id']
    data = json.loads(fitted[-1].content)
    assert data['order_id'] == 1001 and data['eligible'] and data['context_truncated']
    assert info['is_estimate'] and estimate_context_tokens(fitted) <= 600
    assert len(json.loads(messages[-1].content)['item']) == 40000


def test_budget_never_slices_a_user_message_or_a_tool_pair():
    with pytest.raises(ContextBudgetExceeded):
        fit_model_context([SystemMessage('system'), HumanMessage('x' * 10000)], [], budget=300)
    messages = [SystemMessage('sys'), HumanMessage('old' * 1000), AIMessage('answer' * 1000), HumanMessage('new')]
    fitted, info = fit_model_context(messages, [], budget=400)
    assert [message.type for message in fitted] == ['system', 'human']
    assert info['removed_turns'] == 1


def test_budget_exhaustion_keeps_a_created_proposal_and_stops_model_calls(setup, monkeypatch):
    service, engine, _ = setup
    original = base.fit_model_context
    def fail_after_proposal(messages, agent_tools):
        if any(isinstance(message, ToolMessage) and message.name == 'initiate_refund' for message in messages):
            raise ContextBudgetExceeded('test budget')
        return original(messages, agent_tools)
    monkeypatch.setattr(base, 'fit_model_context', fail_after_proposal)
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001的退款', request_id='budget')
    assert result['status'] == 'pending_confirmation' and result['error_type'] == 'context_budget'
    assert len(rows(engine, ActionRecord)) == len(rows(engine, PendingAction)) == 1
    assert next(row for row in rows(engine, Refund) if row.order_id == 1001).status == 'none'
    assert service.chat(A, 'one', 'demo', '帮我申请订单1001的退款', request_id='budget')['replayed']


def test_api_memory_and_profile_require_authentication_versions_and_owned_orders(setup):
    service, _, _ = setup
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    other = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    assert TestClient(api.app).get('/profile').status_code == 401
    body = {'session_id': 'one', 'mode': 'demo', 'order_id': 1001, 'expected_version': 0}
    assert client.patch('/memory/session/order', json=body).status_code == 200
    assert client.patch('/memory/session/order', json=body).status_code == 409
    assert other.patch('/memory/session/order', json=body).status_code == 404
    assert client.patch('/memory/session/order', json={**body, 'user_id': 'customer_b'}).status_code == 422
    saved = client.patch('/profile', json={'fields': {'tone': '简洁'}, 'expected_version': 0})
    assert saved.json()['fields']['tone']['source'] == 'user_api'
    assert other.get('/profile').json()['values'] == {}
    assert client.request('DELETE', '/profile/tone', json={'expected_version': 1}).status_code == 200
    assert client.get('/profile').json()['values'] == {}
    assert client.request('DELETE', '/profile/tone', json={'expected_version': 1}).status_code == 409
    snapshot = client.get('/memory/session', params={'session_id': 'one'}).json()
    assert snapshot['issue']['active_order_id'] == 1001
    assert service.snapshot(B, 'one', 'demo')['issue'] == empty_issue()


def test_chat_cleanup_preserves_audits_profile_actions_and_deduplication(setup):
    service, engine, _ = setup
    service.update_profile(A, {'tone': '简洁'}, 0)
    result = service.chat(A, 'one', 'demo', '帮我申请订单1001的退款', request_id='keep')
    snapshot = service.snapshot(A, 'one', 'demo')
    cleared = service.clear_history(A, 'one', 'demo', snapshot['version'])
    assert cleared['turns'] == [] and cleared['issue'] == snapshot['issue']
    assert len(service.list_pending_actions(A, 'one', 'demo')) == 1 and service.profile(A)['tone'] == '简洁'
    assert len(rows(engine, ActionRecord)) == 1
    assert service.chat(A, 'one', 'demo', '帮我申请订单1001的退款', request_id='keep')['replayed']
    assert service.snapshot(A, 'one', 'demo')['turns'] == []
    assert service.confirm_action(A, result['pending_actions'][0]['action_id'])['status'] == 'completed'


def test_v2_memory_migration_preserves_data_and_marks_legacy_candidates(tmp_path):
    path = tmp_path / 'old.sqlite'
    payload = {'turns': [], 'last_order_id': 1001, 'total_tokens': 17}
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE sessions(user_id TEXT,mode TEXT,session_id TEXT,payload TEXT,updated_at TEXT,version INTEGER,
                           PRIMARY KEY(user_id,mode,session_id));
                           CREATE TABLE profiles(user_id TEXT PRIMARY KEY,payload TEXT,updated_at TEXT);
                           CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY);
                           INSERT INTO schema_migrations VALUES(1),(2);''')
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)', ('customer_a','demo','one',json.dumps(payload),'original',9))
        db.execute('INSERT INTO profiles VALUES(?,?,?)', ('customer_a','{"tone":"brief"}','original'))
    store = SupportStore(path)
    restored = store.load_session('customer_a', 'demo', 'one')
    assert restored['version'] == 9 and restored['total_tokens'] == 17
    assert restored['issue']['active_order_id'] is None and restored['issue']['selection_source'] == 'legacy_last_order'
    assert store.profile_details('customer_a')['fields']['tone']['source'] == 'legacy_user_saved'
    backup = list(tmp_path.glob('old.sqlite.before-v3-*.bak'))
    assert len(backup) == 1
    with sqlite3.connect(backup[0]) as db:
        assert json.loads(db.execute('SELECT payload FROM sessions').fetchone()[0]) == payload
    SupportStore(path)
    assert len(list(tmp_path.glob('old.sqlite.before-v3-*.bak'))) == 1


@pytest.mark.parametrize('change', ['issue', 'profile'])
def test_restart_respects_issue_correction_and_profile_withdrawal(setup, monkeypatch, change):
    service, engine, directory = setup
    service.update_profile(A, {'tone': 'withdraw-me-after-crash'}, 0)
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url))
    code = '''
import os
from src import orchestrator
from src.auth import Principal
from src.service import SupportService
def stop(state): os._exit(23)
orchestrator.tool_step=stop
SupportService().chat(Principal('customer_a'),'one','demo','帮我申请订单1001的退款',request_id='stopped')
'''
    assert subprocess.run([sys.executable, '-c', code], env=env, timeout=20).returncode == 23
    run = service.list_runs(A, 'one', 'demo')[0]
    if change == 'issue':
        service.set_active_order(A, 'one', 'demo', 1002, service.snapshot(A, 'one', 'demo')['version'])
        result = service.resume_run(A, run['run_id'])
        assert result['status'] == 'needs_clarification' and result['error_type'] == 'order_reference_changed'
        assert result['memory_saved'] and service.list_runs(A, 'one', 'demo') == []
        assert not rows(engine, PendingAction)
    else:
        service.delete_profile_field(A, 'tone', 1)
        class CurrentProfileModel(DemoChatModel):
            def bind_tools(self, agent_tools):
                self.tools = {tool.name for tool in agent_tools}
                return self
            def invoke(self, messages):
                assert 'withdraw-me-after-crash' not in messages[0].content
                return super().invoke(messages)
        monkeypatch.setattr(base, 'get_llm', CurrentProfileModel)
        assert service.resume_run(A, run['run_id'])['status'] == 'pending_confirmation'


def test_page_correction_preference_deletion_and_inline_confirmation(setup, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    service, _, directory = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(directory))
    st.cache_resource.clear()
    try:
        page = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=20).run()
        next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-a').run()
        page.chat_input[0].set_value('查一下1001和1002，再退掉那笔').run()
        assert not any(button.label == '确认申请' for button in page.button)
        next(t for t in page.text_input if t.label == '更正当前订单编号').set_value('1001')
        next(button for button in page.button if button.label == '设置当前订单').click().run()
        page.chat_input[0].set_value('帮我申请退款').run()
        assert [button.label for button in page.chat_message[-1].button] == ['确认申请', '取消操作']
        next(t for t in page.text_input if t.label == '回复偏好').set_value('自定义回复偏好')
        next(t for t in page.text_input if t.label == '联系偏好').set_value('仅用于账户A的联系偏好')
        next(button for button in page.button if button.label == '保存客户偏好').click().run()
        assert service.profile(A)['tone'] == '自定义回复偏好'
        next(button for button in page.button if button.label == '删除回复偏好').click().run()
        assert 'tone' not in service.profile(A)
        next(button for button in page.button if button.label == '清除当前会话聊天').click().run()
        assert not page.chat_message and any(button.label == '确认申请' for button in page.button)
        next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-b').run()
        assert not page.chat_message and not any(button.label == '确认申请' for button in page.button)
        assert next(t for t in page.text_input if t.label == '联系偏好').value != '仅用于账户A的联系偏好'
        assert not page.exception
    finally:
        st.cache_resource.clear()
