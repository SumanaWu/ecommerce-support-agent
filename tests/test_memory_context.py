"""Online regression grouped by responsibility; no rule-model fallback."""
import json
from pathlib import Path
import sqlite3
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
from src.memory import empty_issue
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


def test_low_level_proposal_also_rejects_ambiguous_context(setup):
    service, _, _ = setup
    context, _, _ = tools.request_journal().begin_request('customer_a', 'online', 'direct', 'one', 'test')
    from src.runtime_context import active_request
    request_token = active_request.set(context)
    reference_token = active_order_reference.set({'status': 'ambiguous', 'order_id': None})
    try:
        with pytest.raises(AccessError):
            tools.initiate_refund(1001, 'model guessed')
    finally:
        active_order_reference.reset(reference_token)
        active_request.reset(request_token)


def test_issue_edit_uses_session_version_and_clear_does_not_leak_into_new_sessions(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, [plan(kind='clarify', message='请提供付款订单编号。')])
    service, _, _ = setup
    first = service.set_active_order(A, 'one', 1001, 0)
    with pytest.raises(SessionConflict):
        service.set_active_order(A, 'one', 1002, 0)
    cleared = service.set_active_order(A, 'one', None, first['version'])
    assert not cleared['issue']['candidate_order_ids'] and cleared['issue']['active_order_id'] is None
    assert service.chat(A, 'one', '帮我付款')['status'] == 'needs_clarification'


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


def test_recent_dialogue_is_bounded_and_never_split():
    turns = [{'request_id': str(n), 'user': '问题' + str(n), 'reply': '回答' + str(n), 'trace': []} for n in range(12)]
    selected, messages, info = build_context(turns, '继续', empty_issue(), {}, budget=1500)
    assert selected == turns[-6:] and len(messages) == 12
    assert all(isinstance(messages[n], HumanMessage) and isinstance(messages[n + 1], AIMessage) for n in range(0, 12, 2))
    assert info['selected_turns'] == 6 and info['is_estimate']


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


def test_planning_reference_budget_removes_whole_turns_and_keeps_current_context():
    from src.context import fit_structured_context
    context = {'user_input': '1006', 'pending_clarification': {'messages': ['查询付款状态'], 'question': '订单编号？'},
               'issue': empty_issue(), 'profile': {'tone': '简短'}, 'recent_turns': [
                   {'tasks': [{'operation': 'order_details', 'order_ids': list(range(1000000000000, 1000000000020))} for _ in range(6)]},
                   {'tasks': [{'operation': 'payment_status', 'order_ids': [1006]}]}]}
    messages = [SystemMessage('内部规划器，只返回JSON。'), HumanMessage(json.dumps(context, ensure_ascii=False))]
    original = messages[-1].content
    fitted, info = fit_structured_context(messages, budget=600)
    current = json.loads(fitted[-1].content)
    assert current == {**context, 'recent_turns': context['recent_turns'][1:], 'recent_facts': []}
    assert info['removed_turns'] == 1 and info['history_turns'] == 1
    assert info['history_format'] == 'conversation_facts' and estimate_context_tokens(fitted) <= 600
    assert messages[-1].content == original


@pytest.mark.parametrize('field', ['user_input', 'pending_clarification'])
def test_planning_budget_does_not_slice_mandatory_input_or_pending(field):
    from src.context import fit_structured_context
    context = {'user_input': '1006', 'pending_clarification': None, 'recent_turns': []}
    context[field] = '必须完整保留' * 5000
    with pytest.raises(ContextBudgetExceeded):
        fit_structured_context([SystemMessage('只返回JSON。'), HumanMessage(json.dumps(context, ensure_ascii=False))], budget=600)


def test_planning_context_preserves_dialogue_but_only_trusts_tool_facts():
    from src.context import planning_context
    turns = [{'request_id': 'read', 'user': '查付款', 'reply': '模型声称999元',
              'trace': [{'tool': 'get_payment_status', 'result': {'order_id': 1006, 'status': 'paid', 'amount_minor': 5999}}],
              'tasks': [{'operation': 'payment_status', 'order_ids': [1006]}]},
             {'user': '失败请求', 'reply': '技术失败', 'trace': []}]
    result = planning_context(turns)
    assert result['recent_turns'][0]['reply'] == '模型声称999元'
    assert result['recent_facts'][0]['result']['amount_minor'] == 5999
    assert result['recent_facts'][0]['source_request_id'] == 'read' and result['recent_facts'][0]['historical']


def test_planning_context_keeps_legacy_tasks_and_recent_order():
    from src.context import planning_context
    turns = [{'user': '先查付款', 'reply': '已付款', 'tasks': [{'operation': 'payment_status', 'order_ids': [1006]}]},
             {'user': '查详情', 'reply': '详情', 'tasks': [{'operation': 'order_details', 'order_id': 1001}]},
             {'user': '再查付款', 'reply': '查询失败', 'tasks': [{'operation': 'payment_status', 'order_ids': [1006]}]}]
    packet = planning_context(turns)
    assert [turn['order_ids'] for turn in packet['recent_turns']] == [[1006], [1001], [1006]]
    assert packet['recent_facts'] == []


def test_budget_exhaustion_keeps_a_created_proposal_and_stops_model_calls(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, proposal_steps())
    service, engine, _ = setup
    original = base.fit_model_context
    def fail_after_proposal(messages, agent_tools):
        if any(isinstance(message, ToolMessage) and message.name == 'initiate_refund' for message in messages):
            raise ContextBudgetExceeded('test budget')
        return original(messages, agent_tools)
    monkeypatch.setattr(base, 'fit_model_context', fail_after_proposal)
    result = service.chat(A, 'one', '帮我申请订单1001的退款', request_id='budget')
    assert result['status'] == 'pending_confirmation' and result['error_type'] == 'context_budget'
    assert len(rows(engine, ActionRecord)) == len(rows(engine, PendingAction)) == 1
    assert next(row for row in rows(engine, Refund) if row.order_id == 1001).status == 'none'
    assert service.chat(A, 'one', '帮我申请订单1001的退款', request_id='budget')['replayed']


def test_api_memory_and_profile_require_authentication_versions_and_owned_orders(setup):
    service, _, _ = setup
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    other = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    assert TestClient(api.app).get('/profile').status_code == 401
    body = {'session_id': 'one', 'order_id': 1001, 'expected_version': 0}
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
    assert service.snapshot(B, 'one')['issue'] == empty_issue()


def test_chat_cleanup_preserves_audits_profile_actions_and_deduplication(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, proposal_steps())
    service, engine, _ = setup
    service.update_profile(A, {'tone': '简洁'}, 0)
    result = service.chat(A, 'one', '帮我申请订单1001的退款', request_id='keep')
    snapshot = service.snapshot(A, 'one')
    cleared = service.clear_history(A, 'one', snapshot['version'])
    assert cleared['turns'] == [] and cleared['issue'] == snapshot['issue']
    assert len(service.list_pending_actions(A, 'one')) == 1 and service.profile(A)['tone'] == '简洁'
    assert len(rows(engine, ActionRecord)) == 1
    assert service.chat(A, 'one', '帮我申请订单1001的退款', request_id='keep')['replayed']
    assert service.snapshot(A, 'one')['turns'] == []
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
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)', ('customer_a','online','one',json.dumps(payload),'original',9))
        db.execute('INSERT INTO profiles VALUES(?,?,?)', ('customer_a','{"tone":"brief"}','original'))
    store = SupportStore(path)
    restored = store.load_session('customer_a', 'online', 'one')
    assert restored['version'] == 9 and restored['total_tokens'] == 17
    assert restored['issue']['active_order_id'] is None and restored['issue']['selection_source'] == 'legacy_last_order'
    assert store.profile_details('customer_a')['fields']['tone']['source'] == 'legacy_user_saved'
    backup = list(tmp_path.glob('old.sqlite.before-v3-*.bak'))
    assert len(backup) == 1
    with sqlite3.connect(backup[0]) as db:
        assert json.loads(db.execute('SELECT payload FROM sessions').fetchone()[0]) == payload
    SupportStore(path)
    assert len(list(tmp_path.glob('old.sqlite.before-v3-*.bak'))) == 1


def test_page_correction_preference_deletion_and_inline_confirmation(setup, monkeypatch):
    from tests.online_support import install_script, plan, proposal_steps, reference_verdict
    steps = proposal_steps()
    steps[0]['tasks'][0]['context_use'] = 'order'
    steps.insert(1, reference_verdict(1001))
    install_script(monkeypatch, [plan(kind='clarify', message='要给1001还是1002退款？', mentioned=[1001, 1002]), *steps])
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


