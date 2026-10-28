import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import api
from src import actions, tools
from src.auth import AccessError, Principal, principal_scope, resolve_principal
from src.backend.db import ActionRecord, Order, Payment, PendingAction, Refund, get_engine, get_session_factory, init_db
from src.backend.seed import seed
from src.runtime_context import active_confirmation, active_principal, active_request
from src.service import SupportService

A, B = Principal('customer_a'), Principal('customer_b')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(api, 'service', service)
    client_a = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    client_b = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    yield service, engine, client_a, client_b
    engine.dispose()


def propose(service, order_id=1001):
    message = f'帮我申请订单 {order_id} 的退款' if order_id != 1003 else '帮我付款订单 1003'
    result = service.chat(A, 'one', 'demo', message)
    assert result['status'] == 'pending_confirmation'
    return result['pending_actions'][0]['action_id']


def state(engine, order_id=1001):
    with get_session_factory(engine)() as db:
        model = Payment if order_id == 1003 else Refund
        return db.scalars(select(model).where(model.order_id == order_id)).one().status


def test_authentication_and_user_id_hint(setup):
    _, _, client_a, _ = setup
    unauthenticated = TestClient(api.app)
    assert unauthenticated.get('/health').status_code == 200
    for method, url, kwargs in [('post', '/chat', {'json': {'message': '查询订单 1001'}}),
                                ('get', '/actions', {}), ('post', '/actions/missing/confirm', {})]:
        assert getattr(unauthenticated, method)(url, **kwargs).status_code == 401
    invalid = TestClient(api.app, headers={'Authorization': 'Bearer wrong-token'})
    assert invalid.post('/chat', json={'message': 'query'}).status_code == 401
    assert client_a.post('/chat', json={'message': 'query', 'user_id': 'customer_b'}).status_code == 403
    assert client_a.post('/chat', json={'message': '查询订单 1001', 'user_id': 'customer_a'}).status_code == 200
    assert client_a.post('/chat', json={'message': 'query', 'confirmed': True}).status_code == 422


@pytest.mark.parametrize('config', ['', '{}', '[]', '{bad', '{"a":"same","b":"same"}', '{"bad/id":"token"}', '{"a":3}'])
def test_invalid_account_configuration_is_closed(monkeypatch, config):
    monkeypatch.setenv('SUPPORT_ACCOUNT_TOKENS', config)
    with pytest.raises(AccessError) as error:
        resolve_principal('token')
    assert error.value.status_code == 503
    assert config not in str(error.value) if config else True


def test_principal_cannot_be_taken_from_free_form_customer_id(setup):
    service, _, _, _ = setup
    with pytest.raises(AccessError):
        service.chat('customer_a', 'one', 'demo', 'query')
    with pytest.raises(AccessError):
        service.snapshot('customer_b', 'one', 'demo')
    assert resolve_principal('test-token-a') == A


@pytest.mark.parametrize('tool,args', [
    (tools.get_order_status, {}), (tools.get_order_details, {}), (tools.track_shipment, {}),
    (tools.get_payment_status, {}), (tools.check_refund_eligibility, {}),
    (tools.initiate_refund, {'reason': 'test'}), (tools.process_payment, {'method': 'card'}),
])
def test_all_order_tools_hide_foreign_and_missing_orders(setup, tool, args):
    journal = tools.request_journal()
    context, _, _ = journal.begin_request('customer_a', 'demo', 'tool-check', 'one', 'test')
    token = active_request.set(context)
    try:
        foreign = tool(order_id=1008, **args)
        missing = tool(order_id=9999, **args)
    finally:
        active_request.reset(token)
    assert foreign == missing == {'error': '未找到可访问的订单。'}
    with principal_scope(B):
        assert tools.get_order_details(1008)['order_id'] == 1008
    assert active_principal.get() == A


@pytest.mark.parametrize('order_id', [1001, 1003])
def test_proposal_confirmation_and_idempotent_execution(setup, order_id):
    service, engine, _, _ = setup
    action_id = propose(service, order_id)
    initial = 'none' if order_id == 1001 else 'pending'
    assert state(engine, order_id) == initial
    assert service.get_action(A, action_id)['status'] == 'pending'
    first = service.confirm_action(A, action_id)
    second = SupportService(service.store.path).confirm_action(A, action_id)
    assert first['status'] == second['status'] == 'completed'
    assert second['replayed'] and first['reply'] == second['reply']
    assert state(engine, order_id) == ('requested' if order_id == 1001 else 'paid')
    with get_session_factory(engine)() as db:
        assert db.get(Order, order_id).version == 2
        assert len(db.scalars(select(ActionRecord)).all()) == 2
        assert db.get(PendingAction, action_id).status == 'executed'
    assert len(service.snapshot(A, 'one', 'demo')['turns']) == 2
    assert active_confirmation.get() is None


def test_foreign_action_lookup_confirmation_and_cancellation_are_indistinguishable(setup):
    service, engine, _, client_b = setup
    action_id = propose(service)
    for suffix, method in [('', 'get'), ('/confirm', 'post'), ('/cancel', 'post')]:
        foreign = getattr(client_b, method)(f'/actions/{action_id}{suffix}')
        missing = getattr(client_b, method)(f'/actions/missing{suffix}')
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json()
    assert client_b.get('/actions', params={'session_id': 'one'}).json() == {'actions': []}
    assert state(engine) == 'none'


def test_cancel_is_idempotent_and_cannot_be_confirmed(setup):
    service, engine, client_a, _ = setup
    proposal = service.chat(A, 'one', 'demo', '帮我申请订单 1001 的退款', request_id='cancel-source')
    action_id = proposal['pending_actions'][0]['action_id']
    assert client_a.post(f'/actions/{action_id}/cancel').json()['status'] == 'cancelled'
    assert client_a.post(f'/actions/{action_id}/cancel').json()['replayed']
    assert client_a.post(f'/actions/{action_id}/confirm').status_code == 409
    assert state(engine) == 'none'
    replay = service.chat(A, 'one', 'demo', '帮我申请订单 1001 的退款', request_id='cancel-source')
    assert replay['status'] == 'cancelled' and not replay['pending_actions']
    assert '已取消' in replay['reply'] and '已保存' not in replay['reply']
    new_action = propose(service)
    assert new_action != action_id
    assert service.confirm_action(A, new_action)['status'] == 'completed'


def test_executed_action_cannot_be_cancelled(setup):
    service, engine, client_a, _ = setup
    action_id = propose(service)
    service.confirm_action(A, action_id)
    assert client_a.post(f'/actions/{action_id}/cancel').status_code == 409
    assert state(engine) == 'requested'


@pytest.mark.parametrize('change', ['amount', 'version', 'order_status', 'payment_status', 'refund_status'])
def test_confirmation_rejects_stale_business_facts(setup, change):
    service, engine, _, _ = setup
    action_id = propose(service)
    with get_session_factory(engine).begin() as db:
        if change == 'amount':
            db.get(Order, 1001).amount_minor += 100
        elif change == 'version':
            db.get(Order, 1001).version += 1
        elif change == 'order_status':
            db.get(Order, 1001).status = 'cancelled'
        elif change == 'payment_status':
            db.scalars(select(Payment).where(Payment.order_id == 1001)).one().status = 'refunded'
        else:
            db.scalars(select(Refund).where(Refund.order_id == 1001)).one().status = 'requested'
    response = service.confirm_action(A, action_id)
    assert response['status'] == 'rejected'
    assert response['trace'][0]['result']['error_code'] == 'stale_action'
    with get_session_factory(engine)() as db:
        assert db.get(PendingAction, action_id).status == 'rejected'
    assert state(engine) == ('requested' if change == 'refund_status' else 'none')


def test_order_ownership_and_token_are_rechecked_at_confirmation(setup, monkeypatch):
    service, engine, client_a, client_b = setup
    action_id = propose(service)
    monkeypatch.setenv('SUPPORT_ACCOUNT_TOKENS', '{"customer_b":"test-token-b"}')
    assert client_a.post(f'/actions/{action_id}/confirm').status_code == 401
    monkeypatch.setenv('SUPPORT_ACCOUNT_TOKENS', '{"customer_a":"test-token-a","customer_b":"test-token-b"}')
    with get_session_factory(engine).begin() as db:
        db.get(Order, 1001).customer_id = 'customer_b'
    assert client_a.post(f'/actions/{action_id}/confirm').status_code == 404
    assert client_b.post(f'/actions/{action_id}/confirm').status_code == 404
    assert state(engine) == 'none'


def test_confirmation_body_cannot_override_frozen_arguments(setup):
    service, engine, client_a, _ = setup
    action_id = propose(service)
    for forged in [{'amount': 0}, {'order_id': 1008}, {'user_id': 'customer_b'}, {'confirmed': True}]:
        assert client_a.post(f'/actions/{action_id}/confirm', json=forged).status_code == 422
    assert state(engine) == 'none'
    assert client_a.post(f'/actions/{action_id}/confirm', json={}).status_code == 200


def test_raw_apply_functions_require_an_unforgeable_server_grant(setup):
    _, engine, _, _ = setup
    with get_session_factory(engine).begin() as db:
        with pytest.raises(AccessError):
            tools.apply_refund(db, 1001, 'test')
        with pytest.raises(AccessError):
            tools.apply_payment(db, 1003, 'card')
    assert state(engine) == 'none' and state(engine, 1003) == 'pending'


def test_model_cannot_invoke_confirmation_or_extra_identity_parameters(setup, monkeypatch):
    from src.agents import base
    from src.agents.refund_agent import REFUND_TOOLS
    from tests.test_agent_base import FakeLLM, FakeToolCallMessage
    service, engine, _, _ = setup
    for name, args in [('confirm_action', {'action_id': 'fake'}),
                       ('initiate_refund', {'order_id': 1001, 'reason': 'test', 'confirmed': True}),
                       ('initiate_refund', {'order_id': 1001, 'reason': 'test', 'user_id': 'customer_b'})]:
        monkeypatch.setattr(base, 'get_llm', lambda: FakeLLM([
            FakeToolCallMessage(tool_calls=[{'name': name, 'args': args, 'id': 'call'}]),
            FakeToolCallMessage(content='done')]))
        service.chat(A, 'one', 'demo', 'test', runner=lambda *a, **k: base.run_tool_agent('refund', 'test', REFUND_TOOLS, 'test'))
    assert state(engine) == 'none'
    with get_session_factory(engine)() as db:
        assert not db.scalars(select(PendingAction)).all()


def test_backend_corrects_model_claim_of_success_before_confirmation(setup):
    service, engine, _, _ = setup
    def runner(*a, **k):
        result = tools.initiate_refund(1001, 'test')
        return {'reply': '退款已成功', 'agent': 'refund', 'trace': [
            {'agent': 'refund', 'tool': 'initiate_refund', 'args': {'order_id': 1001}, 'result': result}], 'usage': {}}
    response = service.chat(A, 'one', 'demo', 'refund order 1001', runner=runner)
    assert response['status'] == 'pending_confirmation' and '尚未办理' in response['reply']
    assert state(engine) == 'none'


def test_confirmed_business_success_survives_memory_failure(setup, monkeypatch):
    service, engine, _, _ = setup
    action_id = propose(service, 1003)
    def fail(*args):
        raise sqlite3.OperationalError('simulated memory failure')
    with monkeypatch.context() as patch:
        patch.setattr(service.store, '_write_session', fail)
        first = service.confirm_action(A, action_id)
    assert first['status'] == 'completed' and not first['memory_saved']
    assert state(engine, 1003) == 'paid'
    second = service.confirm_action(A, action_id)
    assert second['memory_saved'] and second['replayed']
    assert len(service.snapshot(A, 'one', 'demo')['turns']) == 2
    with get_session_factory(engine)() as db:
        assert db.get(Order, 1003).version == 2


def test_replay_and_history_hide_order_facts_after_ownership_changes(setup):
    service, engine, _, _ = setup
    source = service.chat(A, 'one', 'demo', '查询订单 1001 的详情', request_id='old-owner')
    assert 'Alice' in source['reply']
    with get_session_factory(engine).begin() as db:
        db.get(Order, 1001).customer_id = 'customer_b'
    snapshot = service.snapshot(A, 'one', 'demo')
    assert snapshot['turns'] == [] and snapshot['last_order_id'] is None
    replay = service.chat(A, 'one', 'demo', '查询订单 1001 的详情', request_id='old-owner')
    assert replay['status'] == 'failed' and 'Alice' not in replay['reply']
    assert replay['trace'] == []


def test_migration_backs_up_and_does_not_guess_unknown_owners(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_name VARCHAR(100), item VARCHAR(200), status VARCHAR(20), order_date DATE, amount FLOAT)')
        for order_id in (1001, 1008, 4242):
            db.execute('INSERT INTO orders VALUES(?,?,?,?,?,?)', (order_id, 'Same Name', 'Item', 'delivered', '2026-01-01', 12.34))
    engine = get_engine('sqlite:///' + str(path))
    try:
        init_db(engine)
        init_db(engine)
        backups = list(tmp_path.glob('*.before-v2-*.bak'))
        assert len(backups) == 1
        with sqlite3.connect(backups[0]) as db:
            assert 'customer_id' not in [row[1] for row in db.execute('PRAGMA table_info(orders)')]
            assert db.execute('SELECT count(*) FROM orders').fetchone()[0] == 3
        with get_session_factory(engine)() as db:
            assert [(db.get(Order, i).customer_id, db.get(Order, i).amount_minor, db.get(Order, i).currency) for i in (1001, 1008, 4242)] == [
                ('customer_a', 1234, 'CNY'), ('customer_b', 1234, 'CNY'), (None, 1234, 'CNY')]
    finally:
        engine.dispose()


def test_confirmation_is_idempotent_across_processes(setup):
    service, engine, _, _ = setup
    action_id = propose(service)
    env = dict(os.environ, SUPPORT_DATA_DIR=str(Path(service.store.path).parent), SUPPORT_AGENT_DB_URL=str(engine.url))
    code = "from src.service import SupportService; from src.auth import Principal; r=SupportService().confirm_action(Principal('customer_a'), '" + action_id + "'); assert r['status'] in {'completed','in_progress'}"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20), range(2)))
    assert all(result.returncode == 0 for result in results), [result.stderr for result in results]
    assert service.confirm_action(A, action_id)['replayed']
    assert state(engine) == 'requested'
    with get_session_factory(engine)() as db:
        assert len(db.scalars(select(ActionRecord)).all()) == 2


def test_account_switch_in_page_hides_other_accounts_pending_actions(setup, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    service, engine, _, _ = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(Path(service.store.path).parent))
    st.cache_resource.clear()
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
    assert len(page.chat_input) == 0
    next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-a').run()
    page.chat_input[0].set_value('帮我申请订单 1001 的退款').run()
    assert not page.exception and state(engine) == 'none'
    assert any(button.label == '确认申请' for button in page.button)
    next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-b').run()
    assert not page.exception and not any(button.label == '确认申请' for button in page.button)
    assert len(page.chat_message) == 0
    next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-a').run()
    next(button for button in page.button if button.label == '确认申请').click().run()
    assert not page.exception and state(engine) == 'requested'
    st.cache_resource.clear()


@pytest.mark.parametrize('order_id,label', [(1001, '确认申请'), (1003, '确认付款')])
def test_page_repairs_confirmation_memory_without_repeating_business(setup, monkeypatch, order_id, label):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src.storage import SupportStore
    service, engine, _, _ = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(Path(service.store.path).parent))
    st.cache_resource.clear()
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
    next(t for t in page.text_input if t.label == '测试账户令牌').set_value('test-token-a').run()
    page.chat_input[0].set_value('帮我申请订单 1001 的退款' if order_id == 1001 else '帮我付款订单 1003').run()
    original = SupportStore._write_session
    failures = []
    def fail_confirmation_once(self, db, user_id, mode, session_id, payload):
        if payload['turns'][-1]['user'].startswith('确认动作') and not failures:
            failures.append(1)
            raise sqlite3.OperationalError('simulated confirmation memory error')
        return original(self, db, user_id, mode, session_id, payload)
    monkeypatch.setattr(SupportStore, '_write_session', fail_confirmation_once)
    next(button for button in page.button if button.label == label).click().run()
    assert not page.exception
    assert state(engine, order_id) == ('requested' if order_id == 1001 else 'paid')
    assert page.session_state['pending_decisions']
    next(button for button in page.button if button.label == '重试动作结果').click().run()
    assert not page.exception and not page.session_state['pending_decisions']
    assert len(service.snapshot(A, 'case_1', 'demo')['turns']) == 2
    with get_session_factory(engine)() as db:
        assert db.get(Order, order_id).version == 2
    st.cache_resource.clear()


def test_database_path_uses_current_configured_directory(tmp_path, monkeypatch):
    target = tmp_path / 'custom-data'
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(target))
    monkeypatch.setenv('SUPPORT_AGENT_DB_URL', '')
    engine = get_engine()
    try:
        assert engine.url.database == str(target / 'business.sqlite')
        init_db(engine)
        assert (target / 'business.sqlite').is_file()
    finally:
        engine.dispose()


def test_explicit_reseed_cannot_revalidate_old_pending_action(setup):
    service, engine, _, _ = setup
    action_id = propose(service)
    with get_session_factory(engine)() as db:
        seed(db)
    response = service.confirm_action(A, action_id)
    assert response['status'] == 'rejected'
    assert response['trace'][0]['result']['error_code'] == 'stale_action'
    assert state(engine) == 'none'


@pytest.mark.parametrize('confirm', [True, False])
def test_interrupted_decision_before_write_can_be_resumed_without_a_model(setup, monkeypatch, confirm):
    service, engine, _, _ = setup
    action_id = propose(service)
    operation = 'confirm' if confirm else 'cancel'
    message = ('确认动作 ' if confirm else '取消动作 ') + action_id
    tools.request_journal().begin_request('customer_a', 'demo', operation + '_' + action_id, 'one', message)
    monkeypatch.setattr('src.service.run_orchestrator', lambda *a, **k: pytest.fail('decision called model graph'))
    response = service.confirm_action(A, action_id) if confirm else service.cancel_action(A, action_id)
    assert response['status'] == ('completed' if confirm else 'cancelled')
    assert state(engine) == ('requested' if confirm else 'none')
    assert response['replayed']


def test_permission_failure_during_confirmation_is_not_a_model_error(setup, monkeypatch):
    service, engine, client_a, _ = setup
    action_id = propose(service)
    def revoked(*args, **kwargs):
        raise AccessError('未找到可访问的动作。', 404)
    monkeypatch.setattr(actions, 'decide_action', revoked)
    response = client_a.post(f'/actions/{action_id}/confirm')
    assert response.status_code == 404
    assert response.json() == {'detail': '未找到可访问的动作。'}
    assert state(engine) == 'none'
