from src.auth import Principal
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import api
from src import tools
from src.agents import base
from src.backend.db import ActionRecord, Refund, RequestRecord, SchemaMigration, get_engine, get_session_factory, init_db
from src.backend.seed import seed
from src.demo_model import DemoChatModel
from src.errors import invoke_with_retry
from src.requests import RequestConflict
from src.runtime_context import active_request
from src.service import SupportService


@pytest.fixture
def setup(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    yield service, engine, tmp_path
    engine.dispose()


def rows(engine, model):
    with get_session_factory(engine)() as db:
        return db.scalars(select(model)).all()


def test_refund_survives_failed_model_summary_and_replay(setup, monkeypatch):
    service, engine, _ = setup
    calls = []

    class BrokenSummary(DemoChatModel):
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            from langchain_core.messages import ToolMessage
            calls.append(1)
            results = [m for m in messages if isinstance(m, ToolMessage)]
            if results and results[-1].name == 'initiate_refund':
                raise TimeoutError('simulated summary failure')
            self.tools = {'check_refund_eligibility', 'initiate_refund'}
            return super().invoke(messages)

    monkeypatch.setattr(base, 'get_llm', BrokenSummary)
    monkeypatch.setattr('src.errors.time.sleep', lambda _: None)
    first = service.chat(Principal('customer_a'), 'one', 'demo', '帮我申请订单 1001 的退款', request_id='refund-1')
    replay = SupportService(service.store.path).chat(
        Principal('customer_a'), 'one', 'demo', '帮我申请订单 1001 的退款', request_id='refund-1',
        runner=lambda *_a, **_k: pytest.fail('replay executed graph'))
    assert len(calls) == 4  # eligibility, write, summary and summary-only retry
    assert first['status'] == 'pending_confirmation' and first['reply_source'] == 'fallback'
    assert '待确认' in first['reply'] and '1001' in first['reply']
    assert replay['reply'] == first['reply'] and replay['replayed']
    assert service.confirm_action(Principal('customer_a'),first['pending_actions'][0]['action_id'])['status']=='completed'
    assert len(rows(engine, ActionRecord)) == 2
    assert len(service.snapshot(Principal('customer_a'), 'one', 'demo')['turns']) == 2


def test_memory_failure_rolls_back_receipt_and_replay_repairs_once(setup, monkeypatch):
    service, engine, _ = setup
    original = service.store._write_session

    def fail_after_write(*args):
        original(*args)
        raise sqlite3.OperationalError('simulated disk error')

    with monkeypatch.context() as patch:
        patch.setattr(service.store, '_write_session', fail_after_write)
        first = service.chat(Principal('customer_a'), None, 'demo', '帮我付款订单 1003', request_id='pay-1')
    assert not first['memory_saved']
    assert service.snapshot(Principal('customer_a'), first['session_id'], 'demo')['turns'] == []
    with service.store.connect() as db:
        assert db.execute('SELECT count(*) FROM session_requests').fetchone()[0] == 0
    replay = service.chat(Principal('customer_a'), None, 'demo', '帮我付款订单 1003', request_id='pay-1')
    again = service.chat(Principal('customer_a'), None, 'demo', '帮我付款订单 1003', request_id='pay-1')
    assert replay['session_id'] == first['session_id'] == again['session_id']
    assert replay['memory_saved'] and replay['replayed']
    assert len(rows(engine, ActionRecord)) == 1
    assert len(service.snapshot(Principal('customer_a'), replay['session_id'], 'demo')['turns']) == 1


def test_conflicting_request_does_not_execute(setup):
    service, _, _ = setup
    service.chat(Principal('customer_a'), 'one', 'demo', '查询订单 1001', request_id='one')
    for message, session in [('查询订单 1002', 'one'), ('查询订单 1001', 'two')]:
        with pytest.raises(RequestConflict):
            service.chat(Principal('customer_a'), session, 'demo', message, request_id='one')


def test_duplicate_live_request_is_in_progress(setup):
    service, _, _ = setup
    entered, release = Event(), Event()
    runs = []

    def runner(*_a, **_k):
        runs.append(1)
        entered.set()
        assert release.wait(10)
        return {'reply': 'done', 'agent': 'order', 'trace': [], 'usage': {}}

    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(service.chat, Principal('customer_a'), 'one', 'demo', 'query', runner, 'live')
        assert entered.wait(10)
        try:
            duplicate = SupportService(service.store.path).chat(Principal('customer_a'), 'one', 'demo', 'query', request_id='live')
            assert duplicate['status'] == 'in_progress'
            with pytest.raises(RequestConflict):
                service.chat(Principal('customer_a'), 'one', 'demo', 'different', request_id='live')
        finally:
            release.set()
        assert first.result()['status'] == 'completed'
    assert len(runs) == 1


def test_action_result_and_business_change_are_atomic(setup):
    from src.actions import ConfirmationGrant
    from src.runtime_context import active_confirmation
    from src.requests import canonical
    service, engine, _ = setup
    journal = tools.request_journal()
    context, _, _ = journal.begin_request('customer_a', 'demo', 'rollback', 'one', 'refund')

    def failing_apply(db):
        tools.apply_refund(db, 1001, 'test')
        raise RuntimeError('failure before receipt commit')

    token=active_confirmation.set(ConfirmationGrant('test','customer_a','initiate_refund',canonical({'order_id':1001,'reason':'test'})))
    try:
        with pytest.raises(RuntimeError):
            journal.execute_action_once(context, 'initiate_refund', {'order_id': 1001}, failing_apply)
    finally:
        active_confirmation.reset(token)
    assert not rows(engine, ActionRecord)
    assert next(r for r in rows(engine, Refund) if r.order_id == 1001).status == 'none'


def test_action_parameters_cannot_change_and_success_is_replayed(setup):
    _, engine, _ = setup
    journal = tools.request_journal()
    context, _, _ = journal.begin_request('customer_a', 'demo', 'action', 'one', 'refund')
    token = active_request.set(context)
    try:
        first = tools.initiate_refund(1001, 'original')
        assert tools.initiate_refund(1001, 'original') == first
        with pytest.raises(RequestConflict):
            tools.initiate_refund(1001, 'different')
    finally:
        active_request.reset(token)
    assert len(rows(engine, ActionRecord)) == 1


def test_generic_failure_after_write_does_not_restart_graph(setup, monkeypatch):
    from src import actions
    service, engine, _ = setup
    runs = []
    proposal=service.chat(Principal('customer_a'),'one','demo','帮我付款订单 1003')
    original=actions.decide_action

    def fail_after_commit(*args, **kwargs):
        runs.append(1)
        original(*args, **kwargs)
        raise RuntimeError('failure after commit')

    monkeypatch.setattr(actions,'decide_action',fail_after_commit)
    result = service.confirm_action(Principal('customer_a'),proposal['pending_actions'][0]['action_id'])
    assert len(runs) == 1 and result['status'] == 'partial'
    assert 'paid' in result['reply']
    assert len(rows(engine, ActionRecord)) == 2
    assert active_request.get() is None


def test_lost_response_after_finish_is_replayed(setup, monkeypatch):
    service, engine, _ = setup
    from src.requests import RequestJournal
    original = RequestJournal.finish_request

    def fail_after_finish(*args):
        original(*args)
        raise ConnectionError('simulated lost response')

    with monkeypatch.context() as patch:
        patch.setattr(RequestJournal, 'finish_request', fail_after_finish)
        with pytest.raises(ConnectionError):
            service.chat(Principal('customer_a'), None, 'demo', '帮我申请订单 1001 的退款', request_id='lost')
    replay = service.chat(Principal('customer_a'), None, 'demo', '帮我申请订单 1001 的退款', request_id='lost')
    assert replay['replayed'] and replay['status'] == 'pending_confirmation'
    assert len(rows(engine, ActionRecord)) == 1
    assert len(service.snapshot(Principal('customer_a'), replay['session_id'], 'demo')['turns']) == 1


def test_request_key_is_scoped_to_customer_and_mode(setup):
    service, engine, _ = setup
    runner = lambda *_a, **_k: {'reply': 'ok', 'agent': 'order', 'trace': [], 'usage': {'total_tokens': 10}}
    for user, mode in [('customer_a', 'demo'), ('customer_b', 'demo'), ('customer_a', 'online')]:
        service.chat(Principal(user), 'one', mode, 'query', runner=runner, request_id='same')
    assert len(rows(engine, RequestRecord)) == 3
    service.chat(Principal('customer_a'), 'one', 'demo', 'query', runner=runner, request_id='same')
    assert service.snapshot(Principal('customer_a'), 'one', 'demo')['total_tokens'] == 10


def test_receipt_survives_history_window(setup):
    service, _, _ = setup
    runner = lambda *_a, **_k: {'reply': 'ok', 'agent': 'order', 'trace': [], 'usage': {}}
    for i in range(51):
        service.chat(Principal('customer_a'), 'one', 'demo', str(i), runner=runner, request_id=str(i))
    service.chat(Principal('customer_a'), 'one', 'demo', '0', runner=runner, request_id='0')
    assert [t['user'] for t in service.snapshot(Principal('customer_a'), 'one', 'demo')['turns']] == [str(i) for i in range(1, 51)]


def test_api_replay_auto_session_conflict_and_refusal(setup, monkeypatch):
    service, _, _ = setup
    monkeypatch.setattr(api, 'service', service)
    client = TestClient(api.app, headers={'Authorization':'Bearer test-token-a'})
    body = {'message': '查询订单 1002 的物流', 'request_id': 'api-replay'}
    first = client.post('/chat', json=body)
    second = client.post('/chat', json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()['session_id'] == second.json()['session_id']
    assert second.json()['replayed']
    assert client.post('/chat', json={**body, 'message': 'different'}).status_code == 409
    rejected = client.post('/chat', json={'message': '帮我申请退款', 'session_id': first.json()['session_id']})
    assert rejected.json()['status'] == 'rejected'
    assert client.post('/chat', json={**body, 'request_id': '../bad'}).status_code == 422


def test_api_returns_202_for_running_request(setup, monkeypatch):
    service, _, _ = setup
    monkeypatch.setattr(api, 'service', service)
    journal = tools.request_journal()
    with journal.request_guard('customer_a', 'demo', 'busy'):
        journal.begin_request('customer_a', 'demo', 'busy', 'one', 'query')
        response = TestClient(api.app, headers={'Authorization':'Bearer test-token-a'}).post('/chat', json={'message': 'query', 'session_id': 'one', 'request_id': 'busy'})
    assert response.status_code == 202 and response.json()['request_id'] == 'busy'


def test_process_crash_recovers_committed_action_without_reexecuting(setup):
    from src.backend.db import PendingAction
    service, engine, directory = setup
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url))
    script = '''
import os
from src.service import SupportService
from src.auth import Principal
from src.requests import RequestJournal
s=SupportService(); p=Principal('customer_a')
proposal=s.chat(p,None,'demo','帮我申请订单 1001 的退款',request_id='crashed')
original=RequestJournal.finish_request
def finish(self,context,result):
    if context.request_id.startswith('confirm_'): os._exit(0)
    return original(self,context,result)
RequestJournal.finish_request=finish
s.confirm_action(p,proposal['pending_actions'][0]['action_id'])
'''
    subprocess.run([sys.executable, '-c', script], env=env, check=True, timeout=20)
    result = service.confirm_action(Principal('customer_a'),rows(engine,PendingAction)[0].action_id)
    assert result['replayed'] and result['status'] == 'completed'
    assert '1001' in result['reply'] and 'requested' in result['reply']
    assert len(rows(engine, ActionRecord)) == 2


def test_process_lock_rejects_other_worker(setup):
    _, engine, directory = setup
    env = dict(os.environ, SUPPORT_DATA_DIR=str(directory), SUPPORT_AGENT_DB_URL=str(engine.url))
    journal = tools.request_journal()
    with journal.request_guard('customer_a', 'demo', 'worker'):
        journal.begin_request('customer_a', 'demo', 'worker', 'one', 'query')
        subprocess.run([sys.executable, '-c',
                        "from src.auth import Principal; from src.service import SupportService; r=SupportService().chat(Principal('customer_a'),'one','demo','query',request_id='worker'); assert r['status']=='in_progress'"],
                       env=env, check=True, timeout=20)


def test_request_lock_uses_canonical_database_path(setup):
    from src.requests import RequestJournal
    _, engine, directory = setup
    alias = get_engine('sqlite:///' + os.path.relpath(directory / 'business.sqlite'))
    try:
        first = tools.request_journal()
        second = RequestJournal(get_session_factory(alias), alias)
        with first.request_guard('customer_a', 'demo', 'alias') as acquired:
            assert acquired
            with second.request_guard('customer_a', 'demo', 'alias') as duplicate:
                assert not duplicate
    finally:
        alias.dispose()


def test_interrupted_request_without_actions_is_not_reexecuted(setup):
    service, _, _ = setup
    tools.request_journal().begin_request('customer_a', 'demo', 'empty', 'one', 'query')
    result = service.chat(Principal('customer_a'), 'one', 'demo', 'query', request_id='empty',
                          runner=lambda *_a, **_k: pytest.fail('reran interrupted request'))
    assert result['status'] == 'failed' and result['replayed']


def test_rejected_business_action_does_not_retry_or_create_ticket(setup):
    service, engine, _ = setup
    runs = []
    def runner(*_a, **_k):
        runs.append(1)
        result = tools.process_payment(1001, 'card')
        return {'reply': 'already paid', 'agent': 'payment', 'trace': [
            {'agent': 'payment', 'tool': 'process_payment', 'args': {'order_id': 1001}, 'result': result}], 'usage': {}}
    response = service.chat(Principal('customer_a'), 'one', 'demo', 'pay order 1001', runner=runner, request_id='refused')
    assert response['status'] == 'rejected' and len(runs) == 1
    assert len(rows(engine, ActionRecord)) == 1


def test_configuration_error_is_safe_and_does_not_retry(setup):
    service, engine, _ = setup
    from src.errors import ModelConfigurationError
    runs = []
    def runner(*_a, **_k):
        runs.append(1)
        raise ModelConfigurationError('unit-secret-must-not-leak')
    response = service.chat(Principal('customer_a'), 'one', 'online', 'query', runner=runner)
    assert response['status'] == 'failed' and response['error_type'] == 'configuration'
    assert 'unit-secret' not in response['reply'] and len(runs) == 1
    assert not rows(engine, ActionRecord)


def test_page_retries_same_id_after_memory_failure(setup, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    from src.storage import SupportStore
    _, engine, directory = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(directory))
    st.cache_resource.clear()
    original = SupportStore._write_session
    failures = []
    def fail_once(*args):
        if not failures:
            failures.append(1)
            raise sqlite3.OperationalError('simulated memory failure')
        return original(*args)
    monkeypatch.setattr(SupportStore, '_write_session', fail_once)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
    next(t for t in app.text_input if t.label=='测试账户令牌').set_value('test-token-a').run()
    app.chat_input[0].set_value('帮我申请订单 1001 的退款').run()
    assert not app.exception
    request_id = next(iter(app.session_state['pending_requests'].values()))['request_id']
    next(button for button in app.button if button.label == '重试同一请求').click().run()
    assert not app.exception and app.session_state['pending_requests'] == {}
    assert len(rows(engine, ActionRecord)) == 1
    with get_session_factory(engine)() as db:
        assert db.get(RequestRecord, ('customer_a', 'demo', request_id)) is not None
    assert len(app.chat_message) == 2
    st.cache_resource.clear()


def test_additive_migration_preserves_legacy_business_rows(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_name VARCHAR(100), item VARCHAR(200), status VARCHAR(20), order_date DATE, amount FLOAT)')
        db.execute("INSERT INTO orders VALUES (1001,'Existing','Item','delivered','2026-01-01',12.34)")
    engine = get_engine('sqlite:///' + str(path))
    try:
        init_db(engine)
        init_db(engine)
        with sqlite3.connect(path) as db:
            assert db.execute('SELECT amount_minor,currency FROM orders').fetchall() == [(1234, 'CNY')]
        assert len(rows(engine, SchemaMigration)) == 4
        assert not rows(engine, RequestRecord)
    finally:
        engine.dispose()


def test_memory_migration_preserves_legacy_history_and_profile(tmp_path):
    import json
    from src.storage import SupportStore
    path = tmp_path / 'legacy-memory.sqlite'
    payload = {'turns': [{'user': 'old', 'reply': 'saved', 'agent': 'order'}],
               'last_order_id': 1001, 'total_tokens': 5}
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE sessions (user_id TEXT, mode TEXT, session_id TEXT, payload TEXT, updated_at TEXT, PRIMARY KEY(user_id,mode,session_id)); CREATE TABLE profiles (user_id TEXT PRIMARY KEY, payload TEXT, updated_at TEXT);')
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?)', ('customer_a', 'demo', 'one', json.dumps(payload), 'old'))
        db.execute('INSERT INTO profiles VALUES(?,?,?)', ('customer_a', '{"tone":"brief"}', 'old'))
    store = SupportStore(path)
    restored = store.load_session('customer_a', 'demo', 'one')
    assert {key: restored[key] for key in payload} == payload
    assert restored['version'] == 0 and restored['issue']['candidate_order_ids'] == [1001]
    assert restored['issue']['active_order_id'] is None
    assert store.profile('customer_a') == {'tone': 'brief'}
    with store.connect() as db:
        assert db.execute('SELECT version FROM schema_migrations').fetchall() == [(1,), (2,), (3,)]
    assert len(list(tmp_path.glob('legacy-memory.sqlite.before-v2-*.bak'))) == 1


@pytest.mark.parametrize('error,expected_calls', [(TimeoutError('offline'), 2), (ValueError('bad'), 1), (RuntimeError('bug'), 1)])
def test_only_model_transport_errors_are_retried(error, expected_calls):
    calls = []
    def fail():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        invoke_with_retry(fail, backoff_seconds=0)
    assert len(calls) == expected_calls
