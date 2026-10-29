"""Removed parameters and preserved archived namespaces are API boundaries."""
import inspect
import sqlite3
from uuid import NAMESPACE_URL, uuid5
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
import api
from src import tools
from src.auth import Principal
from src.backend.db import ActionRecord, PendingAction, RequestRecord, get_engine, get_session_factory
from src.backend.seed import seed
from src.requests import canonical
from src.runtime_context import active_request
from src.service import SupportService
from tests.online_support import call, install_script, plan, task


A = Principal('customer_a')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(api, 'service', service)
    yield service, engine
    engine.dispose()


@pytest.mark.parametrize('value', ['demo', 'online', ''])
@pytest.mark.parametrize('path', ['/health', '/actions', '/runs', '/memory/session', '/profile'])
def test_removed_query_mode_is_explicitly_rejected(setup, value, path):
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    response = client.get(path, params={'mode': value})
    assert response.status_code == 422 and 'mode 参数已移除' in response.json()['detail']


@pytest.mark.parametrize('method,path,body', [
    ('post', '/chat', {'message': '查询付款状态'}),
    ('patch', '/memory/session/order', {'session_id': 'one', 'order_id': 1001, 'expected_version': 0}),
    ('delete', '/memory/session/history', {'session_id': 'one', 'expected_version': 0}),
    ('post', '/actions/missing/confirm', {}), ('post', '/actions/missing/cancel', {}),
    ('post', '/runs/missing/resume', {})])
@pytest.mark.parametrize('value', ['demo', 'online'])
def test_removed_body_mode_is_rejected_before_execution(setup, method, path, body, value):
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    response = client.request(method, path, json={**body, 'mode': value})
    assert response.status_code == 422
    assert any(item['loc'][-1] == 'mode' and item['type'] == 'extra_forbidden' for item in response.json()['detail'])


def test_public_contract_has_no_mode_or_rules_model(setup):
    from pathlib import Path
    from src import memory, planning, runtime_context
    from src import llm
    assert not Path('src/demo_model.py').exists() and not Path('scripts/demo.py').exists()
    assert not hasattr(runtime_context, 'active_mode')
    assert not any(hasattr(memory, name) for name in ('extract_order_ids', 'resolve_order_reference'))
    assert not any(hasattr(planning, name) for name in ('allows_write', 'split_task_clauses', 'plan_tasks'))
    assert 'demo' not in llm.SUPPORTED_PROVIDERS
    for method in (SupportService.chat, SupportService.snapshot, SupportService.list_runs, SupportService.list_pending_actions):
        assert 'mode' not in inspect.signature(method).parameters
    schema = api.app.openapi()
    assert 'mode' not in schema['components']['schemas']['ChatRequest']['properties']
    for path in schema['paths'].values():
        for operation in path.values():
            assert not any(parameter['name'] == 'mode' for parameter in operation.get('parameters', []))


def archive_fixture(service, mode, request_id):
    """Seed old persisted records; this is data, never an executable old model."""
    journal = tools.request_journal()
    context, _, _ = journal.begin_request(A.customer_id, mode, request_id, 'same', '旧付款请求')
    token = active_request.set(context)
    try:
        proposal = tools.process_payment(1003, 'card')
    finally:
        active_request.reset(token)
    result = {'reply': '旧待确认正文', 'status': 'pending_confirmation', 'reply_source': 'model', 'agent': 'payment',
        'mode': mode, 'usage': {}, 'session_version': 0,
        'trace': [{'agent': 'payment', 'tool': 'process_payment', 'args': {'order_id': 1003, 'method': 'card'}, 'result': proposal}]}
    journal.finish_request(context, result)
    return context, proposal


def test_archived_data_remains_unchanged_and_all_legacy_entrypoints_are_closed(setup, monkeypatch):
    service, engine = setup
    old, proposal = archive_fixture(service, 'demo', 'shared')
    payload = {'turns': [{'user': '旧离线消息', 'reply': '不能混入在线'}], 'last_order_id': 1003, 'total_tokens': 999}
    service.store.save_session(A.customer_id, 'demo', 'same', payload)
    service.save_profile(A, '保留共享偏好', '文字')
    run = service.store.create_run(old, 1, {'pipeline_version': 'legacy-v1'})
    with sqlite3.connect(run['checkpoint_path']) as db:
        db.execute('CREATE TABLE archived_marker(value TEXT)')
        db.execute("INSERT INTO archived_marker VALUES('保留原检查点')")
    with get_session_factory(engine)() as db:
        before = [tuple(vars(row)[key] for key in row.__table__.columns.keys()) for model in (RequestRecord, ActionRecord, PendingAction)
                  for row in db.scalars(select(model).where(model.mode == 'demo'))]
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    assert client.get('/memory/session', params={'session_id': 'same'}).json()['turns'] == []
    assert client.get('/actions', params={'session_id': 'same'}).json()['actions'] == []
    assert client.get('/runs', params={'session_id': 'same'}).json()['runs'] == []
    for suffix in ('', '/confirm', '/cancel'):
        response = client.get('/actions/' + proposal['action_id']) if not suffix else client.post('/actions/' + proposal['action_id'] + suffix)
        assert response.status_code == 404
    assert client.get('/runs/' + run['run_id']).status_code == 404
    assert client.post('/runs/' + run['run_id'] + '/resume').status_code == 404
    calls = install_script(monkeypatch, [plan(task('payment_status', [1006])), call('get_payment_status', {'order_id': 1006}), '已付款。'])
    result = service.chat(A, 'same', '线上查询', request_id='shared')
    assert result['status'] == 'completed' and not result['replayed'] and len(calls) == 3
    assert service.snapshot(A, 'same')['total_tokens'] != 999 and 'mode' not in result
    assert service.profile(A)['tone'] == '保留共享偏好'
    assert service.store.load_session(A.customer_id, 'demo', 'same')['turns'] == payload['turns']
    with get_session_factory(engine)() as db:
        after = [tuple(vars(row)[key] for key in row.__table__.columns.keys()) for model in (RequestRecord, ActionRecord, PendingAction)
                 for row in db.scalars(select(model).where(model.mode == 'demo'))]
    assert after == before
    with sqlite3.connect(run['checkpoint_path']) as db:
        assert db.execute('SELECT value FROM archived_marker').fetchone()[0] == '保留原检查点'


def test_existing_online_proposal_replays_with_same_action_id_without_model(setup, monkeypatch):
    service, _ = setup
    context, proposal = archive_fixture(service, 'online', 'old-online')
    expected = uuid5(NAMESPACE_URL, canonical([*context.key, 'process_payment', 1003])).hex
    assert proposal['action_id'] == expected
    calls = install_script(monkeypatch, [])
    result = SupportService(service.store.path).chat(A, 'same', '旧付款请求', request_id='old-online')
    assert result['replayed'] and result['pending_actions'][0]['action_id'] == expected and 'mode' not in result
    assert service.confirm_action(A, expected)['status'] == 'completed'
    assert service.confirm_action(A, expected)['replayed'] and not calls
    assert tools.get_payment_status(1003)['status'] == 'paid'
