"""Online regression grouped by responsibility; no rule-model fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select
import api
from src import orchestrator, tools
from src.auth import Principal
from src.backend.db import ActionRecord, PendingAction, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.service import SupportService
from src.storage import SessionConflict


A = Principal('customer_a')
MESSAGE = '帮我申请订单 1001 的退款'


@pytest.fixture
def setup(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    env = dict(os.environ, SUPPORT_DATA_DIR=str(tmp_path), SUPPORT_AGENT_DB_URL=str(engine.url))
    yield service, engine, env, tmp_path
    engine.dispose()


def rows(engine, model):
    with get_session_factory(engine)() as db:
        return db.scalars(select(model)).all()


def test_stale_full_session_save_cannot_overwrite_a_turn(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, [plan(task('order_status', [1001])), call('get_order_status', {'order_id': 1001}), '已送达。'])
    service, _, _, _ = setup
    old = service.store.load_session('customer_a', 'online', 'one')
    service.chat(A, 'one', '查询订单 1001')
    with pytest.raises(SessionConflict):
        service.store.save_session_if_version('customer_a', 'online', 'one', old, old['version'])
    assert len(service.snapshot(A, 'one')['turns']) == 1


def test_independent_process_conflict_is_visible_and_repair_preserves_both_turns(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, [plan(task('shipment', [1002])), call('track_shipment', {'order_id': 1002}), '运输中。'])
    service, engine, env, directory = setup
    code = '''
import json,time
from pathlib import Path
from src.auth import Principal
from src import tools
from src.results import fallback_result
from src.service import SupportService
directory=Path(DIRECTORY)
import pytest
from tests.online_support import install_script, proposal_steps
from src import orchestrator
original=orchestrator.model_step

def pause(state):
    if not (directory/'entered').exists():
        (directory/'entered').write_text('yes')
        deadline=time.monotonic()+15
        while not (directory/'release').exists():
            assert time.monotonic()<deadline
            time.sleep(.02)
    return original(state)
orchestrator.model_step=pause
with pytest.MonkeyPatch.context() as patch:
    install_script(patch, proposal_steps())
    result=SupportService().chat(Principal('customer_a'),'one','refund order 1001',request_id='slow')
(directory/'result.json').write_text(json.dumps(result))
'''.replace('DIRECTORY', repr(str(directory)))
    process = subprocess.Popen([sys.executable, '-c', code], env=env)
    try:
        deadline = time.monotonic() + 15
        while not (directory / 'entered').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        first = service.chat(A, 'one', '查询订单 1002 的物流')
        assert first['memory_saved']
        (directory / 'release').write_text('yes')
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    conflict = json.loads((directory / 'result.json').read_text())
    assert conflict['status'] == 'pending_confirmation'
    assert conflict['memory_status'] == 'conflict' and not conflict['memory_saved']
    assert conflict['current_session_version'] == 1
    assert len(rows(engine, ActionRecord)) == 1
    repaired = service.chat(A, 'one', 'refund order 1001', request_id='slow',
                            runner=lambda *a, **k: pytest.fail('repair reran side effects'))
    assert repaired['memory_saved'] and repaired['memory_rebased']
    again = service.chat(A, 'one', 'refund order 1001', request_id='slow')
    payload = service.snapshot(A, 'one')
    assert again['memory_saved'] and len(payload['turns']) == payload['version'] == 2
    assert payload['last_order_id'] == 1002
    assert len(rows(engine, ActionRecord)) == 1


def test_api_reports_session_conflict_with_preserved_business_result(setup, monkeypatch):
    from tests.online_support import call, install_script, plan, proposal_steps, task
    install_script(monkeypatch, [plan(task('order_status', [1002])), call('get_order_status', {'order_id': 1002}), '订单已发货。'])
    service, _, _, _ = setup
    monkeypatch.setattr(api, 'service', service)
    def runner(*a, **k):
        other = SupportService(service.store.path)
        other.chat(A, 'one', '查询订单 1002')
        return {'reply': 'done', 'agent': 'order', 'trace': [], 'usage': {'total_tokens': 7}}
    monkeypatch.setattr(api, 'run_orchestrator', runner)
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    body = {'message': 'slow', 'session_id': 'one', 'request_id': 'api-conflict'}
    response = client.post('/chat', json=body)
    assert response.status_code == 409
    assert response.json()['status'] == 'completed' and response.json()['memory_status'] == 'conflict'
    assert client.post('/chat', json=body).status_code == 200
    assert service.snapshot(A, 'one')['total_tokens'] == 7


def test_page_restores_interrupted_run_and_still_requires_confirmation(setup, monkeypatch):
    from tests.online_support import install_script, proposal_steps
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    service, engine, _, directory = setup
    install_script(monkeypatch, proposal_steps())
    original = orchestrator.tool_step
    def interrupt_after_proposal(state):
        value = original(state)
        if state['tool_calls'][state['tool_index']]['name'] == 'initiate_refund':
            raise SystemExit('after proposal, before checkpoint')
        return value
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, 'tool_step', interrupt_after_proposal)
        with pytest.raises(SystemExit):
            service.chat(A, 'one', MESSAGE, request_id='page-recovery')
    run = service.list_runs(A, 'one')[0]
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(directory))
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=20).run()
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('test-token-a').run()
        next(t for t in app.text_input if t.label == '会话编号').set_value(run['session_id']).run()
        assert not app.exception
        next(button for button in app.button if button.label == '恢复执行').click().run()
        assert not app.exception
        assert not any(button.label == '恢复执行' for button in app.button)
        assert any(button.label == '确认申请' for button in app.button)
        assert next(r for r in rows(engine, Refund) if r.order_id == 1001).status == 'none'
        next(button for button in app.button if button.label == '确认申请').click().run()
        assert not app.exception
        assert next(r for r in rows(engine, Refund) if r.order_id == 1001).status == 'requested'
        assert len(service.snapshot(A, run['session_id'])['turns']) == 2
    finally:
        st.cache_resource.clear()


@pytest.mark.parametrize('stage', ['before_understanding', 'after_understanding', 'after_model', 'after_read',
                                  'tool_commit_gap', 'after_proposal_checkpoint', 'after_graph', 'after_finish'])
def test_independent_process_online_recovery_preserves_plan_receipts_and_budget(setup, monkeypatch, stage):
    from tests.online_support import call, install_script, proposal_steps
    service, engine, env, _ = setup
    child = '''
import os
import pytest
from src import orchestrator
from src.auth import Principal
from src.requests import RequestJournal
from src.service import SupportService
from tests.online_support import install_script, proposal_steps
stage=os.environ['ONLINE_CRASH_STAGE']
original_route=orchestrator.route_node
original_prepare=orchestrator.prepare_node
original_model=orchestrator.model_step
original_tool=orchestrator.tool_step
original_finish=RequestJournal.finish_request
def route(state):
    if stage=='before_understanding': os._exit(23)
    return original_route(state)
def prepare(state):
    if stage=='after_understanding': os._exit(23)
    return original_prepare(state)
def model(state):
    names=[entry['tool'] for entry in state.get('trace',[])]
    if stage=='after_read' and 'check_refund_eligibility' in names: os._exit(23)
    if stage=='after_proposal_checkpoint' and 'initiate_refund' in names: os._exit(23)
    return original_model(state)
def tool(state):
    if stage=='after_model': os._exit(23)
    result=original_tool(state)
    if stage=='tool_commit_gap' and result['trace'][-1]['tool']=='initiate_refund': os._exit(23)
    return result
def finish(self,context,result):
    if stage=='after_graph': os._exit(23)
    original_finish(self,context,result)
    if stage=='after_finish': os._exit(23)
orchestrator.route_node=route
orchestrator.prepare_node=prepare
orchestrator.model_step=model
orchestrator.tool_step=tool
RequestJournal.finish_request=finish
with pytest.MonkeyPatch.context() as patch:
    install_script(patch,proposal_steps())
    SupportService().chat(Principal('customer_a'),'one','申请1001退款',request_id='real-process')
'''
    process = subprocess.run([sys.executable, '-c', child], env={**env, 'ONLINE_CRASH_STAGE': stage},
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 23, process.stderr
    steps = proposal_steps()
    remaining = {'before_understanding': steps, 'after_understanding': steps[1:],
                 'after_model': steps[2:], 'after_read': steps[2:], 'tool_commit_gap': steps[3:],
                 'after_proposal_checkpoint': steps[3:], 'after_graph': [], 'after_finish': []}[stage]
    calls = install_script(monkeypatch, remaining)
    run = service.list_runs(A, 'one')[0]
    restored = SupportService(service.store.path)
    result = restored.resume_run(A, run['run_id'])
    assert result['status'] == 'pending_confirmation' and result['memory_saved']
    assert len(calls) == len(remaining) and result['execution_budget']['model_calls'] == 4
    assert result['execution_budget']['tool_calls'] == 2
    assert len(rows(engine, ActionRecord)) == len(rows(engine, PendingAction)) == 1
    assert next(row for row in rows(engine, Refund) if row.order_id == 1001).status == 'none'
    action_id = result['pending_actions'][0]['action_id']
    assert restored.confirm_action(A, action_id)['status'] == 'completed'
    assert restored.confirm_action(A, action_id)['replayed']
    assert len(rows(engine, ActionRecord)) == 2


