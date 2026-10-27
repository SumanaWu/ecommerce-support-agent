"""Real process exits and independent writers, always on temporary databases."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
import pytest
from sqlalchemy import select

import api
from src import orchestrator, tools
from src.auth import AccessError, Principal
from src.backend.db import ActionRecord, Order, PendingAction, Refund, Ticket, get_engine, get_session_factory
from src.backend.seed import seed
from src.service import SupportService
from src.storage import SessionConflict, SupportStore


A = Principal('customer_a')
B = Principal('customer_b')
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


CRASH = '''
import os
from src import orchestrator
from src.auth import Principal
from src.service import SupportService
from src.requests import RequestJournal
stage=os.environ['CRASH_STAGE']
original_model=orchestrator.model_step
original_tool=orchestrator.tool_step
original_route=orchestrator.route_node
original_build=orchestrator.build_graph
original_finish=RequestJournal.finish_request
original_escalate=orchestrator.escalate_node
def route(state):
    if stage=='before_route': os._exit(23)
    return original_route(state)
def model(state):
    if stage=='before_model' and not state['trace']: os._exit(23)
    if stage=='after_read_checkpoint' and state['trace']: os._exit(23)
    if stage=='after_tool_checkpoint' and any(t['tool']=='initiate_refund' for t in state['trace']): os._exit(23)
    result=original_model(state)
    if stage=='model_checkpoint_gap' and not state['trace']: os._exit(23)
    return result
def tool(state):
    if stage=='after_model_checkpoint' and not state['trace']: os._exit(23)
    result=original_tool(state)
    if stage=='tool_commit_gap' and result['trace'][-1]['tool']=='initiate_refund': os._exit(23)
    return result
def build(*args,**kwargs):
    if stage=='before_input_checkpoint': os._exit(23)
    return original_build(*args,**kwargs)
def finish(self,context,result):
    if stage=='after_final_checkpoint': os._exit(23)
    return original_finish(self,context,result)
def escalate(state):
    result=original_escalate(state)
    os._exit(23)
orchestrator.route_node=route
orchestrator.model_step=model
orchestrator.tool_step=tool
orchestrator.build_graph=build
RequestJournal.finish_request=finish
message='帮我申请订单 1001 的退款'
if stage=='escalation_commit_gap':
    orchestrator.escalate_node=escalate
    message='我要投诉'
SupportService().chat(Principal('customer_a'),None,'demo',os.environ.get('CRASH_MESSAGE',message),request_id='crash')
'''


def crash(setup, stage, message=None):
    service, _, env, _ = setup
    if message:
        env = {**env, 'CRASH_MESSAGE': message}
    process = subprocess.run([sys.executable, '-c', CRASH],
                             env={**env, 'CRASH_STAGE': stage}, timeout=20)
    assert process.returncode == 23
    runs = service.store.list_runs('customer_a', 'demo',
                                   tools.request_journal().request_input('customer_a', 'demo', 'crash')['session_id'])
    assert len(runs) == 1
    return runs[0]


@pytest.mark.parametrize('stage', ['before_input_checkpoint', 'before_route', 'before_model',
                                  'model_checkpoint_gap', 'after_model_checkpoint', 'tool_commit_gap',
                                  'after_tool_checkpoint', 'after_final_checkpoint'])
def test_process_crash_resumes_saved_steps_without_duplicate_proposal(setup, stage):
    service, engine, _, _ = setup
    run = crash(setup, stage)
    response = service.chat(A, None, 'demo', MESSAGE, request_id='crash')
    assert response['status'] == 'interrupted' and response['run_id'] == run['run_id']
    assert not response['memory_saved']
    result = SupportService(service.store.path).resume_run(A, run['run_id'])
    assert result['status'] == 'pending_confirmation' and result['memory_saved']
    assert result['session_id'] == run['session_id'] and result['replayed']
    assert len(rows(engine, PendingAction)) == len(rows(engine, ActionRecord)) == 1
    assert next(r for r in rows(engine, Refund) if r.order_id == 1001).status == 'none'
    assert len(service.snapshot(A, run['session_id'], 'demo')['turns']) == 1
    assert service.resume_run(A, run['run_id'])['memory_saved']
    assert service.list_runs(A, run['session_id'], 'demo') == []
    # Waiting for confirmation survives another fresh process and still needs an independent decision.
    _, _, env, _ = setup
    code = """
from src.auth import Principal
from src.service import SupportService
s=SupportService(); p=Principal('customer_a')
actions=s.list_pending_actions(p,SESSION,'demo')
assert len(actions)==1
r=s.confirm_action(p,actions[0]['action_id'])
assert r['status']=='completed'
assert s.confirm_action(p,actions[0]['action_id'])['replayed']
""".replace('SESSION', repr(run['session_id']))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=20)
    assert len(rows(engine, ActionRecord)) == 2


def test_saved_model_call_is_not_repeated_and_tool_messages_stay_paired(setup, monkeypatch):
    from src.agents import base
    from src.demo_model import DemoChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    service, _, _, _ = setup
    run = crash(setup, 'after_model_checkpoint')
    class GuardedModel(DemoChatModel):
        def bind_tools(self, agent_tools):
            self.tools = {t.name for t in agent_tools}
            return self

        def invoke(self, messages):
            assert any(isinstance(message, ToolMessage) for message in messages)
            return super().invoke(messages)

    monkeypatch.setattr(base, 'get_llm', GuardedModel)
    monkeypatch.setattr(orchestrator, 'route_node', lambda _: pytest.fail('repeated saved routing'))
    result = service.resume_run(A, run['run_id'])
    assert result['status'] == 'pending_confirmation'
    with SqliteSaver.from_conn_string(run['checkpoint_path']) as saver:
        snapshot = orchestrator.build_graph(saver).get_state({'configurable': {'thread_id': run['run_id']}})
    messages = snapshot.values['messages']
    requested = [call['id'] for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]
    supplied = [message.tool_call_id for message in messages if isinstance(message, ToolMessage)]
    assert requested == supplied and len(requested) == 2
    assert not snapshot.next


def test_completed_checkpoint_does_not_call_a_model_on_resume(setup, monkeypatch):
    service, _, _, _ = setup
    run = crash(setup, 'after_final_checkpoint')
    monkeypatch.setattr(orchestrator, 'get_llm', lambda: pytest.fail('repeated routing'))
    monkeypatch.setattr('src.agents.base.get_llm', lambda: pytest.fail('repeated model'))
    assert service.resume_run(A, run['run_id'])['status'] == 'pending_confirmation'


def test_committed_ticket_is_not_duplicated_by_node_recovery(setup):
    service, engine, _, _ = setup
    run = crash(setup, 'escalation_commit_gap')
    assert len(rows(engine, Ticket)) == 1
    result = service.resume_run(A, run['run_id'])
    assert result['agent'] == 'escalate' and result['memory_saved']
    assert len(rows(engine, Ticket)) == len(rows(engine, ActionRecord)) == 1


def test_resume_requires_same_principal_and_does_not_accept_checkpoint_overrides(setup, monkeypatch):
    service, _, _, _ = setup
    run = crash(setup, 'tool_commit_gap')
    monkeypatch.setattr(api, 'service', service)
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    other = TestClient(api.app, headers={'Authorization': 'Bearer test-token-b'})
    path = '/runs/' + run['run_id']
    assert other.get(path).status_code == other.post(path + '/resume', json={}).status_code == 404
    assert other.get('/runs', params={'session_id': run['session_id']}).json() == {'runs': []}
    assert client.get('/runs', params={'session_id': run['session_id'], 'mode': 'online'}).json() == {'runs': []}
    assert TestClient(api.app).post(path + '/resume').status_code == 401
    for override in [{'user_id': 'customer_b'}, {'checkpoint_id': 'other'}, {'mode': 'online'}, {'session_id': 'other'}]:
        assert client.post(path + '/resume', json=override).status_code == 422
    with tools.request_journal().request_guard('customer_a', 'demo', 'crash'):
        assert client.post(path + '/resume').status_code == 202
    response = client.post(path + '/resume')
    assert response.status_code == 200 and response.json()['status'] == 'pending_confirmation'
    assert 'input' not in client.get(path).json() and 'checkpoint_path' not in client.get(path).json()


def test_resume_rechecks_order_permissions(setup):
    service, engine, _, _ = setup
    run = crash(setup, 'after_tool_checkpoint')
    with get_session_factory(engine).begin() as db:
        db.get(Order, 1001).customer_id = 'customer_b'
    with pytest.raises(AccessError) as exc:
        service.resume_run(A, run['run_id'])
    assert exc.value.status_code == 409
    with pytest.raises(AccessError) as exc:
        service.resume_run(B, run['run_id'])
    assert exc.value.status_code == 404


def test_unavailable_order_error_can_resume_without_disclosing_order_facts(setup):
    service, _, _, _ = setup
    run = crash(setup, 'after_read_checkpoint', '查询订单 9999 的详情')
    result = service.resume_run(A, run['run_id'])
    assert result['status'] == 'failed' and result['memory_saved']
    assert result['trace'][0]['result'] == tools.ORDER_UNAVAILABLE


def test_saved_input_history_is_not_replaced_by_newer_session_on_resume(setup, monkeypatch):
    from src.agents import base
    from src.demo_model import DemoChatModel
    service, _, _, _ = setup
    run = crash(setup, 'before_model')
    service.chat(A, run['session_id'], 'demo', '查询订单 1002 的物流')
    captured = []
    class CapturingModel(DemoChatModel):
        def bind_tools(self, agent_tools):
            self.tools = {t.name for t in agent_tools}
            return self

        def invoke(self, messages):
            captured.append(str(messages))
            return super().invoke(messages)
    monkeypatch.setattr(base, 'get_llm', CapturingModel)
    result = service.resume_run(A, run['run_id'])
    assert result['status'] == 'pending_confirmation' and result['memory_status'] == 'conflict'
    assert all('1002' not in messages for messages in captured)
    repaired = service.resume_run(A, run['run_id'])
    assert repaired['memory_saved'] and repaired['memory_rebased']
    assert service.snapshot(A, run['session_id'], 'demo')['last_order_id'] == 1002


def test_memory_failure_after_graph_completion_repairs_without_model(setup, monkeypatch):
    service, _, _, _ = setup
    with monkeypatch.context() as patch:
        patch.setattr(service.store, '_write_session', lambda *_: (_ for _ in ()).throw(sqlite3.OperationalError()))
        result = service.chat(A, 'one', 'demo', MESSAGE)
    assert not result['memory_saved']
    assert service.list_runs(A, 'one', 'demo')[0]['run_id'] == result['run_id']
    monkeypatch.setattr(orchestrator, 'get_llm', lambda: pytest.fail('repeated graph'))
    repaired = service.resume_run(A, result['run_id'])
    assert repaired['memory_saved'] and repaired['replayed']
    assert service.snapshot(A, 'one', 'demo')['version'] == 1


def test_stale_full_session_save_cannot_overwrite_a_turn(setup):
    service, _, _, _ = setup
    old = service.store.load_session('customer_a', 'demo', 'one')
    service.chat(A, 'one', 'demo', '查询订单 1001')
    with pytest.raises(SessionConflict):
        service.store.save_session_if_version('customer_a', 'demo', 'one', old, old['version'])
    assert len(service.snapshot(A, 'one', 'demo')['turns']) == 1


def test_independent_process_conflict_is_visible_and_repair_preserves_both_turns(setup):
    service, engine, env, directory = setup
    code = '''
import json,time
from pathlib import Path
from src.auth import Principal
from src import tools
from src.results import fallback_result
from src.service import SupportService
directory=Path(DIRECTORY)
def runner(*a,**k):
    (directory/'entered').write_text('yes')
    deadline=time.monotonic()+15
    while not (directory/'release').exists():
        assert time.monotonic()<deadline
        time.sleep(.02)
    result=tools.initiate_refund(1001,'test')
    return fallback_result([{'agent':'refund','tool':'initiate_refund','args':{'order_id':1001,'reason':'test'},'result':result}])
result=SupportService().chat(Principal('customer_a'),'one','demo','refund',runner=runner,request_id='slow')
(directory/'result.json').write_text(json.dumps(result))
'''.replace('DIRECTORY', repr(str(directory)))
    process = subprocess.Popen([sys.executable, '-c', code], env=env)
    try:
        deadline = time.monotonic() + 15
        while not (directory / 'entered').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(.02)
        first = service.chat(A, 'one', 'demo', '查询订单 1002 的物流')
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
    repaired = service.chat(A, 'one', 'demo', 'refund', request_id='slow',
                            runner=lambda *a, **k: pytest.fail('repair reran side effects'))
    assert repaired['memory_saved'] and repaired['memory_rebased']
    again = service.chat(A, 'one', 'demo', 'refund', request_id='slow')
    payload = service.snapshot(A, 'one', 'demo')
    assert again['memory_saved'] and len(payload['turns']) == payload['version'] == 2
    assert payload['last_order_id'] == 1002
    assert len(rows(engine, ActionRecord)) == 1


def test_api_reports_session_conflict_with_preserved_business_result(setup, monkeypatch):
    service, _, _, _ = setup
    monkeypatch.setattr(api, 'service', service)
    def runner(*a, **k):
        other = SupportService(service.store.path)
        other.chat(A, 'one', 'demo', '查询订单 1002')
        return {'reply': 'done', 'agent': 'order', 'trace': [], 'usage': {'total_tokens': 7}}
    monkeypatch.setattr(api, 'run_orchestrator', runner)
    client = TestClient(api.app, headers={'Authorization': 'Bearer test-token-a'})
    body = {'message': 'slow', 'session_id': 'one', 'request_id': 'api-conflict'}
    response = client.post('/chat', json=body)
    assert response.status_code == 409
    assert response.json()['status'] == 'completed' and response.json()['memory_status'] == 'conflict'
    assert client.post('/chat', json=body).status_code == 200
    assert service.snapshot(A, 'one', 'demo')['total_tokens'] == 7


def test_page_restores_interrupted_run_and_still_requires_confirmation(setup, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    service, engine, _, directory = setup
    run = crash(setup, 'tool_commit_gap')
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
        assert len(service.snapshot(A, run['session_id'], 'demo')['turns']) == 2
    finally:
        st.cache_resource.clear()
