from src.auth import Principal
from concurrent.futures import ThreadPoolExecutor
import pytest
from sqlalchemy import select
from src import tools
from src.backend.db import get_engine, get_session_factory, Ticket, Refund
from src.backend.seed import seed
from src.service import SupportService
from src.orchestrator import parse_route

@pytest.fixture
def isolated(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path/'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as s:
        seed(s)
    yield SupportService(tmp_path/'memory.sqlite'), engine
    engine.dispose()


def test_restart_and_reference_resolution(isolated):
    svc, _ = isolated
    svc.chat(Principal('customer_a'),'one','demo','查询订单 1002 的物流')
    new = SupportService(svc.store.path)
    result = new.chat(Principal('customer_a'),'one','demo','帮我申请退款')
    assert result['agent']=='refund'
    assert result['trace'][0]['args']['order_id']==1002
    assert result['trace'][0]['result']['eligible'] is False
    assert len(new.snapshot(Principal('customer_a'),'one','demo')['turns'])==2


def test_profile_cross_session_and_user_scope(isolated):
    svc, _ = isolated
    svc.store.save_profile('customer_a','简洁','文字')
    svc.chat(Principal('customer_a'),'one','demo','查询订单 1001')
    assert svc.snapshot(Principal('customer_a'),'two','demo')['turns']==[]
    assert svc.store.profile('customer_a')['tone']=='简洁'
    assert svc.store.profile('customer_b')=={}
    assert svc.snapshot(Principal('customer_b'),'one','demo')['turns']==[]
    assert svc.snapshot(Principal('customer_a'),'one','online')['turns']==[]


def test_explicit_new_order_overrides_reference(isolated):
    svc,_=isolated
    svc.chat(Principal('customer_a'),'one','demo','查询订单 1002')
    result=svc.chat(Principal('customer_a'),'one','demo','帮我申请订单 1001 的退款')
    assert result['status']=='pending_confirmation'
    assert svc.confirm_action(Principal('customer_a'),result['pending_actions'][0]['action_id'])['status']=='completed'
    assert svc.snapshot(Principal('customer_a'),'one','demo')['last_order_id']==1001


def test_bad_order_not_memorized(isolated):
    svc,_=isolated
    svc.chat(Principal('customer_a'),'one','demo','查询订单 9999')
    assert svc.snapshot(Principal('customer_a'),'one','demo')['last_order_id'] is None


def test_complaint_not_classified_by_old_order(isolated):
    svc,engine=isolated
    svc.chat(Principal('customer_a'),'one','demo','查询订单 1001')
    result=svc.chat(Principal('customer_a'),'one','demo','我要投诉这个问题')
    assert result['agent']=='escalate'
    tid=result['trace'][0]['result']['ticket_id']
    with get_session_factory(engine)() as s:
        assert s.get(Ticket,tid).status=='pending'


def test_concurrent_refund_at_most_one_change(isolated):
    svc,engine=isolated
    actions=[svc.chat(Principal('customer_a'),'one','demo','帮我申请订单 1001 的退款')['pending_actions'][0]['action_id'] for _ in range(6)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda action:svc.confirm_action(Principal('customer_a'),action), actions))
    assert sum(r['status']=='completed' for r in results)==1
    with get_session_factory(engine)() as s:
        rows=s.scalars(select(Refund).where(Refund.order_id==1001)).all()
        assert len(rows)==1 and rows[0].status=='requested'


def test_concurrent_payment_at_most_one_change(isolated):
    svc,_=isolated
    actions=[svc.chat(Principal('customer_a'),'one','demo','帮我付款订单 1003')['pending_actions'][0]['action_id'] for _ in range(6)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda action:svc.confirm_action(Principal('customer_a'),action), actions))
    assert sum(r['status']=='completed' for r in results)==1


def test_history_window_and_storage_limit(isolated):
    svc,_=isolated
    histories=[]
    def fake(message,history=None):
        histories.append(len(history))
        return {'reply':'ok','agent':'order','trace':[],'usage':{}}
    for i in range(53):
        svc.chat(Principal('customer_a'),'one','demo','hi',runner=fake)
    assert max(histories) <= 100 and all(length % 2 == 0 for length in histories)
    assert len(svc.snapshot(Principal('customer_a'),'one','demo')['turns'])==50


def test_context_profile_and_mode_are_reset(isolated):
    from src.runtime_context import active_mode, active_profile
    svc,_=isolated
    svc.store.save_profile('customer_a','简洁','文字')
    seen=[]
    def fake(message,history=None):
        seen.append((active_mode.get(),active_profile.get()))
        return {'reply':'ok','agent':'order','trace':[],'usage':{}}
    svc.chat(Principal('customer_a'),'one','online','hi',runner=fake)
    assert seen[0][0]=='online' and seen[0][1]['tone']=='简洁'
    assert active_mode.get()=='demo' and active_profile.get()=={}


def test_invalid_input_and_ambiguous_route(isolated):
    svc,_=isolated
    for args in [(Principal('a/other'),'one','demo','hi'),(Principal('customer_a'),'one','bad','hi'),(Principal('customer_a'),'one','demo','x'*2001)]:
        with pytest.raises(ValueError):svc.chat(*args)
    assert parse_route('REFUND ORDER')=='CLARIFY'
    assert parse_route('PREORDER')=='CLARIFY'


def test_extra_or_unknown_tool_is_rejected(monkeypatch):
    from langchain_core.tools import tool
    from src.agents import base
    from .test_agent_base import FakeLLM,FakeToolCallMessage
    @tool
    def echo(value:str)->str:
        """Echo."""
        return value
    fake=FakeLLM([FakeToolCallMessage(tool_calls=[{'name':'unknown','args':{},'id':'1'}]),FakeToolCallMessage(content='done')])
    monkeypatch.setattr(base,'get_llm',lambda:fake)
    result=base.run_tool_agent('test','test',[echo],'test')
    assert 'error' in result['trace'][0]['result']
    fake=FakeLLM([FakeToolCallMessage(tool_calls=[{'name':'echo','args':{'value':'x','user_id':'customer_b'},'id':'1'}]),FakeToolCallMessage(content='done')])
    result=base.run_tool_agent('test','test',[echo],'test')
    assert 'error' in result['trace'][0]['result']
