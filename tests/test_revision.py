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
    svc.chat('a','one','demo','查询订单 1002 的物流')
    new = SupportService(svc.store.path)
    result = new.chat('a','one','demo','帮我申请退款')
    assert result['agent']=='refund'
    assert result['trace'][0]['args']['order_id']==1002
    assert result['trace'][0]['result']['eligible'] is False
    assert len(new.snapshot('a','one','demo')['turns'])==2


def test_profile_cross_session_and_user_scope(isolated):
    svc, _ = isolated
    svc.store.save_profile('a','简洁','文字')
    svc.chat('a','one','demo','查询订单 1001')
    assert svc.snapshot('a','two','demo')['turns']==[]
    assert svc.store.profile('a')['tone']=='简洁'
    assert svc.store.profile('b')=={}
    assert svc.snapshot('b','one','demo')['turns']==[]
    assert svc.snapshot('a','one','online')['turns']==[]


def test_explicit_new_order_overrides_reference(isolated):
    svc,_=isolated
    svc.chat('a','one','demo','查询订单 1002')
    result=svc.chat('a','one','demo','帮我申请订单 1001 的退款')
    assert result['trace'][-1]['result']['success'] is True
    assert svc.snapshot('a','one','demo')['last_order_id']==1001


def test_bad_order_not_memorized(isolated):
    svc,_=isolated
    svc.chat('a','one','demo','查询订单 9999')
    assert svc.snapshot('a','one','demo')['last_order_id'] is None


def test_complaint_not_classified_by_old_order(isolated):
    svc,engine=isolated
    svc.chat('a','one','demo','查询订单 1001')
    result=svc.chat('a','one','demo','我要投诉这个问题')
    assert result['agent']=='escalate'
    tid=result['trace'][0]['result']['ticket_id']
    with get_session_factory(engine)() as s:
        assert s.get(Ticket,tid).status=='pending'


def test_concurrent_refund_at_most_one_change(isolated):
    _,engine=isolated
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:tools.initiate_refund(1001,'test'), range(6)))
    assert sum(r['success'] for r in results)==1
    with get_session_factory(engine)() as s:
        rows=s.scalars(select(Refund).where(Refund.order_id==1001)).all()
        assert len(rows)==1 and rows[0].status=='requested'


def test_concurrent_payment_at_most_one_change(isolated):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:tools.process_payment(1003,'card'), range(6)))
    assert sum(r['success'] for r in results)==1


def test_history_window_and_storage_limit(isolated):
    svc,_=isolated
    histories=[]
    def fake(message,history=None):
        histories.append(len(history))
        return {'reply':'ok','agent':'order','trace':[],'usage':{}}
    for i in range(53):
        svc.chat('a','one','demo','hi',runner=fake)
    assert max(histories)==12
    assert len(svc.snapshot('a','one','demo')['turns'])==50


def test_context_profile_and_mode_are_reset(isolated):
    from src.runtime_context import active_mode, active_profile
    svc,_=isolated
    svc.store.save_profile('a','简洁','文字')
    seen=[]
    def fake(message,history=None):
        seen.append((active_mode.get(),active_profile.get()))
        return {'reply':'ok','agent':'order','trace':[],'usage':{}}
    svc.chat('a','one','online','hi',runner=fake)
    assert seen[0][0]=='online' and seen[0][1]['tone']=='简洁'
    assert active_mode.get()=='demo' and active_profile.get()=={}


def test_invalid_input_and_ambiguous_route(isolated):
    svc,_=isolated
    for args in [('a/other','one','demo','hi'),('a','one','bad','hi'),('a','one','demo','x'*2001)]:
        with pytest.raises(ValueError):svc.chat(*args)
    assert parse_route('REFUND ORDER')=='ESCALATE'
    assert parse_route('PREORDER')=='ESCALATE'


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
    fake=FakeLLM([FakeToolCallMessage(tool_calls=[{'name':'echo','args':{'value':'x','user_id':'b'},'id':'1'}]),FakeToolCallMessage(content='done')])
    result=base.run_tool_agent('test','test',[echo],'test')
    assert 'error' in result['trace'][0]['result']
