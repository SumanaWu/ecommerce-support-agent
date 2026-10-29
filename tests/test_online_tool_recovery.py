"""Missing calls require model correction; server-synthesized calls are forbidden."""
from langchain_core.messages import AIMessage, ToolMessage
import pytest
from sqlalchemy import select

from src import orchestrator, tools
from src.agents import base
from src.auth import Principal
from src.backend.db import PendingAction, get_engine, get_session_factory
from src.backend.seed import seed
from src.service import SupportService
from tests.online_support import install_plan, plan, task

A=Principal('customer_a')


@pytest.fixture
def setup(tmp_path,monkeypatch):
    engine=get_engine('sqlite:///'+str(tmp_path/'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db: seed(db)
    service=SupportService(tmp_path/'memory.sqlite')
    yield service,engine
    engine.dispose()


@pytest.mark.parametrize('operation,ids', [('payment_status',[1006]),('order_details',[1001]),('shipment',[1002]),('refund_eligibility',[1001])])
def test_no_tool_response_is_corrected_once_then_reports_technical_incompletion(setup,monkeypatch,operation,ids):
    service,engine=setup
    install_plan(monkeypatch,plan(task(operation,ids)))
    calls=[]
    class Model:
        def bind_tools(self,tools):return self
        def invoke(self,messages):
            calls.append(messages)
            return AIMessage(content='请提供订单编号。')
    monkeypatch.setattr(base,'get_llm',Model)
    result=service.chat(A,'one','用户完整需求')
    assert len(calls)==2 and result['status']=='partial'
    assert result['stop_reason']=='missing_tool_call' and result['error_type']=='model_output'
    assert result['trace']==[] and result['pending_actions']==[]
    assert '请提供订单编号' not in result['reply']
    with get_session_factory(engine)() as db:assert not db.scalars(select(PendingAction)).all()


@pytest.mark.parametrize('message',['1006付了吗','1006付款了吗','1006'])
def test_correction_can_only_execute_a_real_model_tool_request(setup,monkeypatch,message):
    service,_=setup
    install_plan(monkeypatch,plan(task('payment_status',[1006])))
    calls=[]
    class Model:
        def bind_tools(self,tools):return self
        def invoke(self,messages):
            calls.append(messages)
            if len(calls)==1:return AIMessage(content='请补充订单。')
            if len(calls)==2:return AIMessage(content='',tool_calls=[{'name':'get_payment_status','args':{'order_id':1006},'id':'from-model','type':'tool_call'}])
            assert isinstance(messages[-1],ToolMessage) and messages[-1].tool_call_id=='from-model'
            return AIMessage(content='已付款。')
    monkeypatch.setattr(base,'get_llm',Model)
    result=service.chat(A,'one',message)
    assert result['status']=='completed' and len(calls)==3
    assert result['trace'][0]['result']['status']=='paid' and 'source' not in result['trace'][0]
    assert result['execution_budget']['real_model_calls']==4


def test_failed_correction_stays_in_checkpoint_and_replay_does_not_retry_again(setup,monkeypatch):
    service,_=setup
    install_plan(monkeypatch,plan(task('payment_status',[1006])))
    calls=[]
    class Model:
        def bind_tools(self,tools):return self
        def invoke(self,messages):calls.append(1);return AIMessage(content='未查询')
    monkeypatch.setattr(base,'get_llm',Model)
    first=service.chat(A,'one','查付款',request_id='no-call')
    replay=SupportService(service.store.path).chat(A,'one','查付款',request_id='no-call')
    assert replay['replayed'] and len(calls)==2 and replay['stop_reason']==first['stop_reason']=='missing_tool_call'


@pytest.mark.parametrize('required', [True, False])
def test_correction_uses_adapter_capability_instead_of_provider_name(setup, monkeypatch, required):
    service, _ = setup
    install_plan(monkeypatch, plan(task('payment_status', [1006])))
    bindings = []
    class Model:
        provider = 'test-adapter'
        supports_required_tool_choice = required
        def bind_tools(self, tools, **options):
            bindings.append(options)
            return self
        def invoke(self, messages):
            return AIMessage(content='未查询')
    monkeypatch.setattr(base, 'get_llm', Model)
    result = service.chat(A, 'one', '查询1006付款状态')
    assert result['stop_reason'] == 'missing_tool_call'
    assert bindings == [{}, {'tool_choice': 'required'} if required else {}]
