import copy
import json

from langchain_core.messages import AIMessage
from src import orchestrator, dialogue
from src.planning import make_plan
from tests.test_customer_refunds import svc
from tests.test_actions import business


def test_memory_persisted_on_completed_request_and_replay(svc, monkeypatch):
    item=dict(key='detail',value='详细',origin='explicit',operation='set',evidence='以后详细回答')
    monkeypatch.setattr(orchestrator,'make_plan',lambda *a:({'action':'DIRECT','answer':'好的','tasks':[],'memory_candidates':[item]},{}))
    result=svc.chat('customer_a','one','以后详细回答',request_id='pref')
    assert result['memory']['status']=='applied'
    svc.chat('customer_a','one','以后详细回答',request_id='pref')
    assert len(svc.store.memories('customer_a')[0]['evidence'])==1
    assert svc.store.profile('customer_a')['learned_preferences'][0]['value']=='详细'


def test_bad_memory_does_not_invalidate_business_plan():
    class Model:
        def invoke(self,messages):
            return AIMessage(content=json.dumps({'action':'DIRECT','answer':'你好','memory_candidates':'bad'}))
    plan,_=make_plan('你好',model=Model())
    assert not plan['memory_candidates']


def test_knowledge_composition_validates_sources(monkeypatch):
    class Model:
        def invoke(self,messages):
            return AIMessage(content=json.dumps({'answer':'七日内申请','source_ids':['invented'],'insufficient':False}))
    monkeypatch.setattr(dialogue,'get_llm',Model)
    text,_=dialogue.compose('退货期限',{'knowledge':True,'sources':[]})
    assert '未能生成可核验' in text


def test_knowledge_task_and_explanation_share_evidence(svc,monkeypatch):
    source={'id':'id-1','doc_id':'A','title':'退货','location':'具体规定','version':'1','content':'七日内申请'}
    from tests.test_knowledge_memory import Encoder
    monkeypatch.setattr(svc.knowledge,'model',Encoder())
    monkeypatch.setattr(svc.knowledge,'search',lambda **kwargs:{'knowledge':True,'sources':[source]})
    monkeypatch.setattr(orchestrator,'make_plan',lambda *a:({'action':'PLAN','tasks':[
        dict(task_id='t1',operation='knowledge.search',source_text='退货期限',query='退货期限',knowledge_kind='policy',intent='query',targets=[],depends_on=[],conditions=[],condition_mode='all'),
        dict(task_id='t2',operation='answer.explain',source_text='退货期限',intent='query',targets=[],depends_on=['t1'],conditions=[],condition_mode='all')]},{}))
    monkeypatch.setattr(dialogue,'compose',lambda *a:('七日内申请 [id-1]',{}))
    result=svc.chat('customer_a','one','退货期限',request_id='kb')
    assert result['status']=='completed'
    assert result['sources']==[source]
    assert not result['actions']


def test_plan_repair_error_is_not_a_new_user_message():
    from langchain_core.messages import SystemMessage, HumanMessage
    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1: return AIMessage(content='not json')
            assert isinstance(messages[-1], SystemMessage)
            assert [m.content for m in messages if isinstance(m, HumanMessage)] == ['请记住，先说结论']
            return AIMessage(content=json.dumps({'action':'DIRECT','answer':'好的','tasks':[]}))
    plan,_=make_plan('请记住，先说结论',model=Model())
    assert plan['action']=='DIRECT'


def test_ticket_progress_cannot_create_another_ticket():
    from src.planning import ticket_status_only
    assert ticket_status_only('Has that ticket notified a real human? Can you guarantee resolution today?')
    assert ticket_status_only('工单有人接单了吗，多久能解决？')
    assert not ticket_status_only('Please create a new ticket about the unresolved issue.')
    assert not ticket_status_only('原工单未解决，请提交新的正式投诉。')
    question = 'Has that ticket notified a real human?'
    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=json.dumps({'action':'PLAN','tasks':[
                    dict(task_id='t1',operation='escalate.ticket',intent='escalate',reason='human',source_text=question)]}))
            assert '没有授权新建工单' in messages[-1].content
            return AIMessage(content=json.dumps({'action':'DIRECT','answer':'A local ticket does not notify a human.'}))
    plan,_=make_plan(question,model=Model())
    assert plan['action']=='DIRECT' and not plan['tasks']


def test_search_synthesis_shown_once_with_source_and_failure_fallback():
    first=dict(task_id='t1',operation='search.web',source_text='weather',depends_on=[],targets=[])
    second=dict(task_id='t2',operation='answer.explain',source_text='weather',depends_on=['t1'],targets=[])
    state={'plan':{'tasks':[first,second]},'results':{
        't1':{'status':'succeeded','answer':'intermediate weather','facts':[{'sources':[{'url':'https://example.com'}]}]},
        't2':{'status':'succeeded','answer':'combined weather','facts':[]}},'actions':[],'inputs':[]}
    result=orchestrator.response(state)
    assert result['reply'].count('combined weather')==1
    assert 'intermediate weather' not in result['reply']
    assert result['sources']==[{'url':'https://example.com'}]
    state['results']['t2']={'status':'failed','facts':[],'reason':'unavailable'}
    assert 'intermediate weather' in orchestrator.response(state)['reply']


def test_synthesis_without_evidence_is_replanned():
    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, messages):
            self.calls += 1
            tasks=[dict(task_id='t1',operation='order.list',source_text='查询本人订单'),
                   dict(task_id='t2',operation='answer.explain',source_text='查询本人订单',depends_on=[] if self.calls==1 else ['t1'])]
            return AIMessage(content=json.dumps({'action':'PLAN','tasks':tasks}))
    model=Model()
    plan,_=make_plan('查询本人订单',model=model)
    assert model.calls==2 and plan['tasks'][1]['depends_on']==['t1']


def test_ticket_claim_guard_preserves_negative_explanations():
    for text in ['偏好不能授权付款，也不能声称已创建工单。', '工单 #123 尚未通知真人。']:
        assert dialogue.checked_reply(text)==text
    assert dialogue.checked_reply('已创建本地工单。')!='已创建本地工单。'
