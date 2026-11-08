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
