import json
from langchain_core.messages import AIMessage
from src import dialogue
from src.planning import make_plan
from src.knowledge import KnowledgeStore
from tests.test_knowledge_memory import Encoder
from qdrant_client import QdrantClient


def test_english_queries_use_chinese_corpus_and_exact_product(tmp_path):
    source = tmp_path / 'corpus'
    source.mkdir()
    (source / 'policy.md').write_text('# 退货\n\n## 具体规定\n签收七日内申请。')
    (source / 'product.md').write_text('# 摄像头\n\n## 规格\n支持1080p，不支持4K。')
    manifest = {'products': [{'id':'P1', 'name':'摄像头', 'aliases':['Webcam 1080p']}], 'documents': [
        {'id':'POL-1','file':'policy.md','title':'退货','kind':'policy','category':'退货','version':'1'},
        {'id':'SKU-1','file':'product.md','title':'摄像头','kind':'product','category':'摄像头','version':'1','product_id':'P1'}]}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    store = KnowledgeStore(tmp_path / 'db', client=QdrantClient(':memory:'), model=Encoder())
    store.ingest(source)
    for question in ['Show all platform policies', 'Give me a policy overview', 'Overview of platform rules']:
        result = store.search(question, kind='policy')
        assert [s['doc_id'] for s in result['sources']] == ['POL-1']
        assert '签收七日' in result['sources'][0]['content']
    result = store.search('Does it support 4K?', kind='product', product_ref='the Webcam 1080p')
    assert result['sources'][0]['doc_id'] == 'SKU-1'
    assert not store.search('Camera', kind='product', product_ref='unknown Webcam')['sources']


def test_english_composition_keeps_chinese_sources_and_candidates_out(monkeypatch):
    from src.runtime_context import active_profile
    captured = []
    class Model:
        def invoke(self, messages):
            captured.append(messages)
            return AIMessage(content=json.dumps({'answer':'Apply within seven days [source-1].', 'source_ids':['source-1'], 'insufficient':False}))
    monkeypatch.setattr(dialogue, 'get_llm', Model)
    token = active_profile.set({'candidate_preferences':[{'key':'language','value':'候选不应使用'}]})
    try:
        answer,_ = dialogue.compose('When can I return it?', {'knowledge':True,'sources':[
            {'id':'source-1','title':'退货政策','location':'具体规定','version':'1','content':'七日内申请'}]})
    finally:
        active_profile.reset(token)
    assert 'Apply within seven days' in answer
    assert 'Source / 来源：退货政策' in answer
    assert '候选不应使用' not in captured[0][0].content
    assert '英文问题用英文' in captured[0][0].content
    assert '中文回答' not in captured[0][0].content


def test_language_priority():
    from src.dialogue import response_language
    profile={'learned_preferences':[{'key':'language','value':'中文','status':'active'}]}
    assert response_language('What is the return policy?')=='en'
    assert response_language('Wireless Headphones支持有线麦克风吗？')=='zh'
    assert response_language('What is the return policy?',profile)=='zh'
    assert response_language('Please answer in English. What is the return policy?',profile)=='en'


def test_invalid_citation_response_is_repaired(monkeypatch):
    class Model:
        def __init__(self): self.calls=0
        def invoke(self,messages):
            self.calls+=1
            if self.calls==1:return AIMessage(content='not valid JSON',usage_metadata={'input_tokens':2,'output_tokens':1,'total_tokens':3})
            assert messages[-1].type=='system'
            return AIMessage(content=json.dumps({'answer':'Seven days [s1]','source_ids':['s1'],'insufficient':False}),usage_metadata={'input_tokens':2,'output_tokens':1,'total_tokens':3})
    model=Model();monkeypatch.setattr(dialogue,'get_llm',lambda:model)
    answer,usage=dialogue.compose('Return deadline?',{'knowledge':True,'sources':[{'id':'s1','title':'退货','location':'规定','version':'1','content':'七日'}]})
    assert 'Seven days' in answer and model.calls==2
    assert usage['total_tokens']==6


def test_synthesis_focus_retains_original_language_instruction(monkeypatch):
    class Model:
        def invoke(self,messages):
            assert '必须使用English' in messages[0].content
            assert messages[-1].content.endswith('解释是否能访问其他客户偏好')
            return AIMessage(content='I cannot access another customer profile.')
    monkeypatch.setattr(dialogue,'get_llm',Model)
    answer,_=dialogue.compose('请用英文回答，查询订单并解释是否能访问其他客户偏好',{},focus='解释是否能访问其他客户偏好')
    assert answer.startswith('I cannot')
