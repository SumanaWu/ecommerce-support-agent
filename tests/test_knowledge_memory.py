import json
from pathlib import Path

import pytest

from src.memory import memory_candidates
from src.storage import SupportStore
from src.knowledge import KnowledgeStore, parse_document


def test_memory_evidence_revocation_and_ordering(tmp_path):
    store = SupportStore(tmp_path/'memory.sqlite')
    item = dict(key='detail',value='详细',origin='observed',operation='set',evidence='详细一点')
    for n in range(3):
        store.apply_memories('a',str(n),[item],f'2026-01-0{n+1}')
    assert store.profile('a')['learned_preferences'][0]['status'] == 'active'
    store.apply_memories('a','2',[item],'2026-01-03')
    assert len(store.memories('a')[0]['evidence']) == 3
    store.forget_memory('a','detail')
    store.apply_memories('a','new',[item],'2099-01-01')
    assert not store.profile('a').get('learned_preferences')
    assert store.profile('b') == {}


def test_memory_candidate_validation():
    item=dict(key='format',value='先结论',origin='explicit',operation='set',evidence='以后先说结论')
    assert memory_candidates([item],'以后先说结论') == [item]
    assert memory_candidates([item],'你好') == []
    assert memory_candidates(None,'你好') == []


def test_corpus_counts_and_order_aliases():
    root=Path(__file__).resolve().parents[1]/'knowledge'
    data=json.loads((root/'manifest.json').read_text())
    assert len(data['products']) == 200
    assert sum(r['kind']=='policy' for r in data['documents']) == 200
    assert sum(r['kind']=='product' for r in data['documents']) == 200
    assert len({r['id'] for r in data['documents']}) == 400
    from src.backend.seed import ORDERS
    aliases={alias for r in data['products'] for alias in r['aliases']}
    assert {r[2] for r in ORDERS} <= aliases
    for doc in data['documents']:
        assert parse_document(root/doc['file'])


class Tokenizer:
    def encode(self,text,**kwargs):
        return [ord(c) for c in text]
    def decode(self,ids,**kwargs):
        return ''.join(chr(c) for c in ids)


class Encoder:
    tokenizer=Tokenizer()
    def encode(self,texts,**kwargs):
        import numpy as np
        return np.array([[1.]+[0.]*383 for _ in texts])


def test_import_replay_delete_and_product_scope(tmp_path):
    from qdrant_client import QdrantClient
    directory=tmp_path/'source';directory.mkdir()
    (directory/'a.md').write_text('# 退款\n\n七日内申请；定制商品除外。')
    (directory/'b.md').write_text('# Camera\n\n1080p USB camera.')
    manifest={'products':[dict(id='P1',name='摄像头',model='C1',aliases=['Camera'])], 'documents':[
        dict(id='A',file='a.md',title='退款',kind='policy',category='退款',version='1'),
        dict(id='B',file='b.md',title='Camera',kind='product',category='摄像头',version='1',product_id='P1')]}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    store=KnowledgeStore(tmp_path/'db',client=QdrantClient(':memory:'),model=Encoder())
    assert store.ingest(directory)['documents']==2
    count=store.backend().count('support_knowledge').count
    store.ingest(directory)
    assert store.backend().count('support_knowledge').count==count
    assert store.search('摄像头','product','Camera')['sources'][0]['doc_id']=='B'
    assert not store.search('unknown','product','missing')['sources']
    assert store.documents(kind='policy')['total']==1
    manifest['documents']=manifest['documents'][:1]
    (directory/'manifest.json').write_text(json.dumps(manifest))
    store.ingest(directory)
    assert store.documents()['total']==1
    assert not store.search('camera','product','P1')['sources']


def test_pdf_docx_reading(tmp_path):
    from docx import Document
    doc=Document();doc.add_heading('规格',1);doc.add_paragraph('USB-C')
    table=doc.add_table(rows=2,cols=2)
    table.cell(0,0).text='容量';table.cell(1,0).text='500ml'
    path=tmp_path/'a.docx';doc.save(path)
    assert '500ml' in parse_document(path)[0]['text']
    from pypdf import PdfWriter
    writer=PdfWriter();writer.add_blank_page(width=100,height=100)
    path=tmp_path/'a.pdf';writer.write(path)
    assert parse_document(path)[0]['text']==''


def test_import_requires_application_stopped(tmp_path):
    import fcntl
    store=KnowledgeStore(tmp_path)
    with (tmp_path/'knowledge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        with pytest.raises(ValueError,match='停止应用'):
            store.ingest(tmp_path)

