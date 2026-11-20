"""Single-platform document catalog, offline import and local multilingual retrieval."""
import fcntl
import hashlib
import json
import re
import os
import sqlite3
import threading
import unicodedata
import uuid
from pathlib import Path


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class KnowledgeStore:
    def __init__(self, root=None, client=None, model=None):
        self.root = Path(root or os.environ.get('SUPPORT_DATA_DIR', 'data'))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'knowledge.sqlite'
        self.client, self.model = client, model
        self.model_lock = threading.RLock()
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases (alias TEXT, product_id TEXT, payload TEXT, PRIMARY KEY(alias,product_id));
            CREATE TABLE IF NOT EXISTS index_state (id INTEGER PRIMARY KEY, status TEXT, fingerprint TEXT);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def status(self):
        with self.connect() as db:
            row = db.execute('SELECT status FROM index_state WHERE id=1').fetchone()
        return row[0] if row else 'empty'

    def backend(self):
        if self.client is None:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=os.environ.get('SUPPORT_QDRANT_URL', 'http://127.0.0.1:6333'), timeout=30)
        return self.client

    def encoder(self, download=False):
        with self.model_lock:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu',
                    cache_folder=str(self.root / 'models'), local_files_only=not download)
        return self.model

    def documents(self, kind='', category='', product_id='', q='', offset=0, limit=20):
        conditions, args = [], []
        for field, value in (('kind',kind),('category',category),('product_id',product_id)):
            if value:
                conditions.append(f"json_extract(payload,'$.{field}')=?")
                args.append(value)
        if q:
            conditions.append("(id || ' ' || json_extract(payload,'$.title') || ' ' || coalesce(json_extract(payload,'$.product_name'),'') || ' ' || coalesce(json_extract(payload,'$.product_id'),'')) LIKE ?")
            args.append('%'+q+'%')
        where = ' WHERE '+' AND '.join(conditions) if conditions else ''
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM documents'+where,args).fetchone()[0]
            rows = db.execute("SELECT json_remove(payload,'$.sections') FROM documents"+where+' ORDER BY id LIMIT ? OFFSET ?', [*args,limit,offset]).fetchall()
        return {'items':[json.loads(r[0]) for r in rows], 'total':total, 'index_status':self.status()}

    def categories(self, kind=''):
        with self.connect() as db:
            rows = db.execute("SELECT DISTINCT json_extract(payload,'$.category') FROM documents WHERE (? = '' OR json_extract(payload,'$.kind') = ?) ORDER BY 1", (kind, kind)).fetchall()
        return [row[0] for row in rows if row[0]]

    def document(self, document_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM documents WHERE id=?', (document_id,)).fetchone()
        if not row:
            raise ValueError('资料不存在')
        return json.loads(row[0])

    def ingest(self, directory):
        from qdrant_client import models
        directory = Path(directory).resolve()
        with (self.root / 'knowledge.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('请先停止应用再导入资料') from exc
            manifest = json.loads((directory / 'manifest.json').read_text())
            products = {r['id']:r for r in manifest['products']}
            if len(products) != len(manifest['products']):
                raise ValueError('商品编号重复')
            prepared = []
            for meta in manifest['documents']:
                path = (directory / meta['file']).resolve()
                if not path.is_relative_to(directory) or meta['kind'] not in {'policy','product'}:
                    raise ValueError('无效路径或资料类型')
                if meta['kind'] == 'product' and meta.get('product_id') not in products:
                    raise ValueError('未知商品引用')
                sections = parse_document(path)
                if not sections or not any(r['text'].strip() for r in sections):
                    raise ValueError(f'{path.name}没有可提取的文字，扫描件请先OCR')
                item = {**meta, 'sections':sections}
                if meta.get('product_id'):
                    item['product_name'] = products[meta['product_id']]['name']
                item['hash'] = hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                prepared.append(item)
            if len({r['id'] for r in prepared}) != len(prepared):
                raise ValueError('文档编号重复')
            model = self.encoder(download=True)
            model.max_seq_length = 480
            fingerprint = 'multilingual-e5-small:384:chunks-v3-budgeted-60'
            client = self.backend()
            with self.connect() as db:
                old = {r[0]:json.loads(r[1]) for r in db.execute('SELECT id,payload FROM documents')}
                catalog_ids = set(old)
                previous = db.execute('SELECT fingerprint,status FROM index_state WHERE id=1').fetchone()
                db.execute("INSERT OR REPLACE INTO index_state VALUES (1,'importing',?)", (fingerprint,))
            try:
                if not client.collection_exists('support_knowledge') or not previous or previous[0] != fingerprint:
                    if client.collection_exists('support_knowledge'):
                        client.delete_collection('support_knowledge')
                    client.create_collection('support_knowledge', vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE,on_disk=True))
                    for key in ('doc_id','kind','product_id','category'):
                        client.create_payload_index('support_knowledge', key, models.PayloadSchemaType.KEYWORD)
                    old = {}
                if previous and previous[1] != 'ready':
                    old = {}
                expected_points = 0
                for item in prepared:
                    if old.get(item['id'],{}).get('hash') != item['hash']:
                        chunks = []
                        sections = item['sections']
                        if item['kind'] == 'policy':
                            # A policy is a single rule: keep its conditions and exceptions together.
                            sections = [{'id':'policy','location':item['title'],'text':'\n\n'.join(r['text'] for r in sections)}]
                        for section in sections:
                            ids = model.tokenizer.encode(section['text'],add_special_tokens=False)
                            available = max(64,460-len(model.tokenizer.encode('passage: '+item['title'][:40],add_special_tokens=False)))
                            for start in range(0,max(1,len(ids)-60),max(1,available-60)):
                                text = model.tokenizer.decode(ids[start:start+available],skip_special_tokens=True)
                                source_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item['id']+item['hash']+section['id']+str(start)))
                                chunks.append({'id':source_id,'doc_id':item['id'],'title':item['title'],'kind':item['kind'], 'category':item['category'], 'product_id':item.get('product_id',''), 'version':item['version'],'section':section['id'],'location':section['location'],'content':text})
                        client.delete('support_knowledge', models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key='doc_id',match=models.MatchValue(value=item['id']))])),wait=True)
                        for start in range(0,len(chunks),32):
                            batch = chunks[start:start+32]
                            vectors = model.encode(['passage: '+r['title'][:40]+' '+r['content'] for r in batch],normalize_embeddings=True,batch_size=32)
                            client.upsert('support_knowledge',[models.PointStruct(id=r['id'],vector=v.tolist(),payload=r) for r,v in zip(batch,vectors)],wait=True)
                        item['chunk_count'] = len(chunks)
                        with self.connect() as db:
                            db.execute('INSERT OR REPLACE INTO documents VALUES (?,?)',(item['id'],json.dumps(item,ensure_ascii=False)))
                    expected_points += item.get('chunk_count',old.get(item['id'],{}).get('chunk_count',0))
                for removed in catalog_ids-{r['id'] for r in prepared}:
                    client.delete('support_knowledge',models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key='doc_id',match=models.MatchValue(value=removed))])),wait=True)
                if client.count('support_knowledge',exact=True).count != expected_points:
                    raise ValueError('索引片段数量校验失败，请重新导入')
                with self.connect() as db:
                    db.execute('DELETE FROM aliases')
                    for product in products.values():
                        for alias in {product['id'], product['name'],product.get('model',''),*product.get('aliases',[])}-{''}:
                            db.execute('INSERT OR IGNORE INTO aliases VALUES (?,?,?)',(normalized(alias),product['id'],json.dumps(product,ensure_ascii=False)))
                    for removed in catalog_ids-{r['id'] for r in prepared}:
                        db.execute('DELETE FROM documents WHERE id=?',(removed,))
                    db.execute("UPDATE index_state SET status='ready' WHERE id=1")
            except Exception:
                with self.connect() as db:
                    db.execute("UPDATE index_state SET status='failed' WHERE id=1")
                raise
            return {'documents':len(prepared),'products':len(products),'status':'ready'}

    def search(self, query, kind='all', product_ref='', category=''):
        from qdrant_client import models
        if self.status() != 'ready':
            raise ValueError('知识库尚未导入完成，请先运行目录导入')
        if kind != 'product' and not product_ref and (any(word in query for word in ('全部政策','所有政策','政策概览','全部规则','所有规则')) or re.search(r'\b(?:all (?:the )?(?:platform )?(?:policies|rules)|polic(?:y|ies) overview|overview of (?:the )?(?:platform )?(?:policies|rules))\b', query, re.I)):
            with self.connect() as db:
                rows = db.execute("SELECT payload FROM documents WHERE json_extract(payload,'$.kind')='policy' ORDER BY id").fetchall()
            sources, seen = [], set()
            for row in rows:
                document = json.loads(row[0])
                if document['category'] in seen:
                    continue
                seen.add(document['category'])
                section = next((r for r in document['sections'] if r['location'] == '具体规定'),document['sections'][0])
                sources.append({'id':str(uuid.uuid5(uuid.NAMESPACE_URL,document['id']+document['hash']+'overview')), 'doc_id':document['id'],'title':document['category']+'：'+document['title'],'version':document['version'],'location':section['location'],'section':section['id'],'content':section['text']})
            return {'knowledge':True,'sources':sources,'message':'这是分类概览，每类仅选一条示例，不是全部规则。完整规则请在平台资料详情页按分类查看。'}
        conditions = []
        if kind != 'all':
            conditions.append(models.FieldCondition(key='kind',match=models.MatchValue(value=kind)))
        if category and self.documents(category=category,limit=1)['total']:
            conditions.append(models.FieldCondition(key='category',match=models.MatchValue(value=category)))
        if product_ref and kind != 'policy':
            with self.connect() as db:
                matches = db.execute('SELECT product_id,payload FROM aliases WHERE alias=?',(normalized(product_ref),)).fetchall()
                if not matches and re.match(r'^the\s+', product_ref.strip(), re.I):
                    # Permit only an optional English article, never fuzzy product matching.
                    reference = re.sub(r'^the\s+', '', product_ref.strip(), flags=re.I)
                    matches = db.execute('SELECT product_id,payload FROM aliases WHERE alias=?', (normalized(reference),)).fetchall()
            if len(matches) != 1:
                return {'knowledge':True,'sources':[], 'message':'请提供准确商品名称、型号或编号。', 'candidates':[json.loads(r[1]) for r in matches]}
            product_condition = models.FieldCondition(key='product_id',match=models.MatchValue(value=matches[0][0]))
            conditions.append(product_condition if kind == 'product' else models.Filter(should=[product_condition,models.FieldCondition(key='kind',match=models.MatchValue(value='policy'))]))
        elif kind == 'product':
            return {'knowledge':True,'sources':[], 'message':'请提供商品名称、型号或编号。'}
        with self.model_lock:
            vector = self.encoder().encode(['query: '+query],normalize_embeddings=True)[0].tolist()
        hits = self.backend().query_points('support_knowledge',query=vector,query_filter=models.Filter(must=conditions),limit=12).points
        sources, seen = [], set()
        for hit in hits:
            key = (hit.payload['doc_id'],hit.payload['section'])
            if key not in seen and len(sources) < 6:
                sources.append(dict(hit.payload))
                seen.add(key)
        # Include the complete matching section so exceptions survive window boundaries.
        budget = 6000
        for source in sources:
            document = self.document(source['doc_id'])
            section = ({'text':'\n\n'.join(r['text'] for r in document['sections'])} if source['section'] == 'policy' else next(r for r in document['sections'] if r['id'] == source['section']))
            ids = self.encoder().tokenizer.encode(section['text'],add_special_tokens=False)
            if len(ids) <= min(1000,budget):
                source['content'] = section['text']
            budget -= len(self.encoder().tokenizer.encode(source['content'],add_special_tokens=False))
            if budget < 0:
                source['content'] = ''
        sources = [r for r in sources if r['content']]
        return {'knowledge':True,'sources':sources, 'message':'' if sources else '资料库没有找到对应内容。'}


def parse_document(path):
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        return [{'id':str(i+1),'location':f'第{i+1}页','text':page.extract_text() or ''} for i,page in enumerate(PdfReader(path).pages)]
    if suffix == '.docx':
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        texts = []
        for block in Document(path).iter_inner_content():
            texts.append(block.text if isinstance(block,Paragraph) else '\n'.join(' | '.join(c.text for c in row.cells) for row in block.rows) if isinstance(block,Table) else '')
        text = '\n\n'.join(texts)
    elif suffix in {'.txt','.md'}:
        text = path.read_text(encoding='utf-8-sig')
    else:
        raise ValueError(f'不支持文件格式：{suffix}')
    import re
    parts = re.split(r'(?m)(?=^#{1,6} )',text)
    return [{'id':str(i+1),'location':p.splitlines()[0].lstrip('# ').strip()[:100], 'text':p.strip()} for i,p in enumerate(parts) if p.strip()]


from functools import lru_cache


@lru_cache(maxsize=4)
def get_knowledge(root):
    return KnowledgeStore(root)
