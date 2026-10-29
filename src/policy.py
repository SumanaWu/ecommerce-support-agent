"""Versioned Markdown evidence, SQLite keyword search and public-query caching."""
from collections import OrderedDict
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import RLock

from sqlalchemy import case, func, select

from src.backend.db import PolicyChunk, PolicyDocument, PROJECT_ROOT

DEFAULT_POLICY_DIR = PROJECT_ROOT / 'policies'
NO_EVIDENCE = '当前已生效的政策资料没有足够依据回答这个问题，请补充具体问题。'
_cache = OrderedDict()
_cache_lock = RLock()


def _read_document(path):
    text = path.read_text(encoding='utf-8')
    parts = text.split('---', 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f'{path.name} 缺少政策元数据头。')
    metadata = {}
    for line in parts[1].strip().splitlines():
        key, separator, value = line.partition(':')
        if not separator or key.strip() in metadata:
            raise ValueError(f'{path.name} 的政策元数据无效。')
        metadata[key.strip()] = value.strip().strip('"\'')
    if set(metadata) != {'policy_id', 'version', 'title', 'effective_from', 'keywords'}:
        raise ValueError('政策头必须包含 policy_id、version、title、effective_from、keywords。')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', metadata['policy_id']) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,40}', metadata['version']):
        raise ValueError('政策编号或版本格式无效。')
    metadata['effective_from'] = date.fromisoformat(metadata['effective_from'])
    metadata['keywords'] = [word.strip().lower() for word in metadata['keywords'].split(',') if word.strip()]
    if not metadata['title'] or len(metadata['title']) > 200 or not parts[2].strip():
        raise ValueError('政策标题或正文不能为空，标题最多 200 字。')
    try:
        source = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source = path.name
    chunks, section, paragraphs = [], metadata['title'], []
    def flush():
        content = '\n'.join(paragraphs).strip()
        for offset in range(0, len(content), 1000):
            chunks.append((section[:200], content[offset:offset + 1000]))
        paragraphs.clear()
    for line in parts[2].strip().splitlines():
        if line.startswith('#'):
            flush()
            section = line.lstrip('#').strip()
        else:
            paragraphs.append(line)
    flush()
    if not chunks:
        raise ValueError('政策正文必须包含可检索内容。')
    return {**metadata, 'source': source, 'content_hash': hashlib.sha256(text.encode()).hexdigest(), 'chunks': chunks}


def import_policy_documents(directory, session_factory):
    """Append immutable versions; validate the whole directory before writing."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('政策目录不存在。')
    documents = [_read_document(path) for path in sorted(directory.glob('*.md'))]
    keys = [(doc['policy_id'], doc['version']) for doc in documents]
    if len(keys) != len(set(keys)):
        raise ValueError('同一导入批次的政策编号与版本不能重复。')
    with session_factory.begin() as db:
        db.connection().exec_driver_sql('BEGIN IMMEDIATE')
        revision = db.scalar(select(func.max(PolicyDocument.revision))) or 0
        imported = 0
        for doc in documents:
            existing = db.get(PolicyDocument, (doc['policy_id'], doc['version']))
            if existing:
                if existing.content_hash != doc['content_hash']:
                    raise ValueError('同一政策版本不可改写；请递增 version 后重新导入。')
                continue
            revision += 1
            imported += 1
            values = {key: value for key, value in doc.items() if key != 'chunks'}
            values['keywords'] = json.dumps(doc['keywords'], ensure_ascii=False)
            db.add(PolicyDocument(**values, revision=revision))
            for index, (section, content) in enumerate(doc['chunks']):
                db.add(PolicyChunk(policy_id=doc['policy_id'], version=doc['version'], chunk_index=index,
                                   section=section, content=content,
                                   search_text=' '.join([doc['title'], *doc['keywords'], section, content]).lower()))
    return {'imported': imported, 'catalog_revision': revision}


def _query_terms(question):
    terms = re.findall(r'[a-z0-9]+', question.lower())
    for word in re.findall(r'[\u4e00-\u9fff]+', question):
        terms.extend(word[index:index + 2] for index in range(len(word) - 1))
    stop = {'how', 'long', 'does', 'the', 'is', 'what', 'of', 'a', 'an', 'it', 'can', 'i', 'my',
            'policy', 'policies', 'rules', 'rule', 'please', 'tell', 'me', 'take',
            '政策', '规则', '什么', '是否', '可以', '怎么', '一下', '请问', '的是', '么是', '策是', '的政', '关于', '哪些'}
    return list(dict.fromkeys(term for term in terms if term not in stop))[:32]


def search_policy(question, session_factory, *, as_of=None, limit=3):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError('政策问题需为 1 至 2000 字。')
    if not 1 <= limit <= 5:
        raise ValueError('政策检索数量需为 1 至 5。')
    as_of = as_of or datetime.now(timezone.utc).date()
    with session_factory() as db:
        revision = db.scalar(select(func.max(PolicyDocument.revision))) or 0
        database = db.get_bind().url.database
        identity = str(Path(database).resolve()) if database and database != ':memory:' else db.get_bind()
        key = (identity, revision, as_of.isoformat(), question.strip().lower(), limit)
        with _cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return {**deepcopy(_cache[key]), 'cached': True}
        documents = db.scalars(select(PolicyDocument).where(PolicyDocument.effective_from <= as_of)
                               .order_by(PolicyDocument.effective_from.desc(), PolicyDocument.revision.desc())).all()
        current = {}
        for doc in documents:
            current.setdefault(doc.policy_id, doc)
        terms = _query_terms(question)
        sources = []
        if terms and current:
            conditions = [(PolicyChunk.policy_id == doc.policy_id) & (PolicyChunk.version == doc.version) for doc in current.values()]
            from sqlalchemy import or_
            score = sum(case((func.instr(PolicyChunk.search_text, term) > 0, 1), else_=0) for term in terms)
            rows = db.execute(select(PolicyChunk, score.label('score')).where(or_(*conditions), score > 0)
                              .order_by(score.desc(), PolicyChunk.policy_id, PolicyChunk.chunk_index).limit(limit)).all()
            for chunk, _ in rows:
                doc = current[chunk.policy_id]
                sources.append({'policy_id': doc.policy_id, 'version': doc.version, 'title': doc.title,
                                'source': doc.source, 'effective_from': doc.effective_from.isoformat(),
                                'section': chunk.section, 'chunk_index': chunk.chunk_index, 'excerpt': chunk.content})
    result = {'found': bool(sources), 'sources': sources, 'catalog_revision': revision, 'as_of': as_of.isoformat(), 'cached': False}
    with _cache_lock:
        _cache[key] = deepcopy(result)
        while len(_cache) > 128:
            _cache.popitem(last=False)
    return result


def answer_with_sources(question, session_factory, *, as_of=None):
    result = search_policy(question, session_factory, as_of=as_of)
    if not result['found']:
        return {**result, 'answer': NO_EVIDENCE}
    parts = ['相关政策如下：']
    for index, source in enumerate(result['sources'], 1):
        parts.append(f"[{index}] {source['excerpt']}\n来源：{source['source']} · {source['section']} · 版本 {source['version']} · 生效 {source['effective_from']}")
    return {**result, 'answer': '\n\n'.join(parts)}
