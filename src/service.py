import json
import os
import re
import sqlite3
import threading
import weakref
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from sqlalchemy import func, select

from src import tools
from src.backend.db import RequestRecord
from src.memory import ConversationMemory
from src.payment_methods import PAYMENT_METHODS
from src.storage import SupportStore



class ConflictError(ValueError):
    pass


class SupportService:
    def __init__(self, path=None):
        path = Path(path or Path(os.environ.get('SUPPORT_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data'))) / 'memory.sqlite')
        self.store = SupportStore(path)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path.parent / 'checkpoints.sqlite', check_same_thread=False)
        self._close = weakref.finalize(self, self.connection.close)
        from src.orchestrator import build_workflow
        self.graph = build_workflow(SqliteSaver(self.connection))

    def close(self):
        self._close()

    def validate(self, *values):
        if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', v) for v in values):
            raise ValueError('编号只允许1至64位字母、数字、下划线、短横线')

    def _scoped_response(self, user_id, response):
        records = list(response.get('actions') or []) + list((response.get('selection') or {}).get('candidates', []))
        records += (response.get('payment_selection') or {}).get('orders', [])
        records += [e['result'] for e in response.get('trace') or [] if isinstance(e.get('result'), dict) and not e['result'].get('error')]
        records += [i for i in response.get('order_results') or [] if i.get('amount') is not None]
        with tools._session() as session:
            if any(r.get('order_id') and not tools.owned_order(session, r['order_id'], user_id) for r in records):
                return {**response, 'status': 'failed', 'reply': '订单不存在或无权访问',
                        'trace': [], 'actions': [], 'current_action': None, 'selection': None,
                        'payment_selection': None, 'pending_inputs': [], 'tasks': [], 'order_results': [], 'recent_order_ids': []}
        return response

    def _load(self, user_id, request_id):
        with tools._session() as s:
            row = s.get(RequestRecord, (user_id, request_id))
            if row is None:
                return None
            return {'input': json.loads(row.input_data), 'response': self._scoped_response(user_id, json.loads(row.response)) if row.response else None,
                    'session_id': row.session_id}

    def _save(self, user_id, request_id, response):
        with tools._session() as s, s.begin():
            row = s.get(RequestRecord, (user_id, request_id))
            row.response = json.dumps(response, ensure_ascii=False)
            row.tokens = (response.get('usage') or {}).get('total_tokens') or 0

    def snapshot(self, user_id, session_id):
        self.validate(user_id, session_id)
        with tools._session() as s:
            rows = s.scalars(select(RequestRecord).where(RequestRecord.user_id == user_id,
                RequestRecord.session_id == session_id, RequestRecord.kind == 'chat')
                .order_by(RequestRecord.created_at.desc()).limit(50)).all()
            total = s.scalar(select(func.sum(RequestRecord.tokens)).where(RequestRecord.user_id == user_id,
                RequestRecord.session_id == session_id, RequestRecord.kind == 'chat')) or 0
            recent = self.store.load_session(user_id, session_id).get('recent_order_ids', [])
            turns = []
            for row in reversed(rows):
                source = json.loads(row.input_data)
                result = self._scoped_response(user_id, json.loads(row.response)) if row.response else {'reply': '正在处理', 'agent': '', 'status': 'processing'}
                turn = {'user': source['message'], 'request_id': row.request_id, **result}
                turns.append(turn)
                ids = result.get('recent_order_ids')
                if ids is not None:
                    recent = ids
        payload = {'turns': turns, 'recent_order_ids': recent, 'last_order_id': recent[0] if len(recent) == 1 else None, 'total_tokens': total}
        self.store.save_session(user_id, session_id, payload)
        return payload

    def _config(self, user_id, session_id, request_id):
        return {'configurable': {'thread_id': uuid5(NAMESPACE_URL, json.dumps([user_id, session_id, request_id])).hex}, 'recursion_limit': 1000}

    def _invoke(self, value, config):
        for attempt in range(2):
            try:
                return self.graph.invoke(value, config, durability='sync')
            except Exception:
                state = self.graph.get_state(config)
                if attempt or 'plan' not in state.values:
                    raise RuntimeError('处理暂未完成，请继续原请求。')
                interruptions = [i for task in state.tasks for i in task.interrupts]
                if state.values and not interruptions:
                    value = None

    def _graph_result(self, user_id, session_id, request_id, result):
        response = {**result['response'], 'user_id': user_id,
                    'session_id': session_id, 'request_id': request_id}
        inputs = response['pending_inputs']
        orders = next((i for i in inputs if i['type'] == 'orders'), None)
        method = next((i for i in inputs if i['type'] == 'method'), None)
        response['selection'] = {**orders, 'recent_order_ids': self.store.load_session(user_id, session_id).get('recent_order_ids', [])} if orders else None
        response['payment_selection'] = method
        ids = list(dict.fromkeys(f['order_id'] for t in response['tasks'] for f in t.get('facts', []) if f.get('order_id') and not f.get('error')))
        if ids:
            response['recent_order_ids'] = ids
        self._save(user_id, request_id, response)
        return response

    def chat(self, user_id, session_id, message, request_id=None):
        request_id = request_id or uuid4().hex
        session_id = session_id or (self._load(user_id, request_id) or {}).get('session_id') or uuid4().hex
        self.validate(user_id, session_id, request_id)
        if not message.strip() or len(message) > 2000:
            raise ValueError('消息需为1至2000个字符')
        with self.lock:
            old = self._load(user_id, request_id)
            source = {'message': message, 'session_id': session_id}
            if old:
                if old['input'] != source:
                    raise ConflictError('请求编号已用于其他内容')
                if old['response'] and old['response']['status'] != 'processing':
                    return old['response']
            else:
                if any(t['status'] in {'processing', 'awaiting_selection', 'awaiting_payment_method', 'awaiting_confirmation'} for t in self.snapshot(user_id, session_id)['turns']):
                    raise ConflictError('请先完成当前操作的确认或取消')
                with tools._session() as db, db.begin():
                    db.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                        kind='chat', input_data=json.dumps(source)))
            config = self._config(user_id, session_id, request_id)
            state = self.graph.get_state(config)
            memory = ConversationMemory()
            payload = self.snapshot(user_id, session_id)
            for turn in [t for t in payload['turns'] if t['status'] == 'completed'][-6:]:
                memory.add_turn(turn['user'], turn['reply'])
            try:
                value = None if state.values else {'user_id': user_id, 'session_id': session_id,
                    'request_id': request_id, 'message': message, 'history': memory.turns,
                    'profile': self.store.profile(user_id)}
                pending = None
                interruptions = [i for task in state.tasks for i in task.interrupts]
                if interruptions:
                    with tools._session() as db:
                        records = db.scalars(select(RequestRecord).where(RequestRecord.user_id == user_id,
                            RequestRecord.session_id == session_id, RequestRecord.kind == 'decision', RequestRecord.response.is_(None))).all()
                        for row in records:
                            event = json.loads(row.input_data)
                            if event.pop('chat_request_id') != request_id:
                                continue
                            if event['type'] != 'decision' and not event.get('input_id'):
                                candidates = [i for i in interruptions[0].value['pending_inputs'] if i['type'] == event['type']]
                                if len(candidates) != 1:
                                    continue
                                event['input_id'] = candidates[0]['input_id']
                            pending, value = row.request_id, Command(resume=event)
                            break
                result = self._invoke(value, config)
                response = self._graph_result(user_id, session_id, request_id, result)
                if pending:
                    self._save(user_id, pending, response)
                return response
            except Exception:
                self._save(user_id, request_id, {'status': 'processing',
                    'reply': '本次处理尚未完成，请继续处理原请求。', 'agent': 'assistant', 'trace': [], 'usage': {}})
                raise

    def _submit(self, user_id, session_id, chat_request_id, request_id, event):
        self.validate(user_id, session_id, chat_request_id, request_id)
        if event['decision'] not in ({'confirm', 'cancel'} if event['type'] == 'decision' else {'submit', 'cancel'}):
            raise ValueError('无效决定')
        with self.lock:
            base = self.get_request(user_id, session_id, chat_request_id)
            old = self._load(user_id, request_id)
            source = {'chat_request_id': chat_request_id, **event}
            if old:
                if old['input'] != source or old['session_id'] != session_id:
                    raise ConflictError('请求编号已用于其他内容')
                if old['response']:
                    return old['response']
            if event['type'] == 'decision':
                current = base.get('current_action')
                if not current or current['action_id'] != event['action_id']:
                    if any(a['action_id'] == event['action_id'] and a['status'] != 'pending' for a in base.get('actions', [])):
                        return base
                    raise ConflictError('确认操作已变化')
            else:
                candidates = [i for i in base.get('pending_inputs', []) if i['type'] == event['type'] and (not event.get('input_id') or i['input_id'] == event['input_id'])]
                if len(candidates) != 1:
                    raise ConflictError('请选择明确的输入项')
                event = {**event, 'input_id': candidates[0]['input_id']}
                if event['type'] == 'method' and event['decision'] == 'submit' and event.get('method') not in PAYMENT_METHODS:
                    raise ValueError('请选择列表中的支付方式')
                if event['type'] == 'orders' and event['decision'] == 'submit':
                    ids = event['order_ids']
                    if not ids:
                        raise ValueError('请至少选择一笔订单')
                    if not set(ids) <= {i['order_id'] for i in candidates[0]['candidates']}:
                        raise ConflictError('所选订单不在候选集合中')
            if not old:
                with tools._session() as db, db.begin():
                    db.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                        kind='decision', input_data=json.dumps(source)))
            config = self._config(user_id, session_id, chat_request_id)
            state = self.graph.get_state(config)
            interruptions = [i for task in state.tasks for i in task.interrupts]
            if old and interruptions:
                waiting = interruptions[0].value
                consumed = ((event['type'] == 'decision' and (waiting.get('current_action') or {}).get('action_id') != event['action_id'])
                    or (event['type'] != 'decision' and not any(i['input_id'] == event['input_id'] for i in waiting.get('pending_inputs', []))))
                if consumed:
                    response = self._graph_result(user_id, session_id, chat_request_id, state.values)
                    self._save(user_id, request_id, response)
                    return response
            try:
                result = self._invoke(Command(resume=event), config)
            except RuntimeError:
                self._save(user_id, chat_request_id, {**base, 'status': 'processing'})
                raise
            response = self._graph_result(user_id, session_id, chat_request_id, result)
            self._save(user_id, request_id, response)
            return response

    def get_request(self, user_id, session_id, request_id):
        self.validate(user_id, session_id, request_id)
        record = self._load(user_id, request_id)
        if not record or record['session_id'] != session_id or 'message' not in record['input']:
            raise ConflictError('请求不存在')
        return record['response'] or {'status': 'processing', 'request_id': request_id}

    def resume(self, user_id, session_id, chat_request_id, request_id, action_id, decision):
        return self._submit(user_id, session_id, chat_request_id, request_id,
            {'type': 'decision', 'action_id': action_id, 'decision': decision})

    def select_orders(self, user_id, session_id, chat_request_id, request_id, decision, order_ids, input_id=None):
        return self._submit(user_id, session_id, chat_request_id, request_id,
            {'type': 'orders', 'decision': decision, 'order_ids': order_ids, 'input_id': input_id})

    def select_payment_method(self, user_id, session_id, chat_request_id, request_id, decision, method=None, input_id=None):
        return self._submit(user_id, session_id, chat_request_id, request_id,
            {'type': 'method', 'decision': decision, 'method': method, 'input_id': input_id})
