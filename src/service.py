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
from src.actions import prepare_action
from src.backend.db import RequestRecord
from src.confirmation import build_confirmation_graph
from src.llm import get_llm
from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator
from src.runtime_context import active_profile, active_request
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
        self.graph = build_confirmation_graph(SqliteSaver(self.connection))

    def close(self):
        self._close()

    def validate(self, *values):
        if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', v) for v in values):
            raise ValueError('编号只允许1至64位字母、数字、下划线、短横线')

    def _config(self, user_id, request_id):
        return {'configurable': {'thread_id': uuid5(NAMESPACE_URL, json.dumps([user_id, request_id])).hex}}

    def _invoke_confirmation(self, value, config):
        for attempt in range(2):
            try:
                return self.graph.invoke(value, config)
            except Exception:
                if attempt:
                    ticket = tools.escalate_to_human('确认流程执行失败', config['configurable']['thread_id'])
                    raise RuntimeError(ticket['message'])
                state = self.graph.get_state(config)
                interruptions = [i for task in state.tasks for i in task.interrupts]
                same_target = (isinstance(value, Command) and interruptions
                    and isinstance(value.resume, dict)
                    and (interruptions[0].value['action_id'], interruptions[0].value['version'])
                        == (value.resume.get('action_id'), value.resume.get('version')))
                if state.values and not same_target:
                    value = None

    def _load(self, user_id, request_id):
        with tools._session() as s:
            row = s.get(RequestRecord, (user_id, request_id))
            if row is None:
                return None
            return {'input': json.loads(row.input_data), 'response': json.loads(row.response) if row.response else None,
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
            memory = ConversationMemory(last_order_id=self.store.load_session(user_id, session_id)['last_order_id'])
            turns = []
            for row in reversed(rows):
                source = json.loads(row.input_data)
                result = json.loads(row.response) if row.response else {'reply': '正在处理', 'agent': '', 'status': 'processing'}
                turn = {'user': source['message'], 'request_id': row.request_id, **result}
                turns.append(turn)
                memory.update_last_order_id(result.get('trace', []))
        payload = {'turns': turns, 'last_order_id': memory.last_order_id, 'total_tokens': total}
        self.store.save_session(user_id, session_id, payload)
        return payload

    def _graph_result(self, user_id, request_id, base, result):
        base = {**base, 'actions': result.get('actions', []),
                'status': 'awaiting_confirmation' if result.get('__interrupt__') else 'completed'}
        if result.get('__interrupt__'):
            base['current_action'] = result['__interrupt__'][0].value
            base['reply'] = '以下操作尚未执行，请逐项确认。'
        else:
            base['current_action'] = None
            base['reply'] = result.get('reply', base.get('reply', '处理完成'))
        self._save(user_id, request_id, base)
        return base

    def chat(self, user_id, session_id, message, runner=None, request_id=None):
        request_id = request_id or uuid4().hex
        self.validate(user_id, request_id)
        if not message.strip() or len(message) > 2000:
            raise ValueError('消息需为1至2000个字符')
        with self.lock:
            previous = self._load(user_id, request_id)
            session_id = session_id or (previous['session_id'] if previous else uuid4().hex)
            self.validate(session_id)
            source = {'message': message, 'session_id': session_id}
            if previous:
                if previous['input'] != source:
                    raise ConflictError('请求编号已用于其他内容')
                if previous['response'] and previous['response']['status'] != 'processing':
                    return previous['response']
            else:
                if any(t['status'] in {'processing', 'awaiting_confirmation'} for t in self.snapshot(user_id, session_id)['turns']):
                    raise ConflictError('请先完成当前操作的确认或取消')
                with tools._session() as s, s.begin():
                    s.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                                        input_data=json.dumps(source), kind='chat'))
            base = previous['response'] if previous else None
            config = self._config(user_id, request_id)
            token = active_profile.set(self.store.profile(user_id))
            request_token = active_request.set(self._config(user_id, request_id)['configurable']['thread_id'])
            try:
                checkpoint = self.graph.get_state(config)
                if checkpoint.values:
                    result = self._invoke_confirmation(None, config)
                    return self._graph_result(user_id, request_id, base, result)
                if not base:
                    if runner is None or runner is run_orchestrator:
                        get_llm()
                    payload = self.snapshot(user_id, session_id)
                    memory = ConversationMemory(last_order_id=payload['last_order_id'])
                    for turn in [t for t in payload['turns'] if t.get('status') == 'completed'][-6:]:
                        memory.add_turn(turn['user'], turn['reply'])
                    augmented = message if re.findall(r'\b\d{4,8}\b', message) else memory.context_note() + message
                    result = (runner or run_orchestrator)(augmented, history=memory.turns)
                    base = {k: result.get(k) for k in ['reply', 'agent', 'trace', 'usage', 'cached']}
                    base.update(request_id=request_id, user_id=user_id, session_id=session_id, status='processing')
                    self._save(user_id, request_id, base)
                proposals = []
                for entry in base.get('trace') or []:
                    proposal = entry['result'].get('pending_action') if isinstance(entry['result'], dict) else None
                    if proposal and proposal not in proposals:
                        if any(p['order_id'] == proposal['order_id'] for p in proposals):
                            raise ConflictError('同一订单存在冲突方案，请分别提交')
                        proposals.append(proposal)
                pending = [prepare_action(user_id, session_id, request_id, i, p) for i, p in enumerate(proposals)]
                if not pending:
                    base['status'] = 'completed'
                    self._save(user_id, request_id, base)
                    return base
                result = self._invoke_confirmation({'user_id': user_id, 'session_id': session_id,
                                            'actions': pending, 'index': 0}, config)
                return self._graph_result(user_id, request_id, base, result)
            except Exception as exc:
                if base:
                    if isinstance(exc, ValueError):
                        base = {**base, 'status': 'failed', 'reply': str(exc)}
                    self._save(user_id, request_id, base)
                else:
                    self._save(user_id, request_id, {'request_id': request_id, 'user_id': user_id,
                        'session_id': session_id, 'status': 'failed', 'reply': str(exc), 'agent': '', 'trace': [], 'usage': {}})
                raise
            finally:
                active_profile.reset(token)
                active_request.reset(request_token)

    def get_request(self, user_id, session_id, request_id):
        self.validate(user_id, session_id, request_id)
        record = self._load(user_id, request_id)
        if not record or record['session_id'] != session_id or 'message' not in record['input']:
            raise ConflictError('请求不存在')
        return record['response'] or {'status': 'processing', 'request_id': request_id}

    def resume(self, user_id, session_id, chat_request_id, request_id, action_id, version, decision):
        self.validate(user_id, session_id, chat_request_id, request_id, action_id)
        if decision not in {'confirm', 'cancel'}:
            raise ValueError('无效确认决定')
        source = {'chat_request_id': chat_request_id, 'action_id': action_id, 'version': version, 'decision': decision}
        with self.lock:
            old = self._load(user_id, request_id)
            if old and (old['input'] != source or old['session_id'] != session_id):
                raise ConflictError('请求编号已用于其他决定')
            if old and old['response']:
                return old['response']
            with tools._session() as s:
                decisions = s.scalars(select(RequestRecord).where(RequestRecord.user_id == user_id,
                    RequestRecord.session_id == session_id, RequestRecord.kind == 'decision')).all()
                for saved in decisions:
                    previous = json.loads(saved.input_data)
                    if (previous['chat_request_id'], previous['action_id'], previous['version']) == (chat_request_id, action_id, version) and previous['decision'] != decision:
                        raise ConflictError('该操作版本已有其他确认决定')
            base = self.get_request(user_id, session_id, chat_request_id)
            current = base.get('current_action')
            if not current or (current['action_id'], current['version']) != (action_id, version):
                terminal = next((a for a in base.get('actions', []) if a['action_id'] == action_id and a['version'] == version and a['status'] != 'pending'), None)
                if terminal:
                    return base
                raise ConflictError('操作或版本已变化')
            if not old:
                with tools._session() as s, s.begin():
                    s.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                                        input_data=json.dumps(source), kind='decision'))
            config = self._config(user_id, chat_request_id)
            token = active_profile.set(self.store.profile(user_id))
            request_token = active_request.set(config['configurable']['thread_id'])
            try:
                state = self.graph.get_state(config)
                interruptions = [i for task in state.tasks for i in task.interrupts]
                if interruptions and (interruptions[0].value['action_id'], interruptions[0].value['version']) != (action_id, version):
                    # A prior execution advanced the graph before its response was saved.
                    result = {**state.values, '__interrupt__': interruptions}
                else:
                    command = Command(resume={'action_id': action_id, 'version': version, 'decision': decision}) if interruptions else None
                    result = self._invoke_confirmation(command, config)
                response = self._graph_result(user_id, chat_request_id, base, result)
                self._save(user_id, request_id, response)
                return response
            finally:
                active_profile.reset(token)
                active_request.reset(request_token)
