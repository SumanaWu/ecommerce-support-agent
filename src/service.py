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
from src.runtime_context import active_profile, active_request, active_customer
from src.refund_flow import explicit_orders
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
                    and interruptions[0].value['action_id'] == value.resume.get('action_id'))
                if state.values and not same_target:
                    value = None

    def _scoped_response(self, user_id, response):
        records = list(response.get('actions') or []) + list((response.get('selection') or {}).get('candidates', []))
        records += [e['result'] for e in response.get('trace') or [] if isinstance(e.get('result'), dict) and not e['result'].get('error')]
        records += [i for i in response.get('order_results') or [] if i.get('amount') is not None]
        with tools._session() as session:
            if any(r.get('order_id') and not tools.owned_order(session, r['order_id'], user_id) for r in records):
                return {**response, 'status': 'failed', 'reply': '订单不存在或无权访问',
                        'trace': [], 'actions': [], 'current_action': None, 'selection': None,
                        'order_results': [], 'recent_order_ids': []}
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

    def _graph_result(self, user_id, request_id, base, result):
        base = {**base, 'actions': result.get('actions', []),
                'status': 'awaiting_confirmation' if result.get('__interrupt__') else 'completed'}
        if result.get('__interrupt__'):
            base['current_action'] = result['__interrupt__'][0].value
            base['reply'] = '以下操作尚未执行，请逐项确认。'
        else:
            base['current_action'] = None
            base['reply'] = result.get('reply', base.get('reply', '处理完成'))
        if base.get('order_results') is not None:
            for item in base['order_results']:
                action = next((a for a in base['actions'] if a['order_id'] == item['order_id']), None)
                if action:
                    item.update(stage={'pending': '待确认', 'completed': '已提交申请', 'cancelled': '用户取消', 'rejected': '未办理'}[action['status']], action_id=action['action_id'])
                    item['amount'] = action['amount']
                    if action['status'] == 'completed':
                        item['reason'] = '退款申请已记录，尚不代表到账'
                    elif action['status'] == 'cancelled':
                        item['reason'] = '未提交退款申请'
                    if (action.get('result') or {}).get('reason'):
                        item.update(reason=action['result']['reason'], reason_code=action['result'].get('reason_code'))
            base['reply'] = self._refund_summary(base['order_results'])
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
                if any(t['status'] in {'processing', 'awaiting_confirmation', 'awaiting_selection'} for t in self.snapshot(user_id, session_id)['turns']):
                    raise ConflictError('请先完成当前操作的确认或取消')
                with tools._session() as s, s.begin():
                    s.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                                        input_data=json.dumps(source), kind='chat'))
            base = previous['response'] if previous else None
            config = self._config(user_id, request_id)
            customer_token = active_customer.set(user_id)
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
                    augmented = message
                    if payload['last_order_id'] and not explicit_orders(message) and not any(w in message.lower() for w in ['退款', '退钱', 'refund']):
                        augmented = memory.context_note() + message
                    result = (runner or run_orchestrator)(augmented, history=memory.turns)
                    base = {k: result.get(k) for k in ['reply', 'agent', 'trace', 'usage', 'cached', 'refund_intent']}
                    base.update(request_id=request_id, user_id=user_id, session_id=session_id, status='processing')
                    ids = explicit_orders(message)
                    if not ids:
                        ids = list(dict.fromkeys(e.get('args', {}).get('order_id') for e in base.get('trace') or [] if isinstance(e.get('result'), dict) and not e['result'].get('error')))
                    with tools._session() as session:
                        accessible = [i for i in ids if i and tools.owned_order(session, i, user_id)]
                    if ids and base.get('agent') in {'order', 'refund', 'payment'}:
                        base['recent_order_ids'] = accessible
                    self._save(user_id, request_id, base)
                if base.get('refund_intent'):
                    ids = explicit_orders(message)
                    if not ids and base.get('selection_request_id'):
                        ids = self._load(user_id, base['selection_request_id'])['input']['order_ids']
                    recent = self.snapshot(user_id, session_id)['recent_order_ids']
                    if not ids and len(recent) == 1 and base['refund_intent'] == 'query':
                        ids = recent
                    intent = base['refund_intent']
                    if intent == 'unclear':
                        base.update(status='completed', reply='请说明要查询退款资格，还是申请退款。')
                        self._save(user_id, request_id, base)
                        return base
                    if not ids:
                        with tools._session() as session:
                            orders = session.scalars(select(tools.Order).where(tools.Order.customer_id == user_id).order_by(tools.Order.order_id)).all()
                            candidates = [dict(order_id=o.order_id, item=o.item, amount=o.amount, status=o.status,
                                               **{k: v for k, v in tools.refund_eligibility(session, o.order_id).items() if k in {'eligible', 'reason', 'reason_code'}}) for o in orders]
                        candidates.sort(key=lambda order: (not order.get('eligible', False), order['order_id']))
                        base.update(status='awaiting_selection', reply='请选择要退款的订单。', selection={'candidates': candidates, 'recent_order_ids': recent})
                        self._save(user_id, request_id, base)
                        return base
                    return self._prepare_refunds(user_id, session_id, request_id, base, ids, intent == 'apply')
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
                active_customer.reset(customer_token)
                active_profile.reset(token)
                active_request.reset(request_token)

    def get_request(self, user_id, session_id, request_id):
        self.validate(user_id, session_id, request_id)
        record = self._load(user_id, request_id)
        if not record or record['session_id'] != session_id or 'message' not in record['input']:
            raise ConflictError('请求不存在')
        return record['response'] or {'status': 'processing', 'request_id': request_id}

    def resume(self, user_id, session_id, chat_request_id, request_id, action_id, decision):
        self.validate(user_id, session_id, chat_request_id, request_id, action_id)
        if decision not in {'confirm', 'cancel'}:
            raise ValueError('无效确认决定')
        source = {'chat_request_id': chat_request_id, 'action_id': action_id, 'decision': decision}
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
                    if (previous['chat_request_id'], previous['action_id']) == (chat_request_id, action_id) and previous['decision'] != decision:
                        raise ConflictError('该操作已有其他确认决定')
            base = self.get_request(user_id, session_id, chat_request_id)
            current = base.get('current_action')
            if not current or current['action_id'] != action_id:
                terminal = next((a for a in base.get('actions', []) if a['action_id'] == action_id and a['status'] != 'pending'), None)
                if terminal:
                    return base
                raise ConflictError('操作已变化')
            if not old:
                with tools._session() as s, s.begin():
                    s.add(RequestRecord(user_id=user_id, request_id=request_id, session_id=session_id,
                                        input_data=json.dumps(source), kind='decision'))
            config = self._config(user_id, chat_request_id)
            customer_token = active_customer.set(user_id)
            token = active_profile.set(self.store.profile(user_id))
            request_token = active_request.set(config['configurable']['thread_id'])
            try:
                state = self.graph.get_state(config)
                interruptions = [i for task in state.tasks for i in task.interrupts]
                if interruptions and interruptions[0].value['action_id'] != action_id:
                    # A prior execution advanced the graph before its response was saved.
                    result = {**state.values, '__interrupt__': interruptions}
                else:
                    command = Command(resume={'action_id': action_id, 'decision': decision}) if interruptions else None
                    result = self._invoke_confirmation(command, config)
                response = self._graph_result(user_id, chat_request_id, base, result)
                self._save(user_id, request_id, response)
                return response
            finally:
                active_customer.reset(customer_token)
                active_profile.reset(token)
                active_request.reset(request_token)

    @staticmethod
    def _refund_summary(items):
        return '\n\n'.join(f"订单 {i['order_id']}：{i['stage']}。{i.get('reason', '')}" for i in items)

    def _prepare_refunds(self, user_id, session_id, request_id, base, ids, apply=True):
        pending, results = [], []
        base = {**base, 'trace': list(base.get('trace') or [])}
        token = active_customer.set(user_id)
        try:
            for position, order_id in enumerate(ids):
                item = {'order_id': order_id}
                try:
                    eligibility = tools.check_refund_eligibility(order_id)
                    base['trace'].append({'agent': 'refund', 'tool': 'check_refund_eligibility', 'args': {'order_id': order_id}, 'result': eligibility})
                    item.update(reason=eligibility.get('reason', eligibility.get('error', '')), reason_code=eligibility.get('reason_code'), stage='符合条件' if eligibility.get('eligible') else '不符合条件')
                    if apply and eligibility.get('eligible'):
                        proposal = {'type': 'refund', 'order_id': order_id, 'reason': '客户申请退款'}
                        action = prepare_action(user_id, session_id, request_id, position, proposal)
                        pending.append(action)
                        item.update(stage='待确认', action_id=action['action_id'], amount=action['amount'], reason='确认后提交，尚未申请退款')
                        base['trace'].append({'agent': 'refund', 'tool': 'prepare_refund', 'args': proposal, 'result': {'pending_action': proposal}})
                except Exception as exc:
                    item.update(stage='准备失败', reason=str(exc), reason_code='PREPARATION_FAILED')
                results.append(item)
            base.update(order_results=results, reply=self._refund_summary(results), status='processing', selection=None)
            with tools._session() as session:
                base['recent_order_ids'] = [i for i in ids if tools.owned_order(session, i, user_id)]
            self._save(user_id, request_id, base)
            if not pending:
                base['status'] = 'completed'
                self._save(user_id, request_id, base)
                return base
            result = self._invoke_confirmation({'user_id': user_id, 'session_id': session_id, 'actions': pending, 'index': 0}, self._config(user_id, request_id))
            return self._graph_result(user_id, request_id, base, result)
        finally:
            active_customer.reset(token)

    def select_orders(self, user_id, session_id, chat_request_id, request_id, decision, order_ids):
        self.validate(user_id, session_id, chat_request_id, request_id)
        if decision not in {'submit', 'cancel'} or any(type(i) is not int for i in order_ids):
            raise ValueError('无效选单参数')
        ids = list(dict.fromkeys(order_ids))
        if decision == 'submit' and not ids:
            raise ValueError('请至少选择一笔订单')
        source = {'chat_request_id': chat_request_id, 'decision': decision, 'order_ids': ids}
        with self.lock:
            previous = self._load(user_id, request_id)
            if previous:
                if previous['input'] != source or previous['session_id'] != session_id:
                    raise ConflictError('请求编号已用于其他内容')
                if previous['response'] and previous['response']['status'] != 'processing':
                    return previous['response']
            base = self.get_request(user_id, session_id, chat_request_id)
            if base['status'] != 'awaiting_selection':
                # Recover a response lost after preparation completed.
                if previous and base.get('selection_request_id') == request_id:
                    if base['status'] == 'processing':
                        config = self._config(user_id, chat_request_id)
                        if self.graph.get_state(config).values:
                            base = self._graph_result(user_id, chat_request_id, base, self._invoke_confirmation(None, config))
                        else:
                            base = self._prepare_refunds(user_id, session_id, chat_request_id, base, ids, base['refund_intent'] == 'apply')
                    self._save(user_id, request_id, base)
                    return base
                raise ConflictError('当前请求不在选单阶段')
            candidates = {o['order_id'] for o in base['selection']['candidates']}
            if not set(ids) <= candidates:
                raise ConflictError('所选订单不在候选集合中')
            if not previous:
                with tools._session() as session, session.begin():
                    session.add(RequestRecord(user_id=user_id, session_id=session_id, request_id=request_id, kind='selection', input_data=json.dumps(source)))
            base['selection_request_id'] = request_id
            if decision == 'cancel':
                base.update(status='completed', selection=None, reply='已取消选单，未申请退款。')
                self._save(user_id, chat_request_id, base)
                result = base
            else:
                result = self._prepare_refunds(user_id, session_id, chat_request_id, base, ids, base['refund_intent'] == 'apply')
            self._save(user_id, request_id, result)
            return result
