import os
import re
import threading
from pathlib import Path
from uuid import uuid4

from src import tools
from src import actions
from src.auth import AccessError, principal_scope, require_principal
from src.errors import classify_error, error_message
from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator
from src.results import execution_status, fallback_result
from src.runtime_context import active_mode, active_profile, active_request, active_run, record_event
from src.storage import SessionConflict, SupportStore


class SupportService:
    def __init__(self, path=None):
        path = path or Path(os.environ.get('SUPPORT_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data'))) / 'memory.sqlite'
        self.store = SupportStore(path)
        self.lock = threading.RLock()

    def validate(self, user_id, session_id, mode):
        if mode not in {'demo', 'online'}:
            raise ValueError('运行模式必须为 demo 或 online')
        for value in [user_id, session_id]:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
                raise ValueError('编号只允许 1 至 40 位字母、数字、下划线、短横线')

    def snapshot(self, principal, session_id, mode):
        principal = require_principal(principal)
        self.validate(principal.customer_id, session_id, mode)
        with principal_scope(principal):
            payload = self.store.load_session(principal.customer_id, mode, session_id)
            payload['turns'] = [turn for turn in payload['turns'] if self._visible_trace(turn.get('trace', []))]
            if payload['last_order_id'] and not tools.can_access_order(payload['last_order_id']):
                payload['last_order_id'] = None
            return payload

    def profile(self, principal):
        return self.store.profile(require_principal(principal).customer_id)

    def save_profile(self, principal, tone, contact):
        self.store.save_profile(require_principal(principal).customer_id, tone, contact)

    def _visible_trace(self, trace):
        for entry in trace:
            result = entry.get('result') or {}
            order_id = (entry.get('args') or {}).get('order_id')
            if isinstance(result, dict):
                order_id = order_id or result.get('order_id')
                if result.get('error'):
                    continue
            if order_id is not None and not tools.can_access_order(order_id):
                return False
        return True

    def _present_actions(self, principal, result):
        if not self._visible_trace(result.get('trace', [])):
            return {**result, 'reply': '未找到可访问的订单。', 'trace': [], 'usage': {},
                    'status': 'failed', 'reply_source': 'fallback', 'actions': [], 'pending_actions': []}
        views = []
        for entry in result.get('trace', []):
            action_id = (entry.get('result') or {}).get('action_id') if isinstance(entry.get('result'), dict) else None
            if action_id and action_id not in {view['action_id'] for view in views}:
                views.append(actions.get_action(principal, action_id))
        pending = [view for view in views if view['status'] == 'pending']
        result = {**result, 'actions': views, 'pending_actions': pending}
        if pending:
            result.update(status='pending_confirmation', reply_source='fallback', reply='\n\n'.join(
                f"订单 {view['order_id']} 的{'退款申请' if view['tool'] == 'initiate_refund' else '付款'}待确认，本地样例金额 {view['amount']:.2f}。请检查后点击确认；当前尚未办理。"
                for view in pending))
        elif views and result.get('status') == 'pending_confirmation':
            trace = [{'agent': 'refund' if view['tool'] == 'initiate_refund' else 'payment',
                      'tool': view['tool'], 'args': view['args'], 'result': view['result']} for view in views]
            current = fallback_result(trace)
            result.update(reply=current['reply'], status=current['status'], reply_source='fallback')
        return result

    def _save_turn(self, context, message, result, *, repair=False):
        try:
            expected = result.get('session_version', 0)
            if repair:
                expected = self.store.load_session(context.user_id, context.mode, context.session_id)['version']
            self.store.append_turn_once(context.user_id, context.mode, context.session_id,
                                        context.request_id, message, result, expected_version=expected,
                                        preserve_context=repair and expected != result.get('session_version', 0))
            result['memory_saved'] = True
            result['memory_status'] = 'saved'
            result['memory_rebased'] = repair and expected != result.get('session_version', 0)
        except SessionConflict as exc:
            result.update(memory_saved=False, memory_status='conflict', current_session_version=exc.current,
                          memory_message=str(exc))
            record_event('session_conflict', expected_version=exc.expected, current_version=exc.current)
        except Exception:
            # The completed business response is already durable. Same-ID replay
            # repairs memory; a failed memory write must not rerun the graph.
            result['memory_saved'] = False
            result['memory_status'] = 'failed'
            record_event('memory_save_failed')
        return result

    def chat(self, principal, session_id, mode, message, runner=None, request_id=None):
        principal = require_principal(principal)
        if request_id and request_id.startswith(('confirm_', 'cancel_')):
            raise ValueError('该请求编号前缀保留给动作确认与取消。')
        return self._request(principal, session_id, mode, message, runner, request_id)

    def _request(self, principal, session_id, mode, message, runner=None, request_id=None, *, resuming=False):
        with principal_scope(principal):
            return self._chat(principal, session_id, mode, message, runner, request_id, resuming=resuming)

    def _run_request(self, principal, run_id):
        run = self.store.get_run(require_principal(principal).customer_id, run_id)
        request = tools.request_journal().request_input(run['user_id'], run['mode'], run['request_id'])
        if request['session_id'] != run['session_id']:
            raise AccessError('运行身份或会话不匹配。', 403)
        return run, request

    def get_run(self, principal, run_id):
        run, request = self._run_request(principal, run_id)
        return {**{key: run[key] for key in ('run_id', 'request_id', 'session_id', 'mode')},
                'status': request['result']['status'] if request['result'] else 'unfinished',
                'memory_saved': self.store.has_turn(run['user_id'], run['mode'], run['session_id'], run['request_id']),
                'resumable': request['result'] is None}

    def list_runs(self, principal, session_id, mode):
        self.validate(require_principal(principal).customer_id, session_id, mode)
        runs = [self.get_run(principal, run['run_id']) for run in
                self.store.list_runs(principal.customer_id, mode, session_id)]
        return [run for run in runs if run['resumable'] or not run['memory_saved']]

    def resume_run(self, principal, run_id):
        run, request = self._run_request(principal, run_id)
        return self._request(principal, request['requested_session'], run['mode'], request['message'],
                             request_id=run['request_id'], resuming=True)

    def get_action(self, principal, action_id):
        return actions.get_action(principal, action_id)

    def list_pending_actions(self, principal, session_id, mode):
        self.validate(require_principal(principal).customer_id, session_id, mode)
        return actions.list_actions(principal, session_id=session_id, mode=mode)

    def confirm_action(self, principal, action_id):
        return self._decide(principal, action_id, confirm=True)

    def cancel_action(self, principal, action_id):
        return self._decide(principal, action_id, confirm=False)

    def _decide(self, principal, action_id, *, confirm):
        view = actions.get_action(principal, action_id)
        def runner(*_args, **_kwargs):
            trace = [actions.decide_action(action_id, confirm=confirm)]
            return fallback_result(trace)
        operation = 'confirm' if confirm else 'cancel'
        return self._request(principal, view['session_id'], view['mode'],
                             ('确认动作 ' if confirm else '取消动作 ') + action_id,
                             runner, operation + '_' + action_id)

    def _chat(self, principal, session_id, mode, message, runner=None, request_id=None, *, resuming=False):
        user_id = principal.customer_id
        self.validate(user_id, session_id if session_id is not None else 'new', mode)
        if not message.strip() or len(message) > 2000:
            raise ValueError('消息需为 1 至 2000 个字符')
        request_id = request_id if request_id is not None else uuid4().hex
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request_id):
            raise ValueError('请求编号只允许 1 至 64 位字母、数字、下划线、短横线')
        journal = tools.request_journal()
        with journal.request_guard(user_id, mode, request_id) as acquired:
            context, result, created = journal.begin_request(
                user_id, mode, request_id, session_id, message, create=acquired)
            if not acquired:
                return {'request_id': request_id, 'session_id': None if created else context.session_id,
                        'user_id': user_id, 'mode': mode, 'status': 'in_progress',
                        'reply': '同一请求正在处理中，请使用相同请求编号稍后查询。',
                        'agent': None, 'trace': [], 'usage': {}, 'cached': False,
                        'reply_source': 'fallback', 'replayed': False, 'memory_saved': False}
            token = active_request.set(context)
            try:
                if result is not None:
                    record_event('request_replayed')
                    return self._save_turn(context, message, self._present_actions(principal, {**result, 'replayed': True}), repair=True)
                with self.lock:
                    if not created:
                        run = self.store.find_run(context)
                        if run is not None:
                            if not resuming:
                                return {**self.get_run(principal, run['run_id']), 'user_id': user_id,
                                        'status': 'interrupted',
                                        'reply': '请求曾中断，请点击恢复执行，从已保存步骤继续；已提交动作不会重复办理。',
                                        'agent': 'system', 'trace': [], 'usage': {}, 'cached': False,
                                        'reply_source': 'fallback', 'replayed': True, 'memory_saved': False}
                            result = self._run(context, message, run_orchestrator, journal, resume=True)
                        elif request_id.startswith(('confirm_', 'cancel_')):
                            # Only trusted decision entrypoints can use these
                            # IDs. Their runner is a single idempotent transaction,
                            # not a model graph; receipts make recovery safe.
                            result = self._run(context, message, runner, journal)
                        else:
                            trace = journal.action_trace(context)
                            result = fallback_result(trace, interrupted=True) if trace else {
                                'reply': '请求曾中断，没有可回放的已提交动作；未自动重跑，请核对后使用新请求编号重试。',
                                'agent': 'system', 'trace': [], 'usage': {}, 'cached': False,
                                'status': 'failed', 'reply_source': 'fallback'}
                    else:
                        record_event('request_started')
                        result = self._run(context, message, runner or run_orchestrator, journal)
                    result = self._present_actions(principal, result)
                    result = {**result, 'request_id': request_id, 'session_id': context.session_id,
                              'user_id': user_id, 'mode': mode, 'replayed': not created}
                    journal.finish_request(context, result)
                    return self._save_turn(context, message, result)
            finally:
                active_request.reset(token)

    def _run(self, context, message, runner, journal, *, resume=False):
        payload = self.snapshot(require_principal(), context.session_id, context.mode)
        run = self.store.find_run(context) if resume else None
        if run is not None:
            saved = run['input']
            if any(not self._visible_trace(turn.get('trace', [])) for turn in saved['turns']) or (
                    saved['last_order_id'] and not tools.can_access_order(saved['last_order_id'])):
                raise AccessError('运行涉及的订单权限已变化，不能继续恢复。', 409)
            turns, augmented, profile = saved['turns'], saved['augmented'], saved['profile']
            version = run['session_version']
        else:
            turns = payload['turns'][-6:]
            memory = ConversationMemory(last_order_id=payload['last_order_id'])
            explicit = re.findall(r'\b\d{4,8}\b', message)
            augmented = message if explicit else memory.context_note() + message
            profile = self.store.profile(context.user_id)
            version = payload['version']
            if runner is run_orchestrator:
                run = self.store.create_run(context, version, {
                    'turns': turns, 'augmented': augmented, 'profile': profile,
                    'last_order_id': payload['last_order_id']})
        memory = ConversationMemory(last_order_id=payload['last_order_id'])
        for turn in turns:
            memory.add_turn(turn['user'], turn['reply'])
        mode_token = active_mode.set(context.mode)
        profile_token = active_profile.set(profile)
        run_token = active_run.set(run)
        try:
            if run:
                record_event('run_resumed' if resume else 'run_started', run_id=run['run_id'], session_version=version)
            try:
                result = runner(augmented, history=memory.turns)
                result = {k: v for k, v in result.items() if k in {
                    'reply', 'agent', 'trace', 'usage', 'cached', 'status', 'reply_source', 'error_type'}}
                result.setdefault('status', execution_status(result.get('trace', [])))
                result.setdefault('reply_source', 'demo' if context.mode == 'demo' else 'model')
                result['session_version'] = version
                if run:
                    result.update(run_id=run['run_id'], resumable=False)
                return result
            except AccessError:
                raise
            except Exception as exc:
                kind = classify_error(exc)
                record_event('execution_failed', error_type=kind, exception_type=type(exc).__name__)
                trace = journal.action_trace(context)
                if trace:
                    result = fallback_result(trace, interrupted=True)
                else:
                    result = {'reply': error_message(kind), 'agent': 'system', 'trace': [],
                              'usage': {}, 'cached': False, 'status': 'failed', 'reply_source': 'fallback'}
                    if kind in {'network', 'quota', 'program'}:
                        ticket = tools.escalate_to_human('Agent pipeline failed', kind)
                        result['reply'] += '\n\n' + ticket['message']
                        result['agent'] = 'escalate'
                        result['trace'] = journal.action_trace(context)
                result['error_type'] = kind
                result['session_version'] = version
                if run:
                    result.update(run_id=run['run_id'], resumable=False)
                return result
        finally:
            active_run.reset(run_token)
            active_profile.reset(profile_token)
            active_mode.reset(mode_token)
