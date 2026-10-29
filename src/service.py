import os
import re
import threading
from pathlib import Path
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage

from src import tools
from src import actions
from src.auth import AccessError, principal_scope, require_principal
from src.errors import classify_error, error_message
from src.memory import empty_issue
from src.context import ContextBudgetExceeded, build_context
from src.budget import ExecutionBudgetExceeded, RequestBudget, check_execution_budget
from src.planning import summarize_task_results, iter_order_results
from src.understanding import PIPELINE_VERSION
from src.orchestrator import run_orchestrator
from src.results import compose_response, execution_status, fallback_result
from src.runtime_context import STORAGE_MODE, active_profile, active_request, active_run, active_order_reference, active_budget, record_event
from src.storage import SessionConflict, SupportStore


def _context_order_ids(turns, issue, pending=None):
    ids = [*issue.get('candidate_order_ids', []), *(pending or {}).get('candidate_order_ids', [])]
    for turn in turns:
        ids.extend(turn.get('context_order_ids', []))
        ids.extend(number for entry in turn.get('trace', []) for number, fact in iter_order_results(entry.get('result'))
                   if not fact.get('error'))
    return list(dict.fromkeys(ids))


class SupportService:
    def __init__(self, path=None):
        path = path or Path(os.environ.get('SUPPORT_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data'))) / 'memory.sqlite'
        self.store = SupportStore(path)
        self.lock = threading.RLock()

    def validate(self, user_id, session_id):
        for value in [user_id, session_id]:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
                raise ValueError('编号只允许 1 至 40 位字母、数字、下划线、短横线')

    def snapshot(self, principal, session_id, *, for_planning=False):
        principal = require_principal(principal)
        self.validate(principal.customer_id, session_id)
        with principal_scope(principal):
            payload = self.store.load_session(principal.customer_id, STORAGE_MODE, session_id)
            visible, legacy_context = [], set()
            for turn in payload['turns']:
                # Older turns did not record which historical facts the model saw.
                legacy_context.update(number for entry in turn.get('trace', [])
                                      for number, fact in iter_order_results(entry.get('result')) if not fact.get('error'))
                dependencies = turn.get('context_order_ids', legacy_context)
                if self._visible_result(turn) and all(tools.can_access_order(number) for number in dependencies):
                    # UI cards reflect today's action status; dialogue keeps the original reply.
                    visible.append(turn if for_planning else self._present_actions(principal, turn))
            payload['turns'] = visible
            if payload['last_order_id'] and not tools.can_access_order(payload['last_order_id']):
                payload['last_order_id'] = None
            issue = payload['issue']
            issue['candidate_order_ids'] = [number for number in issue['candidate_order_ids'] if tools.can_access_order(number)]
            if issue['active_order_id'] not in issue['candidate_order_ids']:
                issue['active_order_id'] = None
            for group in issue['order_groups']:
                # Keep positions stable even if an order is no longer accessible.
                group['unavailable_order_ids'] = [n for n in group['order_ids'] if not tools.can_access_order(n)]
            pending = payload.get('pending_clarification')
            if pending and any(not tools.can_access_order(number) for number in pending.get('candidate_order_ids', [])):
                payload['pending_clarification'] = None
            return payload

    def profile(self, principal):
        return self.store.profile(require_principal(principal).customer_id)

    def save_profile(self, principal, tone, contact):
        return self.store.save_profile(require_principal(principal).customer_id, tone, contact)

    def profile_details(self, principal):
        return self.store.profile_details(require_principal(principal).customer_id)

    def update_profile(self, principal, fields, expected_version, *, source='user_form'):
        return self.store.update_profile(require_principal(principal).customer_id, fields, expected_version, source=source)

    def delete_profile_field(self, principal, field, expected_version, *, source='user_form'):
        return self.store.delete_profile_field(require_principal(principal).customer_id, field, expected_version, source=source)

    def set_active_order(self, principal, session_id, order_id, expected_version):
        principal = require_principal(principal)
        self.validate(principal.customer_id, session_id)
        with principal_scope(principal):
            if order_id is not None and not tools.can_access_order(order_id):
                raise AccessError('未找到可访问的订单。', 404)
            self.store.set_active_order(principal.customer_id, STORAGE_MODE, session_id, order_id, expected_version)
        return self.snapshot(principal, session_id)

    def clear_history(self, principal, session_id, expected_version):
        self.validate(require_principal(principal).customer_id, session_id)
        self.store.clear_history(principal.customer_id, STORAGE_MODE, session_id, expected_version)
        return self.snapshot(principal, session_id)

    def _visible_trace(self, trace):
        for entry in trace:
            result = entry.get('result') or {}
            if isinstance(result, dict) and result.get('error'):
                continue
            ids = [number for number, fact in iter_order_results(result) if not fact.get('error')]
            if not ids and (entry.get('args') or {}).get('order_id'):
                ids = [entry['args']['order_id']]
            if any(not tools.can_access_order(number) for number in ids):
                return False
        return True

    def _visible_result(self, result):
        reference = result.get('order_reference') or {}
        pending = result.get('pending_clarification') or {}
        ids = [*result.get('context_order_ids', []),
               *reference.get('issue', {}).get('candidate_order_ids', []),
               *pending.get('candidate_order_ids', [])]
        return self._visible_trace(result.get('trace', [])) and all(tools.can_access_order(number) for number in ids)

    def _present_actions(self, principal, result):
        result = {key: value for key, value in result.items() if key != 'mode'}
        context = active_request.get()
        if context and 'context_order_ids' not in result:
            # Completed older requests can still depend on historical tool facts.
            run = self.store.find_run(context)
            if run:
                saved = run['input']
                result = {**result, 'context_order_ids': _context_order_ids(
                    saved['turns'], saved.get('issue_at_start', {}), saved.get('pending_clarification'))}
        reference = result.get('order_reference')
        if not self._visible_result(result):
            return {**result, 'reply': '未找到可访问的订单。', 'trace': [], 'usage': {},
                    'status': 'failed', 'reply_source': 'fallback', 'actions': [], 'pending_actions': [], 'tasks': [], 'sources': [],
                    'batch_results': [], 'understanding': {}, 'pending_clarification': None,
                    'model_reply': '', 'clarification_question': None, 'context_order_ids': [], 'context_info': {},
                    'order_reference': {'status': 'invalid', 'order_id': None,
                                        'issue': {**empty_issue(), 'revision': reference['issue']['revision'] if reference else 0}}}
        views = []
        for entry in result.get('trace', []):
            action_id = (entry.get('result') or {}).get('action_id') if isinstance(entry.get('result'), dict) else None
            if action_id and action_id not in {view['action_id'] for view in views}:
                views.append(actions.get_action(principal, action_id))
        pending = [view for view in views if view['status'] == 'pending']
        result = {**result, 'actions': views, 'pending_actions': pending}
        result.setdefault('tasks', [])
        result.setdefault('stop_reason', 'completed')
        if len(result['tasks']) > 1 and any(view['status'] != 'pending' for view in views):
            refreshed = []
            for task in result['tasks']:
                ids = {entry['result'].get('action_id') for entry in task.get('trace', []) if isinstance(entry.get('result'), dict)}
                terminal = [view for view in views if view['action_id'] in ids and view['status'] != 'pending']
                if terminal:
                    current = fallback_result([{'agent': 'refund' if view['tool'] == 'initiate_refund' else 'payment',
                        'tool': view['tool'], 'args': view['args'], 'result': view['result']} for view in terminal])
                    task = {**task, 'reply': current['reply'], 'status': current['status']}
                refreshed.append(task)
            result.update(tasks=refreshed, **summarize_task_results(refreshed))
        if pending:
            notice = '\n\n'.join(
                f"订单 {view['order_id']} 的{'退款申请' if view['tool'] == 'initiate_refund' else '付款'}待确认，金额 {view['amount_display']} {view['currency']}。请检查后点击确认；当前尚未办理。"
                for view in pending)
            partial = result['stop_reason'] == 'budget_exhausted' or len(result['tasks']) > 1 and result.get('status') == 'partial'
            reply = result['reply'] + '\n\n待确认操作：\n' + notice if len(result['tasks']) > 1 else notice
            result.update(status='partial' if partial else 'pending_confirmation', reply_source='fallback', reply=reply)
            if result.get('error_type') == 'order_reference_changed':
                result['reply'] = '当前事项已更正，原运行已停止继续执行。以下仅为中断前已保存的动作，请单独核对或取消：\n\n' + result['reply']
        elif views and result.get('status') == 'pending_confirmation':
            trace = [{'agent': 'refund' if view['tool'] == 'initiate_refund' else 'payment',
                      'tool': view['tool'], 'args': view['args'], 'result': view['result']} for view in views]
            current = fallback_result(trace)
            result.update(reply=current['reply'], status=current['status'], reply_source='fallback')
        evidence_rows = [entry['result'] for entry in result.get('trace', []) if entry['tool'] == 'answer_faq'
                         and isinstance(entry.get('result'), dict) and 'answer' in entry['result']]
        if result.get('agent') == 'faq' or evidence_rows:
            result.update(sources=[], policy_found=False, catalog_revision=None, policy_cached=False)
            sources = {}
            for evidence in evidence_rows:
                for source in evidence['sources']:
                    sources[(source['policy_id'], source['version'], source['chunk_index'])] = source
                result.update(policy_found=result['policy_found'] or evidence['found'], catalog_revision=evidence['catalog_revision'])
            result.update(sources=list(sources.values()), policy_cached=bool(evidence_rows) and all(row['cached'] for row in evidence_rows))
        return compose_response(result)

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

    def chat(self, principal, session_id, message, runner=None, request_id=None):
        principal = require_principal(principal)
        if request_id and request_id.startswith(('confirm_', 'cancel_')):
            raise ValueError('该请求编号前缀保留给动作确认与取消。')
        return self._request(principal, session_id, message, runner, request_id)

    def _request(self, principal, session_id, message, runner=None, request_id=None, *, resuming=False):
        with principal_scope(principal):
            return self._chat(principal, session_id, message, runner, request_id, resuming=resuming)

    def _run_request(self, principal, run_id):
        run = self.store.get_run(require_principal(principal).customer_id, run_id)
        if run['mode'] != STORAGE_MODE:
            raise AccessError('未找到可访问的运行。', 404)
        request = tools.request_journal().request_input(run['user_id'], STORAGE_MODE, run['request_id'])
        if request['session_id'] != run['session_id']:
            raise AccessError('运行身份或会话不匹配。', 403)
        return run, request

    def get_run(self, principal, run_id):
        run, request = self._run_request(principal, run_id)
        return {**{key: run[key] for key in ('run_id', 'request_id', 'session_id')},
                'status': request['result']['status'] if request['result'] else 'unfinished',
                'memory_saved': self.store.has_turn(run['user_id'], run['mode'], run['session_id'], run['request_id']),
                'resumable': request['result'] is None}

    def list_runs(self, principal, session_id):
        self.validate(require_principal(principal).customer_id, session_id)
        runs = [self.get_run(principal, run['run_id']) for run in
                self.store.list_runs(principal.customer_id, STORAGE_MODE, session_id)]
        return [run for run in runs if run['resumable'] or not run['memory_saved']]

    def resume_run(self, principal, run_id):
        run, request = self._run_request(principal, run_id)
        return self._request(principal, request['requested_session'], request['message'],
                             request_id=run['request_id'], resuming=True)

    def get_action(self, principal, action_id):
        return actions.get_action(principal, action_id)

    def list_pending_actions(self, principal, session_id):
        self.validate(require_principal(principal).customer_id, session_id)
        return actions.list_actions(principal, session_id=session_id)

    def confirm_action(self, principal, action_id):
        return self._decide(principal, action_id, confirm=True)

    def cancel_action(self, principal, action_id):
        return self._decide(principal, action_id, confirm=False)

    def _decide(self, principal, action_id, *, confirm):
        view = actions.get_action(principal, action_id)
        def runner(*_args, **_kwargs):
            if not tools.request_journal().action_trace(active_request.get()):
                check_execution_budget('tool', consume=True)
            trace = [actions.decide_action(action_id, confirm=confirm)]
            return fallback_result(trace)
        operation = 'confirm' if confirm else 'cancel'
        return self._request(principal, view['session_id'],
                             ('确认动作 ' if confirm else '取消动作 ') + action_id,
                             runner, operation + '_' + action_id)

    def _chat(self, principal, session_id, message, runner=None, request_id=None, *, resuming=False):
        user_id = principal.customer_id
        self.validate(user_id, session_id if session_id is not None else 'new')
        if not message.strip() or len(message) > 2000:
            raise ValueError('消息需为 1 至 2000 个字符')
        request_id = request_id if request_id is not None else uuid4().hex
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request_id):
            raise ValueError('请求编号只允许 1 至 64 位字母、数字、下划线、短横线')
        journal = tools.request_journal()
        with journal.request_guard(user_id, STORAGE_MODE, request_id) as acquired:
            context, result, created = journal.begin_request(
                user_id, STORAGE_MODE, request_id, session_id, message, create=acquired)
            if not acquired:
                return {'request_id': request_id, 'session_id': None if created else context.session_id,
                        'user_id': user_id, 'status': 'in_progress',
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
                              'user_id': user_id, 'replayed': not created}
                    journal.finish_request(context, result)
                    return self._save_turn(context, message, result)
            finally:
                active_request.reset(token)

    def _run(self, context, message, runner, journal, *, resume=False):
        budget = RequestBudget(context, journal.sessions)
        payload = self.snapshot(require_principal(), context.session_id, for_planning=True)
        profile = self.store.profile(context.user_id)
        run = self.store.find_run(context) if resume else None
        if run is not None:
            saved = run['input']
            if not self._visible_result({'trace': [], 'context_order_ids': saved.get('context_order_ids', _context_order_ids(
                                         saved['turns'], saved.get('issue_at_start', {}), saved.get('pending_clarification'))),
                                         'order_reference': saved.get('order_reference'),
                                         'pending_clarification': saved.get('pending_clarification')}) or any(
                    not self._visible_result(turn) for turn in saved['turns']) or (
                    saved['last_order_id'] and not tools.can_access_order(saved['last_order_id'])):
                raise AccessError('运行涉及的订单权限已变化，不能继续恢复。', 409)
            if payload['issue']['revision'] != saved.get('issue_at_start', {}).get('revision', 0):
                trace = journal.action_trace(context)
                result = fallback_result(trace, interrupted=True) if trace else {
                    'reply': '当前事项已更正，原运行已停止继续执行。请根据当前订单重新发起请求，尚未生成新的写入提案。',
                    'status': 'needs_clarification', 'agent': 'system', 'trace': [], 'usage': {},
                    'cached': False, 'reply_source': 'fallback'}
                return {**result, 'error_type': 'order_reference_changed', 'session_version': payload['version'],
                        'run_id': run['run_id'], 'resumable': False, 'stop_reason': 'order_reference_changed',
                        'execution_budget': budget.snapshot()}
            turns, augmented = saved['turns'], saved['augmented']
            reference = saved.get('order_reference')
            info = saved.get('context_info', {})
            version = run['session_version']
            history = [item for turn in turns for item in
                       (HumanMessage(content=turn['user']), AIMessage(content=turn['reply']))]
        else:
            version = payload['version']
            decision = context.request_id.startswith(('confirm_', 'cancel_'))
            reference = None if decision else {'status': 'context', 'order_id': None, 'issue': payload['issue']}
            augmented = message
            turns, history, info = build_context(payload['turns'], augmented,
                                                 reference['issue'] if reference else payload['issue'], profile,
                                                 pending=payload.get('pending_clarification'))
            context_ids = _context_order_ids(turns, payload['issue'], payload.get('pending_clarification'))
            if runner is run_orchestrator:
                run = self.store.create_run(context, version, {
                    'turns': turns, 'augmented': augmented, 'profile': profile,
                    'last_order_id': payload['last_order_id'], 'issue_at_start': payload['issue'],
                    'order_reference': reference, 'context_info': info, 'pending_clarification': payload.get('pending_clarification'),
                    'context_order_ids': context_ids,
                    'pipeline_version': PIPELINE_VERSION})
        budget_token = active_budget.set(budget)
        profile_token = active_profile.set(profile)
        run_token = active_run.set(run)
        reference_token = active_order_reference.set(reference)
        try:
            if run:
                record_event('run_resumed' if resume else 'run_started', run_id=run['run_id'], session_version=version)
            try:
                result = runner(augmented, history=history)
                result = {k: v for k, v in result.items() if k in {
                    'reply', 'agent', 'trace', 'usage', 'cached', 'status', 'reply_source', 'error_type', 'context_info',
                    'tasks', 'stop_reason', 'clarification_question', 'planning_usage', 'pipeline_version',
                    'understanding', 'order_reference', 'pending_clarification', 'batch_results', 'model_diagnostics'}}
                result.setdefault('status', execution_status(result.get('trace', [])))
                result.setdefault('reply_source', 'model')
                result['session_version'] = version
                result['context_info'] = {**info, **result.get('context_info', {})}
                if reference:
                    result.setdefault('order_reference', reference)
                if run:
                    result.update(run_id=run['run_id'], resumable=False)
                result['execution_budget'] = budget.snapshot()
                result['context_order_ids'] = run['input'].get('context_order_ids', []) if run else context_ids
                return result
            except AccessError:
                raise
            except Exception as exc:
                kind = 'context_budget' if isinstance(exc, ContextBudgetExceeded) else classify_error(exc)
                record_event('execution_failed', error_type=kind, exception_type=type(exc).__name__)
                trace = journal.action_trace(context)
                if trace:
                    result = fallback_result(trace, interrupted=True)
                else:
                    result = {'reply': str(exc) if kind == 'context_budget' else error_message(kind), 'agent': 'system', 'trace': [],
                              'usage': getattr(exc, 'model_usage', {}), 'cached': False, 'status': 'partial' if kind == 'budget' else 'failed', 'reply_source': 'fallback'}
                result['error_type'] = kind
                result['model_diagnostics'] = getattr(exc, 'model_diagnostics', [])
                result['context_info'] = {**info, **getattr(exc, 'context_info', {})}
                result['pipeline_version'] = PIPELINE_VERSION
                result['context_order_ids'] = run['input'].get('context_order_ids', []) if run else context_ids
                result['session_version'] = version
                result['execution_budget'] = budget.snapshot()
                result['stop_reason'] = 'budget_exhausted' if kind == 'budget' else 'model_error'
                if kind == 'budget':
                    result['tasks'] = [{'task_id': 'task_' + str(index), 'message': clause, 'status': 'not_started',
                                        'reply': '请求预算不足，未开始执行。', 'trace': [], 'stop_reason': 'budget_exhausted'}
                                       for index, clause in enumerate([message], 1)]
                if reference:
                    result.setdefault('order_reference', reference)
                if run:
                    result.update(run_id=run['run_id'], resumable=False)
                return result
        finally:
            active_order_reference.reset(reference_token)
            active_run.reset(run_token)
            active_profile.reset(profile_token)
            active_budget.reset(budget_token)
