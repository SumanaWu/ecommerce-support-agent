import logging
import sqlite3
from contextlib import closing, contextmanager
from copy import deepcopy
from typing import TypedDict
import json
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents import base, faq_agent, order_agent, payment_agent, refund_agent
from src.auth import AccessError, require_principal
from src.errors import classify_error, error_message
from src.logging_config import configure_logging
from src.tools import escalate_to_human
from src.runtime_context import active_request, active_run, active_order_reference, active_task, active_profile
from src.budget import ExecutionBudgetExceeded
from src.planning import summarize_task_results, compile_task_plan, check_task_conditions, iter_order_results
from src.understanding import PIPELINE_VERSION, OPERATIONS, understand_request
from src.memory import empty_issue, remember_order_groups
from src.context import planning_context, task_context, fit_structured_context, context_budget

configure_logging()
logger = logging.getLogger("support_agent.orchestrator")

class SupportState(TypedDict, total=False):
    user_message: str
    history: list
    route: str
    reply: str
    trace: list
    agent: str
    usage: dict
    cached: bool
    status: str
    reply_source: str
    error_type: str
    binding: list
    messages: list
    tool_calls: list
    tool_index: int
    rounds: int
    max_rounds: int
    seen_calls: list
    done: bool
    model_reply: str
    correction_phases: list
    needs_tool_correction: bool
    understanding: dict
    pipeline_version: str
    order_reference: dict
    pending_clarification: dict | None
    batch_results: list
    condition_result: str | None
    context_info: dict
    stop_reason: str | None
    clarification_question: str | None
    task_plan: dict
    task_index: int
    task_results: list
    completed_trace: list
    completed_usage: dict
    tasks: list
    planning_usage: dict
    model_diagnostics: list
    completed_diagnostics: list
    conversation_context: dict
    exposed_tools: list | None


def route_node(state: SupportState) -> dict:
    issue = (active_order_reference.get() or {}).get('issue', empty_issue())
    pending = (active_run.get() or {}).get('input', {}).get('pending_clarification')
    if pending and 'orders' in pending.get('missing_fields', []) and len(pending.get('candidate_order_ids', [])) == 1:
        # Keep the stored source for access checks, but do not resolve a new query to a suggested old order.
        pending = {**pending, 'candidate_order_ids': []}
    context_ids = set(issue['candidate_order_ids'])
    for turn in (active_run.get() or {}).get('input', {}).get('turns', []):
        for entry in turn.get('trace', []):
            context_ids.update(number for number, result in iter_order_results(entry.get('result')) if not result.get('error'))
    if pending:
        context_ids.update(pending.get('candidate_order_ids', []))
    packet = planning_context((active_run.get() or {}).get('input', {}).get('turns', []))
    parsed = understand_request(state['user_message'], **packet, issue=issue,
                               pending=pending, profile=active_profile.get(), context_order_ids=context_ids)
    plan = compile_task_plan(parsed['understanding'], state['user_message'], usage=parsed['usage'], context_order_ids=context_ids).to_dict()
    display = {**parsed['understanding'], 'tasks': [{**task, 'expert': OPERATIONS[task['operation']][0],
               'intent': OPERATIONS[task['operation']][1]} for task in parsed['understanding']['tasks']]}
    return {'route': plan['tasks'][0]['route'], 'usage': {}, 'task_plan': plan,
            'task_index': 0, 'task_results': [], 'completed_trace': [], 'completed_usage': plan['usage'],
            'understanding': display, 'pipeline_version': PIPELINE_VERSION, 'conversation_context': packet,
            'completed_diagnostics': parsed.get('model_diagnostics', []),
            'context_info': parsed.get('context_info', {})}


SPECIALISTS = {
    'ORDER': ('order', order_agent.SYSTEM_PROMPT, order_agent.ORDER_TOOLS),
    'REFUND': ('refund', refund_agent.SYSTEM_PROMPT, refund_agent.REFUND_TOOLS),
    'PAYMENT': ('payment', payment_agent.SYSTEM_PROMPT, payment_agent.PAYMENT_TOOLS),
    'FAQ': ('faq', faq_agent.SYSTEM_PROMPT, faq_agent.FAQ_TOOLS),
}


def prepare_node(state):
    name, prompt, _ = SPECIALISTS[state['route']]
    with task_scope(state) as task:
        prepared = base.prepare_agent(name, prompt, task['message'] if task else state['user_message'])
    return {**prepared, 'reply': None, 'status': None, 'error_type': None, 'stop_reason': None,
            'model_reply': '', 'tool_calls': [], 'clarification_question': None,
            'usage': {} if state.get('task_plan') else state.get('usage', {})}


@contextmanager
def task_scope(state):
    tasks = state.get('task_plan', {}).get('tasks', [])
    task = deepcopy(tasks[state['task_index']]) if tasks else None
    if task is None or task.get('pipeline_version') != PIPELINE_VERSION:
        raise ValueError('旧在线运行缺少兼容计划，不能继续执行。')
    if task.get('pipeline_version') == PIPELINE_VERSION:
        task['prior_facts'] = [entry for entry in state.get('completed_trace', [])
                              if any(number in task['order_ids'] for number, _ in iter_order_results(entry.get('result')))]
        task['conversation_context'] = task_context(state.get('conversation_context', {'recent_turns': [], 'recent_facts': []}),
                                                  task['order_ids'], intent=task.get('intent', 'read'))
        fitted, _ = fit_structured_context([HumanMessage(content=json.dumps({
            **task['conversation_context'], 'user_input': task['message']}, ensure_ascii=False))],
            budget=max(256, context_budget() - 1536))
        task['conversation_context'] = {key: value for key, value in json.loads(fitted[-1].text).items() if key != 'user_input'}
        task['current_user_input'] = state['user_message']
    reference = deepcopy(active_order_reference.get() or {'issue': empty_issue()})
    reference.update(status='resolved' if task['order_id'] else 'irrelevant', order_id=task['order_id'])
    if task['order_id']:
        reference['issue']['active_order_id'] = task['order_id']
        reference['issue']['candidate_order_ids'] = [task['order_id']]
    elif task.get('order_ids'):
        reference['issue']['active_order_id'] = None
        reference['issue']['candidate_order_ids'] = task['order_ids']
    task_token, reference_token = active_task.set(task), active_order_reference.set(reference)
    try:
        yield task
    finally:
        active_order_reference.reset(reference_token)
        active_task.reset(task_token)


def model_step(state):
    with task_scope(state):
        try:
            return base.model_step(state, SPECIALISTS[state['route']][2])
        except Exception as exc:
            if len(state.get('task_plan', {}).get('tasks', [])) <= 1:
                raise
            kind = classify_error(exc)
            return {'reply': error_message(kind), 'reply_source': 'fallback', 'status': 'failed',
                    'done': True, 'error_type': kind, 'stop_reason': 'model_error'}


def tool_step(state):
    # Write tools consult the transaction's action receipt before doing work.
    # Re-running this node after a commit/checkpoint gap never repeats a write.
    with task_scope(state):
        return base.tool_step(state, SPECIALISTS[state['route']][2])


def finalize_step(state):
    with task_scope(state):
        return base.finalize_step(state)


def escalate_node(state: SupportState) -> dict:
    with task_scope(state) as task:
        try:
            result = escalate_to_human(
                reason="Could not classify or resolve the request automatically",
                context=task['message'] if task else state['user_message'])
        except ExecutionBudgetExceeded as exc:
            return {'reply': str(exc), 'status': 'partial', 'reply_source': 'fallback', 'agent': 'escalate',
                    'trace': [], 'stop_reason': 'budget_exhausted', 'error_type': 'budget'}
    return {
        "reply": result["message"],
        "trace": [{"agent": "escalate", "tool": "escalate_to_human", "args": {}, "result": result}],
        "agent": "escalate",
        "usage": state.get("usage", {}),
        'status': 'completed', 'reply_source': 'fallback', 'stop_reason': 'completed', 'error_type': None,
    }


def clarify_node(state):
    with task_scope(state) as task:
        question = task['clarification_question']
    return {'reply': question, 'clarification_question': question, 'agent': 'clarify', 'trace': [],
            'status': 'needs_clarification', 'reply_source': 'fallback', 'stop_reason': 'needs_clarification', 'error_type': None}


def complete_task(state):
    task = state['task_plan']['tasks'][state['task_index']]
    trace = [{**entry, 'task_id': task['task_id']} for entry in state.get('trace', [])]
    result = {key: task[key] for key in ('task_id', 'message', 'route', 'order_id', 'depends_on')}
    result.update(status=state.get('status') or 'completed', reply=state['reply'], trace=trace,
                  stop_reason=state.get('stop_reason') or 'completed', usage=state.get('usage', {}))
    if state.get('clarification_question'):
        result['clarification_question'] = state['clarification_question']
    if task.get('pipeline_version') == PIPELINE_VERSION:
        result.update(order_ids=task['order_ids'], operation=task.get('operation'), expert=task.get('expert'),
                      intent=task.get('intent'), answer_mode=task.get('answer_mode', 'result'),
                      presentation_intent=task.get('presentation_intent', 'result'),
                      conditions=task['conditions'], condition_result=state.get('condition_result'))
    if state.get('error_type'):
        result['error_type'] = state['error_type']
    results = state['task_results'] + [result]
    index = state['task_index'] + 1
    if result['stop_reason'] == 'budget_exhausted':
        for remaining in state['task_plan']['tasks'][index:]:
            results.append({**{key: remaining[key] for key in ('task_id', 'message', 'route', 'order_id', 'depends_on')},
                            'status': 'not_started', 'reply': '前一步未完成，尚未执行；请核对后单独发起。',
                            'trace': [], 'usage': {}, 'stop_reason': 'dependency_unresolved'})
        index = len(state['task_plan']['tasks'])
    return {'task_results': results, 'task_index': index,
            'completed_trace': state['completed_trace'] + trace,
            'completed_diagnostics': state.get('completed_diagnostics', []) + state.get('model_diagnostics', []),
            'completed_usage': base.sum_usage(state['completed_usage'], state.get('usage'))}


def execute_tasks(state):
    """Select a model-planned task; dependency checks never reinterpret language."""
    task = state['task_plan']['tasks'][state['task_index']]
    condition = check_task_conditions(task, state['task_results']) if task.get('pipeline_version') == PIPELINE_VERSION else 'true'
    return {'condition_result': condition, 'route': state['task_plan']['tasks'][state['task_index']]['route'], 'trace': [],
            'usage': {}, 'reply': None, 'status': None, 'stop_reason': None, 'error_type': None,
            'clarification_question': None, 'model_diagnostics': []}


def reply_node(state):
    task = state['task_plan']['tasks'][state['task_index']]
    return {'reply': task['response'], 'agent': 'conversation', 'status': 'completed', 'trace': [],
            'reply_source': 'model', 'stop_reason': 'completed'}


def skip_node(state):
    met = state['condition_result']
    return {'reply': '前置工具事实不满足条件；未生成办理提案。' if met == 'false' else '前置任务没有可验证结果，后续任务未执行。',
            'agent': state['route'].lower(), 'trace': [], 'status': 'rejected' if met == 'false' else 'blocked',
            'reply_source': 'fallback', 'stop_reason': 'condition_false' if met == 'false' else 'dependency_unresolved'}


def summarize_node(state):
    results = state['task_results']
    summary = summarize_task_results(results) if len(results) > 1 else {
        'reply': results[0]['reply'], 'status': results[0]['status']}
    output = {**summary, 'tasks': results, 'trace': state['completed_trace'], 'usage': state['completed_usage'],
            'model_diagnostics': state.get('completed_diagnostics', []),
            'planning_usage': state['task_plan']['usage'],
            'agent': 'multi' if len(results) > 1 else state['agent'], 'cached': False,
            'reply_source': 'fallback' if len(results) > 1 else state.get('reply_source', 'fallback'),
            'stop_reason': state.get('stop_reason') or 'completed'}
    if state.get('pipeline_version') == PIPELINE_VERSION:
        if output['status'] == 'partial':
            output['stop_reason'] = next((task['stop_reason'] for task in results if task['stop_reason'] != 'completed'), 'partial_results')
        if any(task.get('error_type') for task in results):
            output['error_type'] = next(task['error_type'] for task in results if task.get('error_type'))
        from src import tools
        from datetime import datetime, timezone
        value = state['understanding']
        selected_ids = [number for task in state['task_plan']['tasks'] for number in task.get('order_ids', [])]
        ids = list(dict.fromkeys(selected_ids or value.get('mentioned_order_ids', [])))
        ids = [number for number in ids if tools.can_access_order(number)]
        previous = (active_order_reference.get() or {}).get('issue', empty_issue())
        issue = {**previous}
        if value['kind'] == 'tasks':
            operations = {task['operation'] for task in value['tasks']}
            issue['operation'] = next(iter(operations)) if len(operations) == 1 else None
            issue['order_groups'] = remember_order_groups(previous.get('order_groups', []),
                state['task_plan']['tasks'], active_request.get().request_id)
        if ids and value['kind'] != 'reply':
            active = ids[0] if len(ids) == 1 else None
            issue.update(candidate_order_ids=ids, active_order_id=active, selection_source='model_understanding',
                         updated_at=datetime.now(timezone.utc).isoformat(),
                         revision=previous['revision'] + int(active != previous['active_order_id'] or ids != previous['candidate_order_ids']))
        pending = (active_run.get() or {}).get('input', {}).get('pending_clarification')
        if value['kind'] == 'clarify':
            messages = pending['messages'] if pending and value.get('continues_pending') else []
            pending = {'messages': messages + [state['user_message']], 'question': value['message'], 'candidate_order_ids': ids,
                       'missing_fields': value.get('missing_fields', []), 'operation': value.get('pending_operation'),
                       'parameter': value.get('clarification_parameter')}
        elif value['kind'] != 'reply':
            pending = None
        output.update(pipeline_version=PIPELINE_VERSION, understanding=value, pending_clarification=pending,
                      order_reference={'status': 'resolved' if issue['active_order_id'] else 'context',
                                       'order_id': issue['active_order_id'], 'issue': issue},
                      batch_results=[{'task_id': entry['task_id'], 'tool': entry['tool'], 'items': entry['result']['items']}
                                     for entry in state['completed_trace'] if isinstance(entry['result'], dict) and 'items' in entry['result']])
    return output


def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)
    graph.add_node("route", route_node)
    graph.add_node('prepare', prepare_node)
    graph.add_node('model', model_step)
    graph.add_node('tool', tool_step)
    graph.add_node('finalize', finalize_step)
    graph.add_node("escalate", escalate_node)
    graph.add_node('clarify', clarify_node)
    graph.add_node('reply', reply_node)
    graph.add_node('skip', skip_node)
    graph.add_node('complete_task', complete_task)
    graph.add_node('execute_tasks', execute_tasks)
    graph.add_node('summarize', summarize_node)

    graph.add_edge(START, "route")
    graph.add_edge('route', 'execute_tasks')
    graph.add_edge('prepare', 'model')
    graph.add_conditional_edges('model', lambda state: 'finalize' if state['done'] else 'model' if state.get('needs_tool_correction') else 'tool')
    graph.add_conditional_edges('tool', lambda state: 'finalize' if state.get('done') else 'tool' if state['tool_index'] < len(state['tool_calls'])
                                else 'finalize' if state['agent'] == 'faq' else 'model')
    for name in ('finalize', 'escalate', 'clarify', 'reply', 'skip'):
        graph.add_conditional_edges(name, lambda state: 'complete_task' if state.get('task_plan') else END)
    graph.add_conditional_edges('complete_task', lambda state: 'execute_tasks' if state['task_index'] < len(state['task_plan']['tasks']) else 'summarize')
    graph.add_conditional_edges('execute_tasks', lambda state: 'skip' if state['condition_result'] != 'true' else
                                'prepare' if state['route'] in SPECIALISTS else 'escalate' if state['route'] == 'ESCALATE' else
                                'reply' if state['route'] == 'REPLY' else 'clarify')
    graph.add_edge('summarize', END)

    return graph.compile(checkpointer=checkpointer)


_GRAPH = build_graph()


def run_orchestrator(user_message: str, history: list | None = None) -> dict:
    value = {'user_message': user_message, 'history': history or [], 'trace': []}
    run = active_run.get()
    if run is None:
        result = _GRAPH.invoke(value, {'recursion_limit': 256})
    else:
        context = active_request.get()
        binding = [require_principal().customer_id, context.mode, context.session_id, context.request_id]
        if binding != [run['user_id'], run['mode'], run['session_id'], run['request_id']]:
            raise AccessError('运行身份或会话不匹配。', 403)
        with closing(sqlite3.connect(run['checkpoint_path'], timeout=20, check_same_thread=False)) as connection:
            saver = SqliteSaver(connection)
            graph = build_graph(saver)
            config = {'configurable': {'thread_id': run['run_id']}, 'recursion_limit': 256}
            snapshot = graph.get_state(config)
            if run['input'].get('pipeline_version') != PIPELINE_VERSION:
                from src import tools
                from src.results import fallback_result
                trace = tools.request_journal().action_trace(context)
                result = fallback_result(trace, interrupted=True) if trace else {'reply': '旧在线流程与当前版本不兼容，未自动重跑；请重新发起剩余需求。',
                    'agent': 'system', 'trace': [], 'usage': {}, 'status': 'failed', 'reply_source': 'fallback'}
                return {**result, 'stop_reason': 'pipeline_incompatible', 'error_type': 'pipeline_incompatible'}
            if snapshot.values:
                if snapshot.values.get('binding') != binding:
                    raise AccessError('运行身份或会话不匹配。', 403)
                from src import tools
                # A saved answer must not reveal an order whose owner changed.
                for entry in [*snapshot.values.get('completed_trace', []), *snapshot.values.get('trace', [])]:
                    if isinstance(entry.get('result'), dict) and entry['result'].get('error'):
                        continue
                    ids = [number for number, fact in iter_order_results(entry.get('result')) if not fact.get('error')]
                    if not ids and (entry.get('args') or {}).get('order_id'):
                        ids = [entry['args']['order_id']]
                    if any(not tools.can_access_order(number) for number in ids):
                        raise AccessError('运行涉及的订单权限已变化，不能继续恢复。', 409)
                result = graph.invoke(None, config, durability='sync') if snapshot.next else snapshot.values
            else:
                result = graph.invoke({**value, 'binding': binding}, config, durability='sync')

    logger.info(
        "agent_decision",
        extra={
            "extra_fields": {
                "message_chars": len(user_message),
                "route": result.get("route"),
                "agent": result.get("agent"),
                "tools_called": [t["tool"] for t in result.get("trace", [])],
                "usage": result.get("usage"),
            }
        },
    )
    return result

