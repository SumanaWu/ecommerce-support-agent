"""Estimated input budgets; whole history turns and paired tool messages only."""
import json
import math
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ContextBudgetExceeded(ValueError):
    pass


def context_budget():
    value = int(os.environ.get('SUPPORT_CONTEXT_TOKEN_BUDGET', '12000'))
    if not 256 <= value <= 128000:
        raise ValueError('SUPPORT_CONTEXT_TOKEN_BUDGET 必须为 256 至 128000 的整数。')
    return value


def estimate_context_tokens(messages, tools=None):
    values = [{'role': message.type, 'content': message.content,
               'tool_calls': getattr(message, 'tool_calls', []),
               'tool_call_id': getattr(message, 'tool_call_id', None)} for message in messages]
    schemas = [{'name': tool.name, 'description': tool.description, 'args': tool.args} for tool in tools or []]
    return math.ceil(len(json.dumps([values, schemas], ensure_ascii=False).encode('utf-8')) / 3)


def context_info(budget, **fields):
    return {'budget_tokens': budget, 'is_estimate': True, 'estimator': 'utf8_bytes_div3', **fields}


def planning_context(turns):
    """Recent dialogue and bounded, sourced snapshots; never current eligibility."""
    from src.planning import iter_order_results
    recent, facts, events = [], [], []
    for turn in turns:
        if turn.get('request_id', '').startswith(('confirm_', 'cancel_')):
            actions = turn.get('actions') or [entry['result'] for entry in turn.get('trace', [])
                                            if isinstance(entry.get('result'), dict) and entry['result'].get('action_id')]
            events.append({'source_request_id': turn['request_id'], 'status': turn.get('status'),
                           'actions': [{key: action[key] for key in ('action_id', 'order_id', 'status') if key in action}
                                       for action in actions], 'historical': True})
            continue
        tasks = [{key: task[key] for key in ('operation', 'order_ids', 'order_id', 'answer_mode') if key in task}
                 for task in turn.get('tasks', []) if task.get('operation')]
        ids = list(dict.fromkeys([number for task in tasks for number in
                   (task.get('order_ids') or ([task['order_id']] if task.get('order_id') else []))] +
                   (turn.get('understanding') or {}).get('mentioned_order_ids', [])))
        for entry in turn.get('trace', []):
            for number, result in iter_order_results(entry.get('result')):
                if result.get('error') or result.get('executed') is False:
                    continue
                if number not in ids:
                    ids.append(number)
                keys = {'order_id', 'status', 'order_status', 'payment_status', 'eligible', 'reason_code', 'reason',
                        'checks', 'rule_version', 'amount_minor', 'currency', 'item', 'method', 'action_id',
                        'requires_confirmation', 'success', 'tracking_number', 'carrier'}
                facts.append({'order_id': number, 'tool': entry['tool'],
                              'result': {key: value for key, value in result.items() if key in keys},
                              'source_request_id': turn.get('request_id'), 'historical': True})
        if not turn.get('_facts_only'):
            recent.append({'user': turn.get('user', ''), 'reply': turn.get('reply', ''),
                           'tasks': tasks, 'order_ids': ids, 'source_request_id': turn.get('request_id')})
    return {'recent_turns': recent[-6:], 'recent_facts': facts[-8:], 'recent_events': events[-3:]}


def task_context(packet, order_ids, *, intent='read'):
    ids = set(order_ids)
    recent = [turn for turn in packet['recent_turns'] if ids.intersection(turn['order_ids'])]
    facts = [fact for fact in packet['recent_facts'] if fact['order_id'] in ids]
    events = [event for event in packet.get('recent_events', [])
              if any(action.get('order_id') in ids for action in event['actions'])]
    if intent == 'propose':
        # The planner has already resolved the new request. Old proposals are not execution state.
        recent = [turn for turn in recent if not any(task['operation'] in {'propose_refund', 'propose_payment'}
                                                    for task in turn.get('tasks', []))]
        facts = [fact for fact in facts if fact['tool'] not in {'initiate_refund', 'process_payment'}]
        events = []
    return {'recent_turns': recent, 'recent_facts': facts, 'recent_events': events}


def build_context(turns, message, issue, profile, budget=None, *, pending=None):
    budget = context_budget() if budget is None else budget
    # Keep older factual sources and authenticated action events outside the six Q/A pairs.
    factual = [index for index, turn in enumerate(turns) if turn.get('trace')][-8:]
    decisions = [i for i, turn in enumerate(turns) if turn.get('request_id', '').startswith(('confirm_', 'cancel_'))][-3:]
    recent = [i for i in range(len(turns)) if not
              turns[i].get('request_id', '').startswith(('confirm_', 'cancel_'))][-6:]
    selected = [{**turns[index], **({'_facts_only': True} if index not in recent else {})}
                for index in sorted(set(factual + recent + decisions))]
    packet = planning_context(selected)
    fixed = {'user_input': message, 'issue': issue, 'profile': profile, 'pending_clarification': pending, **packet}
    try:
        fitted, _ = fit_structured_context([HumanMessage(content=json.dumps(fixed, ensure_ascii=False))], budget=max(256, budget - 1024))
        kept_count = len(json.loads(fitted[-1].text)['recent_turns'])
    except ContextBudgetExceeded:
        kept_count = 0  # The planner reports mandatory-input overflow inside the request's error handler.
    kept_indices = set(recent[-kept_count:]) if kept_count else set()
    selected = [{**turns[index], **({'_facts_only': True} if index not in kept_indices else {})}
                for index in sorted(set(factual + decisions) | kept_indices)]
    history = [item for turn in selected if not turn.get('_facts_only')
               for item in (HumanMessage(content=turn['user']), AIMessage(content=turn['reply']))]
    return selected, history, context_info(budget, selected_turns=len(history) // 2, available_turns=len(turns),
                                         estimated_history_tokens=estimate_context_tokens(history))


def fit_model_context(messages, tools, budget=None):
    budget = context_budget() if budget is None else budget
    messages = list(messages)
    removed, shortened = 0, False
    # The latest human message starts the current tool loop. Older history is H/A pairs.
    current = max(index for index, message in enumerate(messages) if isinstance(message, HumanMessage))
    while estimate_context_tokens(messages, tools) > budget and current > 1:
        del messages[1:3]
        current -= 2
        removed += 1
    if estimate_context_tokens(messages, tools) > budget:
        for index, message in enumerate(messages):
            if not isinstance(message, ToolMessage) or len(str(message.content)) <= 500:
                continue
            data = json.loads(message.content)
            # Preserve the fields used by business decisions. Original results stay in trace/DB.
            if isinstance(data, dict):
                compact = {key: value if not isinstance(value, str) else value[:300]
                           for key, value in data.items() if value is None or isinstance(value, (str, bool, int, float))}
                if isinstance(data.get('items'), list):
                    keys = {'order_id', 'status', 'eligible', 'reason_code', 'amount_minor', 'currency',
                            'requires_confirmation', 'action_id', 'error', 'error_code', 'success'}
                    compact['items'] = [{'order_id': item['order_id'], 'result': {
                        key: value[:300] if isinstance(value, str) else value
                        for key, value in item['result'].items() if key in keys}}
                        for item in data['items']]
            else:
                compact = {'excerpt': str(data)[:300]}
            compact['context_truncated'] = True
            messages[index] = message.model_copy(update={'content': json.dumps(compact, ensure_ascii=False)})
            shortened = True
    size = estimate_context_tokens(messages, tools)
    if size > budget:
        raise ContextBudgetExceeded('必要消息与工具结果超出上下文估算预算，已停止继续调用模型。')
    return messages, context_info(budget, estimated_input_tokens=size, removed_turns=removed, tool_results_truncated=shortened)


def fit_structured_context(messages, budget=None):
    """Trim whole optional dialogue turns/facts after the schema is included."""
    budget = context_budget() if budget is None else budget
    try:
        current = json.loads(messages[-1].text)
    except (ValueError, TypeError):
        return fit_model_context(messages, [], budget)
    if not isinstance(current, dict) or 'recent_turns' not in current:
        return fit_model_context(messages, [], budget)
    recent = list(current['recent_turns'])
    facts = list(current.get('recent_facts', []))
    original = len(recent)
    original_facts = len(facts)
    while True:
        fitted = [*messages[:-1], HumanMessage(content=json.dumps({**current, 'recent_turns': recent, 'recent_facts': facts}, ensure_ascii=False))]
        size = estimate_context_tokens(fitted)
        if size <= budget:
            return fitted, context_info(budget, estimated_input_tokens=size, removed_turns=original - len(recent),
                history_turns=len(recent), history_format='conversation_facts', removed_facts=original_facts - len(facts),
                fact_count=len(facts), tool_results_truncated=False)
        if facts:
            facts.pop(0)
        elif recent:
            recent.pop(0)
        else:
            raise ContextBudgetExceeded('必要规划上下文超出估算预算，当前输入和待澄清事项不能截断，已停止调用模型。')
