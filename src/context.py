"""Estimated input budgets; whole history turns and paired tool messages only."""
import json
import math
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ContextBudgetExceeded(ValueError):
    pass


def context_budget():
    value = int(os.environ.get('SUPPORT_CONTEXT_TOKEN_BUDGET', '6000'))
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


def build_context(turns, message, issue, profile, budget=None):
    budget = context_budget() if budget is None else budget
    ids = set(issue['candidate_order_ids'])
    if issue['active_order_id']:
        ids = {issue['active_order_id']}
    topics = ('退款', '支付', '付款', '物流', 'refund', 'payment', 'tracking')
    def score(turn):
        orders = {(entry.get('args') or {}).get('order_id') for entry in turn.get('trace', [])}
        common = sum(word in message.lower() and word in turn['user'].lower() for word in topics)
        return 4 * len(ids & orders) + common
    ranked = sorted(range(len(turns)), key=lambda index: (score(turns[index]), index), reverse=True)
    fixed = HumanMessage(content=message + json.dumps({'issue': issue, 'profile': profile}, ensure_ascii=False))
    available = max(0, budget - 1024 - estimate_context_tokens([fixed]))
    chosen, used = [], 0
    for index in ranked:
        pair = [HumanMessage(content=turns[index]['user']), AIMessage(content=turns[index]['reply'])]
        size = estimate_context_tokens(pair)
        if used + size <= available:
            chosen.append(index)
            used += size
    selected = [turns[index] for index in sorted(chosen)]
    history = [message for turn in selected for message in
               (HumanMessage(content=turn['user']), AIMessage(content=turn['reply']))]
    return selected, history, context_info(budget, selected_turns=len(selected), available_turns=len(turns),
                                         estimated_history_tokens=used)


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
            else:
                compact = {'excerpt': str(data)[:300]}
            compact['context_truncated'] = True
            messages[index] = message.model_copy(update={'content': json.dumps(compact, ensure_ascii=False)})
            shortened = True
    size = estimate_context_tokens(messages, tools)
    if size > budget:
        raise ContextBudgetExceeded('必要消息与工具结果超出上下文估算预算，已停止继续调用模型。')
    return messages, context_info(budget, estimated_input_tokens=size, removed_turns=removed, tool_results_truncated=shortened)
