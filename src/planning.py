"""Small sequential plans over explicit user clauses, never model-authored permissions."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import re

from src.memory import empty_issue, extract_order_ids, resolve_order_reference
from src.runtime_context import active_order_reference


CLARIFICATION = '请说明你想查询订单或物流、查看退款资格、办理付款、了解政策，还是转人工；办理业务时请明确订单。'
SPECIALIST_ROUTES = {'ORDER', 'REFUND', 'PAYMENT', 'FAQ'}


def negative_action(message, route=None):
    actions = {'REFUND': r'退款|退钱|退掉|refund', 'PAYMENT': r'付款|支付|pay(?:ment)?',
               'ESCALATE': r'(?:转|找|联系)?人工|投诉|human|complaint'}
    action = actions.get(route, '|'.join(actions.values()))
    return bool(re.search(r"(?:不要|不用|不想|不需要|别|暂不|不|\bdo not\b|\bdon't\b|\bnever\b|\bno\b)"
                         r'[^,，;；。\n]{0,40}(?:' + action + ')', message, re.IGNORECASE))


def explicit_escalation(message):
    return not negative_action(message, 'ESCALATE') and bool(re.search(
        r'投诉|转人工|找人工|人工客服|人工处理|要人工|\bhuman\b|complaint|speak to.*(?:manager|agent)', message, re.IGNORECASE))


def allows_write(message, route):
    if negative_action(message, route) or re.search(r'只查|只看|仅查|仅看|能否|是否|资格|eligib|status|only.*(?:check|query)', message, re.IGNORECASE):
        return False
    pattern = {
        'REFUND': r'(?:帮我|给我|请|我要|我想|申请|办理|直接).*(?:退款|退钱|退掉)|退款吧|退掉|退那笔|退这笔|refund it|want.*refund|request.*refund|like.*return|get my money back',
        'PAYMENT': r'(?:帮我|给我|请|我要|我想|办理|执行|直接).*(?:付款|支付)|(?:支付|付款)吧|^(?:付款|支付)|process.*payment|make.*payment|(?:want|like|please|help).*pay|pay it',
    }.get(route)
    return bool(pattern and re.search(pattern, message, re.IGNORECASE))


def split_task_clauses(message):
    message = message.split('\n', 1)[-1] if message.startswith('(Context:') else message
    parts = re.split(r'(?:[，,；;。]\s*|然后|接着|并且|同时|以及|(?:再|并|和)(?=查|看|帮|申请|办理|付|退)|\b(?:and then|then|and)\b)', message, flags=re.IGNORECASE)
    clauses = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Amounts, reasons and methods belong to the preceding task.
        if clauses and re.match(r'(?:金额|原因|因为|理由|使用|用\s*(?:paypal|card)|using\b|because\b|get my money back)', part, re.IGNORECASE):
            clauses[-1] += '，' + part
        else:
            clauses.append(part)
    return clauses or [message.strip()]


def separate_order_tasks(message):
    clauses = split_task_clauses(message)
    ids = [extract_order_ids(clause) for clause in clauses]
    return len(clauses) > 1 and all(len(numbers) <= 1 for numbers in ids) and len({number for numbers in ids for number in numbers}) > 1


@dataclass
class TaskPlan:
    tasks: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def plan_tasks(message, classify):
    """The classifier returns a route + usage; server code resolves IDs/permissions."""
    from src import tools
    clauses = split_task_clauses(message)
    if len(clauses) > 6 or re.search(r'如果.*(?:再|就)|成功后|确认后|到账后|\bif\b.*\bthen\b|\bafter\b.*(?:refund|payment|confirm)', message, re.IGNORECASE):
        return TaskPlan([{'task_id': 'task_1', 'message': message, 'route': 'CLARIFY', 'order_id': None,
                          'allow_write': False, 'depends_on': [],
                          'clarification_question': '请将任务缩减到六项以内；若后续动作依赖退款、付款或确认成功，请先完成并核对前一步，再单独发起后续请求。'}])
    plan = TaskPlan()
    issue = deepcopy((active_order_reference.get() or {}).get('issue', empty_issue()))
    for clause in clauses:
        if len(clauses) > 1 and negative_action(clause) and not re.search(r'查|看|track|check|query|政策|policy', clause, re.IGNORECASE):
            continue
        route, usage = classify(clause)
        for key, value in usage.items():
            plan.usage[key] = plan.usage.get(key, 0) + (value or 0)
        if route == 'ESCALATE' and not explicit_escalation(clause):
            route = 'CLARIFY'
        reference = resolve_order_reference(clause, issue, tools.can_access_order, force_order_task=route in {'ORDER', 'REFUND', 'PAYMENT'})
        question = CLARIFICATION
        if route in {'ORDER', 'REFUND', 'PAYMENT'}:
            if reference['status'] in {'ambiguous', 'missing', 'invalid'}:
                from src.memory import clarify_order_reference
                question = '未找到可访问的订单，请核对编号。' if reference['status'] == 'invalid' else clarify_order_reference(reference)
                route = 'CLARIFY'
            else:
                issue = reference['issue']
        task = {'task_id': 'task_' + str(len(plan.tasks) + 1), 'message': clause, 'route': route,
                'order_id': reference['order_id'], 'allow_write': allows_write(clause, route),
                'depends_on': [plan.tasks[-1]['task_id']] if plan.tasks else []}
        if route == 'CLARIFY':
            task['clarification_question'] = question
        duplicate = next((item for item in plan.tasks if (route == 'ESCALATE' or task['allow_write']) and
                          item['route'] == route and item['order_id'] == task['order_id']), None)
        if duplicate:
            duplicate['message'] += '；' + clause
            continue
        plan.tasks.append(task)
    if not plan.tasks:
        plan.tasks.append({'task_id': 'task_1', 'message': message, 'route': 'CLARIFY', 'order_id': None,
                           'allow_write': False, 'depends_on': [], 'clarification_question': CLARIFICATION})
    return plan


def summarize_task_results(results):
    statuses = [task['status'] for task in results]
    unfinished = {'partial', 'failed', 'needs_clarification', 'not_started', 'blocked'}
    if statuses and all(status == 'needs_clarification' for status in statuses):
        status = 'needs_clarification'
    elif any(value in unfinished for value in statuses) or 'rejected' in statuses and len(set(statuses)) > 1:
        status = 'partial'
    elif 'pending_confirmation' in statuses:
        status = 'pending_confirmation'
    elif statuses and all(value == 'rejected' for value in statuses):
        status = 'rejected'
    elif statuses and all(value == 'cancelled' for value in statuses):
        status = 'cancelled'
    else:
        status = 'completed'
    return {'status': status, 'reply': '\n\n'.join(
        f"{index}. {task['message']}（{task['status']}）\n{task['reply']}" for index, task in enumerate(results, 1))}
