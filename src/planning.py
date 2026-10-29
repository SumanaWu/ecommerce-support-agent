"""Compile and execute model-authored plans under capability and resource checks."""
from dataclasses import asdict, dataclass, field


@dataclass
class TaskPlan:
    tasks: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


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


def compile_task_plan(understanding, message, *, usage=None, context_order_ids=()):
    from src.understanding import PIPELINE_VERSION, OPERATIONS, expand_order_selection, validate_understanding
    value = validate_understanding(understanding, context_order_ids=context_order_ids)
    if value.kind != 'tasks':
        route = {'clarify': 'CLARIFY', 'reply': 'REPLY', 'escalate': 'ESCALATE'}[value.kind]
        return TaskPlan([{'task_id': 'task_1', 'message': message, 'route': route, 'order_id': None,
            'order_ids': [], 'allow_write': False, 'depends_on': [], 'conditions': [],
            'pipeline_version': PIPELINE_VERSION, 'clarification_question': value.message,
            'response': value.message}], usage or {})
    tasks = []
    for item in value.tasks:
        ids = expand_order_selection(item.orders)
        expert, intent, _ = OPERATIONS[item.operation]
        tasks.append({'task_id': item.task_id, 'message': item.instruction, 'route': expert, 'expert': expert, 'intent': intent,
            'operation': item.operation, 'order_ids': ids, 'order_id': ids[0] if len(ids) == 1 else None,
            'selection_kind': item.orders.kind, 'allow_write': intent == 'propose', 'answer_mode': item.answer_mode,
            'presentation_intent': item.presentation_intent,
            'refund_reason': item.refund_reason, 'payment_method': item.payment_method,
            'allowed_tools': sorted(OPERATIONS[item.operation][2]), 'depends_on': item.depends_on,
            'conditions': [condition.model_dump() for condition in item.conditions], 'pipeline_version': PIPELINE_VERSION})
    return TaskPlan(tasks, usage or {})


def iter_order_results(result):
    """Yield structured order facts, including per-item unavailable batch results."""
    if not isinstance(result, dict):
        return
    if 'items' in result:
        for item in result['items']:
            yield item['order_id'], item['result']
    elif 'order_id' in result:
        yield result['order_id'], result


def check_task_conditions(task, prior_results):
    prior = {result['task_id']: result for result in prior_results}
    fact_dependencies = {condition['task_id'] for condition in task.get('conditions', [])}
    for dependency in task['depends_on']:
        if dependency not in prior or (prior[dependency]['status'] != 'completed' and not (
                prior[dependency]['status'] == 'partial' and dependency in fact_dependencies)):
            return 'unknown'
    for condition in task.get('conditions', []):
        source = prior[condition['task_id']]
        tool, field = {'eligible': ('check_refund_eligibility', 'eligible'),
                       'payment_status': ('get_payment_status', 'status'),
                       'order_status': ('get_order_status', 'status')}[condition['field']]
        values = [result[field] for entry in source['trace'] if entry['tool'] in {tool, tool + '_batch'}
                  for number, result in iter_order_results(entry.get('result'))
                  if number == condition['order_id'] and not result.get('error') and field in result]
        if not values:
            return 'unknown'
        if type(values[-1]) is not type(condition['equals']) or values[-1] != condition['equals']:
            return 'false'
    return 'true'


class ToolValidationError(ValueError):
    def __init__(self, message, code='invalid_arguments'):
        super().__init__(message)
        self.code = code


def task_stage(task, trace):
    """Only verified results from this request advance a proposal stage."""
    operation = (task or {}).get('operation')
    if operation not in {'propose_refund', 'propose_payment'}:
        return 'read'
    if task_evidence_complete(task, trace):
        return 'done'
    prerequisite, field, expected = ('check_refund_eligibility', 'eligible', True) if operation == 'propose_refund' else ('get_payment_status', 'status', 'pending')
    for entry in reversed(trace):
        if entry['tool'] not in {prerequisite, prerequisite + '_batch'}:
            continue
        for number, result in iter_order_results(entry.get('result')):
            if number in task['order_ids'] and not result.get('error') and field in result:
                return 'propose' if result[field] == expected else 'done'
    return 'prerequisite'


def stage_tools(task, trace):
    if not task or not task.get('allow_write'):
        return (task or {}).get('allowed_tools', [])
    stage = task_stage(task, trace)
    return ({'propose_refund': {'prerequisite': ['check_refund_eligibility'], 'propose': ['initiate_refund']},
             'propose_payment': {'prerequisite': ['get_payment_status'], 'propose': ['process_payment']}}
            [task['operation']].get(stage, []))


def validate_tool_scope(name, args, task, *, trace=()):
    from src.understanding import PIPELINE_VERSION
    if not task or task.get('pipeline_version') != PIPELINE_VERSION:
        return
    if name not in task['allowed_tools']:
        raise ValueError('工具不属于模型已校验任务的能力范围。')
    ids = args.get('order_ids') if 'order_ids' in args else [args['order_id']] if 'order_id' in args else []
    if not isinstance(ids, list) or len(ids) > 20 or any(type(number) is not int or number <= 0 for number in ids):
        raise ValueError('工具订单参数必须为最多20个正整数。')
    if not set(ids) <= set(task['order_ids']):
        raise ValueError('工具订单超出本任务范围。')
    if task['order_ids'] and not ids:
        raise ValueError('业务工具缺少订单参数。')
    if name in {'initiate_refund', 'process_payment'}:
        if not task['allow_write'] or len(ids) != 1:
            raise ValueError('只读任务不能创建办理提案。')
        if name == 'initiate_refund' and (not isinstance(args.get('reason'), str) or not 1 <= len(args['reason']) <= 300):
            raise ValueError('退款理由必须为1至300字，不能截断后改变已保存动作参数。')
        parameter, field = ('reason', 'refund_reason') if name == 'initiate_refund' else ('method', 'payment_method')
        if task.get(field) is not None and args.get(parameter) != task[field]:
            raise ToolValidationError('提案参数必须与当前任务的' + field + '逐字一致；请使用已校验字段修正参数，不得自行改写。', 'action_parameter_mismatch')
        prerequisite, field, expected = ('check_refund_eligibility', 'eligible', True) if name == 'initiate_refund' else ('get_payment_status', 'status', 'pending')
        facts = [result for entry in trace if entry['tool'] in {prerequisite, prerequisite + '_batch'}
                 for number, result in iter_order_results(entry.get('result')) if number == ids[0] and not result.get('error')]
        if not facts or facts[-1].get(field) != expected:
            raise ToolValidationError('创建提案前必须调用本订单的资格/付款状态工具；请自行查询后继续，不能要求客户提供数据库状态。', 'missing_prerequisite')
    if name not in stage_tools(task, trace):
        raise ToolValidationError('该工具不属于当前执行阶段，请使用当前开放的工具继续。', 'invalid_stage')


def task_evidence_complete(task, trace):
    if not task or not task.get('operation'):
        return bool(trace)
    operation = task['operation']
    required = {'order_status': {'get_order_status', 'get_order_status_batch'},
                'order_details': {'get_order_details', 'get_order_details_batch'},
                'shipment': {'track_shipment', 'track_shipments_batch'},
                'payment_status': {'get_payment_status', 'get_payment_status_batch'},
                'refund_eligibility': {'check_refund_eligibility'},
                'propose_refund': {'initiate_refund'}, 'propose_payment': {'process_payment'},
                'policy': {'answer_faq'}}[operation]
    matching = [entry for entry in trace if entry['tool'] in required and
                isinstance(entry.get('result'), dict) and not entry['result'].get('error') and entry['result'].get('executed') is not False]
    if operation == 'policy':
        return any(isinstance(entry['result'], dict) and 'answer' in entry['result'] for entry in matching)
    covered = {number for entry in matching for number, result in iter_order_results(entry.get('result'))}
    for entry in trace:
        # A real unavailable-order result is terminal, unlike a local validation error.
        if entry['tool'] in task.get('allowed_tools', []) and isinstance(entry.get('result'), dict) and entry['result'].get('error') == '未找到可访问的订单。' and entry['result'].get('executed') is not False:
            covered.update(entry.get('args', {}).get('order_ids', [entry['args']['order_id']] if 'order_id' in entry.get('args', {}) else []))
    # A business refusal is a verified outcome, not a missing write call.
    if task['allow_write']:
        for entry in trace:
            if entry['tool'] in {'check_refund_eligibility', 'get_payment_status'}:
                for number, result in iter_order_results(entry.get('result')):
                    if result.get('eligible') is False or operation == 'propose_payment' and result.get('status') in {'paid', 'refunded'}:
                        covered.add(number)
    return set(task['order_ids']) <= covered
