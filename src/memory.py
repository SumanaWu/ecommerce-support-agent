"""Structured session facts; language and references are understood by the model."""


def empty_issue():
    return {'candidate_order_ids': [], 'active_order_id': None, 'selection_source': None,
            'updated_at': None, 'revision': 0, 'order_groups': []}


def remember_order_groups(groups, tasks, request_id):
    """Keep ordered batch sources independent of the current single-order focus."""
    groups = list(groups)
    for task in tasks:
        ids = list(dict.fromkeys(task.get('order_ids', [])))
        if len(ids) < 2 or task.get('intent', 'read') != 'read':
            continue
        if not any(group['source_request_id'] == request_id and group.get('source_task_id', 'task_1') == task.get('task_id', 'task_1') for group in groups):
            sequence = max((int(group['group_id'][5:]) for group in groups if group['group_id'].startswith('list_') and group['group_id'][5:].isdigit()), default=0) + 1
            group_id = 'list_' + str(sequence)
            groups.append({'group_id': group_id, 'order_ids': ids, 'operation': task.get('operation'),
                           'source_request_id': request_id, 'source_task_id': task.get('task_id', 'task_1')})
    return groups[-6:]


def session_issue(payload):
    if 'issue' in payload:
        issue = {**empty_issue(), **payload['issue']}
    else:
        issue = empty_issue()
    if 'issue' not in payload and payload.get('last_order_id'):
        issue.update(candidate_order_ids=[payload['last_order_id']], selection_source='legacy_last_order')
    if 'order_groups' not in payload.get('issue', {}):
        for turn in payload.get('turns', []):
            issue['order_groups'] = remember_order_groups(issue['order_groups'], turn.get('tasks', []), turn.get('request_id'))
    return issue
