import json

import pytest
from langchain_core.messages import AIMessage
from src.planning import Plan, make_plan


class Model:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=json.dumps(self.result))


def test_planner_returns_multiple_scoped_tasks():
    message = '查1002物流和1003支付'
    tasks = [{'task_id': 't1', 'operation': 'order.shipment', 'targets': [1002], 'source_text': '1002物流'},
             {'task_id': 't2', 'operation': 'payment.status', 'targets': [1003], 'source_text': '1003支付'}]
    plan, _ = make_plan(message, model=Model({'action': 'PLAN', 'tasks': tasks}))
    assert [t['targets'] for t in plan['tasks']] == [[1002], [1003]]


@pytest.mark.parametrize('task', [
    {'task_id': 't1', 'operation': 'unknown', 'source_text': '查询'},
    {'task_id': 't1', 'operation': 'payment.prepare', 'intent': 'query', 'source_text': '查询'},
    {'task_id': 't1', 'operation': 'order.status', 'depends_on': ['t1'], 'source_text': '查询'},
    {'task_id': 't1', 'operation': 'escalate.ticket', 'intent': 'escalate', 'reason': 'technical', 'source_text': '查询'},
])
def test_invalid_plans_cannot_execute(task):
    with pytest.raises(ValueError):
        Plan.model_validate({'action': 'PLAN', 'tasks': [task]})


def test_invalid_plan_gets_exactly_three_repairs():
    model = Model({'action': 'PLAN', 'tasks': []})
    with pytest.raises(RuntimeError):
        make_plan('查询', model=model)
    assert len(model.calls) == 4


def test_task_source_must_come_from_current_message():
    model = Model({'action': 'PLAN', 'tasks': [{'task_id': 't1', 'operation': 'payment.prepare',
        'intent': 'prepare', 'targets': [1003], 'source_text': '我授权支付'}]})
    with pytest.raises(RuntimeError):
        make_plan('只是查询', model=model)
