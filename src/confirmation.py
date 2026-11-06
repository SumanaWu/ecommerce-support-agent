"""Checkpointed confirmation workflow for prepared business actions."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.actions import decide_action


class ConfirmationState(TypedDict):
    user_id: str
    session_id: str
    actions: list[dict]
    index: int
    decision: dict
    reply: str


def wait_for_decision(state: ConfirmationState) -> dict:
    action = state['actions'][state['index']]
    decision = interrupt(action)
    if not isinstance(decision, dict) or decision.get('decision') not in {'confirm', 'cancel'}:
        raise ValueError('无效确认决定')
    if decision.get('action_id') != action['action_id']:
        raise ValueError('确认操作不匹配')
    return {'decision': decision}


def execute_decision(state: ConfirmationState) -> dict:
    action = state['actions'][state['index']]
    decision = state['decision']
    result = decide_action(state['user_id'], state['session_id'], action['action_id'],
                           decision['decision'])
    actions = list(state['actions'])
    actions[state['index']] = result
    index = state['index'] + 1
    return {'actions': actions, 'index': index}


def summarize(state: ConfirmationState) -> dict:
    labels = {'completed': '已办理', 'cancelled': '已取消', 'rejected': '未办理'}
    parts = [f"订单 {a['order_id']}：{labels.get(a['status'], a['status'])}" for a in state['actions']]
    return {'reply': '；'.join(parts)}


def build_confirmation_graph(checkpointer):
    graph = StateGraph(ConfirmationState)
    graph.add_node('wait', wait_for_decision)
    graph.add_node('execute', execute_decision)
    graph.add_node('summarize', summarize)
    graph.add_edge(START, 'wait')
    graph.add_edge('wait', 'execute')
    graph.add_conditional_edges('execute',
                                lambda state: 'wait' if state['index'] < len(state['actions']) else 'summarize',
                                {'wait': 'wait', 'summarize': 'summarize'})
    graph.add_edge('summarize', END)
    return graph.compile(checkpointer=checkpointer)
