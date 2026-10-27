import logging
import re
import sqlite3
from contextlib import closing
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents import base, faq_agent, order_agent, payment_agent, refund_agent
from src.agents.base import as_text
from src.auth import AccessError, require_principal
from src.errors import invoke_with_retry
from src.llm import extract_usage, get_llm
from src.logging_config import configure_logging
from src.tools import escalate_to_human
from src.runtime_context import active_request, active_run

configure_logging()
logger = logging.getLogger("support_agent.orchestrator")

ROUTES = ("ORDER", "REFUND", "PAYMENT", "FAQ", "ESCALATE")

ROUTER_PROMPT = """You are a customer-support router. Read the user's message and decide which
specialist should handle it. Respond with ONLY one word from this list:
- ORDER    : order status, tracking, order details
- REFUND   : refund requests, refund status, eligibility
- PAYMENT  : payment status, making a payment
- FAQ      : general policy questions (shipping times, return policy)
- ESCALATE : anything unclear, abusive, or outside the above

User message: {user_message}
Answer with one word only."""


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


def parse_route(text: str) -> str:
    text = text.strip().upper()
    hits = {r for r in ROUTES if re.search(r"\b" + r + r"\b", text)}
    return next(iter(hits)) if len(hits) == 1 else "ESCALATE"


def route_node(state: SupportState) -> dict:
    llm = get_llm()
    result = invoke_with_retry(lambda: llm.invoke(ROUTER_PROMPT.format(user_message=state["user_message"])))
    route = parse_route(as_text(result.content))
    return {"route": route, "usage": extract_usage(result)}


SPECIALISTS = {
    'ORDER': ('order', order_agent.SYSTEM_PROMPT, order_agent.ORDER_TOOLS),
    'REFUND': ('refund', refund_agent.SYSTEM_PROMPT, refund_agent.REFUND_TOOLS),
    'PAYMENT': ('payment', payment_agent.SYSTEM_PROMPT, payment_agent.PAYMENT_TOOLS),
    'FAQ': ('faq', faq_agent.SYSTEM_PROMPT, faq_agent.FAQ_TOOLS),
}


def prepare_node(state):
    name, prompt, _ = SPECIALISTS[state['route']]
    return base.prepare_agent(name, prompt, state['user_message'], state.get('history'))


def model_step(state):
    return base.model_step(state, SPECIALISTS[state['route']][2])


def tool_step(state):
    # Write tools consult the transaction's action receipt before doing work.
    # Re-running this node after a commit/checkpoint gap never repeats a write.
    return base.tool_step(state, SPECIALISTS[state['route']][2])


def finalize_step(state):
    return base.finalize_step(state)


def escalate_node(state: SupportState) -> dict:
    result = escalate_to_human(
        reason="Could not classify or resolve the request automatically",
        context=state["user_message"],
    )
    return {
        "reply": result["message"],
        "trace": [{"agent": "escalate", "tool": "escalate_to_human", "args": {}, "result": result}],
        "agent": "escalate",
        "usage": state.get("usage", {}),
    }


def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)
    graph.add_node("route", route_node)
    graph.add_node('prepare', prepare_node)
    graph.add_node('model', model_step)
    graph.add_node('tool', tool_step)
    graph.add_node('finalize', finalize_step)
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {route: 'prepare' for route in SPECIALISTS} | {'ESCALATE': 'escalate'},
    )
    graph.add_edge('prepare', 'model')
    graph.add_conditional_edges('model', lambda state: 'finalize' if state['done'] else 'tool')
    graph.add_conditional_edges('tool', lambda state: 'tool' if state['tool_index'] < len(state['tool_calls']) else 'model')
    graph.add_edge('finalize', END)
    graph.add_edge('escalate', END)

    return graph.compile(checkpointer=checkpointer)


_GRAPH = build_graph()


def run_orchestrator(user_message: str, history: list | None = None) -> dict:
    value = {'user_message': user_message, 'history': history or [], 'trace': []}
    run = active_run.get()
    if run is None:
        result = _GRAPH.invoke(value, {'recursion_limit': 64})
    else:
        context = active_request.get()
        binding = [require_principal().customer_id, context.mode, context.session_id, context.request_id]
        if binding != [run['user_id'], run['mode'], run['session_id'], run['request_id']]:
            raise AccessError('运行身份或会话不匹配。', 403)
        with closing(sqlite3.connect(run['checkpoint_path'], timeout=20, check_same_thread=False)) as connection:
            saver = SqliteSaver(connection)
            graph = build_graph(saver)
            config = {'configurable': {'thread_id': run['run_id']}, 'recursion_limit': 64}
            snapshot = graph.get_state(config)
            if snapshot.values:
                if snapshot.values.get('binding') != binding:
                    raise AccessError('运行身份或会话不匹配。', 403)
                from src import tools
                # A saved answer must not reveal an order whose owner changed.
                for entry in snapshot.values.get('trace', []):
                    if isinstance(entry.get('result'), dict) and entry['result'].get('error'):
                        continue
                    order_id = (entry.get('args') or {}).get('order_id')
                    if order_id is not None and not tools.can_access_order(order_id):
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

