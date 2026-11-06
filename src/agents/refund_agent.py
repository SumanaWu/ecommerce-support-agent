from langchain_core.tools import tool

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def check_refund_eligibility(order_id: int) -> dict:
    """Check whether an order is eligible for a refund (rule: must be delivered, no existing refund)."""
    return backend_tools.check_refund_eligibility(order_id)


@tool
def prepare_refund(order_id: int, reason: str) -> dict:
    """Prepare a refund for user confirmation; this does not submit a refund."""
    eligibility = backend_tools.check_refund_eligibility(order_id)
    if not eligibility.get('eligible'):
        return eligibility
    order = backend_tools.get_order_details(order_id)
    return {'pending_action': {'type': 'refund', 'order_id': order_id,
                               'reason': reason[:300], 'amount': order['amount']},
            'message': '等待用户确认，尚未申请退款。'}


@tool
def get_order_status(order_id: int) -> dict:
    """Look up the current status of an order (useful context before discussing a refund)."""
    return backend_tools.get_order_status(order_id)


REFUND_TOOLS = [check_refund_eligibility, prepare_refund, get_order_status]

SYSTEM_PROMPT = (
    "You are the Refund Agent for an e-commerce support system. "
    "Before preparing any refund, you must call check_refund_eligibility first. "
    "Only call prepare_refund if the order is eligible; the action requires user confirmation. "
    "If it isn't eligible, explain why to the customer instead of attempting the refund. "
    "Always call a tool to get real data before answering; never guess. "
    "Reply concisely and naturally to the customer."
)


def run_refund_agent(user_message: str, history: list | None = None) -> dict:
    return run_tool_agent("refund", SYSTEM_PROMPT, REFUND_TOOLS, user_message, history)

