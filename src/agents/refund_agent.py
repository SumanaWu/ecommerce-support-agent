from langchain_core.tools import tool
from typing import Annotated
from pydantic import Field

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def check_refund_eligibility(order_id: int) -> dict:
    """Check whether an order is eligible for a refund (rule: must be delivered, no existing refund)."""
    return backend_tools.check_refund_eligibility(order_id)


@tool
def initiate_refund(order_id: int, reason: Annotated[str, Field(min_length=1, max_length=300,
                    description='客户申请退款的原始原因，已提供时逐字保留；仅未提供时使用默认未提供原因说明。')]) -> dict:
    """Propose a refund for an eligible owned order. Does not execute; the user must click a separate confirmation."""
    return backend_tools.initiate_refund(order_id, reason)


@tool
def get_order_status(order_id: int) -> dict:
    """Look up the current status of an order (useful context before discussing a refund)."""
    return backend_tools.get_order_status(order_id)


REFUND_TOOLS = [check_refund_eligibility, initiate_refund, get_order_status]

SYSTEM_PROMPT = (
    "You are the Refund Agent for an e-commerce support system. "
    "Before initiating any refund, you must call check_refund_eligibility first. "
    "Only call initiate_refund if the order is eligible (delivered, no existing refund). "
    "initiate_refund only creates a pending proposal. Never claim a refund was applied before user confirmation. "
    "If it isn't eligible, explain why to the customer instead of attempting the refund. "
    "Always call a tool to get real data before answering; never guess. "
    "The task already specifies the order. Query its eligibility; do not ask the customer to supply database facts. "
    "If no refund reason was given, use 用户申请退款（未提供具体原因）. "
    "Never invent a product defect or use the eligibility explanation as the customer's reason. "
    "Reply concisely and naturally to the customer."
)


def run_refund_agent(user_message: str, history: list | None = None) -> dict:
    return run_tool_agent("refund", SYSTEM_PROMPT, REFUND_TOOLS, user_message, history)

