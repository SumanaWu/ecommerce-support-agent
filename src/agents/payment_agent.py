from langchain_core.tools import tool
from typing import Literal

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def get_payment_status(order_id: int) -> dict:
    """Look up the payment status (pending/paid/refunded) for an order."""
    return backend_tools.get_payment_status(order_id)


@tool
def process_payment(order_id: int, method: Literal['card', 'paypal']) -> dict:
    """Propose payment for an owned pending order. Does not execute; the user must click a separate confirmation."""
    return backend_tools.process_payment(order_id, method)


@tool
def get_payment_status_batch(order_ids: list[int]) -> dict:
    """Read payment status for up to 20 specified orders. Inaccessible and missing orders have the same result."""
    return backend_tools.get_payment_status_batch(order_ids)


PAYMENT_TOOLS = [get_payment_status, process_payment, get_payment_status_batch]

SYSTEM_PROMPT = (
    "You are the Payment Agent for an e-commerce support system. "
    "You can check payment status and process a pending payment. "
    "Only process a payment that is currently pending; if it's already paid or refunded, explain that instead. "
    "process_payment only creates a pending proposal. Never claim payment was executed before user confirmation. "
    "The payment method must come from the customer's current request or their answer to a pending clarification. "
    "A stored payment method is historical data, not authorization to choose it for this request. "
    "Always call a tool to get real data before answering; never guess. "
    "Reply concisely and naturally to the customer."
)


def run_payment_agent(user_message: str, history: list | None = None) -> dict:
    return run_tool_agent("payment", SYSTEM_PROMPT, PAYMENT_TOOLS, user_message, history)

