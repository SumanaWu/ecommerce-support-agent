from langchain_core.tools import tool

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def get_order_status(order_id: int) -> dict:
    """Look up the current status (placed/shipped/delivered/cancelled) of an order."""
    return backend_tools.get_order_status(order_id)


@tool
def get_order_details(order_id: int) -> dict:
    """Get full details of an order: customer, item, status, date, amount."""
    return backend_tools.get_order_details(order_id)


@tool
def track_shipment(order_id: int) -> dict:
    """Get shipment tracking info for an order based on its current status."""
    return backend_tools.track_shipment(order_id)


@tool
def get_order_status_batch(order_ids: list[int]) -> dict:
    """Read current status for up to 20 specified orders. Inaccessible and missing orders have the same result."""
    return backend_tools.get_order_status_batch(order_ids)


@tool
def get_order_details_batch(order_ids: list[int]) -> dict:
    """Read details for up to 20 specified orders. Inaccessible and missing orders have the same result."""
    return backend_tools.get_order_details_batch(order_ids)


@tool
def track_shipments_batch(order_ids: list[int]) -> dict:
    """Read shipment tracking for up to 20 specified orders. Inaccessible and missing orders have the same result."""
    return backend_tools.track_shipments_batch(order_ids)


ORDER_TOOLS = [get_order_status, get_order_details, track_shipment, get_order_status_batch, get_order_details_batch, track_shipments_batch]

SYSTEM_PROMPT = (
    "You are the Order Agent for an e-commerce support system. "
    "You can look up order status, full order details, and shipment tracking. "
    "Always call a tool to get real data before answering; never guess. "
    "Reply concisely and naturally to the customer."
)


def run_order_agent(user_message: str, history: list | None = None) -> dict:
    return run_tool_agent("order", SYSTEM_PROMPT, ORDER_TOOLS, user_message, history)

