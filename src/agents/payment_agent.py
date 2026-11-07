from langchain_core.tools import tool

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def get_payment_status(order_id: int) -> dict:
    """Look up the payment status (pending/paid/refunded) for an order."""
    return backend_tools.get_payment_status(order_id)


@tool
def prepare_payment(order_id: int) -> dict:
    """Open payment method selection for a pending order; this does not process it."""
    payment = backend_tools.get_payment_status(order_id)
    if payment.get('error'):
        return payment
    if payment.get('status') != 'pending':
        return {'error': 'Only pending payments can be processed.'}
    return {'pending_action': {'type': 'payment', 'order_id': order_id,
                               'amount': payment['amount']},
            'message': '请在界面选择支付方式，选择后再确认金额；尚未办理支付。'}


PAYMENT_TOOLS = [get_payment_status, prepare_payment]

SYSTEM_PROMPT = (
    "You are the Payment Agent for an e-commerce support system. "
    "You can check payment status and open payment method selection for a pending payment. "
    "Only process a payment that is currently pending; if it's already paid or refunded, explain that instead. "
    "Always call a tool to get real data before answering; never guess. "
    "The stored method is a recorded value, not a restriction on payment choices. "
    "When asked to pay, call prepare_payment with only the order_id, even if the user names a method. "
    "The user selects the method in the UI, then separately confirms payment. "
    "Do not choose a method, reject a method mentioned in conversation, or ask for its spelling. "
    "Reply concisely and naturally to the customer."
)


def run_payment_agent(user_message: str, history: list | None = None) -> dict:
    return run_tool_agent("payment", SYSTEM_PROMPT, PAYMENT_TOOLS, user_message, history)

