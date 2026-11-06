from typing import Literal

from langchain_core.tools import tool

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def answer_faq(topic: Literal['shipping', 'return', 'cancel', 'refund', 'contact']) -> str:
    """Get the sample policy for shipping, returns, cancellations, refunds, or contact."""
    return backend_tools.answer_faq(topic)


FAQ_TOOLS = [answer_faq]

SYSTEM_PROMPT = (
    "You are the FAQ Agent for an e-commerce support system, answering general policy questions "
    "about shipping, returns, cancellations, refunds, and how to contact support. "
    "Always call answer_faq to get the official policy text before answering; never guess. "
    "Reply concisely and naturally to the customer."
)

# 只缓存纯政策查询，不缓存含客户上下文的完整模型回答。
def run_faq_agent(user_message: str, history: list | None = None) -> dict:
    result = run_tool_agent("faq", SYSTEM_PROMPT, FAQ_TOOLS, user_message, history)
    return {**result, "cached": False}
