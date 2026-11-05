"""Tool functions the agents call. Each hits the SQLite backend (read or write).

Docstrings are written for the LLM to read when deciding which tool to call.
"""
import logging
import uuid
from functools import lru_cache
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from src.backend.db import Order, Payment, Refund, Ticket, get_engine, get_session_factory, init_db
from src.backend.seed import seed as _seed_data

logger = logging.getLogger("support_agent.tools")

_engine = get_engine()
init_db(_engine)
_SessionLocal = get_session_factory(_engine)


def _ensure_seeded() -> None:
    """Deployed environments (e.g. Streamlit Cloud) only ever run app.py, with
    no separate seed step - so seed on first import if the DB is empty."""
    with _SessionLocal() as s:
        if s.query(Order).count() == 0:
            _seed_data(s)
            logger.info("Auto-seeded empty database on startup.")


_ensure_seeded()


def configure_db(engine) -> None:
    """Point tools at a different engine (used by tests)."""
    global _engine, _SessionLocal
    _engine = engine
    init_db(_engine)
    _SessionLocal = get_session_factory(_engine)


def _session():
    return _SessionLocal()


def get_order_status(order_id: int) -> dict:
    """Look up the current status (placed/shipped/delivered/cancelled) of an order."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        return {"order_id": order.order_id, "status": order.status}


def get_order_details(order_id: int) -> dict:
    """Get full details of an order: customer, item, status, date, amount."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        return {
            "order_id": order.order_id,
            "customer_name": order.customer_name,
            "item": order.item,
            "status": order.status,
            "order_date": order.order_date.isoformat(),
            "amount": order.amount,
        }


def track_shipment(order_id: int) -> dict:
    """Get shipment tracking info for an order based on its current status."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        messages = {
            "placed": "Order placed, not yet shipped.",
            "shipped": "In transit.",
            "delivered": "Delivered.",
            "cancelled": "Order was cancelled; no shipment in progress.",
        }
        return {
            "order_id": order.order_id,
            "status": order.status,
            "tracking_message": messages.get(order.status, "Unknown status."),
        }


def check_refund_eligibility(order_id: int) -> dict:
    """Check whether an order is eligible for a refund. Rule: order must be
    'delivered' and must not already have a requested/approved/completed refund."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        refund = s.query(Refund).filter_by(order_id=order_id).first()
        refund_status = refund.status if refund else "none"

        if order.status != "delivered":
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": f"Order status is '{order.status}'; only delivered orders are refund-eligible.",
            }
        if refund_status != "none":
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": f"A refund is already '{refund_status}' for this order.",
            }
        return {"order_id": order_id, "eligible": True, "reason": "Order is delivered and has no existing refund."}


def initiate_refund(order_id: int, reason: str) -> dict:
    """申请本地样例退款；工具内重查规则，用条件更新避免同订单重复申请。"""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        if order.status != "delivered":
            return {"success": False, "order_id": order_id, "reason": "Only delivered orders are refund-eligible."}
        refund = s.query(Refund).filter_by(order_id=order_id).first()
        if refund:
            changed = s.execute(update(Refund).where(Refund.order_id == order_id, Refund.status == "none").values(status="requested", reason=reason[:300]))
            if changed.rowcount != 1:
                return {"success": False, "order_id": order_id, "reason": "A refund is already requested/approved/completed."}
            refund_id = refund.refund_id
        else:
            refund = Refund(order_id=order_id, status="requested", reason=reason[:300])
            s.add(refund)
            try:
                s.flush()
            except IntegrityError:
                s.rollback()
                return {"success": False, "order_id": order_id, "reason": "A refund is already requested."}
            refund_id = refund.refund_id
        s.commit()
        return {"success": True, "order_id": order_id, "refund_id": refund_id, "status": "requested"}


def get_payment_status(order_id: int) -> dict:
    """Look up the payment status (pending/paid/refunded) for an order."""
    with _session() as s:
        payment = s.query(Payment).filter_by(order_id=order_id).first()
        if not payment:
            return {"error": f"No payment found for order {order_id}"}
        return {
            "order_id": order_id,
            "status": payment.status,
            "method": payment.method,
            "amount": payment.amount,
        }


def process_payment(order_id: int, method: str) -> dict:
    """仅把本地 pending 支付状态改为 paid；不连接支付网关。"""
    if method not in {"card", "paypal"}:
        return {"success": False, "order_id": order_id, "reason": "Unsupported payment method."}
    with _session() as s:
        payment = s.query(Payment).filter_by(order_id=order_id).first()
        if not payment:
            return {"error": f"No payment found for order {order_id}"}
        changed = s.execute(update(Payment).where(Payment.order_id == order_id, Payment.status == "pending").values(status="paid", method=method))
        if changed.rowcount != 1:
            return {"success": False, "order_id": order_id, "reason": "Only pending payments can be processed; it may already be paid or refunded."}
        s.commit()
        return {"success": True, "order_id": order_id, "status": "paid", "method": method}


_FAQ = {
    "shipping": "Standard shipping takes 3-5 business days. Expedited shipping (1-2 days) is available at checkout.",
    "return": "本地样例退款规则：订单已送达且不存在已有退款申请。本示例不校验退货期限、商品类别或包装状态，不能当作真实商家的退货政策。",
    "cancel": "Orders can be cancelled free of charge before they ship. Once shipped, they must be returned instead.",
    "refund": "Refunds are issued to the original payment method within 5-7 business days of approval.",
    "contact": "You can reach human support at support@example.com, Mon-Fri 9am-6pm.",
}


@lru_cache(maxsize=128)
def answer_faq(question: str) -> str:
    """Answer a general policy question (shipping, returns, cancellations, refunds, contact info)."""
    q = question.lower()
    for alias, keyword in [("配送", "shipping"), ("物流", "shipping"), ("快递", "shipping"), ("退货", "return"), ("退款规则", "return"), ("取消", "cancel"), ("退款到账", "refund"), ("联系", "contact")]:
        if alias in q:
            q = keyword
            break
    for keyword, answer in _FAQ.items():
        if keyword in q:
            return answer
    return (
        "I don't have a specific policy answer for that. "
        "You can reach human support at support@example.com for more detail."
    )


def escalate_to_human(reason: str, context: str) -> dict:
    """Escalate the conversation to a human agent when the request can't be resolved automatically."""
    ticket_id = uuid.uuid4().hex[:8]
    with _session() as s:
        s.add(Ticket(ticket_id=ticket_id, reason=reason[:300], context=context[:1000], status="pending"))
        s.commit()
    logger.warning("ESCALATION ticket=%s", ticket_id)
    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "message": f"已生成本地待处理工单 #{ticket_id}。尚未连接人工通知渠道，不代表真人已接单。",
    }

