"""Tool functions the agents call. Each hits the SQLite backend (read or write).

Docstrings are written for the LLM to read when deciding which tool to call.
"""
import logging
import uuid
from functools import lru_cache
from sqlalchemy import select, update

from src.backend.db import Order, Payment, Refund, Ticket, get_engine, get_session_factory, init_db
from src.backend.seed import seed as _seed_data
from src.requests import RequestJournal
from src.runtime_context import active_request
from src.auth import require_principal

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


def request_journal():
    return RequestJournal(_SessionLocal, _engine)


ORDER_UNAVAILABLE = {'error': '未找到可访问的订单。'}


def get_owned_order(db, order_id):
    principal = require_principal()
    return db.scalars(select(Order).where(Order.order_id == order_id, Order.customer_id == principal.customer_id)).first()


def can_access_order(order_id):
    with _session() as db:
        return get_owned_order(db, order_id) is not None


def get_order_status(order_id: int) -> dict:
    """Look up the current status (placed/shipped/delivered/cancelled) of an order."""
    with _session() as s:
        order = get_owned_order(s, order_id)
        if not order:
            return dict(ORDER_UNAVAILABLE)
        return {"order_id": order.order_id, "status": order.status}


def get_order_details(order_id: int) -> dict:
    """Get full details of an order: customer, item, status, date, amount."""
    with _session() as s:
        order = get_owned_order(s, order_id)
        if not order:
            return dict(ORDER_UNAVAILABLE)
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
        order = get_owned_order(s, order_id)
        if not order:
            return dict(ORDER_UNAVAILABLE)
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
        order = get_owned_order(s, order_id)
        if not order:
            return dict(ORDER_UNAVAILABLE)
        return refund_eligibility(s, order)


def refund_eligibility(s, order):
    refund = s.query(Refund).filter_by(order_id=order.order_id).first()
    refund_status = refund.status if refund else 'none'
    if order.status != 'delivered':
        return {'order_id': order.order_id, 'eligible': False,
                'reason': f"Order status is '{order.status}'; only delivered orders are refund-eligible."}
    if refund_status != 'none':
        return {'order_id': order.order_id, 'eligible': False, 'reason': f"A refund is already '{refund_status}' for this order."}
    return {'order_id': order.order_id, 'eligible': True, 'reason': 'Order is delivered and has no existing refund.'}


def initiate_refund(order_id: int, reason: str) -> dict:
    """只生成本地退款提案；用户必须通过独立确认入口才能执行。"""
    from src.actions import propose_action
    return propose_action('initiate_refund', {'order_id': order_id, 'reason': reason[:300]})


def apply_refund(s, order_id, reason):
    from src.actions import require_confirmation
    require_confirmation('initiate_refund', {'order_id': order_id, 'reason': reason})
    order = get_owned_order(s, order_id)
    if not order:
        return dict(ORDER_UNAVAILABLE)
    if order.status != "delivered":
        return {"success": False, "order_id": order_id, "reason": "Only delivered orders are refund-eligible."}
    refund = s.query(Refund).filter_by(order_id=order_id).first()
    if refund:
        changed = s.execute(update(Refund).where(Refund.order_id == order_id, Refund.status == "none").values(status="requested", reason=reason[:300]))
        if changed.rowcount != 1:
            return {"success": False, "order_id": order_id, "reason": "A refund is already requested/approved/completed."}
    else:
        refund = Refund(order_id=order_id, status="requested", reason=reason[:300])
        s.add(refund)
        s.flush()
    order.version += 1
    return {"success": True, "order_id": order_id, "refund_id": refund.refund_id, "status": "requested"}


def get_payment_status(order_id: int) -> dict:
    """Look up the payment status (pending/paid/refunded) for an order."""
    with _session() as s:
        if get_owned_order(s, order_id) is None:
            return dict(ORDER_UNAVAILABLE)
        payment = s.query(Payment).filter_by(order_id=order_id).first()
        if not payment:
            return dict(ORDER_UNAVAILABLE)
        return {
            "order_id": order_id,
            "status": payment.status,
            "method": payment.method,
            "amount": payment.amount,
        }


def process_payment(order_id: int, method: str) -> dict:
    """只生成本地付款提案，不会直接改变支付状态。"""
    from src.actions import propose_action
    return propose_action('process_payment', {'order_id': order_id, 'method': method})


def apply_payment(s, order_id, method):
    from src.actions import require_confirmation
    require_confirmation('process_payment', {'order_id': order_id, 'method': method})
    order = get_owned_order(s, order_id)
    if order is None:
        return dict(ORDER_UNAVAILABLE)
    payment = s.query(Payment).filter_by(order_id=order_id).first()
    if not payment:
        return dict(ORDER_UNAVAILABLE)
    changed = s.execute(update(Payment).where(Payment.order_id == order_id, Payment.status == "pending").values(status="paid", method=method))
    if changed.rowcount != 1:
        return {"success": False, "order_id": order_id, "reason": "Only pending payments can be processed; it may already be paid or refunded."}
    order.version += 1
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
    require_principal()
    def apply(s):
        ticket_id = uuid.uuid4().hex[:8]
        s.add(Ticket(ticket_id=ticket_id, reason=reason[:300], context=context[:1000], status="pending"))
        return {"escalated": True, "ticket_id": ticket_id,
                "message": f"已生成本地待处理工单 #{ticket_id}。尚未连接人工通知渠道，不代表真人已接单。"}
    result = request_journal().execute_action_once(
        active_request.get(), 'escalate_to_human', {'reason': reason, 'context': context}, apply)
    logger.warning("ESCALATION ticket=%s", result['ticket_id'])
    return result

