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
        try:
            result = _initiate_refund(s, order_id, reason)
        except IntegrityError:
            s.rollback()
            return {"success": False, "order_id": order_id, "reason": "A refund is already requested."}
        s.commit()
        return result


def _initiate_refund(s, order_id: int, reason: str) -> dict:
    """Apply a refund within the caller's transaction, without committing it."""
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
    else:
        refund = Refund(order_id=order_id, status="requested", reason=reason[:300])
        s.add(refund)
        s.flush()
    return {"success": True, "order_id": order_id, "refund_id": refund.refund_id, "status": "requested"}


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
    with _session() as s:
        result = _process_payment(s, order_id, method)
        s.commit()
        return result


def _process_payment(s, order_id: int, method: str) -> dict:
    """Apply a payment within the caller's transaction, without committing it."""
    if method not in {"card", "paypal"}:
        return {"success": False, "order_id": order_id, "reason": "Unsupported payment method."}
    payment = s.query(Payment).filter_by(order_id=order_id).first()
    if not payment:
        return {"error": f"No payment found for order {order_id}"}
    changed = s.execute(update(Payment).where(Payment.order_id == order_id, Payment.status == "pending").values(status="paid", method=method))
    if changed.rowcount != 1:
        return {"success": False, "order_id": order_id, "reason": "Only pending payments can be processed; it may already be paid or refunded."}
    return {"success": True, "order_id": order_id, "status": "paid", "method": method}


_FAQ = {
    "shipping": "问：配送需要多久？答：普通配送示例时效为3–5个工作日，并非实际预计送达时间。\n问：如何查看运输状态？答：请提供订单编号查询本地样例状态。\n问：能加急或修改配送吗？答：本系统尚未接入这些功能。",
    "return": "问：退款有什么条件？答：样例订单必须已送达且没有已有退款申请。\n问：未送达可以退款吗？答：当前样例规则不允许。\n问：有退货期限或包装要求吗？答：本系统未校验期限、包装或商品类别。",
    "cancel": "问：可以取消订单吗？答：本系统没有取消订单工具。\n问：已发货如何取消？答：当前不能办理取消或安排退货物流。\n问：查询取消政策是否代表已取消？答：不代表，订单状态不会因此改变。",
    "refund": "问：申请退款后就到账了吗？答：requested 只表示本地申请已记录。\n问：可以重复申请吗？答：已有申请时不能重复申请。\n问：什么时候到账？答：本系统未连接真实退款渠道，无法承诺到账时间。",
    "contact": "问：如何联系人工？答：系统可以生成本地待处理工单。\n问：工单会通知真人吗？答：尚未连接通知或接单渠道。\n问：客服邮箱是什么？答：support@example.com 仅为示例占位，不代表可联系的邮箱。",
}


@lru_cache(maxsize=128)
def answer_faq(question: str) -> str:
    """Answer a general policy question (shipping, returns, cancellations, refunds, contact info)."""
    return _FAQ.get(question, "没有找到对应的样例政策。")


def escalate_to_human(reason: str, context: str) -> dict:
    """Escalate the conversation to a human agent when the request can't be resolved automatically."""
    from src.runtime_context import active_request
    key = active_request.get()
    ticket_id = uuid.uuid5(uuid.NAMESPACE_URL, 'ticket:' + key).hex if key else uuid.uuid4().hex
    with _session() as s:
        if s.get(Ticket, ticket_id) is None:
            s.add(Ticket(ticket_id=ticket_id, reason=reason[:300], context=context[:1000], status="pending"))
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                if s.get(Ticket, ticket_id) is None:
                    raise
    logger.warning("ESCALATION ticket=%s", ticket_id)
    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "message": f"已生成本地待处理工单 #{ticket_id}。尚未连接人工通知渠道，不代表真人已接单。",
    }

