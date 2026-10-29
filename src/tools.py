"""Tool functions the agents call. Each hits the SQLite backend (read or write).

Docstrings are written for the LLM to read when deciding which tool to call.
"""
import logging
import os
import uuid
from sqlalchemy import select, update

from src.backend.db import Order, Payment, Refund, Ticket, get_engine, get_session_factory, init_db
from src.backend.seed import seed as _seed_data
from src.requests import RequestJournal
from src.runtime_context import active_request, active_task
from src.auth import require_principal
from src.money import money_view
from src.rules import evaluate_refund_policy
from src.policy import DEFAULT_POLICY_DIR, answer_with_sources, import_policy_documents

logger = logging.getLogger("support_agent.tools")

_engine = get_engine()
init_db(_engine)
_SessionLocal = get_session_factory(_engine)
import_policy_documents(os.environ.get('SUPPORT_POLICY_DIR') or DEFAULT_POLICY_DIR, _SessionLocal)


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
    import_policy_documents(os.environ.get('SUPPORT_POLICY_DIR') or DEFAULT_POLICY_DIR, _SessionLocal)


def _session():
    return _SessionLocal()


def request_journal():
    return RequestJournal(_SessionLocal, _engine)


ORDER_UNAVAILABLE = {'error': '未找到可访问的订单。'}


def get_owned_order(db, order_id):
    principal = require_principal()
    task = active_task.get()
    if task and task.get('pipeline_version') == 'online-v3' and order_id not in task['order_ids']:
        raise ValueError('订单超出当前任务范围。')
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
            **money_view(order),
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
    return evaluate_refund_policy(order, refund)


def initiate_refund(order_id: int, reason: str) -> dict:
    """只生成退款提案；用户必须通过独立确认入口才能执行。"""
    from src.actions import propose_action
    return propose_action('initiate_refund', {'order_id': order_id, 'reason': reason[:300]})


def apply_refund(s, order_id, reason):
    from src.actions import require_confirmation
    require_confirmation('initiate_refund', {'order_id': order_id, 'reason': reason})
    order = get_owned_order(s, order_id)
    if not order:
        return dict(ORDER_UNAVAILABLE)
    eligibility = refund_eligibility(s, order)
    if not eligibility['eligible']:
        return {**eligibility, 'success': False}
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
    return {"success": True, "order_id": order_id, "refund_id": refund.refund_id, "status": "requested",
            "rule_version": eligibility['rule_version'], **money_view(order)}


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
            **money_view(payment),
        }


def process_payment(order_id: int, method: str) -> dict:
    """只生成付款提案，不会直接改变支付状态。"""
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
    return {"success": True, "order_id": order_id, "status": "paid", "method": method, **money_view(payment)}


def answer_faq(question: str) -> dict:
    """Retrieve effective local policy excerpts with versions and sources; never execute policy prose."""
    return answer_with_sources(question, _SessionLocal)


def escalate_to_human(reason: str, context: str) -> dict:
    """Escalate the conversation to a human agent when the request can't be resolved automatically."""
    require_principal()
    from src.budget import check_execution_budget
    request = active_request.get()
    args = {'reason': reason, 'context': context}
    if request is None or request_journal().get_action_result(request, 'escalate_to_human:ticket', args) is None:
        check_execution_budget('tool', consume=True)
    def apply(s):
        ticket_id = uuid.uuid4().hex[:8]
        s.add(Ticket(ticket_id=ticket_id, reason=reason[:300], context=context[:1000], status="pending"))
        return {"escalated": True, "ticket_id": ticket_id,
                "message": f"已生成待处理工单 #{ticket_id}。"}
    result = request_journal().execute_action_once(
        active_request.get(), 'escalate_to_human', {'reason': reason, 'context': context}, apply)
    logger.warning("ESCALATION ticket=%s", result['ticket_id'])
    return result



def _read_batch(order_ids, read):
    if not isinstance(order_ids, list) or not order_ids or len(order_ids) > 20 or any(type(n) is not int or n <= 0 for n in order_ids):
        raise ValueError('批量查询只接受1至20个正整数订单编号。')
    return {'items': [{'order_id': number, 'result': read(number)} for number in dict.fromkeys(order_ids)]}


def get_order_status_batch(order_ids: list[int]) -> dict:
    return _read_batch(order_ids, get_order_status)


def get_order_details_batch(order_ids: list[int]) -> dict:
    return _read_batch(order_ids, get_order_details)


def track_shipments_batch(order_ids: list[int]) -> dict:
    return _read_batch(order_ids, track_shipment)


def get_payment_status_batch(order_ids: list[int]) -> dict:
    return _read_batch(order_ids, get_payment_status)
