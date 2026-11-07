"""Persistent, transaction-bound business action preparation and execution."""
import json
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select, text

from src import tools
from src.backend.db import Payment, PendingAction
from src.payment_methods import PAYMENT_METHODS


def _amount(value):
    return Decimal(str(value)).quantize(Decimal('0.01'))


def _lock_transaction(session):
    if session.bind.dialect.name == 'sqlite':
        session.execute(text('BEGIN IMMEDIATE'))


def _serialize(row):
    return {'action_id': row.action_id, 'request_id': row.request_id,
            'order_id': row.order_id, 'type': row.kind,
            'amount': format(_amount(row.amount), '.2f'),
            'status': row.status, 'parameters': json.loads(row.parameters),
            'result': json.loads(row.result) if row.result else None}


def prepare_action(user_id, session_id, request_id, position, proposal):
    """Store a stable proposal, reading its amount from the business database."""
    kind = proposal.get('type')
    order_id = proposal.get('order_id')
    if kind not in {'refund', 'payment'} or not isinstance(order_id, int):
        raise ValueError('无效操作方案')
    if kind == 'payment':
        method = proposal.get('method')
        if method not in PAYMENT_METHODS:
            raise ValueError('不支持的支付方式')
        params = {'method': method}
    else:
        reason = proposal.get('reason')
        if not isinstance(reason, str):
            raise ValueError('退款原因无效')
        params = {'reason': reason[:300]}
    parameters = json.dumps(params, sort_keys=True, ensure_ascii=False)
    action_id = uuid5(NAMESPACE_URL, json.dumps([user_id, request_id, position])).hex
    with tools._session() as session, session.begin():
        _lock_transaction(session)
        order = tools.owned_order(session, order_id, user_id)
        if not order:
            raise ValueError('订单不存在或无权访问')
        existing = session.get(PendingAction, action_id)
        if existing:
            if (existing.user_id, existing.session_id, existing.kind, existing.order_id,
                existing.parameters) != (user_id, session_id, kind, order_id, parameters):
                raise ValueError('请求编号已用于其他操作方案')
            return _serialize(existing)
        if kind == 'refund':
            eligibility = tools.refund_eligibility(session, order_id, user_id)
            if not eligibility.get('eligible'):
                raise ValueError(eligibility.get('reason', eligibility.get('error')))
            amount = float(eligibility['refundable_amount'])
        else:
            payment = session.scalar(select(Payment).where(Payment.order_id == order_id))
            if not payment or payment.status != 'pending':
                raise ValueError('订单不符合支付条件')
            amount = payment.amount
        row = PendingAction(action_id=action_id, request_id=request_id, user_id=user_id,
                            session_id=session_id, position=position, kind=kind,
                            order_id=order_id, amount=amount, parameters=parameters,
                            status='pending')
        session.add(row)
        session.flush()
        return _serialize(row)


def decide_action(user_id, session_id, action_id, decision):
    """Atomically confirm or cancel one saved action; repeated decisions replay its result."""
    if decision not in {'confirm', 'cancel'}:
        raise ValueError('无效确认决定')
    with tools._session() as session, session.begin():
        _lock_transaction(session)
        row = session.scalar(select(PendingAction).where(PendingAction.action_id == action_id).with_for_update())
        if not row or (row.user_id, row.session_id) != (user_id, session_id):
            raise ValueError('待确认操作不存在')
        if not tools.owned_order(session, row.order_id, user_id):
            raise ValueError('订单不存在或无权访问')
        if row.status != 'pending':
            return _serialize(row)
        if decision == 'cancel':
            row.status = 'cancelled'
            row.result = json.dumps({'cancelled': True})
            return _serialize(row)
        payment = session.scalar(select(Payment).where(Payment.order_id == row.order_id))
        eligibility = tools.refund_eligibility(session, row.order_id, user_id) if row.kind == 'refund' else {}
        eligible = eligibility.get('eligible') if row.kind == 'refund' else payment and payment.status == 'pending'
        if not eligible:
            row.status = 'rejected'
            row.result = json.dumps({**eligibility, 'success': False, 'reason': eligibility.get('reason', '业务状态已变化')}, ensure_ascii=False)
            return _serialize(row)
        params = json.loads(row.parameters)
        if row.kind == 'refund':
            token = tools.active_customer.set(user_id)
            try:
                result = tools._initiate_refund(session, row.order_id, params['reason'])
            finally:
                tools.active_customer.reset(token)
        else:
            token = tools.active_customer.set(user_id)
            try:
                result = tools._process_payment(session, row.order_id, params['method'])
            finally:
                tools.active_customer.reset(token)
        row.status = 'completed' if result.get('success') else 'rejected'
        row.result = json.dumps(result, ensure_ascii=False)
        return _serialize(row)
