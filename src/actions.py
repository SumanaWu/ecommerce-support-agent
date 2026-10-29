"""Model tools propose; authenticated entrypoints alone confirm or cancel."""
from dataclasses import dataclass
import json
import time
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from src.auth import AccessError, principal_scope, require_principal
from src.backend.db import Payment, PendingAction, Refund
from src.requests import canonical
from src.money import money_view
from src.rules import business_fingerprint
from src.runtime_context import STORAGE_MODE, active_confirmation, active_request, active_order_reference, active_task


@dataclass(frozen=True)
class ConfirmationGrant:
    action_id: str
    user_id: str
    tool: str
    args: str


def require_confirmation(tool, args):
    principal = require_principal()
    grant = active_confirmation.get()
    if not isinstance(grant, ConfirmationGrant) or grant.user_id != principal.customer_id or grant.tool != tool or grant.args != canonical(args):
        raise AccessError('写操作需要有效的用户确认。', 403)


def business_version(db, order):
    payment = db.scalars(select(Payment).where(Payment.order_id == order.order_id)).first()
    refund = db.scalars(select(Refund).where(Refund.order_id == order.order_id)).first()
    return business_fingerprint(vars(order), vars(payment) if payment else None, vars(refund) if refund else None)


def action_view(row):
    return {'action_id': row.action_id, 'order_id': row.order_id, 'tool': row.tool,
            **money_view(row), 'args': json.loads(row.args), 'status': row.status,
            'session_id': row.session_id,
            'result': json.loads(row.result) if row.result else None}


def _owned_action(db, action_id):
    from src import tools
    principal = require_principal()
    row = db.get(PendingAction, action_id)
    if row is None or row.mode != STORAGE_MODE or row.user_id != principal.customer_id or tools.get_owned_order(db, row.order_id) is None:
        raise AccessError('未找到可访问的动作。', 404)
    return row


def get_action(principal, action_id):
    from src import tools
    with principal_scope(principal), tools._session() as db:
        return action_view(_owned_action(db, action_id))


def list_actions(principal, *, session_id=None, source_request_id=None, pending_only=True):
    from src import tools
    principal = require_principal(principal)
    with principal_scope(principal), tools._session() as db:
        query = select(PendingAction).where(PendingAction.user_id == principal.customer_id, PendingAction.mode == STORAGE_MODE)
        if session_id is not None:
            query = query.where(PendingAction.session_id == session_id)
        if source_request_id is not None:
            query = query.where(PendingAction.source_request_id == source_request_id)
        if pending_only:
            query = query.where(PendingAction.status == 'pending')
        return [action_view(row) for row in db.scalars(query.order_by(PendingAction.created_at))
                if tools.get_owned_order(db, row.order_id) is not None]


def propose_action(tool, args):
    from src import tools
    principal = require_principal()
    context = active_request.get()
    if context is None or context.user_id != principal.customer_id:
        raise AccessError('请通过已认证的客服请求创建待确认动作。', 403)
    task = active_task.get()
    if task is not None and (not task['allow_write'] or
            tool != {'REFUND': 'initiate_refund', 'PAYMENT': 'process_payment'}.get(task['route'])):
        raise AccessError('本任务没有用户明确的办理请求，只能查询；模型不能自行生成写入提案。', 403)
    reference = active_order_reference.get()
    if reference is not None and (reference['status'] != 'resolved' or reference['order_id'] != args['order_id']):
        raise AccessError('本轮订单尚未明确或与已确定订单不一致，请先澄清或更正当前订单。', 403)

    def propose(db):
        order = tools.get_owned_order(db, args['order_id'])
        if order is None:
            return dict(tools.ORDER_UNAVAILABLE)
        price = order
        if tool == 'initiate_refund':
            eligibility = tools.refund_eligibility(db, order)
            if not eligibility['eligible']:
                return {**eligibility, 'success': False}
        else:
            payment = db.scalars(select(Payment).where(Payment.order_id == order.order_id)).first()
            if payment is None or payment.status != 'pending':
                return {'success': False, 'order_id': order.order_id, 'reason': 'Only pending payments can be processed; it may already be paid or refunded.'}
            if args['method'] not in {'card', 'paypal'}:
                return {'success': False, 'order_id': order.order_id, 'reason': 'Unsupported payment method.'}
            price = payment
        action_id = uuid5(NAMESPACE_URL, canonical([*context.key, tool, order.order_id])).hex
        row = PendingAction(action_id=action_id, user_id=principal.customer_id, mode=context.mode,
                            session_id=context.session_id, source_request_id=context.request_id,
                            tool=tool, order_id=order.order_id, args=canonical(args),
                            business_version=business_version(db, order), amount_minor=price.amount_minor, currency=price.currency,
                            status='pending', created_at=time.time())
        db.add(row)
        return {'order_id': order.order_id, 'action_id': action_id, 'status': 'pending_confirmation',
                'requires_confirmation': True, 'message': '已生成待确认动作；请检查订单和金额后点击确认，尚未办理。'}

    return tools.request_journal().execute_action_once(context, tool, args, propose)


def validate_action_preconditions(db, row):
    from src import tools
    order = tools.get_owned_order(db, row.order_id)
    return order is not None and business_version(db, order) == row.business_version


def decide_action(action_id, *, confirm):
    from src import tools
    principal = require_principal()
    context = active_request.get()
    with tools._session() as db:
        row = _owned_action(db, action_id)
        tool, args = row.tool, json.loads(row.args)

    def apply(db):
        row = _owned_action(db, action_id)
        if row.status == 'executed':
            return json.loads(row.result) if confirm else {
                'success': False, 'order_id': row.order_id, 'action_id': action_id,
                'error_code': 'already_executed', 'reason': '已执行的动作不能取消。'}
        if row.status == 'cancelled':
            return {'success': not confirm, 'status': 'cancelled', 'order_id': row.order_id,
                    'action_id': action_id, 'error_code': 'cancelled' if confirm else None,
                    'message': '动作已取消，未执行业务操作。'}
        if row.status == 'rejected':
            return json.loads(row.result)
        if not confirm:
            row.status = 'cancelled'
            result = {'success': True, 'status': 'cancelled', 'order_id': row.order_id,
                      'action_id': action_id, 'message': '动作已取消，未执行业务操作。'}
        elif not validate_action_preconditions(db, row):
            row.status = 'rejected'
            result = {'success': False, 'order_id': row.order_id, 'action_id': action_id,
                      'error_code': 'stale_action', 'reason': '订单数据或业务状态已变化，请重新查询并创建动作。'}
        else:
            grant = active_confirmation.set(ConfirmationGrant(action_id, principal.customer_id, tool, canonical(args)))
            try:
                result = tools.apply_refund(db, **args) if tool == 'initiate_refund' else tools.apply_payment(db, **args)
            finally:
                active_confirmation.reset(grant)
            result = {**result, 'action_id': action_id}
            row.status = 'executed' if result.get('success') else 'rejected'
        row.result = canonical(result)
        return result

    receipt_tool = tool if confirm else 'cancel_action'
    receipt_args = args if confirm else {'order_id': args['order_id'], 'action_id': action_id}
    result = tools.request_journal().execute_action_once(context, receipt_tool, receipt_args, apply)
    agent = 'refund' if tool == 'initiate_refund' else 'payment'
    return {'agent': agent, 'tool': receipt_tool, 'args': receipt_args, 'result': result}
