"""单次请求的身份、模式、客户偏好、确认凭据与图运行上下文。"""
from contextvars import ContextVar
import logging

active_mode = ContextVar('active_mode', default='demo')
active_profile = ContextVar('active_profile', default={})
active_request = ContextVar('active_request', default=None)
active_principal = ContextVar('active_principal', default=None)
active_confirmation = ContextVar('active_confirmation', default=None)
active_run = ContextVar('active_run', default=None)
active_order_reference = ContextVar('active_order_reference', default=None)
active_task = ContextVar('active_task', default=None)
active_budget = ContextVar('active_budget', default=None)


def record_event(event, **fields):
    context = active_request.get()
    if context is not None:
        fields = {'request_id': context.request_id, 'session_id': context.session_id, **fields}
    logging.getLogger('support_agent.execution').info(event, extra={'extra_fields': fields})
