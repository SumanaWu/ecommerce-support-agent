"""单次请求的客户偏好；不是认证上下文。"""
from contextvars import ContextVar

active_profile = ContextVar('active_profile', default={})
active_request = ContextVar('active_request', default=None)
