"""单次请求的偏好、请求编号与本地客户访问范围；不提供登录认证。"""
from contextvars import ContextVar

active_profile = ContextVar('active_profile', default={})
active_request = ContextVar('active_request', default=None)

active_customer = ContextVar('active_customer', default=None)
