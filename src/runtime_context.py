"""单次请求的运行模式与客户偏好；不是认证上下文。"""
from contextvars import ContextVar

active_mode = ContextVar('active_mode', default='demo')
active_profile = ContextVar('active_profile', default={})
