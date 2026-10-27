"""Local test-account tokens. Principal objects are created by trusted entrypoints."""
from contextlib import contextmanager
from dataclasses import dataclass
import hmac
import json
import os
import re
from dotenv import load_dotenv

from src.runtime_context import active_principal

load_dotenv()


class AccessError(ValueError):
    def __init__(self, message='需要有效的测试账户令牌。', status_code=401):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Principal:
    customer_id: str


def require_principal(principal=None):
    principal = principal if principal is not None else active_principal.get()
    if not isinstance(principal, Principal) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', principal.customer_id):
        raise AccessError()
    return principal


def resolve_principal(token):
    try:
        accounts = json.loads(os.environ.get('SUPPORT_ACCOUNT_TOKENS') or '{}')
        if not isinstance(accounts, dict) or not accounts:
            raise ValueError()
        values = list(accounts.values())
        if any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError()
        for customer_id in accounts:
            require_principal(Principal(customer_id))
    except (ValueError, TypeError):
        raise AccessError('请在 .env 配置有效的 SUPPORT_ACCOUNT_TOKENS 测试账户映射。', 503) from None
    if not isinstance(token, str) or not token:
        raise AccessError()
    matched = None
    for customer_id, expected in accounts.items():
        if hmac.compare_digest(token.encode(), expected.encode()):
            matched = Principal(customer_id)
    if matched is None:
        raise AccessError()
    return matched


@contextmanager
def principal_scope(principal):
    token = active_principal.set(require_principal(principal))
    try:
        yield principal
    finally:
        active_principal.reset(token)
