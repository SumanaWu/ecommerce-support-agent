"""Isolate imports too: tools initialize a database during test collection."""
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

_data = TemporaryDirectory(prefix='support-agent-tests-')
os.environ['SUPPORT_DATA_DIR'] = _data.name
os.environ['SUPPORT_AGENT_DB_URL'] = 'sqlite:///' + str(Path(_data.name) / 'business.sqlite')
os.environ['SUPPORT_ACCOUNT_TOKENS'] = json.dumps({'customer_a': 'test-token-a', 'customer_b': 'test-token-b'})
os.environ['LLM_FALLBACKS'] = '[]'


@pytest.fixture(autouse=True)
def forbid_live_model_http(monkeypatch):
    import httpx
    import httpx2
    def deny(*args, **kwargs):
        raise AssertionError('自动测试禁止真实 HTTP 请求；请使用 MockTransport。')
    async def deny_async(*args, **kwargs):
        deny()
    for module in (httpx, httpx2):
        monkeypatch.setattr(module.HTTPTransport, 'handle_request', deny)
        monkeypatch.setattr(module.AsyncHTTPTransport, 'handle_async_request', deny_async)


@pytest.fixture(autouse=True)
def trusted_test_principal():
    from src.auth import Principal, principal_scope
    with principal_scope(Principal('customer_a')):
        yield


def pytest_sessionfinish(session, exitstatus):
    from src import tools
    tools._engine.dispose()
    _data.cleanup()
