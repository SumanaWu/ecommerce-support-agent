from fastapi.testclient import TestClient

import api
from src.service import SupportService
from tests.test_actions import business
from tests.test_flow import runner


def test_http_confirmation_and_idempotency(business, tmp_path, monkeypatch):
    service = SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(api, 'service', service)
    monkeypatch.setattr(api, 'run_orchestrator', runner)
    client = TestClient(api.app)
    request = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'chat', 'message': '申请退款'}
    initial = client.post('/chat', json=request)
    assert initial.status_code == 200
    assert client.post('/chat', json=request).json() == initial.json()
    action = initial.json()['current_action']
    payload = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'decision',
               'action_id': action['action_id'], 'decision': 'confirm'}
    result = client.post('/requests/chat/resume', json=payload)
    assert result.status_code == 200
    assert result.json()['current_action']['order_id'] == 1006
    assert client.post('/requests/chat/resume', json=payload).json() == result.json()
    assert client.get('/requests/chat', params={'user_id': 'customer_b', 'session_id': 'one'}).status_code == 409
    assert client.post('/chat', json={**request, 'message': 'different'}).status_code == 409
    service.close()
