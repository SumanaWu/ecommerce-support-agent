import pytest
from concurrent.futures import ThreadPoolExecutor

from src import tools
from src.service import ConflictError, SupportService
from tests.test_actions import business


def runner(message, history=None):
    return {'reply': '待确认', 'agent': 'refund', 'usage': {'total_tokens': 10},
            'trace': [{'args': {'order_id': oid}, 'result': {'pending_action': {
                'type': 'refund', 'order_id': oid, 'reason': 'test'}}} for oid in [1001, 1006]]}


def test_service_restart_sequential_confirmation_and_replay(business, tmp_path):
    service = SupportService(tmp_path / 'memory.sqlite')
    first = service.chat('customer_a', 'one', '申请退款', runner=runner, request_id='chat')
    assert first['status'] == 'awaiting_confirmation'
    assert service.chat('customer_a', 'one', '申请退款', runner=lambda *a, **k: pytest.fail('replayed LLM'), request_id='chat') == first
    with pytest.raises(ConflictError):
        service.chat('customer_a', 'one', '新问题', runner=runner, request_id='other')
    service.close()
    service = SupportService(tmp_path / 'memory.sqlite')
    action = first['current_action']
    second = service.resume('customer_a', 'one', 'chat', 'confirm1', action['action_id'], 'confirm')
    assert second['status'] == 'awaiting_confirmation'
    assert second['current_action']['order_id'] == 1006
    assert service.resume('customer_a', 'one', 'chat', 'confirm1', action['action_id'], 'confirm') == second
    old = service.resume('customer_a', 'one', 'chat', 'anotherclick', action['action_id'], 'confirm')
    assert old['current_action']['order_id'] == 1006
    action = second['current_action']
    final = service.resume('customer_a', 'one', 'chat', 'cancel2', action['action_id'], 'cancel')
    assert final['status'] == 'completed'
    assert [a['status'] for a in final['actions']] == ['completed', 'cancelled']
    assert len(service.snapshot('customer_a', 'one')['turns']) == 1
    assert service.snapshot('customer_a', 'one')['total_tokens'] == 10
    assert not tools.check_refund_eligibility(1001)['eligible']
    assert tools.check_refund_eligibility(1006)['eligible']
    with pytest.raises(ConflictError):
        service.get_request('customer_b', 'one', 'chat')
    service.close()


def test_request_input_and_decision_conflicts(business, tmp_path):
    service = SupportService(tmp_path / 'memory.sqlite')
    result = service.chat('customer_a', 'one', '申请退款', runner=runner, request_id='chat')
    with pytest.raises(ConflictError):
        service.chat('customer_a', 'one', '不同输入', runner=runner, request_id='chat')
    action = result['current_action']
    service.resume('customer_a', 'one', 'chat', 'decision', action['action_id'], 'cancel')
    with pytest.raises(ConflictError):
        service.resume('customer_a', 'one', 'chat', 'decision', action['action_id'], 'confirm')
    service.close()


def test_lost_response_does_not_confirm_next_action(business, tmp_path, monkeypatch):
    service = SupportService(tmp_path / 'memory.sqlite')
    initial = service.chat('customer_a', 'one', '申请退款', runner=runner, request_id='chat')
    action = initial['current_action']
    original = service._graph_result
    def fail_save(*args, **kwargs):
        raise RuntimeError('lost response')
    monkeypatch.setattr(service, '_graph_result', fail_save)
    with pytest.raises(RuntimeError, match='lost response'):
        service.resume('customer_a', 'one', 'chat', 'decision', action['action_id'], 'confirm')
    monkeypatch.setattr(service, '_graph_result', original)
    recovered = service.resume('customer_a', 'one', 'chat', 'decision', action['action_id'], 'confirm')
    assert recovered['current_action']['order_id'] == 1006
    assert tools.check_refund_eligibility(1006)['eligible']
    service.close()


def test_concurrent_chat_replays_one_model_result(business, tmp_path):
    service = SupportService(tmp_path / 'memory.sqlite')
    calls = []
    def read_only(message, history=None):
        calls.append(message)
        return {'reply': 'ok', 'agent': 'order', 'trace': [], 'usage': {'total_tokens': 5}}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: service.chat('customer_a', 'one', '查询订单',
                                runner=read_only, request_id='chat'), range(4)))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
    assert len(service.snapshot('customer_a', 'one')['turns']) == 1
    assert service.snapshot('customer_a', 'one')['total_tokens'] == 5
    service.close()


def test_resume_retries_unconsumed_decision_immediately(business, tmp_path, monkeypatch):
    service = SupportService(tmp_path / 'memory.sqlite')
    initial = service.chat('customer_a', 'one', '申请退款', runner=runner, request_id='chat')
    original = service.graph.invoke
    calls = []
    def flaky(value, config):
        calls.append(value)
        if len(calls) == 1:
            raise RuntimeError('before resume')
        return original(value, config)
    monkeypatch.setattr(service.graph, 'invoke', flaky)
    action = initial['current_action']
    result = service.resume('customer_a', 'one', 'chat', 'decision', action['action_id'], 'confirm')
    assert len(calls) == 2
    assert result['current_action']['order_id'] == 1006
    assert not tools.check_refund_eligibility(1001)['eligible']
    service.close()
