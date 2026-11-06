from pathlib import Path

import pytest
from sqlalchemy import update, select
from streamlit.testing.v1 import AppTest
from fastapi.testclient import TestClient

from src import tools, actions
from src.backend.db import Order, Payment, Refund, PendingAction
from src.runtime_context import active_customer
from src.service import SupportService, ConflictError
from tests.test_actions import business


@pytest.fixture
def svc(business, tmp_path):
    service = SupportService(tmp_path / 'memory.sqlite')
    yield service
    service.close()


def refund_runner(message, history=None):
    return {'agent': 'refund', 'refund_intent': 'query' if '能' in message else 'apply',
            'reply': '', 'trace': [], 'usage': {}}


def ask(svc, text, user='customer_a', request=None):
    return svc.chat(user, 'one', text, runner=refund_runner, request_id=request)


def test_scope_every_tool_and_missing_identity(business):
    for function in [tools.get_order_details, tools.get_order_status, tools.track_shipment,
                     tools.check_refund_eligibility, tools.get_payment_status]:
        result = function(1009)
        assert result['reason_code'] == 'ORDER_NOT_ACCESSIBLE'
        assert 'amount' not in result and 'status' not in result and 'customer_name' not in result
    assert tools.initiate_refund(1009, 'test')['reason_code'] == 'ORDER_NOT_ACCESSIBLE'
    assert tools.process_payment(1008, 'card')['reason_code'] == 'ORDER_NOT_ACCESSIBLE'
    token = active_customer.set(None)
    try:
        assert tools.get_order_details(1001)['reason_code'] == 'ORDER_NOT_ACCESSIBLE'
    finally:
        active_customer.reset(token)
    with pytest.raises(ValueError):
        actions.prepare_action('customer_b', 'one', 'bad', 0, {'type': 'refund', 'order_id': 1006, 'reason': 'x'})


@pytest.mark.parametrize('oid,code', [(1001, 'ELIGIBLE'), (1002, 'STATUS_NOT_ELIGIBLE'),
    (1003, 'UNPAID'), (1004, 'REFUND_IN_PROGRESS'), (1005, 'ALREADY_REFUNDED')])
def test_reason_precedence(business, oid, code):
    assert tools.check_refund_eligibility(oid)['reason_code'] == code


def test_paid_placed_and_zero_balance(business):
    with tools._session() as session, session.begin():
        session.execute(update(Payment).where(Payment.order_id == 1003).values(status='paid'))
    assert tools.check_refund_eligibility(1003)['eligible']
    assert tools.initiate_refund(1003, 'x')['success']
    with tools._session() as session, session.begin():
        session.execute(update(Payment).where(Payment.order_id == 1006).values(amount=0))
    assert tools.check_refund_eligibility(1006)['reason_code'] == 'NO_REFUNDABLE_BALANCE'


def test_batch_partial_success_and_restart(svc):
    result = ask(svc, '退款1006和1005', request='batch')
    assert result['status'] == 'awaiting_confirmation'
    assert [i['stage'] for i in result['order_results']] == ['待确认', '不符合条件']
    assert result['order_results'][1]['reason_code'] == 'ALREADY_REFUNDED'
    assert len(result['trace']) == 3
    ask(svc, '退款1009', user='customer_b', request='other')
    restored = SupportService(svc.store.path)
    try:
        current = restored.get_request('customer_a', 'one', 'batch')['current_action']
        final = restored.resume('customer_a', 'one', 'batch', 'confirm', current['action_id'], 'confirm')
        assert final['order_results'][0]['stage'] == '已提交申请'
        assert final['order_results'][1]['reason_code'] == 'ALREADY_REFUNDED'
        assert '已完成退款' in final['reply']
    finally:
        restored.close()


def test_selection_candidates_and_replay(svc):
    def orders(message, history=None):
        return {'reply': 'orders', 'agent': 'order', 'trace': [
            {'args': {'order_id': i}, 'result': tools.get_order_details(i)} for i in [1001, 1006]], 'usage': {}}
    svc.chat('customer_a', 'one', '查询1001和1006', runner=orders)
    result = ask(svc, '退款', request='select')
    assert result['status'] == 'awaiting_selection'
    assert result['selection']['recent_order_ids'] == [1001, 1006]
    assert [o['order_id'] for o in result['selection']['candidates']] == [1001, 1006, 1002, 1003, 1004, 1005, 1007]
    assert svc.snapshot('customer_a', 'one')['last_order_id'] is None
    with tools._session() as session:
        assert not session.scalars(select(PendingAction)).all()
    with pytest.raises(ValueError):
        svc.select_orders('customer_a', 'one', 'select', 'zero', 'submit', [])
    with pytest.raises(ConflictError):
        svc.select_orders('customer_a', 'one', 'select', 'foreign', 'submit', [1009])
    prepared = svc.select_orders('customer_a', 'one', 'select', 'selection', 'submit', [1006, 1006, 1005])
    assert len(prepared['order_results']) == 2
    assert svc.select_orders('customer_a', 'one', 'select', 'selection', 'submit', [1006, 1006, 1005]) == prepared
    with pytest.raises(ConflictError):
        svc.select_orders('customer_a', 'one', 'select', 'selection', 'submit', [1001])


def test_single_query_and_selection_cancel(svc):
    result = ask(svc, '1006能退款吗')
    assert result['status'] == 'completed' and not result.get('actions')
    result = ask(svc, '退款')
    assert result['status'] == 'awaiting_selection'
    assert len(result['selection']['candidates']) == 7
    fresh = svc.chat('customer_b', 'two', '退款', runner=refund_runner, request_id='empty')
    assert fresh['status'] == 'awaiting_selection'
    assert {o['order_id'] for o in fresh['selection']['candidates']} == {1008, 1009, 1010}
    cancelled = svc.select_orders('customer_b', 'two', 'empty', 'cancel', 'cancel', [])
    assert cancelled['status'] == 'completed'


def test_confirmation_rechecks_ownership_and_eligibility(svc, business):
    result = ask(svc, '退款1006', request='r')
    action = result['current_action']
    with tools._session() as session, session.begin():
        session.execute(update(Payment).where(Payment.order_id == 1006).values(status='refunded'))
    final = svc.resume('customer_a', 'one', 'r', 'confirm', action['action_id'], 'confirm')
    assert final['order_results'][0]['reason_code'] == 'ALREADY_REFUNDED'
    action = actions.prepare_action('customer_a', 'two', 'own', 0, {'type': 'refund', 'order_id': 1001, 'reason': 'x'})
    with tools._session() as session, session.begin():
        session.execute(update(Order).where(Order.order_id == 1001).values(customer_id='customer_b'))
    with pytest.raises(ValueError):
        actions.decide_action('customer_a', 'two', action['action_id'], 'confirm')


def test_batch_technical_failure_isolated(svc, monkeypatch):
    from src import service as module
    original = module.prepare_action
    def fail(user, session, request, position, proposal):
        if proposal['order_id'] == 1001:
            raise RuntimeError('temporary failure')
        return original(user, session, request, position, proposal)
    monkeypatch.setattr(module, 'prepare_action', fail)
    result = ask(svc, '退款1001和1006')
    assert [r['stage'] for r in result['order_results']] == ['准备失败', '待确认']
    assert result['current_action']['order_id'] == 1006


def test_http_selection_and_validation(svc, monkeypatch):
    import api
    monkeypatch.setattr(api, 'service', svc)
    monkeypatch.setattr(api, 'run_orchestrator', refund_runner)
    monkeypatch.setattr('src.service.get_llm', lambda: None)
    client = TestClient(api.app)
    result = client.post('/chat', json={'message': '退款', 'session_id': 'one', 'request_id': 'select'})
    assert result.json()['status'] == 'awaiting_selection'
    source = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'choice', 'decision': 'submit', 'order_ids': [1006]}
    assert client.post('/requests/select/select-orders', json={**source, 'order_ids': []}).status_code == 422
    assert client.post('/requests/select/select-orders', json={**source, 'order_ids': ['1006']}).status_code == 422
    assert client.post('/requests/select/select-orders', json={**source, 'order_ids': [1009]}).status_code == 409
    assert client.post('/requests/select/select-orders', json=source).json()['current_action']['order_id'] == 1006


def test_ui_customer_switch_and_selection(svc, monkeypatch):
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', refund_runner)
    monkeypatch.setattr('src.service.get_llm', lambda: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('退款').run()
    assert not app.exception
    assert len(app.toggle) == 2 and not any(t.value for t in app.toggle)
    app.toggle[1].set_value(True).run()
    app.toggle[1].set_value(False).run()
    assert all(not t.value for t in app.toggle)
    app.toggle[1].set_value(True).run()
    app.sidebar.text_input[0].set_value('customer_b').run()
    assert not app.chat_message and not app.exception
    app.chat_input[0].set_value('退款1009').run()
    app.sidebar.text_input[0].set_value('customer_a').run()
    assert app.toggle[1].value
    next(b for b in app.button if b.label == '准备所选订单退款').click().run()
    assert not app.exception
    assert any('1006' in t.value for t in app.markdown)
    app.sidebar.text_input[0].set_value('customer_b').run()
    assert any('1009' in t.value for t in app.markdown)
    app.sidebar.text_input[0].set_value('customer_a').run()
    next(b for b in app.button if b.label == '确认办理').click().run()
    assert not app.exception and not app.chat_input[0].disabled


def test_selection_recovers_after_graph_failure(svc, monkeypatch):
    ask(svc, '退款', request='select')
    original = svc._invoke_confirmation
    monkeypatch.setattr(svc, '_invoke_confirmation', lambda *a: (_ for _ in ()).throw(RuntimeError('lost')))
    with pytest.raises(RuntimeError):
        svc.select_orders('customer_a', 'one', 'select', 'choice', 'submit', [1006])
    monkeypatch.setattr(svc, '_invoke_confirmation', original)
    recovered = svc.select_orders('customer_a', 'one', 'select', 'choice', 'submit', [1006])
    assert recovered['status'] == 'awaiting_confirmation'
    with tools._session() as session:
        assert len(session.scalars(select(PendingAction)).all()) == 1


def test_ui_current_sessions_are_independent_and_persist(svc, monkeypatch):
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.sidebar.text_input[1].set_value('a_custom').run()
    app.sidebar.text_input[0].set_value('customer_b').run()
    assert app.sidebar.text_input[1].value == 'case_1'
    app.sidebar.text_input[1].set_value('b_custom').run()
    app.sidebar.text_input[0].set_value('customer_a').run()
    assert app.sidebar.text_input[1].value == 'a_custom'
    next(b for b in app.button if b.label == '新建会话').click().run()
    a_new = app.sidebar.text_input[1].value
    assert a_new != 'a_custom'
    app.sidebar.text_input[0].set_value('customer_b').run()
    assert app.sidebar.text_input[1].value == 'b_custom'
    fresh = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    assert fresh.sidebar.text_input[1].value == a_new


def test_replayed_result_rechecks_order_scope(svc, business):
    result = ask(svc, '退款1006', request='r')
    with tools._session() as session, session.begin():
        session.execute(update(Order).where(Order.order_id == 1006).values(customer_id='customer_b'))
    replay = svc.get_request('customer_a', 'one', 'r')
    assert replay['status'] == 'failed' and not replay['actions'] and not replay['trace']
    assert not svc.snapshot('customer_a', 'one')['turns'][0]['actions']


def test_initial_seed_and_normal_restart(svc):
    assert svc.store.current_session('customer_a', 'custom') == 'custom'
    result = ask(svc, '退款1006', request='r')
    a = result['current_action']
    svc.resume('customer_a', 'one', 'r', 'confirm', a['action_id'], 'confirm')
    tools.init_db(tools._engine)
    tools._ensure_seeded()
    assert tools.check_refund_eligibility(1006)['reason_code'] == 'REFUND_IN_PROGRESS'
    restored = SupportService(svc.store.path)
    try:
        assert restored.store.current_session('customer_a') == 'custom'
        assert restored.get_request('customer_a', 'one', 'r')['status'] == 'completed'
    finally:
        restored.close()


def test_policy_does_not_replace_context(svc):
    ask(svc, '1006能退款吗')
    svc.chat('customer_a', 'one', '2026退款政策', runner=lambda *a, **k: {'reply': 'policy', 'agent': 'faq', 'trace': [], 'usage': {}})
    assert svc.snapshot('customer_a', 'one')['recent_order_ids'] == [1006]


def test_chat_recovers_chosen_batch_after_preparation_failure(svc, monkeypatch):
    ask(svc, '退款', request='select')
    original = svc._invoke_confirmation
    monkeypatch.setattr(svc, '_invoke_confirmation', lambda *a: (_ for _ in ()).throw(RuntimeError('lost')))
    with pytest.raises(RuntimeError):
        svc.select_orders('customer_a', 'one', 'select', 'choice', 'submit', [1001, 1006])
    monkeypatch.setattr(svc, '_invoke_confirmation', original)
    recovered = ask(svc, '退款', request='select')
    assert recovered['status'] == 'awaiting_confirmation'
    assert [a['order_id'] for a in recovered['actions']] == [1001, 1006]


def test_refund_intent_is_conservative(monkeypatch):
    from src.refund_flow import parse_intent, explicit_orders
    from langchain_core.messages import AIMessage
    class Model:
        def invoke(self, text):
            return AIMessage(content='not JSON')
    monkeypatch.setattr('src.refund_flow.get_llm', Model)
    assert parse_intent('不要退款')[0] == 'unclear'
    assert explicit_orders('退款1001到1003和1006和1001') == [1001, 1002, 1003, 1006]


def test_ui_refund_preview_and_expand(svc, monkeypatch):
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', refund_runner)
    monkeypatch.setattr('src.service.get_llm', lambda: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('退款').run()
    assert not app.exception
    assert [t.label.split(' · ')[0] for t in app.toggle] == ['订单 1001', '订单 1006']
    rows = [m.value for m in app.markdown if m.value.startswith('订单 ') and ' · ' in m.value]
    assert [r.split(' · ')[0] for r in rows] == ['订单 1002']
    assert next(b for b in app.button if b.label == '准备所选订单退款').disabled
    app.toggle[1].set_value(True).run()
    app.checkbox[0].set_value(True).run()
    assert len(app.toggle) == 2 and app.toggle[1].value
    rows = [m.value for m in app.markdown if m.value.startswith('订单 ') and ' · ' in m.value]
    assert [r.split(' · ')[0] for r in rows] == ['订单 1002', '订单 1003', '订单 1004', '订单 1005', '订单 1007']
    app.checkbox[0].set_value(False).run()
    assert app.toggle[1].value


def test_ui_refund_preview_without_eligible_orders(svc, monkeypatch):
    with tools._session() as session, session.begin():
        session.execute(update(Payment).where(Payment.order_id.in_([1001, 1006])).values(status='refunded'))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', refund_runner)
    monkeypatch.setattr('src.service.get_llm', lambda: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('退款').run()
    assert not app.exception and not app.toggle
    rows = [m.value for m in app.markdown if m.value.startswith('订单 ') and ' · ' in m.value]
    assert [r.split(' · ')[0] for r in rows] == ['订单 1001', '订单 1002', '订单 1003']
    assert next(b for b in app.button if b.label == '准备所选订单退款').disabled
    app.checkbox[0].set_value(True).run()
    assert not app.toggle
    rows = [m.value for m in app.markdown if m.value.startswith('订单 ') and ' · ' in m.value]
    assert len(rows) == 7


def test_ui_refund_hidden_selection_survives_collapse(svc, monkeypatch):
    with tools._session() as session, session.begin():
        session.execute(update(Payment).where(Payment.order_id == 1003).values(status='paid'))
        session.execute(update(Order).where(Order.order_id == 1007).values(status='delivered'))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', refund_runner)
    monkeypatch.setattr('src.service.get_llm', lambda: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('退款').run()
    assert [t.label.split(' · ')[0] for t in app.toggle] == ['订单 1001', '订单 1003', '订单 1006']
    app.checkbox[0].set_value(True).run()
    assert len(app.toggle) == 4
    app.toggle[3].set_value(True).run()
    app.checkbox[0].set_value(False).run()
    assert len(app.toggle) == 3
    assert any(c.value == '已选 1 笔' for c in app.caption)
    app.checkbox[0].set_value(True).run()
    assert app.toggle[3].value
    next(b for b in app.button if b.label == '准备所选订单退款').click().run()
    assert not app.exception
    current = svc.snapshot('customer_a', 'case_1')['turns'][-1]['current_action']
    assert current['order_id'] == 1007
