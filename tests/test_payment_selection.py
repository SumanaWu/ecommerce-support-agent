from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src import tools
from src.agents.payment_agent import prepare_payment
from src.backend.db import Payment, PendingAction
from src.payment_methods import PAYMENT_METHODS
from src.service import ConflictError, SupportService
from tests.test_customer_refunds import svc
from tests.test_actions import business


def payment_runner(message, history=None):
    result = prepare_payment.invoke({'order_id': 1003})
    return {'agent': 'payment', 'reply': '', 'trace': [
        {'agent': 'payment', 'tool': 'prepare_payment', 'args': {'order_id': 1003}, 'result': result}], 'usage': {}}


def start(svc, message='帮我用PayPal付款'):
    return svc.chat('customer_a', 'one', message, runner=payment_runner, request_id='pay')


@pytest.mark.parametrize('method', PAYMENT_METHODS)
def test_selected_method_requires_separate_confirmation_and_is_recorded(svc, method):
    initial = start(svc)
    assert initial['status'] == 'awaiting_payment_method'
    assert initial['payment_selection']['methods'] == PAYMENT_METHODS
    assert initial['payment_selection']['orders'] == [{'order_id': 1003, 'amount': '34.50'}]
    assert not initial.get('actions')
    with tools._session() as session:
        assert not session.scalars(select(PendingAction)).all()
    selected = svc.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', method)
    action = selected['current_action']
    assert selected['status'] == 'awaiting_confirmation'
    assert action['parameters']['method'] == method
    assert action['amount'] == '34.50'
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert tools.get_payment_status(1003)['method'] == 'card'
    assert svc.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', method) == selected
    result = svc.resume('customer_a', 'one', 'pay', 'confirm', action['action_id'], 'confirm')
    assert result['status'] == 'completed'
    assert tools.get_payment_status(1003)['status'] == 'paid'
    assert tools.get_payment_status(1003)['method'] == method
    assert svc.resume('customer_a', 'one', 'pay', 'confirm', action['action_id'], 'confirm') == result


def test_cancel_selection_and_cancel_confirmation_leave_payment_unchanged(svc):
    start(svc)
    cancelled = svc.select_payment_method('customer_a', 'one', 'pay', 'cancel', 'cancel')
    assert cancelled['status'] == 'completed' and not cancelled.get('actions')
    assert svc.select_payment_method('customer_a', 'one', 'pay', 'cancel', 'cancel') == cancelled
    svc.chat('customer_a', 'one', '我要付款', runner=payment_runner, request_id='pay_again')
    chosen = svc.select_payment_method('customer_a', 'one', 'pay_again', 'choose', 'submit', 'alipay')
    svc.resume('customer_a', 'one', 'pay_again', 'cancel_action', chosen['current_action']['action_id'], 'cancel')
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert tools.get_payment_status(1003)['method'] == 'card'


def test_selection_rejects_missing_method_foreign_owner_and_replacement(svc):
    start(svc)
    with pytest.raises(ValueError, match='请选择'):
        svc.select_payment_method('customer_a', 'one', 'pay', 'missing', 'submit')
    with pytest.raises(ValueError, match='请选择'):
        svc.select_payment_method('customer_a', 'one', 'pay', 'invalid', 'submit', 'arbitrary')
    with pytest.raises(ConflictError):
        svc.select_payment_method('customer_b', 'one', 'pay', 'foreign', 'submit', 'paypal')
    with pytest.raises(ConflictError):
        svc.chat('customer_a', 'one', '另外一件事', runner=payment_runner)
    svc.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', 'paypal')
    with pytest.raises(ConflictError):
        svc.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', 'card')
    with pytest.raises(ConflictError):
        svc.select_payment_method('customer_a', 'one', 'pay', 'other_choose', 'submit', 'card')


def test_selection_survives_restart_and_rechecks_payment(svc):
    start(svc)
    restored = SupportService(svc.store.path)
    try:
        assert restored.snapshot('customer_b', 'one')['turns'] == []
        assert restored.snapshot('customer_a', 'one')['turns'][0]['status'] == 'awaiting_payment_method'
        with tools._session() as session, session.begin():
            session.scalar(select(Payment).where(Payment.order_id == 1003)).status = 'paid'
        result = restored.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', 'paypal')
        assert result['tasks'][0]['status'] == 'succeeded'
        assert '已付款，无需再次支付' in result['reply']
        with tools._session() as session:
            assert not session.scalars(select(PendingAction)).all()
    finally:
        restored.close()


def test_selection_retry_after_preparation_failure_uses_original_choice(svc, monkeypatch):
    import src.orchestrator as module
    start(svc)
    original = module.prepare_action
    def fail(*args):
        raise RuntimeError('temporary failure')
    monkeypatch.setattr(module, 'prepare_action', fail)
    with pytest.raises(RuntimeError):
        svc.select_payment_method('customer_a', 'one', 'pay', 'choose', 'submit', 'wechat_pay')
    monkeypatch.setattr(module, 'prepare_action', original)
    retry = start(svc)
    assert retry['current_action']['parameters']['method'] == 'wechat_pay'
    assert tools.get_payment_status(1003)['status'] == 'pending'


def test_http_payment_selection(svc, monkeypatch):
    import api
    monkeypatch.setattr(api, 'service', svc)
    monkeypatch.setattr(api, 'run_orchestrator', payment_runner, raising=False)
    client = TestClient(api.app)
    initial = client.post('/chat', json={'user_id': 'customer_a', 'session_id': 'one',
        'request_id': 'pay', 'message': '帮我付款'}).json()
    assert initial['status'] == 'awaiting_payment_method'
    body = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'choose', 'decision': 'submit'}
    endpoint = '/requests/pay/select-payment-method'
    assert client.post(endpoint, json=body).status_code == 422
    assert client.post(endpoint, json={**body, 'method': 'not_a_method'}).status_code == 422
    result = client.post(endpoint, json={**body, 'method': 'paypal'})
    assert result.status_code == 200
    assert result.json()['current_action']['parameters']['method'] == 'paypal'


def test_ui_inline_method_without_default_and_single_confirmation(svc, monkeypatch):
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', payment_runner, raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('帮我用PayPal付款').run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.selectbox[0].value is None
    assert app.selectbox[0].options == list(PAYMENT_METHODS.values())
    assert next(b for b in app.button if b.label == '确认付款').disabled
    assert not any(b.label == '下一步：确认金额' for b in app.button)
    app.selectbox[0].select('paypal').run()
    assert not app.exception
    assert any(m.value == '支付方式：PayPal' for m in app.markdown)
    assert tools.get_payment_status(1003)['status'] == 'pending'
    next(b for b in app.button if b.label == '确认付款').click().run()
    assert not app.exception and not app.chat_input[0].disabled
    assert tools.get_payment_status(1003)['method'] == 'paypal'
