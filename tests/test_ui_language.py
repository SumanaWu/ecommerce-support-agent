from pathlib import Path
from streamlit.testing.v1 import AppTest
from tests.test_customer_refunds import svc
from tests.test_actions import business
from tests.test_flow import runner


def test_language_switch_keeps_customer_and_action(svc, monkeypatch):
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', runner, raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('申请退款').run()
    assert not app.exception
    original = svc.snapshot('customer_a', 'case_1')['turns'][-1]['current_action']['action_id']
    next(r for r in app.radio if r.key == 'ui_language').set_value('English').run()
    assert not app.exception
    assert app.title[0].value == 'Ecommerce Support Assistant'
    assert app.chat_input[0].disabled
    assert svc.snapshot('customer_a', 'case_1')['turns'][-1]['current_action']['action_id'] == original
    assert any(b.label == 'Confirm refund' for b in app.button)
    assert any(b.label == 'Save preferences' for b in app.button)
    next(b for b in app.button if b.label == 'Cancel this action').click().run()
    assert not app.exception
    next(r for r in app.radio if r.key == 'ui_language').set_value('中文').run()
    assert app.title[0].value == '电商订单与售后助手'
    next(t for t in app.text_input if t.key == 'customer_id').set_value('customer_b').run()
    next(r for r in app.radio if r.key == 'ui_language').set_value('English').run()
    assert next(t for t in app.text_input if t.key == 'customer_id').value == 'customer_b'


def test_english_payment_keeps_method_identifiers(svc, monkeypatch):
    from tests.test_payment_selection import payment_runner
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    monkeypatch.setattr('src.service.run_orchestrator', payment_runner, raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    next(r for r in app.radio if r.key == 'ui_language').set_value('English').run()
    app.chat_input[0].set_value('帮我用PayPal付款').run()
    assert not app.exception
    methods = next(s for s in app.selectbox if s.label == 'Payment method')
    assert methods.value is None
    assert 'WeChat Pay' in methods.options
    methods.select('paypal').run()
    assert not app.exception
    assert svc.snapshot('customer_a', 'case_1')['turns'][-1]['current_action']['parameters']['method'] == 'paypal'
    next(b for b in app.button if b.label == 'Confirm payment').click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
