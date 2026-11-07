import copy

import pytest

from src import tools, orchestrator as workflow
from src.planning import Plan
from src.service import SupportService
from tests.test_customer_refunds import svc
from tests.test_actions import business


def task(operation, targets, **extra):
    return {'operation': operation, 'targets': targets, 'source_text': '模拟指令',
        'intent': 'prepare' if operation.endswith('.prepare') else 'query', **extra}


def install(monkeypatch, *tasks):
    plan = Plan.model_validate({'action': 'PLAN', 'tasks': [
        {'task_id': f't{i+1}', **t} for i, t in enumerate(tasks)]}).model_dump()
    monkeypatch.setattr(workflow, 'make_plan', lambda *a: (copy.deepcopy(plan), {}))


def ask(svc):
    return svc.chat('customer_a', 'one', '模拟指令', request_id='chat')


def confirm(svc, result, request='decision', decision='confirm'):
    return svc.resume('customer_a', 'one', 'chat', request,
        result['current_action']['action_id'], decision)


def test_independent_domain_queries_have_both_tools(svc, monkeypatch):
    install(monkeypatch, task('order.shipment', [1002]), task('payment.status', [1003]))
    result = ask(svc)
    assert result['agents'] == ['order', 'payment']
    assert result['status'] == 'completed'
    assert {t['tool'] for t in result['trace']} == {'track_shipment', 'get_payment_status'}
    assert not result['actions']


@pytest.mark.parametrize('oid,expected', [(1001, 'awaiting_confirmation'), (1002, 'completed')])
def test_conditional_refund_preserves_target(svc, monkeypatch, oid, expected):
    install(monkeypatch, task('order.shipment', [oid]), task('refund.prepare', [oid],
        conditions=[{'task': 't1', 'field': 'order_status', 'value': 'delivered'}]))
    result = ask(svc)
    assert result['status'] == expected
    assert all(a['order_id'] == oid for a in result['actions'])


def test_mixed_actions_overview_separate_confirmation_and_replay(svc, monkeypatch):
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    result = ask(svc)
    assert result['status'] == 'awaiting_confirmation'
    assert result['current_action']['order_id'] == 1001
    assert not result['pending_inputs']
    assert len(result['actions']) == 1
    first = confirm(svc, result, decision='cancel')
    assert first['status'] == 'awaiting_payment_method'
    assert confirm(svc, result, request='replay', decision='cancel') == first
    chosen = svc.select_payment_method('customer_a', 'one', 'chat', 'method', 'submit', 'wechat_pay')
    assert chosen['current_action']['order_id'] == 1003
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert confirm(svc, chosen, request='pay')['status'] == 'completed'
    assert tools.get_payment_status(1003)['method'] == 'wechat_pay'
    assert tools.check_refund_eligibility(1001)['eligible']

def test_payment_unlocks_fresh_refund_query_after_restart(svc, monkeypatch):
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.check', [1003], depends_on=['t1']))
    result = ask(svc)
    chosen = svc.select_payment_method('customer_a', 'one', 'chat', 'method', 'submit', 'paypal')
    restored = SupportService(svc.store.path)
    try:
        result = confirm(restored, chosen)
        assert result['status'] == 'completed'
        assert result['tasks'][1]['facts'][0]['eligible']
        assert result['tasks'][1]['facts'][0]['payment_status'] == 'paid'
        assert len(result['actions']) == 1
    finally:
        restored.close()


def test_cancelled_payment_skips_dependent_query(svc, monkeypatch):
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.check', [1003], depends_on=['t1']))
    ask(svc)
    result = svc.select_payment_method('customer_a', 'one', 'chat', 'method', 'cancel')
    assert result['tasks'][1]['status'] == 'skipped'
    assert tools.get_payment_status(1003)['status'] == 'pending'


def test_query_after_paid_cannot_infer_unpaid_from_placed(svc, monkeypatch):
    tools.process_payment(1003, 'wechat_pay')
    install(monkeypatch, task('order.details', [1003]), task('payment.status', [1003]))
    result = ask(svc)
    assert '已付款' in result['reply'] and '微信支付' in result['reply']
    assert '待付款' not in result['reply'] and '待支付' not in result['reply']


def test_dependency_validation_rejects_cycles():
    with pytest.raises(ValueError, match='依赖'):
        Plan.model_validate({'action': 'PLAN', 'tasks': [{'task_id': 't1',
            **task('payment.status', [1003], depends_on=['t1'])}]})


def test_list_then_domain_filter_is_generic(svc, monkeypatch):
    install(monkeypatch, task('order.list', []), task('payment.status', [], target_from='t1'))
    result = ask(svc)
    assert len(result['tasks'][1]['facts']) == 7
    assert all(f['order_id'] <= 1007 for f in result['tasks'][1]['facts'])


def test_filter_then_prepare_uses_only_matching_orders(svc, monkeypatch):
    install(monkeypatch, task('order.list', []), task('refund.check', [], target_from='t1'),
        task('refund.prepare', [], target_from='t2', conditions=[{'task': 't2', 'field': 'eligible', 'value': True}]))
    result = ask(svc)
    assert [a['order_id'] for a in result['actions']] == [1001, 1006]
    assert result['pending_inputs'] == []


def test_multiple_inputs_are_explicit_and_do_not_pay(svc, monkeypatch):
    from sqlalchemy import update
    from src.backend.db import Payment
    with tools._session() as db, db.begin():
        db.execute(update(Payment).where(Payment.order_id == 1006).values(status='pending'))
    install(monkeypatch, task('payment.prepare', [1003, 1006]))
    result = ask(svc)
    assert len(result['pending_inputs']) == 2
    from src.service import ConflictError
    with pytest.raises(ConflictError):
        svc.select_payment_method('customer_a', 'one', 'chat', 'ambiguous', 'submit', 'paypal')
    result = svc.select_payment_method('customer_a', 'one', 'chat', 'first', 'submit', 'paypal', result['pending_inputs'][0]['input_id'])
    result = svc.select_payment_method('customer_a', 'one', 'chat', 'second', 'cancel', input_id=result['pending_inputs'][0]['input_id'])
    assert len(result['actions']) == 1 and result['current_action']['order_id'] == 1003
    assert tools.get_payment_status(1003)['status'] == tools.get_payment_status(1006)['status'] == 'pending'


def test_skip_propagates_through_multiple_dependencies(svc, monkeypatch):
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.check', [1003], depends_on=['t1']),
        task('order.details', [1003], depends_on=['t2']))
    result = ask(svc)
    result = svc.select_payment_method('customer_a', 'one', 'chat', 'cancel', 'cancel')
    assert [t['status'] for t in result['tasks']] == ['cancelled', 'skipped', 'skipped']


def test_expert_failure_keeps_independent_result(svc, monkeypatch):
    install(monkeypatch, task('order.shipment', [1002]), task('payment.status', [1003]))
    def fail(order_id):
        raise RuntimeError('本地模拟失败')
    monkeypatch.setattr(tools, 'get_payment_status', fail)
    result = ask(svc)
    assert result['tasks'][0]['status'] == 'succeeded'
    assert result['tasks'][1]['status'] == 'failed'
    assert '运输中' in result['reply']


def test_action_committed_but_response_lost_is_not_repeated(svc, monkeypatch):
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    result = ask(svc)
    original = svc._graph_result
    monkeypatch.setattr(svc, '_graph_result', lambda *a: (_ for _ in ()).throw(RuntimeError('lost response')))
    with pytest.raises(RuntimeError):
        confirm(svc, result)
    monkeypatch.setattr(svc, '_graph_result', original)
    replay = confirm(svc, result)
    assert replay['status'] == 'awaiting_payment_method'
    assert replay['actions'][0]['status'] == 'completed'
    assert tools.get_payment_status(1003)['status'] == 'pending'

def test_mixed_plan_ui_completes_refund_before_payment(svc, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('模拟指令').run()
    assert not app.exception and app.chat_input[0].disabled
    assert not app.selectbox
    assert any(b.label == '确认办理' for b in app.button)
    next(b for b in app.button if b.label == '取消本项').click().run()
    assert not app.exception and app.chat_input[0].disabled
    assert app.selectbox[0].value is None
    app.selectbox[0].select('wechat_pay').run()
    assert not app.exception
    next(b for b in app.button if b.label == '确认付款').click().run()
    assert not app.exception and not app.chat_input[0].disabled
    assert tools.get_payment_status(1003)['method'] == 'wechat_pay'
    assert tools.check_refund_eligibility(1001)['eligible']

def test_cancelled_payment_selection_keeps_card_after_restart(svc, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    path = str(Path(__file__).resolve().parents[1] / 'app.py')
    app = AppTest.from_file(path, default_timeout=15).run()
    app.chat_input[0].set_value('模拟指令').run()
    cards = [m.value for m in app.markdown if m.value.startswith('**订单')]
    assert len(cards) == 2 and '1001' in cards[0] and '1003' in cards[1]
    assert any(c.value == '状态：等待前一项处理，尚未核实' for c in app.caption)
    assert not app.selectbox
    next(b for b in app.button if b.label == '取消本项').click().run()
    cards = [m.value for m in app.markdown if m.value.startswith('**订单')]
    assert len(cards) == 2 and '1001' in cards[0] and '1003' in cards[1]
    next(b for b in app.button if b.label == '取消支付').click().run()
    assert not app.exception
    cards = [m.value for m in app.markdown if m.value.startswith('**订单')]
    assert len(cards) == 2 and '1001' in cards[0] and '1003' in cards[1]
    assert any('订单 1001 · 金额 79.99 · 退款申请' in m.value for m in app.markdown)
    assert sum('订单 1003 · 金额 34.50 · 支付' in m.value for m in app.markdown) == 1
    assert any(c.value == '状态：已取消，未支付' for c in app.caption)
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert tools.check_refund_eligibility(1001)['eligible']
    restored = SupportService(svc.store.path)
    try:
        monkeypatch.setattr('src.service.SupportService', lambda: restored)
        restarted = AppTest.from_file(path, default_timeout=15).run()
        assert not restarted.exception
        cards = [m.value for m in restarted.markdown if m.value.startswith('**订单')]
        assert len(cards) == 2 and '1001' in cards[0] and '1003' in cards[1]
        assert sum('订单 1003 · 金额 34.50 · 支付' in m.value for m in restarted.markdown) == 1
        assert any(c.value == '状态：已取消，未支付' for c in restarted.caption)
    finally:
        restored.close()


def test_payment_first_cards_keep_order_through_confirmation(svc, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.prepare', [1001]))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=15).run()
    app.chat_input[0].set_value('模拟指令').run()
    cards = [m.value for m in app.markdown if m.value.startswith('**订单')]
    assert len(cards) == 2 and '1003' in cards[0] and '1001' in cards[1]
    assert not any(b.label == '确认办理' for b in app.button)
    app.selectbox[0].select('paypal').run()
    next(b for b in app.button if b.label == '确认付款').click().run()
    assert not app.exception
    cards = [m.value for m in app.markdown if m.value.startswith('**订单')]
    assert len(cards) == 2 and '1003' in cards[0] and '1001' in cards[1]
    assert any(b.label == '确认办理' for b in app.button)


def test_status_conditions_use_backend_enumerations():
    from src.planning import Condition
    with pytest.raises(ValueError):
        Condition(task='t1', field='payment_status', value='unpaid')
    assert Condition(task='t1', field='payment_status', value='pending').value == 'pending'


def test_duplicate_queries_merge_and_dependencies_follow_original():
    plan = Plan.model_validate({'action': 'PLAN', 'tasks': [
        {'task_id': 't1', **task('payment.status', [1003])},
        {'task_id': 't2', **task('payment.status', [1003])},
        {'task_id': 't3', **task('refund.check', [1003], depends_on=['t2'])}]})
    assert [t.task_id for t in plan.tasks] == ['t1', 't3']
    assert plan.tasks[-1].depends_on == ['t1']


def test_http_multi_input_submission_and_customer_scope(svc, monkeypatch):
    import api
    from fastapi.testclient import TestClient
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    monkeypatch.setattr(api, 'service', svc)
    client = TestClient(api.app)
    result = client.post('/chat', json={'message': '模拟指令', 'session_id': 'one', 'request_id': 'chat'}).json()
    assert result['current_action']['order_id'] == 1001 and not result['pending_inputs']
    decision = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'cancel',
                'decision': 'cancel', 'action_id': result['current_action']['action_id']}
    result = client.post('/requests/chat/resume', json=decision).json()
    entry = result['pending_inputs'][0]
    body = {'user_id': 'customer_a', 'session_id': 'one', 'request_id': 'choose',
        'decision': 'submit', 'method': 'wechat_pay', 'input_id': entry['input_id']}
    assert client.post('/requests/chat/select-payment-method', json={**body, 'user_id': 'customer_b'}).status_code == 409
    chosen = client.post('/requests/chat/select-payment-method', json=body)
    assert chosen.status_code == 200 and len(chosen.json()['actions']) == 2
    assert client.post('/requests/chat/select-payment-method', json=body).json() == chosen.json()
    assert tools.get_payment_status(1003)['status'] == 'pending'

def test_same_order_payment_then_refund_has_distinct_actions(svc, monkeypatch):
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.prepare', [1003], depends_on=['t1']))
    ask(svc)
    chosen = svc.select_payment_method('customer_a', 'one', 'chat', 'choose', 'submit', 'visa')
    paid = confirm(svc, chosen, request='pay')
    assert paid['current_action']['type'] == 'refund'
    assert paid['current_action']['action_id'] != chosen['current_action']['action_id']
    assert all(a['order_id'] == 1003 for a in paid['actions'])
    replay = confirm(svc, chosen, request='old_card')
    assert replay['current_action']['type'] == 'refund'
    assert tools.check_refund_eligibility(1003)['eligible']
    final = confirm(svc, paid, request='refund')
    assert final['status'] == 'completed'
    assert tools.check_refund_eligibility(1003)['reason_code'] == 'REFUND_IN_PROGRESS'


def test_repeated_refund_facts_are_summarized(svc, monkeypatch):
    install(monkeypatch, task('order.details', [1001]), task('payment.status', [1001]),
        task('refund.check', [1001]), task('refund.prepare', [1001]))
    result = ask(svc)
    assert result['reply'].count('**订单 1001**') == 1
    assert result['reply'].count('可退金额') == 1
    assert result['reply'].count('符合退款条件') <= 1
    assert len([t for t in result['trace'] if t['tool'] == 'check_refund_eligibility']) == 2
    assert result['current_action']['order_id'] == 1001


def test_paid_conditional_payment_has_business_explanation(svc, monkeypatch):
    from src.backend.db import Payment
    with tools._session() as db, db.begin():
        db.query(Payment).filter_by(order_id=1003).one().status = 'paid'
    install(monkeypatch, task('payment.status', [1003]), task('payment.prepare', [1003],
        conditions=[{'task': 't1', 'field': 'payment_status', 'value': 'pending'}]))
    result = ask(svc)
    assert '已付款，无需再次支付' in result['reply']
    assert '缺少' not in result['reply'] and '条件未满足' not in result['reply']
    assert result['tasks'][1]['status'] == 'skipped'
    assert not result['actions'] and not result['pending_inputs']


def test_reverse_payment_selection_keeps_plan_confirmation_order(svc, monkeypatch):
    install(monkeypatch, task('payment.prepare', [1003]), task('refund.prepare', [1001]))
    result = ask(svc)
    result = svc.select_payment_method('customer_a', 'one', 'chat', 'choice', 'submit', 'paypal')
    assert [a['order_id'] for a in result['actions']] == [1003]
    assert result['current_action']['order_id'] == 1003
    result = confirm(svc, result)
    assert result['current_action']['order_id'] == 1001


def test_large_order_summary_uses_table(svc, monkeypatch):
    install(monkeypatch, task('order.details', [1001, 1002, 1003, 1004]))
    result = ask(svc)
    assert '| 订单 | 详情 |' in result['reply']
    assert all(result['reply'].count(f'**{oid}**') == 1 for oid in range(1001, 1005))


def test_partial_conditional_payment_explains_excluded_order(svc, monkeypatch):
    install(monkeypatch, task('payment.status', [1001, 1003]), task('payment.prepare', [1001, 1003],
        conditions=[{'task': 't1', 'field': 'payment_status', 'value': 'pending'}]))
    result = ask(svc)
    assert '已付款，无需再次支付' in result['reply']
    assert result['pending_inputs'][0]['orders'][0]['order_id'] == 1003
    assert result['reply'].count('**订单 1001**') == 1


def test_restart_between_ordered_actions(svc, monkeypatch):
    install(monkeypatch, task('refund.prepare', [1001]), task('payment.prepare', [1003]))
    first = ask(svc)
    restored = SupportService(svc.store.path)
    try:
        assert restored.get_request('customer_a', 'one', 'chat')['current_action']['order_id'] == 1001
        result = restored.resume('customer_a', 'one', 'chat', 'confirm', first['current_action']['action_id'], 'confirm')
        assert result['status'] == 'awaiting_payment_method'
        assert result['pending_inputs'][0]['orders'][0]['order_id'] == 1003
    finally:
        restored.close()
