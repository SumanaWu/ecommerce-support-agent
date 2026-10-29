"""Factual online replies and authorized memory views, using scripted models only."""
import json
from pathlib import Path

import pytest

from src import orchestrator, tools
from src.auth import AccessError, Principal
from src.backend.db import Order, PendingAction, RequestRecord, get_engine, get_session_factory
from src.backend.seed import seed
from src.context import build_context
from src.memory import empty_issue
from src.results import compose_response
from src.service import SupportService
from tests.online_support import call, install_script, plan, proposal_steps, task

A = Principal('customer_a')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    yield service, engine
    engine.dispose()


def transfer(engine, order_id):
    with get_session_factory(engine).begin() as db:
        db.get(Order, order_id).customer_id = 'customer_b'


def chat(service, message='查询付款状态', request_id=None):
    return service.chat(A, 'one', message, request_id=request_id)


@pytest.mark.parametrize('number,label,amount', [(1005, '已退款', '24.99'), (1006, '已付款', '59.99'), (1003, '待付款', '34.50')])
def test_single_payment_reply_ignores_model_invented_numbers_and_status(setup, monkeypatch, number, label, amount):
    service, _ = setup
    calls = install_script(monkeypatch, [plan(task('payment_status', [number])),
        call('get_payment_status', {'order_id': number}), '{"order_id":9999,"status":"paid","amount":888.88}'])
    result = chat(service)
    assert result['reply'] == f'订单 {number}：当前{label}，金额 {amount} CNY，支付方式银行卡。'
    assert '888.88' not in result['reply'] and '{' not in result['reply']
    assert result['reply_source'] == 'facts' and len(calls) == 3
    assert '888.88' in result['model_reply']  # Explicit debug data, never the business body.
    assert service.snapshot(A, 'one')['turns'][0]['reply'] == result['reply']


@pytest.mark.parametrize('number,payment_status', [(1001, 'paid'), (1005, 'refunded')])
def test_payment_proposal_for_terminal_payment_is_business_rejection(setup, monkeypatch, number, payment_status):
    from src.backend.db import ActionRecord, Payment
    service, engine = setup
    calls = install_script(monkeypatch, [plan(task('propose_payment', [number])),
        call('get_payment_status', {'order_id': number}), '已经付款成功。'])
    result = chat(service, f'用银行卡支付订单{number}')
    assert result['status'] == 'rejected' and result['stop_reason'] == 'business_rejected'
    assert result['tasks'][0]['status'] == 'rejected' and result['pending_actions'] == []
    assert [entry['tool'] for entry in result['trace']] == ['get_payment_status']
    assert result['trace'][0]['result']['status'] == payment_status and len(calls) == 3
    assert 'missing_action' not in result['stop_reason'] and '本项只完成了查询' not in result['reply']
    assert '付款成功' not in result['reply'] and '待确认' not in result['reply']
    assert '不能再次付款' in result['reply'] and '未创建新的付款操作' in result['reply']
    with get_session_factory(engine)() as db:
        assert db.query(PendingAction).count() == db.query(ActionRecord).count() == 0
        assert db.query(Payment).filter_by(order_id=number).one().status == payment_status


@pytest.mark.parametrize('status,explanation', [('placed', '尚未发货'), ('cancelled', '没有配送中的包裹')])
def test_shipment_reply_answers_whether_a_package_was_sent(status, explanation):
    result = compose_response({'status': 'completed', 'trace': [
        {'tool': 'track_shipment', 'result': {'order_id': 1003, 'status': status}}]})
    assert explanation in result['reply']


@pytest.mark.parametrize('answer_mode', ['result', 'explain'])
def test_requested_refund_fact_and_reply_distinguish_application_from_received_money(setup, monkeypatch, answer_mode):
    from src.backend.db import Payment, Refund
    service, engine = setup
    calls = install_script(monkeypatch, [plan(task('refund_eligibility', [1004], answer_mode=answer_mode)),
        call('check_refund_eligibility', {'order_id': 1004}), '退款已到账，放心。'])
    result = chat(service, '查1004退款申请现在怎么样，申请等于钱到账了吗')
    fact = result['trace'][0]['result']
    assert fact['refund_status'] == 'requested' and fact['eligible'] is False
    assert fact['reason_code'] == 'existing_refund' and result['status'] == 'completed'
    assert '已有退款申请' in result['reply'] and '申请不代表款项已到账' in result['reply']
    assert '不能重复申请' in result['reply'] and '退款已到账' not in result['reply']
    assert result['pending_actions'] == [] and len(calls) == 3
    with get_session_factory(engine)() as db:
        assert db.query(Refund).filter_by(order_id=1004).one().status == 'requested'
        assert db.query(Payment).filter_by(order_id=1004).one().status == 'paid'


@pytest.mark.parametrize('intent,operation,tool,fact,note', [
    ('delivery_timing', 'shipment', 'track_shipment', {'order_id': 1002, 'status': 'shipped'}, '未提供预计送达日期'),
    ('proposal_effect', 'order_status', 'get_order_status', {'order_id': 1003, 'status': 'placed'}, '创建工单，都不等于取消订单'),
    ('refund_progress', 'refund_eligibility', 'check_refund_eligibility',
     {'order_id': 1004, 'order_status': 'delivered', 'eligible': False, 'reason_code': 'existing_refund',
      'refund_status': 'requested'}, '未提供银行到账凭据或预计到账日期'),
])
def test_structured_presentation_adds_only_fact_supported_explanations(intent, operation, tool, fact, note):
    trace = [{'tool': tool, 'result': fact}]
    result = compose_response({'status': 'completed', 'reply_source': 'model', 'reply': '保证今天到账和送达。',
        'trace': trace, 'tasks': [{'operation': operation, 'status': 'completed', 'trace': trace,
                                 'presentation_intent': intent}]})
    assert note in result['reply'] and result['reply'].count(note) == 1
    assert note in result['tasks'][0]['reply'] and result['trace'] == trace
    assert '保证今天到账和送达' not in result['reply']
    assert compose_response(result)['reply'] == result['reply']
    plain = compose_response({'status': 'completed', 'trace': trace})
    assert note not in plain['reply']


@pytest.mark.parametrize('intent,tool', [('delivery_timing', 'track_shipment'),
                                      ('proposal_effect', 'get_order_status'),
                                      ('refund_progress', 'check_refund_eligibility')])
def test_presentation_does_not_add_business_explanation_without_current_evidence(intent, tool):
    result = compose_response({'status': 'failed', 'presentation_intent': intent,
        'trace': [{'tool': tool, 'args': {'order_id': 1008}, 'result': {'error': '未找到可访问的订单。'}}]})
    assert '预计送达日期' not in result['reply'] and '创建工单' not in result['reply']
    assert '银行到账凭据' not in result['reply']
    wrong_tool = compose_response({'status': 'completed', 'presentation_intent': intent,
        'trace': [{'tool': 'get_payment_status', 'result': {'order_id': 1001, 'status': 'paid'}}]})
    assert '预计送达日期' not in wrong_tool['reply'] and '创建工单' not in wrong_tool['reply']
    assert '银行到账凭据' not in wrong_tool['reply']


def test_nested_batch_table_retains_one_delivery_explanation_per_task():
    shipment = {'tool': 'track_shipments_batch', 'result': {'items': [
        {'order_id': 1002, 'result': {'order_id': 1002, 'status': 'shipped'}},
        {'order_id': 1008, 'result': {'error': '未找到可访问的订单。'}}]}}
    payment = {'tool': 'get_payment_status', 'result': {'order_id': 1003, 'status': 'pending'}}
    result = compose_response({'status': 'partial', 'trace': [shipment, payment], 'tasks': [
        {'operation': 'shipment', 'status': 'partial', 'trace': [shipment], 'presentation_intent': 'delivery_timing'},
        {'operation': 'payment_status', 'status': 'completed', 'trace': [payment]}]})
    assert '\n\n  | 订单 | 物流状态 |' in result['reply']
    assert '| 1002 | 运输中 |' in result['reply'] and '| 1008 | 未找到可访问的订单 |' in result['reply']
    assert result['reply'].count('本次查询未提供预计送达日期') == 1
    assert '订单 1003：当前待付款' in result['reply']


def test_online_presentation_intent_survives_planning_and_saved_chat(setup, monkeypatch):
    service, _ = setup
    request = {**task('shipment', [1002]), 'presentation_intent': 'delivery_timing'}
    install_script(monkeypatch, [plan(request), call('track_shipment', {'order_id': 1002}), '肯定今天送到。'])
    result = chat(service, '1002今天能到吗')
    assert result['status'] == 'completed' and '运输中' in result['reply']
    assert result['tasks'][0]['presentation_intent'] == 'delivery_timing'
    assert '无法保证具体哪天送达' in result['reply'] and '肯定今天送到' not in result['reply']
    saved = service.snapshot(A, 'one')['turns'][0]
    assert saved['reply'] == result['reply']
    assert saved['tasks'][0]['presentation_intent'] == 'delivery_timing'


def test_batch_table_preserves_order_and_unavailable_rows(setup, monkeypatch):
    service, _ = setup
    ids = [1006, 1005, 1003, 1008, 9999]
    calls = install_script(monkeypatch, [plan(task('payment_status', ids)),
        call('get_payment_status_batch', {'order_ids': ids}), '所有订单付款成功，金额999元。'])
    result = chat(service)
    body = result['reply']
    assert result['status'] == 'partial' and '| 订单 | 付款状态 | 金额 |' in body
    assert '| 1005 | 已退款 | 24.99 CNY |' in body
    assert '| 1008 | 未找到可访问的订单 | — |' in body
    assert '| 9999 | 未找到可访问的订单 | — |' in body
    assert [body.index('| ' + str(number) + ' |') for number in ids] == sorted(body.index('| ' + str(number) + ' |') for number in ids)
    assert '999元' not in body and len(calls) == 3


def test_amount_comes_from_integer_minor_units_and_business_text_is_escaped():
    result = compose_response({'reply': 'wrong', 'reply_source': 'model', 'status': 'completed', 'trace': [
        {'tool': 'get_order_details_batch', 'result': {'items': [{'order_id': 1, 'result': {
            'status': 'delivered', 'amount_minor': 1234, 'currency': 'USD', 'amount_display': '999.99',
            'item': '商品|名称\n[link](https://wrong.invalid)', 'order_date': '2026-10-01'}}]}}]})
    assert '12.34 USD' in result['reply'] and '999.99' not in result['reply']
    assert '商品\\|名称 ' in result['reply'] and '\\[link\\]' in result['reply']


def test_single_detail_bullets_keep_all_tool_fields_and_escape_business_text():
    result = compose_response({'status': 'completed', 'trace': [
        {'tool': 'get_order_details', 'result': {'order_id': 1001, 'status': 'delivered',
            'amount_minor': 7999, 'currency': 'CNY', 'item': '杯子|套装\n[链接](https://wrong.invalid)',
            'order_date': '2026-10-01', 'customer_name': 'Alice_测试'}}]})
    body = result['reply']
    assert '订单 1001：订单状态为已送达，金额 79.99 CNY。' in body
    assert '\n\n- 商品：杯子\\|套装 \\[链接\\]' in body
    assert '\n- 下单日期：2026-10-01\n- 收件人：Alice\\_测试' in body
    assert body.count('- 商品：') == 1 and '| 订单 |' not in body


def test_multi_task_bullets_keep_nested_fact_table_and_pending_parameters():
    details = {'tool': 'get_order_details_batch', 'result': {'items': [
        {'order_id': 1001, 'result': {'order_id': 1001, 'status': 'delivered', 'amount_minor': 7999,
            'currency': 'CNY', 'item': 'Coffee Mug', 'order_date': '2026-10-01', 'customer_name': 'Alice'}}]}}
    payment = {'tool': 'process_payment', 'result': {'order_id': 1003, 'action_id': 'test-payment',
        'status': 'pending_confirmation', 'requires_confirmation': True, 'amount_minor': 3450,
        'currency': 'CNY', 'method': 'paypal'}}
    result = compose_response({'status': 'pending_confirmation', 'trace': [details, payment], 'tasks': [
        {'operation': 'order_details', 'status': 'completed', 'trace': [details]},
        {'operation': 'propose_payment', 'status': 'pending_confirmation', 'trace': [payment]}]})
    body = result['reply']
    assert '- **订单详情**：' in body and '- **付款办理**：' in body
    assert '\n\n  | 订单 | 订单状态 | 金额 | 商品 | 下单日期 | 收件人 |' in body
    assert '| 1001 | 已送达 | 79.99 CNY | Coffee Mug | 2026-10-01 | Alice |' in body
    assert '⏳ 订单 1003：付款待确认，金额 34.50 CNY。' in body
    assert '支付方式：PayPal' in body and '尚未办理' in body and '付款成功' not in body
    assert result['trace'] == [details, payment]


@pytest.mark.parametrize('exception,kind', [(TimeoutError('private network detail'), 'network'),
                                         (RuntimeError('private internal detail'), 'program')])
def test_technical_failures_do_not_create_unsolicited_human_tickets(setup, exception, kind):
    from src.backend.db import Ticket
    service, engine = setup
    def fail(*_args, **_kwargs):
        raise exception
    result = service.chat(A, 'technical-failure', '帮我查一下订单1001', runner=fail,
                          request_id='technical-failure')
    assert result['status'] == 'failed' and result['error_type'] == kind
    assert result['agent'] == 'system' and result['trace'] == []
    assert '工单' not in result['reply'] and 'private' not in result['reply']
    replay = service.chat(A, 'technical-failure', '帮我查一下订单1001', runner=fail,
                          request_id='technical-failure')
    assert replay['replayed'] and replay['reply'] == result['reply']
    with get_session_factory(engine)() as db:
        assert db.query(Ticket).count() == 0


@pytest.mark.parametrize('end,status', [(1006, 'completed'), (1010, 'partial')])
@pytest.mark.parametrize('operations', [('order_status', 'order_details', 'shipment'), ('order_details', 'shipment', 'order_status')])
def test_redundant_order_tasks_show_details_once_and_keep_execution_records(setup, monkeypatch, end, status, operations):
    service, _ = setup
    ids = list(range(1001, end + 1))
    tool_names = {'order_status': 'get_order_status_batch', 'order_details': 'get_order_details_batch', 'shipment': 'track_shipments_batch'}
    steps = [plan(*(task(operation, ids, task_id=f'task_{index}') for index, operation in enumerate(operations, 1)))]
    for operation in operations:
        steps += [call(tool_names[operation], {'order_ids': ids}), '查询完成。']
    calls = install_script(monkeypatch, steps)
    message = f'查询1001到{end}所有订单'
    result = chat(service, message, request_id='order-list')
    body = result['reply']
    assert result['status'] == status and body.count('| 订单 |') == 1
    assert '| 订单 | 订单状态 | 金额 | 商品 | 下单日期 | 收件人 |' in body
    assert '| 1001 | 已送达 | 79.99 CNY | Wireless Headphones | 2026-06-01 | Alice Chen |' in body
    assert body.count('| 1001 |') == 1 and '物流状态' not in body
    if end == 1010:
        assert body.count('| 1008 | 未找到可访问的订单') == 1 and 'Webcam' not in body
    assert len(result['tasks']) == len(result['trace']) == len(result['batch_results']) == 3
    assert all(len(entry['result']['items']) == len(ids) for entry in result['trace'])
    assert service.snapshot(A, 'one')['turns'][0]['reply'] == body
    assert chat(service, message, request_id='order-list')['reply'] == body
    assert len(calls) == 7  # Presentation and replay perform no additional model calls.


@pytest.mark.parametrize('operation,name,label', [('order_status', 'get_order_status_batch', '订单状态'), ('shipment', 'track_shipments_batch', '物流状态')])
def test_explicit_status_queries_do_not_show_an_empty_amount_column(setup, monkeypatch, operation, name, label):
    service, _ = setup
    ids = [1001, 1002, 1008]
    install_script(monkeypatch, [plan(task(operation, ids)), call(name, {'order_ids': ids}), '已查询。'])
    result = chat(service, '只查询指定订单的' + label)
    assert f'| 订单 | {label} |' in result['reply'] and '金额' not in result['reply']
    assert '| 1001 | 已送达 |' in result['reply']
    assert '| 1008 | 未找到可访问的订单 |' in result['reply']
    assert [entry['tool'] for entry in result['trace']] == [name]


def test_order_details_do_not_hide_other_orders_or_payment_queries(setup, monkeypatch):
    service, _ = setup
    install_script(monkeypatch, [plan(task('order_status', [1001, 1002]), task('order_details', [1001], task_id='task_2'),
                                     task('payment_status', [1001], task_id='task_3')),
        call('get_order_status_batch', {'order_ids': [1001, 1002]}), '状态已查询。',
        call('get_order_details_batch', {'order_ids': [1001]}), '详情已查询。',
        call('get_payment_status', {'order_id': 1001}), '已付款。'])
    result = chat(service)
    assert '| 1001 | 已送达 |\n' not in result['reply']
    assert '| 1002 | 运输中 |' in result['reply']
    assert '| 1001 | 已送达 | 79.99 CNY |' in result['reply']
    assert '当前已付款' in result['reply']
    assert len(result['trace'][0]['result']['items']) == 2


def test_conflicting_status_or_failed_detail_result_is_not_hidden():
    result = compose_response({'status': 'partial', 'trace': [
        {'tool': 'get_order_status_batch', 'result': {'items': [
            {'order_id': 1001, 'result': {'order_id': 1001, 'status': 'placed'}},
            {'order_id': 1002, 'result': {'order_id': 1002, 'status': 'shipped'}}]}},
        {'tool': 'get_order_details_batch', 'result': {'items': [
            {'order_id': 1001, 'result': {'order_id': 1001, 'status': 'delivered', 'amount_minor': 7999, 'currency': 'CNY'}},
            {'order_id': 1002, 'result': {'error': '查询失败'}}]}}]})
    assert result['reply'].count('| 订单 |') == 2
    assert '| 1001 | 已下单 |' in result['reply'] and '| 1001 | 已送达 | 79.99 CNY |' in result['reply']
    assert '| 1002 | 运输中 |' in result['reply'] and '| 1002 | 未完成 |' in result['reply']


def test_confirm_cancel_and_replay_show_current_action_state_without_new_model_calls(setup, monkeypatch):
    service, _ = setup
    calls = install_script(monkeypatch, proposal_steps('propose_payment', 1003))
    original = chat(service, '付款1003', request_id='proposal')
    action = original['pending_actions'][0]['action_id']
    assert '尚未办理' in original['reply'] and '34.50 CNY' in original['reply']
    decided = service.confirm_action(A, action)
    assert '付款成功' in decided['reply'] and '34.50 CNY' in decided['reply']
    replay = chat(service, '付款1003', request_id='proposal')
    assert replay['replayed'] and replay['status'] == 'completed'
    assert '待确认' not in replay['reply'] and '付款成功' in replay['reply'] and len(calls) == 4
    forbidden = service.cancel_action(A, action)
    assert forbidden['status'] == 'rejected' and '不能取消' in forbidden['reply'] and '付款成功' not in forbidden['reply']
    install_script(monkeypatch, proposal_steps())
    refund = chat(service, '申请退款1001')
    cancelled = service.cancel_action(A, refund['pending_actions'][0]['action_id'])
    assert cancelled['status'] == 'cancelled' and '已取消' in cancelled['reply']
    forbidden = service.confirm_action(A, refund['pending_actions'][0]['action_id'])
    assert forbidden['status'] == 'rejected' and '不能确认执行' in forbidden['reply']


def test_stale_confirmation_does_not_look_like_pending_or_success(setup, monkeypatch):
    service, engine = setup
    install_script(monkeypatch, proposal_steps('propose_payment', 1003))
    proposal = chat(service, '用银行卡支付1003')
    with get_session_factory(engine).begin() as db:
        db.get(Order, 1003).version += 1
    result = service.confirm_action(A, proposal['pending_actions'][0]['action_id'])
    assert result['status'] == 'rejected' and '状态已变化' in result['reply'] and '未执行' in result['reply']
    assert '待确认' not in result['reply'].split('请重新查询')[0] and '付款成功' not in result['reply']


def test_toolless_followup_loses_visibility_when_its_context_order_changes_owner(setup, monkeypatch):
    service, engine = setup
    install_script(monkeypatch, [plan(task('payment_status', [1006])), call('get_payment_status', {'order_id': 1006}), '已付款。',
                                plan(kind='reply', message='该订单支付金额为59.99元。')])
    chat(service, request_id='first')
    second = chat(service, '重复一下金额', request_id='followup')
    assert not second['trace'] and second['context_order_ids'] == [1006]
    transfer(engine, 1006)
    visible = service.snapshot(A, 'one')
    assert visible['turns'] == [] and visible['issue']['active_order_id'] is None
    replay = chat(service, '重复一下金额', request_id='followup')
    assert '59.99' not in json.dumps(replay, ensure_ascii=False) and not replay.get('model_reply')
    stored = tools.request_journal().request_input('customer_a', 'online', 'followup')['result']
    assert '59.99' in stored['reply']  # Audit record is retained, view alone is filtered.


def test_legacy_completed_reply_recovers_its_permission_dependencies_from_saved_run(setup, monkeypatch):
    service, engine = setup
    install_script(monkeypatch, [plan(task('payment_status', [1006])), call('get_payment_status', {'order_id': 1006}), '已付款。',
                                plan(kind='reply', message='刚才金额为59.99元。')])
    chat(service)
    service.set_active_order(A, 'one', None, service.snapshot(A, 'one')['version'])
    second = chat(service, '刚才金额是多少', request_id='old-completed')
    assert not second['order_reference']['issue']['candidate_order_ids']
    with get_session_factory(engine).begin() as db:
        row = db.get(RequestRecord, ('customer_a', 'online', 'old-completed'))
        data = json.loads(row.result)
        data.pop('context_order_ids')
        row.result = json.dumps(data)
    transfer(engine, 1006)
    calls = install_script(monkeypatch, [])
    result = chat(service, '刚才金额是多少', request_id='old-completed')
    assert result['replayed'] and '59.99' not in json.dumps(result, ensure_ascii=False) and not calls


def test_failed_model_after_proposal_keeps_facts_and_specific_error(setup, monkeypatch):
    from src.errors import ModelConfigurationError
    service, _ = setup
    install_script(monkeypatch, [*proposal_steps('propose_payment', 1003)[:-1], ModelConfigurationError('missing config')])
    result = chat(service, '用银行卡支付1003')
    assert result['pending_actions'] and result['status'] == 'pending_confirmation'
    assert '34.50 CNY' in result['reply'] and '尚未办理' in result['reply'] and '模型配置不可用' in result['reply']


def test_lost_pending_context_is_hidden_and_never_sent_to_next_model(setup, monkeypatch):
    service, engine = setup
    calls = install_script(monkeypatch, [plan(kind='clarify', message='订单1006的付款金额59.99，需要查什么？', mentioned=[1006]),
                                       plan(kind='reply', message='请描述新的需求。')])
    chat(service)
    transfer(engine, 1006)
    assert service.snapshot(A, 'one')['pending_clarification'] is None
    result = chat(SupportService(service.store.path), '换一个问题')
    assert result['status'] == 'completed'
    assert [message.type for message in calls[-1]] == ['system', 'human']
    current = json.loads(calls[-1][-1].content)
    assert current['user_input'] == '换一个问题' and current['pending_clarification'] is None
    assert current['issue']['candidate_order_ids'] == [] and current['issue']['active_order_id'] is None


def test_resume_rechecks_pending_context_before_any_model_request(setup, monkeypatch):
    service, engine = setup
    install_script(monkeypatch, [plan(kind='clarify', message='1006需要查看哪项？', mentioned=[1006])])
    chat(service)
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, 'route_node', lambda state: (_ for _ in ()).throw(SystemExit('before route')))
        with pytest.raises(SystemExit):
            chat(service, '付款状态', request_id='interrupted')
    run = service.list_runs(A, 'one')[0]
    transfer(engine, 1006)
    calls = install_script(monkeypatch, [])
    with pytest.raises(AccessError, match='权限已变化'):
        service.resume_run(A, run['run_id'])
    assert calls == []


def test_lost_order_hides_pending_action_and_cannot_confirm_it(setup, monkeypatch):
    service, engine = setup
    install_script(monkeypatch, proposal_steps())
    proposal = chat(service, '申请1001退款')
    action = proposal['pending_actions'][0]['action_id']
    transfer(engine, 1001)
    assert not service.list_pending_actions(A, 'one')
    assert not service.snapshot(A, 'one')['turns']
    with pytest.raises(AccessError):
        service.confirm_action(A, action)
    with get_session_factory(engine)() as db:
        assert db.get(PendingAction, action).status == 'pending'


def test_context_keeps_older_batch_facts_without_replaying_old_dialogue():
    from src.context import planning_context
    old = {'user': '之前的批量查询', 'reply': '结果', 'trace': [{'tool': 'get_order_status_batch',
        'args': {'order_ids': [1001, 1006]}, 'result': {'items': [
            {'order_id': 1001, 'result': {'order_id': 1001, 'status': 'delivered'}},
            {'order_id': 1006, 'result': {'order_id': 1006, 'status': 'delivered'}}]}}]}
    turns = [old] + [{'user': '其他内容' * 200, 'reply': '回复' * 200, 'trace': []} for _ in range(10)]
    issue = {**empty_issue(), 'candidate_order_ids': [1006], 'active_order_id': 1006}
    selected, messages, info = build_context(turns, '继续刚才的事项', issue, {}, budget=1500,
                                            pending={'messages': ['查之前那笔'], 'question': '哪一笔？'})
    packet = planning_context(selected)
    assert [fact['order_id'] for fact in packet['recent_facts']] == [1001, 1006]
    assert all('之前的批量查询' != turn['user'] for turn in packet['recent_turns'])
    assert len(messages) % 2 == 0 and info['is_estimate']


def test_compact_batch_tool_message_keeps_every_order_status_and_exact_amount():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from src.context import fit_model_context
    items = [{'order_id': number, 'result': {'order_id': number, 'status': 'paid', 'amount_minor': 1234,
              'currency': 'CNY', 'item': '大段商品信息' * 1000}} for number in range(1001, 1007)]
    messages = [SystemMessage('工具事实'), HumanMessage('查询'),
        AIMessage(content='', tool_calls=[{'id': 'batch', 'name': 'get_payment_status_batch', 'args': {'order_ids': list(range(1001, 1007))}}]),
        ToolMessage(content=json.dumps({'items': items}), tool_call_id='batch')]
    fitted, info = fit_model_context(messages, [], budget=1500)
    compact = json.loads(fitted[-1].content)
    assert len(compact['items']) == 6 and info['tool_results_truncated']
    assert all(item['result']['amount_minor'] == 1234 and item['result']['status'] == 'paid' for item in compact['items'])
    assert fitted[-1].tool_call_id == fitted[-2].tool_calls[0]['id'] == 'batch'


def test_deleted_profile_is_not_reintroduced_from_a_saved_model_checkpoint(setup, monkeypatch):
    service, _ = setup
    service.update_profile(A, {'tone': 'only-this-withdrawn-preference'}, 0)
    install_script(monkeypatch, [plan(task('payment_status', [1006]))])
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, 'model_step', lambda state: (_ for _ in ()).throw(SystemExit('before expert')))
        with pytest.raises(SystemExit):
            chat(service, request_id='profile-interrupted')
    service.delete_profile_field(A, 'tone', 1)
    calls = install_script(monkeypatch, [call('get_payment_status', {'order_id': 1006}), '已付款。'])
    run = service.list_runs(A, 'one')[0]
    result = SupportService(service.store.path).resume_run(A, run['run_id'])
    assert result['status'] == 'completed' and len(calls) == 2
    assert all('only-this-withdrawn-preference' not in str(messages) for messages in calls)


def test_page_is_online_with_fact_table_and_inline_confirmation(setup, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    service, _ = setup
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(Path(service.store.path).parent))
    install_script(monkeypatch, [plan(task('payment_status', [1005, 1006])),
        call('get_payment_status_batch', {'order_ids': [1005, 1006]}), '都未付款', *proposal_steps('propose_payment', 1003)])
    st.cache_resource.clear()
    try:
        page = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=20).run()
        next(item for item in page.text_input if item.label == '测试账户令牌').set_value('test-token-a').run()
        assert not any(item.label == '运行模式' for item in page.selectbox)
        page.chat_input[0].set_value('查看两笔付款状态').run()
        assert not page.exception
        assert '| 1005 | 已退款 | 24.99 CNY |' in page.chat_message[-1].markdown[0].value
        page.chat_input[0].set_value('付款1003').run()
        assert [item.label for item in page.chat_message[-1].button] == ['确认付款', '取消操作']
        next(item for item in page.button if item.label == '确认付款').click().run()
        assert not page.exception and '付款成功' in page.chat_message[-1].markdown[0].value
        assert service.snapshot(A, 'case_1')['turns']
    finally:
        st.cache_resource.clear()
