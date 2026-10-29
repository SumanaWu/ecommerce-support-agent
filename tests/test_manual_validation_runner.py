"""Verify manual-conversation evaluation against isolated, scripted real service calls."""
from copy import deepcopy
import json

import pytest

from scripts import validate_manual_conversations as manual
from src import tools
from src.auth import Principal
from src.backend.db import Order, PendingAction, get_engine, get_session_factory
from src.backend.seed import seed
from src.service import SupportService
from tests.online_support import call, install_script, plan, proposal_steps, task


A = Principal('customer_a')


def test_reason_sentence_punctuation_does_not_change_expected_meaning():
    assert manual.matches_action_args({'reason': '声音断断续续的。'}, {'reason': '声音断断续续的'})
    assert not manual.matches_action_args({'reason': '声音断断续续。'}, {'reason': '声音断断续续的'})
    assert not manual.matches_action_args({'method': 'paypal'}, {'method': 'card'})


def test_reason_punctuation_keeps_original_confirmed_parameters(service, monkeypatch):
    support, engine = service
    responses = proposal_steps()
    responses[2] = call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了。'})
    install_script(monkeypatch, responses)
    view = support.chat(A, 'manual', '1001申请退款，个人不需要了。')['pending_actions'][0]
    measured = manual.click(support, A, 'manual', {'action': 'confirm', 'order_id': 1001,
        'tool': 'initiate_refund', 'args': {'reason': '个人不需要了'}}, engine)
    assert measured['passed'] and support.get_action(A, view['action_id'])['args']['reason'] == '个人不需要了。'
    assert measured['receipts'][0]['args']['reason'] == '个人不需要了。'


@pytest.fixture
def service(tmp_path):
    previous = tools._engine, tools._SessionLocal
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    try:
        yield SupportService(tmp_path / 'memory.sqlite'), engine
    finally:
        engine.dispose()
        tools._engine, tools._SessionLocal = previous


@pytest.mark.parametrize('operation,number,decision', [('propose_refund', 1001, 'confirm'),
    ('propose_refund', 1001, 'cancel'), ('propose_payment', 1003, 'confirm'), ('propose_payment', 1003, 'cancel')])
def test_button_checks_frozen_money_exact_business_changes_and_separate_receipt(service, monkeypatch, operation, number, decision):
    support, engine = service
    calls = install_script(monkeypatch, proposal_steps(operation, number))
    result = support.chat(A, 'manual', f'请给订单{number}生成待确认提案')
    view = result['pending_actions'][0]
    model_calls = len(calls)
    measured = manual.click(support, A, 'manual', {'action': decision, 'order_id': number,
                            'tool': view['tool'], 'args': view['args']}, engine)
    assert measured['passed'] and all(measured['checks'].values())
    assert measured['frozen_money'] == {'amount_minor': 7999 if number == 1001 else 3450, 'currency': 'CNY'}
    assert len(measured['receipts']) == 1 and len(calls) == model_calls


def test_button_checks_detect_unrelated_business_mutation(service, monkeypatch):
    support, engine = service
    install_script(monkeypatch, proposal_steps())
    view = support.chat(A, 'manual', '给1001申请退款')['pending_actions'][0]
    confirm = support.confirm_action
    def accidental_write(*args):
        result = confirm(*args)
        with get_session_factory(engine).begin() as db:
            db.get(Order, 1008).amount_minor += 1
        return result
    monkeypatch.setattr(support, 'confirm_action', accidental_write)
    measured = manual.click(support, A, 'manual', {'action': 'confirm', 'order_id': 1001, 'tool': view['tool']}, engine)
    assert not measured['passed'] and not measured['checks']['only_expected_business_changes']
    assert measured['mismatched_business_tables'] == ['orders']


def test_button_checks_detect_corrupted_frozen_amount_and_refuse_wrong_target(service, monkeypatch):
    support, engine = service
    install_script(monkeypatch, proposal_steps())
    view = support.chat(A, 'manual', '给1001申请退款')['pending_actions'][0]
    wrong = manual.click(support, A, 'manual', {'action': 'confirm', 'order_id': 1006, 'tool': view['tool']}, engine)
    assert not wrong['passed'] and support.get_action(A, view['action_id'])['status'] == 'pending'
    with get_session_factory(engine).begin() as db:
        db.get(PendingAction, view['action_id']).amount_minor += 1
    measured = manual.click(support, A, 'manual', {'action': 'confirm', 'order_id': 1001, 'tool': view['tool']}, engine)
    assert not measured['passed'] and not measured['checks']['frozen_amount_and_currency']


def test_evaluator_saves_chat_and_each_independent_button_progress(monkeypatch, tmp_path):
    install_script(monkeypatch, [plan(task('payment_status', [1006])),
        call('get_payment_status', {'order_id': 1006}), '已付款。', *proposal_steps()])
    cases = [{'id': 'two_turns', 'turns': [
        {'message': '1006付了吗', 'expected_text': '已付款', 'expect': {'kind': 'tasks', 'status': 'completed',
            'tasks': [{'operation': 'payment_status', 'order_ids': [1006], 'expert': 'PAYMENT', 'intent': 'read'}]}},
        {'message': '给1001申请退款', 'expected_text': '等待独立按钮', 'expect': {'kind': 'tasks',
            'status': 'pending_confirmation', 'pending_orders': [1001],
            'tasks': [{'operation': 'propose_refund', 'order_ids': [1001], 'expert': 'REFUND', 'intent': 'propose'}]},
         'after_actions': [{'action': 'cancel', 'order_id': 1001, 'tool': 'initiate_refund'}]}]}]
    report = {'cases': [], 'actual_turns': 0, 'actual_button_actions': 0}
    progress, save = [], manual.save_report
    def record(value, output):
        progress.append(deepcopy(value))
        save(value, output)
    monkeypatch.setattr(manual, 'save_report', record)
    output = tmp_path / 'report.json'
    manual.evaluate(cases, tmp_path, report, output, 2)
    measured = json.loads(output.read_text())
    assert measured['actual_turns'] == 2 and measured['actual_button_actions'] == 1
    assert measured['cases'][0]['passed'] and all(turn['passed'] for turn in measured['cases'][0]['turns'])
    assert any(value['actual_turns'] == 2 and value['actual_button_actions'] == 0 for value in progress)
    assert any(value['actual_turns'] == 2 and value['actual_button_actions'] == 1 for value in progress)
    assert not output.with_name('report.json.tmp').exists()


def test_real_dataset_dry_run_checks_three_rounds_without_business_database(tmp_path, monkeypatch):
    unopened = tmp_path / 'must-not-open'
    monkeypatch.setenv('SUPPORT_DATA_DIR', str(unopened))
    monkeypatch.setenv('SUPPORT_AGENT_DB_URL', 'sqlite:///' + str(unopened / 'business.sqlite'))
    output = tmp_path / 'dry.json'
    assert manual.main(['--dry-run', '--round', '2', '--max-turns', '1', '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['dataset_rounds'] == 3 and report['dataset_chat_turns'] == 75
    assert report['actual_turns'] == 0 and report['selected_rounds'] == ['manual_round_2']
    assert not unopened.exists()
