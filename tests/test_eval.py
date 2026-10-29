"""Evaluation is isolated, explicit, and scores structured results rather than phrases."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from eval import run_eval
from src.errors import ModelConfigurationError
from tests.online_support import call, install_script, plan, task


def test_dataset_contains_core_and_varied_multi_turn_cases():
    cases = run_eval.load_dataset()
    assert len(cases) >= 60 and len({case['id'] for case in cases}) == len(cases)
    assert any(len(case['turns']) > 1 for case in cases)
    tags = {tag for case in cases for tag in case['tags']}
    assert tags >= {'range', 'colloquial', 'negation', 'condition', 'multi_turn', 'permissions', 'future'}
    messages = {turn['message'] for case in cases for turn in case['turns']}
    assert messages >= {'展示1001到1006的付款状态', '1006付了吗', '1006付款了吗', '1005付款了吗'}


@pytest.mark.parametrize('arguments', [[], ['--dry-run']])
def test_default_and_dry_run_do_not_create_business_directory(tmp_path, arguments):
    target = tmp_path / 'must-not-create'
    env = dict(os.environ, SUPPORT_DATA_DIR=str(target), SUPPORT_AGENT_DB_URL='sqlite:///' + str(target / 'business.sqlite'))
    result = subprocess.run([sys.executable, 'eval/run_eval.py', *arguments], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and 'selected=10' in result.stdout and '未调用模型或打开业务库' in result.stdout
    assert not target.exists()


def test_old_mode_cli_is_rejected(tmp_path):
    result = subprocess.run([sys.executable, 'eval/run_eval.py', '--mode', 'online'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr


def test_live_report_limits_cases_tracks_actual_settings_and_restores_data(monkeypatch, tmp_path):
    from src import llm, tools
    original = tools._engine, tools._SessionLocal
    expected = {'kind': 'tasks', 'status': 'completed', 'tasks': [
        {'operation': 'payment_status', 'expert': 'PAYMENT', 'intent': 'read', 'order_ids': [1006]}],
        'facts': [{'tool': 'get_payment_status', 'order_id': 1006, 'status': 'paid'}]}
    dataset = [{'id': str(index), 'turns': [{'message': '查询1006支付情况', 'expect': expected}]} for index in range(12)]
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(dataset))
    output = tmp_path / 'report.json'
    calls = install_script(monkeypatch, [response for _ in range(10) for response in (
        plan(task('payment_status', [1006])), call('get_payment_status', {'order_id': 1006}), '付款状态。')])
    monkeypatch.setattr(llm, 'create_chat_model', lambda: SimpleNamespace(provider='unit-provider', model_name='configured-model'))
    assert run_eval.main(['--live', '--dataset', str(path), '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['provider'] == 'unit-provider' and report['model'] == 'configured-model'
    assert report['actual_cases'] == report['actual_turns'] == 10 and report['dataset_cases'] == 12
    assert len(calls) == 30 and all(case['passed'] for case in report['cases'])
    assert not report['usage_complete'] and report['data_isolation'] == 'temporary_databases'
    assert (tools._engine, tools._SessionLocal) == original


def test_live_configuration_failure_reports_no_attempted_cases_or_secrets(monkeypatch, tmp_path):
    from src import llm
    def missing():
        raise ModelConfigurationError('secret-must-not-leak')
    monkeypatch.setattr(llm, 'create_chat_model', missing)
    path = tmp_path / 'failed.json'
    assert run_eval.main(['--live', '--output', str(path)]) == 2
    report = json.loads(path.read_text())
    assert report['actual_cases'] == 0 and report['error_type'] == 'configuration'
    assert 'secret-must-not-leak' not in path.read_text()


def test_structured_metrics_detect_wrong_intent_objects_facts_and_side_effects():
    before = {'orders': [], 'payments': [], 'refunds': [], 'pending': 0, 'tickets': 0, 'receipts': 0}
    expected = {'kind': 'tasks', 'status': 'completed', 'tasks': [
        {'operation': 'payment_status', 'expert': 'PAYMENT', 'intent': 'read', 'order_ids': [1006]}],
        'facts': [{'tool': 'get_payment_status', 'order_id': 1006, 'status': 'paid'}]}
    result = {'status': 'completed', 'understanding': plan({**task('payment_status', [1006]), 'expert': 'PAYMENT', 'intent': 'read'}), 'pending_actions': [],
        'trace': [{'tool': 'get_payment_status', 'args': {'order_id': 1006}, 'result': {'order_id': 1006, 'status': 'paid'}}]}
    assert all(run_eval.score_turn(expected, result, before, before).values())
    wrong = deepcopy(result)
    wrong['understanding']['tasks'][0].update(expert='REFUND', intent='propose', orders={'kind': 'single', 'ids': [1003]})
    wrong['trace'][0]['result']['status'] = 'pending'
    wrong['status'] = 'partial'
    after = {**before, 'payments': [('unexpected', 'paid')]}
    metrics = run_eval.score_turn(expected, wrong, before, after)
    assert not any(metrics[key] for key in ('expert_selection', 'objects', 'write_intent', 'tool_facts', 'task_completion', 'side_effects'))


def test_clarification_metrics_detect_wrong_pending_business_even_with_correct_question():
    before = {'orders': [], 'payments': [], 'refunds': [], 'pending': 0, 'tickets': 0, 'receipts': 0}
    expected = {'kind': 'clarify', 'status': 'needs_clarification', 'missing_fields': ['orders'],
                'pending_operation': 'order_details'}
    result = {'status': 'needs_clarification', 'clarification_question': '要查哪笔订单？',
              'understanding': {'kind': 'clarify', 'tasks': [], 'missing_fields': ['orders'],
                                'pending_operation': 'refund_eligibility'}}
    assert not run_eval.score_turn(expected, result, before, before)['clarification']


@pytest.mark.parametrize('change', ['duplicate', 'empty_turns', 'invalid_expert'])
def test_invalid_dataset_is_rejected_before_model_access(tmp_path, change):
    cases = run_eval.load_dataset()[:2]
    if change == 'duplicate':
        cases[1]['id'] = cases[0]['id']
    elif change == 'empty_turns':
        cases[0]['turns'] = []
    else:
        cases[0]['turns'][0]['expect']['tasks'][0]['expert'] = 'UNKNOWN'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError):
        run_eval.load_dataset(path)
