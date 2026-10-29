"""Clarifications may request only parameters supported by the selected operation."""
import json

import pytest
from sqlalchemy import select

from src import tools, understanding
from src.auth import Principal
from src.backend.db import PendingAction, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.errors import ModelOutputError
from src.service import SupportService
from tests.online_support import call, install_script, plan, task


A = Principal('customer_a')


def clarification(operation, parameter):
    return {**plan(kind='clarify', message='请补充当前操作支持的参数。', missing=['parameters'],
                   clarification_parameter=parameter), 'pending_operation': operation}


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


@pytest.mark.parametrize('operation,parameter', [('propose_payment', 'payment_method'),
    ('propose_refund', 'refund_reason_length'), ('propose_refund', 'order_selection'),
    ('payment_status', 'order_selection'), ('payment_status', 'request_limit')])
def test_supported_parameter_clarifications_remain_valid(operation, parameter):
    result = understanding.validate_understanding(clarification(operation, parameter))
    assert result.clarification_parameter == parameter and result.pending_operation == operation


@pytest.mark.parametrize('operation,parameter', [('propose_refund', 'payment_method'),
    ('propose_payment', 'refund_reason_length'), ('order_status', 'payment_method'),
    ('policy', 'order_selection'), ('propose_payment', None)])
def test_unsupported_or_unspecified_clarification_parameter_is_rejected(operation, parameter):
    with pytest.raises(ModelOutputError) as failure:
        understanding.validate_understanding(clarification(operation, parameter))
    issue = failure.value.diagnostics['issues'][0]
    assert issue['path'] == ['clarification_parameter']
    assert issue['type'] == ('missing_clarification_parameter' if parameter is None else 'unsupported_clarification_parameter')
    assert issue['pending_operation'] == operation


@pytest.mark.parametrize('value', [
    plan(kind='reply', message='你好', clarification_parameter='payment_method'),
    {**plan(kind='clarify', message='哪笔订单？', missing=['orders'], clarification_parameter='payment_method'),
     'pending_operation': 'propose_payment'},
    plan(task('payment_status', [1003]), clarification_parameter='payment_method')])
def test_non_parameter_results_must_clear_clarification_parameter(value):
    with pytest.raises(ModelOutputError) as failure:
        understanding.validate_understanding(value)
    assert failure.value.diagnostics['issues'][0]['type'] == 'unexpected_clarification_parameter'


@pytest.mark.parametrize('operation', ['refund_eligibility', 'propose_refund'])
def test_invented_refund_method_question_is_replanned_into_real_query_or_proposal(service, monkeypatch, operation):
    support, engine = service
    wrong = {**clarification('propose_refund', 'payment_method'), 'message': '退款要退到银行卡还是PayPal？'}
    reason = '声音断断续续的'
    steps = [wrong, plan(task(operation, [1006], refund_reason=reason if operation == 'propose_refund' else None)),
             call('check_refund_eligibility', {'order_id': 1006})]
    if operation == 'propose_refund':
        steps.append(call('initiate_refund', {'order_id': 1006, 'reason': reason}))
    steps.append('请核对当前结果。')
    calls = install_script(monkeypatch, steps)
    result = support.chat(A, 'contract', '请处理订单1006的退款需求，原因是声音断断续续的')
    assert result['status'] == ('pending_confirmation' if operation == 'propose_refund' else 'completed')
    assert '退到银行卡' not in result['reply']
    assert result['understanding']['tasks'][0]['operation'] == operation
    feedback = json.loads(calls[1][-1].text)['format_correction']['diagnostics']['issues'][0]
    assert feedback['type'] == 'unsupported_clarification_parameter'
    assert 'payment_method' not in feedback['allowed_values']
    assert result['model_diagnostics'][0]['issues'][0]['type'] == 'unsupported_clarification_parameter'
    assert result['execution_budget']['model_calls'] == len(calls)
    with get_session_factory(engine)() as db:
        proposals = db.scalars(select(PendingAction)).all()
        assert len(proposals) == int(operation == 'propose_refund')
        assert db.scalars(select(Refund).where(Refund.order_id == 1006)).one().status == 'none'
        if proposals:
            assert json.loads(proposals[0].args) == {'order_id': 1006, 'reason': reason}


def test_repeated_unsupported_refund_parameter_stops_after_one_correction(service, monkeypatch):
    support, engine = service
    wrong = clarification('propose_refund', 'payment_method')
    calls = install_script(monkeypatch, [wrong, wrong])
    result = support.chat(A, 'bounded', '给1006申请退款')
    assert result['status'] == 'failed' and result['error_type'] == 'model_output'
    assert len(calls) == result['execution_budget']['model_calls'] == 2
    assert not result['trace'] and not result['pending_actions']
    with get_session_factory(engine)() as db:
        assert not db.scalars(select(PendingAction)).all()


@pytest.mark.parametrize('intent,wrong,correct', [('refund_progress', 'order_status', 'refund_eligibility'),
    ('delivery_timing', 'payment_status', 'shipment'), ('proposal_effect', 'refund_eligibility', 'order_status')])
def test_presentation_intent_requires_operation_with_relevant_facts(intent, wrong, correct):
    invalid = plan({**task(wrong, [1004]), 'presentation_intent': intent})
    with pytest.raises(ModelOutputError) as failure:
        understanding.validate_understanding(invalid)
    issue = failure.value.diagnostics['issues'][0]
    assert issue['type'] == 'invalid_presentation_operation' and correct in issue['allowed_values']
    assert invalid['tasks'][0]['operation'] == wrong  # Validation never silently changes the operation.
    valid = understanding.validate_understanding(plan({**task(correct, [1004]), 'presentation_intent': intent}))
    assert valid.tasks[0].operation == correct and valid.tasks[0].presentation_intent == intent


def test_refund_progress_operation_mismatch_is_corrected_before_any_tool_call(service, monkeypatch):
    support, _ = service
    calls = install_script(monkeypatch, [
        plan({**task('order_status', [1004]), 'presentation_intent': 'refund_progress'}),
        plan({**task('refund_eligibility', [1004]), 'presentation_intent': 'refund_progress'}),
        call('check_refund_eligibility', {'order_id': 1004}), '退款申请仍在处理。'])
    result = support.chat(A, 'progress', '1004退款申请现在怎么样，钱到账了吗？')
    assert result['status'] == 'completed'
    assert [entry['tool'] for entry in result['trace']] == ['check_refund_eligibility']
    assert result['trace'][0]['result']['refund_status'] == 'requested'
    assert result['understanding']['tasks'][0]['operation'] == 'refund_eligibility'
    assert result['model_diagnostics'][0]['issues'][0]['type'] == 'invalid_presentation_operation'
    assert result['execution_budget']['model_calls'] == len(calls)
