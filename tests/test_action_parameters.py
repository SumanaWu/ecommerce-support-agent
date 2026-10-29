"""Planner parameters remain unchanged through expert calls, or require model correction."""
import json

from pydantic import ValidationError
import pytest
from sqlalchemy import select

from src import tools
from src.auth import Principal
from src.backend.db import PendingAction, get_engine, get_session_factory
from src.backend.seed import seed
from src.planning import ToolValidationError, compile_task_plan, validate_tool_scope
from src.service import SupportService
from src.understanding import UnderstoodTask
from tests.online_support import call, install_script, plan, task


A = Principal('customer_a')
CASES = [('propose_refund', 1006, 'refund_reason', '声音断断续续的', 'initiate_refund', 'reason',
          '声音断断续续', 'check_refund_eligibility', {'eligible': True}),
         ('propose_payment', 1003, 'payment_method', 'paypal', 'process_payment', 'method',
          'card', 'get_payment_status', {'status': 'pending'})]


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


@pytest.mark.parametrize('operation,number,field,value,write,parameter,changed,read,fact', CASES)
def test_compiled_parameters_reject_expert_rewrite_and_allow_exact_copy(operation, number, field, value, write, parameter, changed, read, fact):
    compiled = compile_task_plan(plan(task(operation, [number], **{field: value})), '原始需求').tasks[0]
    assert compiled[field] == value
    trace = [{'tool': read, 'result': {'order_id': number, **fact}}]
    with pytest.raises(ToolValidationError) as failure:
        validate_tool_scope(write, {'order_id': number, parameter: changed}, compiled, trace=trace)
    assert failure.value.code == 'action_parameter_mismatch'
    validate_tool_scope(write, {'order_id': number, parameter: value}, compiled, trace=trace)


@pytest.mark.parametrize('operation,number,field,value,write,parameter,changed,read,fact', CASES)
@pytest.mark.parametrize('repair', [True, False])
def test_tool_parameter_mismatch_requires_real_model_correction_before_proposal(service, monkeypatch,
        operation, number, field, value, write, parameter, changed, read, fact, repair):
    support, engine = service
    from eval.run_eval import business_snapshot
    before = business_snapshot(engine)
    bad = {'order_id': number, parameter: changed}
    correct = {'order_id': number, parameter: value}
    responses = [plan(task(operation, [number], **{field: value})), call(read, {'order_id': number}),
                 call(write, bad, call_id='invalid-parameter'),
                 call(write, correct if repair else bad, call_id='model-correction'), '请核对卡片。']
    calls = install_script(monkeypatch, responses)
    result = support.chat(A, 'parameters', f'给订单{number}办理', request_id='parameter-check')
    failures = [entry for entry in result['trace'] if entry['result'].get('error_code') == 'action_parameter_mismatch']
    assert len(failures) == 1  # Identical invalid retries stop at the repeated-call guard.
    assert all(entry['result']['executed'] is False and entry['result']['recoverable'] for entry in failures)
    with get_session_factory(engine)() as db:
        proposals = db.scalars(select(PendingAction)).all()
        assert len(proposals) == int(repair)
        if repair:
            assert json.loads(proposals[0].args) == correct
    assert result['status'] == ('pending_confirmation' if repair else 'partial')
    after = business_snapshot(engine)
    assert all(before[key] == after[key] for key in ('orders', 'payments', 'refunds'))
    count = len(calls)
    replay = support.chat(A, 'parameters', f'给订单{number}办理', request_id='parameter-check')
    assert replay['replayed'] and len(calls) == count


@pytest.mark.parametrize('field,value', [('refund_reason', ''), ('refund_reason', '长' * 301),
                                       ('refund_reason', 1), ('payment_method', 'cash')])
def test_invalid_structured_parameters_are_rejected_before_execution(field, value):
    with pytest.raises(ValidationError):
        UnderstoodTask.model_validate({**task('propose_refund', [1001]), field: value})


@pytest.mark.parametrize('operation,number,field,value,write,parameter,changed,read,fact', CASES)
def test_legacy_plan_without_structured_parameters_keeps_existing_tool_contract(operation, number, field, value, write, parameter, changed, read, fact):
    compiled = compile_task_plan(plan(task(operation, [number])), '旧版脚本需求').tasks[0]
    assert compiled['refund_reason'] is None and compiled['payment_method'] is None
    validate_tool_scope(write, {'order_id': number, parameter: changed}, compiled,
                        trace=[{'tool': read, 'result': {'order_id': number, **fact}}])
