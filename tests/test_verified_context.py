"""Independent model agreement may resolve known history, never invent its source."""
import pytest

from src import tools
from src.auth import Principal
from src.backend.db import PendingAction, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.errors import ModelOutputError
from src.memory import empty_issue
from src.service import SupportService
from src.understanding import UnderstandingResult, understand_request, validate_context_binding
from tests.online_support import call, install_script, plan, reference_verdict, task


A = Principal('customer_a')


@pytest.fixture
def setup(tmp_path):
    previous = tools._engine, tools._SessionLocal
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    try:
        yield SupportService(tmp_path / 'memory.sqlite'), engine
    finally:
        tools._engine, tools._SessionLocal = previous
        engine.dispose()


def current_issue():
    return {**empty_issue(), 'candidate_order_ids': [1001], 'active_order_id': 1001,
            'operation': 'order_details', 'order_groups': [
                {'group_id': 'list_1', 'source_request_id': 'list', 'operation': 'order_details',
                 'order_ids': [1001, 1006]}]}


def refund_plan(number):
    return plan({**task('propose_refund', [number], refund_reason='低音有杂音'),
                 'context_use': 'topic', 'instruction': '为另一笔订单申请退款，原因是低音有杂音'})


def test_independently_verified_other_order_reaches_only_that_orders_tools(setup, monkeypatch):
    service, engine = setup
    calls = install_script(monkeypatch, [
        plan(task('order_details', [1001, 1006])),
        call('get_order_details_batch', {'order_ids': [1001, 1006]}), '两笔详情。',
        plan(task('order_details', [1001])), call('get_order_details', {'order_id': 1001}), '第一笔详情。',
        refund_plan(1006), reference_verdict(1006),
        call('check_refund_eligibility', {'order_id': 1006}),
        call('initiate_refund', {'order_id': 1006, 'reason': '低音有杂音'}), '等待独立确认。'])
    service.chat(A, 'one', '查1001和1006的详情', request_id='list')
    service.chat(A, 'one', '单独看看1001详情', request_id='single')
    issue = service.snapshot(A, 'one')['issue']
    assert issue['active_order_id'] == 1001 and issue['candidate_order_ids'] == [1001]
    assert any(group['order_ids'] == [1001, 1006] for group in issue['order_groups'])

    result = service.chat(A, 'one', '另一笔给我退一下，原因是低音有杂音', request_id='other')
    assert result['status'] == 'pending_confirmation' and len(calls) == 11
    assert [(entry['tool'], entry['args']['order_id']) for entry in result['trace']] == [
        ('check_refund_eligibility', 1006), ('initiate_refund', 1006)]
    assert [action['order_id'] for action in result['pending_actions']] == [1006]
    with get_session_factory(engine)() as db:
        assert [action.order_id for action in db.query(PendingAction).all()] == [1006]
        assert all(row.status == 'none' for row in db.query(Refund).filter(Refund.order_id.in_([1001, 1006])))


def test_historical_order_outside_current_selection_still_requires_independent_verification():
    value = UnderstandingResult.model_validate(refund_plan(1006))
    for verified in (None, {'task_1': [1001]}):
        with pytest.raises(ModelOutputError) as error:
            validate_context_binding(value, '另一笔给我退一下，原因是低音有杂音', current_issue(), None,
                                     verified_references=verified)
        assert error.value.diagnostics['issues'][0]['type'] == 'ungrounded_order_selection'


def test_agreeing_models_cannot_ground_an_order_absent_from_known_history(setup, monkeypatch):
    _, engine = setup
    calls = install_script(monkeypatch, [refund_plan(1007), reference_verdict(1007),
                                         refund_plan(1007), reference_verdict(1007)])
    with pytest.raises(ModelOutputError) as error:
        understand_request('另一笔给我退一下，原因是低音有杂音', issue=current_issue(),
            context_order_ids=[1001, 1006], recent_turns=[
                {'user': '查1001和1006的详情', 'reply': '两笔详情。'},
                {'user': '单独看看1001详情', 'reply': '第一笔详情。'}])
    assert len(calls) == 4
    assert error.value.diagnostics['issues'][0]['type'] == 'ungrounded_order_selection'
    assert error.value.diagnostics['issues'][0]['selected_order_ids'] == [1007]
    with get_session_factory(engine)() as db:
        assert db.query(PendingAction).count() == 0
