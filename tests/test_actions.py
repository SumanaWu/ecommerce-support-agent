from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest
from sqlalchemy import update

from src import actions, tools
from src.backend.db import Order, get_engine, get_session_factory
from src.backend.seed import seed
from src.confirmation import build_confirmation_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command


@pytest.fixture
def business(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield engine
    engine.dispose()


def test_refund_confirmation_is_durable_and_replayed(business):
    proposal = {'type': 'refund', 'order_id': 1001, 'reason': 'test', 'amount': 1}
    action = actions.prepare_action('a', 'one', 'request', 0, proposal)
    assert action['amount'] == '79.99'
    assert tools.check_refund_eligibility(1001)['eligible']
    replay = actions.prepare_action('a', 'one', 'request', 0, proposal)
    assert replay['action_id'] == action['action_id']
    result = actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')
    assert result['status'] == 'completed'
    assert actions.decide_action('a', 'one', action['action_id'], 1, 'confirm') == result
    assert not tools.check_refund_eligibility(1001)['eligible']


def test_cancel_and_wrong_owner_cannot_write(business):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'payment', 'order_id': 1003, 'method': 'card'})
    with pytest.raises(ValueError):
        actions.decide_action('b', 'one', action['action_id'], 1, 'confirm')
    assert actions.decide_action('a', 'one', action['action_id'], 1, 'cancel')['status'] == 'cancelled'
    assert actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')['status'] == 'cancelled'
    assert tools.get_payment_status(1003)['status'] == 'pending'


def test_changed_amount_requires_new_confirmation(business):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'refund', 'order_id': 1001, 'reason': 'test'})
    with get_session_factory(business)() as session:
        session.execute(update(Order).where(Order.order_id == 1001).values(amount=80))
        session.commit()
    updated = actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')
    assert updated['status'] == 'pending' and updated['version'] == 2
    assert updated['amount'] == '80.00'
    with pytest.raises(ValueError):
        actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')
    assert tools.check_refund_eligibility(1001)['eligible']


def test_concurrent_confirmation_changes_payment_once(business):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'payment', 'order_id': 1003, 'method': 'card'})
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: actions.decide_action('a', 'one', action['action_id'], 1, 'confirm'), range(4)))
    assert all(result == results[0] for result in results)
    assert results[0]['status'] == 'completed'
    assert tools.get_payment_status(1003)['status'] == 'paid'


def test_confirmation_graph_resumes_after_restart_and_handles_actions_in_order(business, tmp_path):
    proposals = [
        {'type': 'refund', 'order_id': 1001, 'reason': 'test'},
        {'type': 'refund', 'order_id': 1006, 'reason': 'test'},
    ]
    pending = [actions.prepare_action('a', 'one', 'request', i, p)
               for i, p in enumerate(proposals)]
    config = {'configurable': {'thread_id': 'request'}}
    path = tmp_path / 'checkpoints.sqlite'
    with sqlite3.connect(path, check_same_thread=False) as conn:
        graph = build_confirmation_graph(SqliteSaver(conn))
        result = graph.invoke({'user_id': 'a', 'session_id': 'one',
                               'actions': pending, 'index': 0}, config)
        assert result['__interrupt__'][0].value['action_id'] == pending[0]['action_id']
    with sqlite3.connect(path, check_same_thread=False) as conn:
        graph = build_confirmation_graph(SqliteSaver(conn))
        first = {'action_id': pending[0]['action_id'], 'version': 1, 'decision': 'confirm'}
        result = graph.invoke(Command(resume=first), config)
        assert result['__interrupt__'][0].value['action_id'] == pending[1]['action_id']
        second = {'action_id': pending[1]['action_id'], 'version': 1, 'decision': 'cancel'}
        result = graph.invoke(Command(resume=second), config)
    assert [a['status'] for a in result['actions']] == ['completed', 'cancelled']
    assert not tools.check_refund_eligibility(1001)['eligible']
    assert tools.check_refund_eligibility(1006)['eligible']


def test_confirmation_graph_replays_already_committed_action(business, tmp_path):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'payment', 'order_id': 1003, 'method': 'card'})
    config = {'configurable': {'thread_id': 'request'}}
    with sqlite3.connect(tmp_path / 'checkpoints.sqlite', check_same_thread=False) as conn:
        graph = build_confirmation_graph(SqliteSaver(conn))
        graph.invoke({'user_id': 'a', 'session_id': 'one', 'actions': [action], 'index': 0}, config)
        actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')
        result = graph.invoke(Command(resume={'action_id': action['action_id'], 'version': 1,
                                              'decision': 'confirm'}), config)
    assert result['actions'][0]['status'] == 'completed'
    assert tools.get_payment_status(1003)['status'] == 'paid'


def test_confirmation_graph_interrupts_again_after_amount_change(business, tmp_path):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'refund', 'order_id': 1001, 'reason': 'test'})
    config = {'configurable': {'thread_id': 'request'}}
    with sqlite3.connect(tmp_path / 'checkpoints.sqlite', check_same_thread=False) as conn:
        graph = build_confirmation_graph(SqliteSaver(conn))
        graph.invoke({'user_id': 'a', 'session_id': 'one', 'actions': [action], 'index': 0}, config)
        with get_session_factory(business)() as session:
            session.execute(update(Order).where(Order.order_id == 1001).values(amount=80))
            session.commit()
        result = graph.invoke(Command(resume={'action_id': action['action_id'], 'version': 1,
                                              'decision': 'confirm'}), config)
        revised = result['__interrupt__'][0].value
        assert revised['version'] == 2 and revised['amount'] == '80.00'
        assert tools.check_refund_eligibility(1001)['eligible']
        result = graph.invoke(Command(resume={'action_id': action['action_id'], 'version': 2,
                                              'decision': 'confirm'}), config)
    assert result['actions'][0]['status'] == 'completed'


def test_action_failure_rolls_back_business_and_action_log(business, monkeypatch):
    action = actions.prepare_action('a', 'one', 'request', 0,
                                    {'type': 'refund', 'order_id': 1001, 'reason': 'test'})
    original = tools._initiate_refund
    def fail_after_write(*args):
        original(*args)
        raise RuntimeError('crash before commit')
    monkeypatch.setattr(tools, '_initiate_refund', fail_after_write)
    with pytest.raises(RuntimeError):
        actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')
    assert tools.check_refund_eligibility(1001)['eligible']
    monkeypatch.setattr(tools, '_initiate_refund', original)
    assert actions.decide_action('a', 'one', action['action_id'], 1, 'confirm')['status'] == 'completed'
