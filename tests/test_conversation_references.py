"""Regression for the user's full conversations; scripted decisions test execution, not model quality."""
import json

import pytest
from langchain_core.messages import AIMessage

from src.context import build_context, context_budget, planning_context
from src.memory import empty_issue, session_issue
from src.understanding import UnderstandingResult, understand_request, validate_context_binding
from tests.online_support import plan, task


def batch_turn(ids, request_id='batch'):
    return {'request_id': request_id, 'user': '查询详情', 'reply': '列表',
            'tasks': [{'task_id': 'task_1', 'operation': 'order_details', 'order_ids': ids}], 'trace': []}


def test_six_final_pairs_and_action_events_have_separate_channels(monkeypatch):
    monkeypatch.delenv('SUPPORT_CONTEXT_TOKEN_BUDGET', raising=False)
    turns = [{'request_id': str(i), 'user': '问题' + str(i), 'reply': '最终回答' + str(i),
              'model_reply': '内部原始输出', 'model_diagnostics': [{'internal': True}]} for i in range(8)]
    turns.insert(6, {'request_id': 'cancel_action', 'user': '取消动作 action', 'reply': '已取消',
                     'status': 'cancelled', 'actions': [{'action_id': 'action', 'order_id': 1006, 'status': 'cancelled'}]})
    selected, history, _ = build_context(turns, '第一笔退款', empty_issue(), {})
    packet = planning_context(selected)
    assert context_budget() == 12000
    assert [t['user'] for t in packet['recent_turns']] == ['问题' + str(i) for i in range(2, 8)]
    assert len(history) == 12 and packet['recent_events'][0]['actions'][0]['order_id'] == 1006
    assert '内部原始输出' not in json.dumps(packet, ensure_ascii=False)
    assert '取消动作' not in json.dumps(packet['recent_turns'], ensure_ascii=False)


def test_new_proposal_expert_gets_resolved_request_without_old_action_state():
    from src.context import task_context
    packet = {'recent_turns': [
        {'user': '查询1003', 'reply': '待付款', 'order_ids': [1003], 'tasks': [{'operation': 'payment_status'}]},
        {'user': '付款', 'reply': '旧提案待确认', 'order_ids': [1003], 'tasks': [{'operation': 'propose_payment'}]}],
        'recent_facts': [
            {'order_id': 1003, 'tool': 'get_payment_status', 'result': {'status': 'pending'}},
            {'order_id': 1003, 'tool': 'process_payment', 'result': {'action_id': 'old'}}],
        'recent_events': [{'actions': [{'order_id': 1003, 'status': 'cancelled'}]}]}
    scoped = task_context(packet, [1003], intent='propose')
    assert [turn['user'] for turn in scoped['recent_turns']] == ['查询1003']
    assert [fact['tool'] for fact in scoped['recent_facts']] == ['get_payment_status']
    assert not scoped['recent_events']
    assert task_context(packet, [1003]) == packet  # The planner/read-only explanations retain history.


@pytest.mark.parametrize('ids', [[1001, 1006], [1006, 1001]])
def test_legacy_lists_keep_source_order_after_single_query_and_cancel(ids):
    payload = {'issue': {**empty_issue(), 'candidate_order_ids': [1006], 'active_order_id': 1006},
               'turns': [batch_turn(ids), batch_turn([1006], 'single'),
                         {'request_id': 'cancel_action', 'user': '取消动作 action', 'reply': '已取消'}]}
    del payload['issue']['order_groups']
    issue = session_issue(payload)
    assert issue['order_groups'][0]['order_ids'] == ids
    assert issue['order_groups'][0]['source_request_id'] == 'batch'
    assert issue['candidate_order_ids'] == [1006]
    assert session_issue({**payload, 'issue': issue}) == issue


def test_new_explicit_business_is_not_forced_to_old_operation():
    value = UnderstandingResult.model_validate(plan(task('payment_status', [1003])))
    value.tasks[0].context_use = 'topic'  # Wrong metadata must not change an explicit new business.
    validate_context_binding(value, '查询1003付款状态',
                             {**empty_issue(), 'operation': 'refund_eligibility', 'candidate_order_ids': [1007]}, None)


def test_independent_task_identity_is_compiler_metadata_not_a_semantic_failure():
    from src.planning import compile_task_plan
    first = task('payment_status', [1003]); del first['task_id']
    result = understand_request('查询1003付款状态', model=Scripted(plan(first)))
    assert result['understanding']['tasks'][0]['task_id'] == 'task_1'
    assert compile_task_plan(result['understanding'], '查询1003付款状态').tasks[0]['order_id'] == 1003
    assert not result['model_diagnostics']


def test_missing_independent_identity_does_not_collide_with_explicit_id():
    from src.understanding import validate_understanding
    first = task('order_details', [1001]); del first['task_id']
    result = validate_understanding(plan(first, task('payment_status', [1003])))
    assert [task.task_id for task in result.tasks] == ['task_2', 'task_1']


def test_dependency_identity_cannot_be_invented():
    from src.understanding import validate_understanding
    first = task('refund_eligibility', [1001]); del first['task_id']
    second = task('propose_refund', [1001], task_id='task_2', depends_on=['task_1'])
    with pytest.raises(ValueError, match='依赖'):
        validate_understanding(plan(first, second))


def test_repeated_business_query_cannot_pass_as_a_reply_from_historical_facts():
    stale = plan(kind='reply', message='1007已经查询过了，尚未送达。', mentioned=[1007])
    result = understand_request('只查1007能不能退款，不要申请',
        recent_turns=[{'user': '只查1007能不能退款，不要申请', 'reply': '尚未送达。'}],
        model=Scripted(stale, plan(task('refund_eligibility', [1007]))))
    assert result['understanding']['kind'] == 'tasks'
    assert result['understanding']['tasks'][0]['operation'] == 'refund_eligibility'
    assert result['model_diagnostics'][0]['issues'][0]['type'] == 'business_reply_without_tools'


def test_reply_metadata_repair_does_not_force_old_business_on_a_thank_you():
    model = Scripted(plan(kind='reply', message='不客气。', mentioned=[1007]), plan(kind='reply', message='不客气。'))
    result = understand_request('谢谢', issue={'operation': 'refund_eligibility', 'candidate_order_ids': [1007]}, model=model)
    assert result['understanding']['kind'] == 'reply' and result['understanding']['tasks'] == []


class Scripted:
    def __init__(self, *values):
        self.values, self.calls = list(values), []

    def invoke_structured(self, messages, schema, **kwargs):
        self.calls.append((schema.__name__, json.loads(messages[-1].text)))
        value = self.values.pop(0)
        return {'parsed': schema.model_validate(value), 'usage': {}, 'raw': AIMessage(json.dumps(value))}


def verdict(ids=(), *, status='unique', message='请明确要申请退款的订单。'):
    return {'selections': [{'task_id': 'task_1',
        'orders': None if status == 'ambiguous' else {'kind': 'single', 'ids': list(ids)}, 'message': message}]}


def test_repeated_ambiguous_write_is_checked_independently_and_clarified():
    model = Scripted(plan(task('propose_refund', [1006])), verdict(status='ambiguous'))
    # Even an overconfident planner cannot create a proposal after an unanswered question.
    model.values[0]['tasks'][0]['context_use'] = 'pending'
    result = understand_request('给那笔申请退款', model=model,
        pending={'operation': 'propose_refund', 'candidate_order_ids': [1001, 1006],
                 'messages': ['给那笔申请退款'], 'question': '1001还是1006？'},
        recent_turns=[{'user': '你好', 'reply': '你好！'}])
    assert result['understanding']['kind'] == 'clarify'
    assert result['understanding']['missing_fields'] == ['orders']
    checker = model.calls[-1][1]
    assert checker['tasks'] == [{'task_id': 'task_1', 'operation': 'propose_refund'}]
    assert 'instruction' not in json.dumps(checker['tasks'])  # No planner-selected target to anchor the check.


@pytest.mark.parametrize('ids', [[1001, 1006], [1006, 1001]])
def test_model_selects_reference_program_resolves_position(ids):
    group = {'group_id': 'batch:task_1', 'source_request_id': 'batch', 'operation': 'order_details', 'order_ids': ids}
    value = plan(task('propose_refund', selection={'kind': 'reference', 'group_id': group['group_id'], 'positions': [1]}))
    value['tasks'][0]['context_use'] = 'pending'
    checked = verdict([ids[0]])
    checked['selections'][0]['orders'] = {'kind': 'reference', 'group_id': group['group_id'], 'positions': [1]}
    model = Scripted(value, checked)
    result = understand_request('第一笔，原因是个人不需要了', model=model,
        issue={**empty_issue(), 'active_order_id': 1006, 'candidate_order_ids': [1006], 'order_groups': [group]},
        pending={'operation': 'propose_refund', 'candidate_order_ids': ids, 'messages': ['给那笔退款']})
    selected = result['understanding']['tasks'][0]['orders']
    assert selected['ids'] == [ids[0]] and selected['group_id'] == group['group_id'] and selected['positions'] == [1]
    assert any(d['stage'] == 'reference_check' for d in result['model_diagnostics'])


def test_checker_disagreement_replans_without_executing_wrong_target():
    group = {'group_id': 'batch:task_1', 'source_request_id': 'batch', 'operation': 'order_details', 'order_ids': [1001, 1006]}
    wrong = task('propose_refund', [1006]); wrong['context_use'] = 'order'
    correct = task('propose_refund', selection={'kind': 'reference', 'group_id': group['group_id'], 'positions': [1]})
    correct['context_use'] = 'order'
    model = Scripted(plan(wrong), verdict([1001]), plan(correct), verdict([1001]))
    result = understand_request('给第一笔退款，原因个人不需要了', model=model,
        issue={**empty_issue(), 'active_order_id': 1006, 'candidate_order_ids': [1006], 'order_groups': [group]})
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1001]
    assert any(d.get('reason') == 'reference_mismatch' for d in result['model_diagnostics'])


@pytest.mark.parametrize('kind', ['single', 'context'])
def test_correct_order_can_use_independently_verified_list_provenance_before_focus_validation(kind):
    group = {'group_id': 'list_1', 'source_request_id': 'batch', 'order_ids': [1001, 1006]}
    current = task('propose_refund', [1001]); current['context_use'] = 'order'
    current['orders']['kind'] = kind
    checked = verdict([1001]); checked['selections'][0]['orders'] = {'kind': 'reference', 'group_id': 'list_1', 'positions': [1]}
    model = Scripted(plan(current), checked)
    result = understand_request('第一笔，原因是个人不需要了', model=model,
        issue={**empty_issue(), 'active_order_id': 1006, 'candidate_order_ids': [1006], 'order_groups': [group]})
    selected = result['understanding']['tasks'][0]['orders']
    assert selected['ids'] == [1001] and selected['group_id'] == 'list_1' and selected['positions'] == [1]
    assert len(model.calls) == 2
    assert [item['reason'] for item in result['model_diagnostics']] == ['checked']


def test_reference_uniqueness_is_derived_from_the_models_selection():
    from src.understanding import ReferenceSelection
    assert 'status' not in ReferenceSelection.model_json_schema()['properties']
    unresolved = ReferenceSelection.model_validate({'task_id': 'task_1', 'orders': None, 'message': '要给哪笔订单退款？'})
    assert unresolved.status == 'ambiguous'
    group = {'group_id': 'list_1', 'source_request_id': 'batch', 'order_ids': [1006, 1001]}
    selected = {'kind': 'reference', 'group_id': 'list_1', 'positions': [2]}
    proposal = task('propose_refund', selection=selected)
    check = {'selections': [{'task_id': 'task_1', 'orders': selected, 'message': '客户选择原列表第二个位置。'}]}
    result = understand_request('第二笔，原因是个人不需要了', model=Scripted(plan(proposal), check),
        issue={**empty_issue(), 'order_groups': [group]})
    assert result['understanding']['kind'] == 'tasks'
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1001]
    assert result['model_diagnostics'][0]['selections'][0]['status'] == 'unique'


def test_repeated_independent_disagreement_requests_order_instead_of_proposing():
    guessed = task('propose_refund', [1006]); guessed['context_use'] = 'pending'
    model = Scripted(plan(guessed), verdict([1001]), plan(guessed), verdict([1001]))
    result = understand_request('给那笔退款', model=model,
        pending={'operation': 'propose_refund', 'candidate_order_ids': [1001, 1006]})
    assert result['understanding']['kind'] == 'clarify' and not result['understanding']['tasks']
    assert result['understanding']['missing_fields'] == ['orders']


def test_invalid_reference_check_preserves_usage_and_stage():
    class InvalidCheck(Scripted):
        def invoke_structured(self, messages, schema, **kwargs):
            if schema.__name__ == 'ReferenceCheck':
                from src.errors import ModelOutputError
                return {'parsed': None, 'parsing_error': ModelOutputError('无效核验结构'), 'usage': {'total_tokens': 11}}
            return {'parsed': schema.model_validate(plan({**task('propose_refund', [1006]), 'context_use': 'pending'})),
                    'usage': {'total_tokens': 5}}
    from src.errors import ModelOutputError
    with pytest.raises(ModelOutputError) as caught:
        understand_request('给那笔退款', model=InvalidCheck(), pending={'candidate_order_ids': [1001, 1006]})
    assert caught.value.model_usage['total_tokens'] == 32
    assert all(item['stage'] == 'reference_check' for item in caught.value.model_diagnostics)


@pytest.mark.parametrize('selection', [
    {'kind': 'reference', 'group_id': 'missing', 'positions': [1]},
    {'kind': 'reference', 'group_id': 'batch:task_1', 'positions': [0]},
    {'kind': 'reference', 'group_id': 'batch:task_1', 'positions': [3]},
    {'kind': 'reference', 'group_id': 'batch:task_1', 'positions': [1], 'ids': [1006]},
])
def test_reference_bounds_and_source_are_validated(selection):
    from src.understanding import resolve_references
    value = UnderstandingResult.model_validate(plan(task('order_details', selection=selection)))
    with pytest.raises(ValueError):
        resolve_references(value, {'order_groups': [{'group_id': 'batch:task_1', 'order_ids': [1001, 1006]}]})


@pytest.fixture
def service(tmp_path):
    from src import tools
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src.service import SupportService
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    yield SupportService(tmp_path / 'memory.sqlite')
    engine.dispose()


@pytest.mark.parametrize('ids,cancelled', [([1001, 1006], 1006), ([1006, 1001], 1001)])
def test_batch_single_cancel_then_first_uses_source_and_fresh_checks(service, monkeypatch, ids, cancelled):
    from src.auth import Principal
    from tests.online_support import call, install_script, proposal_steps
    principal = Principal('customer_a')
    install_script(monkeypatch, [plan(task('order_details', ids)),
        call('get_order_details_batch', {'order_ids': ids}), '已列出。'])
    service.chat(principal, 'one', '按给出的顺序查询' + '、'.join(map(str, ids)), request_id='batch')
    original = service.snapshot(principal, 'one')['issue']['order_groups']
    install_script(monkeypatch, proposal_steps(order_id=cancelled))
    proposed = service.chat(principal, 'one', '申请' + str(cancelled) + '退款')
    action_id = proposed['pending_actions'][0]['action_id']
    service.cancel_action(principal, action_id)
    assert service.snapshot(principal, 'one')['issue']['order_groups'] == original
    planning = service.snapshot(principal, 'one', for_planning=True)
    assert '待确认' in planning['turns'][1]['reply']
    assert '已取消' in service.snapshot(principal, 'one')['turns'][1]['reply']
    selection = {'kind': 'reference', 'group_id': original[0]['group_id'], 'positions': [1]}
    current = task('propose_refund', selection=selection)
    current['instruction'] = '第一笔退款，原因个人不需要了'
    install_script(monkeypatch, [plan(current), verdict([ids[0]]),
        call('check_refund_eligibility', {'order_id': ids[0]}),
        call('initiate_refund', {'order_id': ids[0], 'reason': '个人不需要了'}), '待确认。'])
    result = service.chat(principal, 'one', '第一笔，原因是个人不需要了')
    assert result['status'] == 'pending_confirmation'
    assert result['pending_actions'][0]['order_id'] == ids[0]
    assert result['pending_actions'][0]['action_id'] != action_id
    assert [e['tool'] for e in result['trace']] == ['check_refund_eligibility', 'initiate_refund']


def test_model_reference_ambiguity_blocks_all_proposal_tools(service, monkeypatch):
    from src.auth import Principal
    from tests.online_support import install_script
    principal = Principal('customer_a')
    payload = service.store.load_session('customer_a', 'online', 'one')
    payload['issue'].update(candidate_order_ids=[1001, 1006])
    payload['pending_clarification'] = {'operation': 'propose_refund', 'messages': ['给那笔申请退款'],
                                      'question': '1001还是1006？', 'candidate_order_ids': [1001, 1006]}
    service.store.save_session_if_version('customer_a', 'online', 'one', payload, 0)
    guessed = task('propose_refund', [1006]); guessed['context_use'] = 'pending'
    install_script(monkeypatch, [plan(guessed), verdict([1001, 1006], status='ambiguous')])
    result = service.chat(principal, 'one', '给那笔申请退款')
    assert result['status'] == 'needs_clarification' and not result['trace']
    assert service.list_pending_actions(principal, 'one') == []


def test_cancelled_payment_is_not_passed_as_new_experts_execution_state(service, monkeypatch):
    from src.auth import Principal
    from tests.online_support import call, install_script, proposal_steps
    principal = Principal('customer_a')
    install_script(monkeypatch, [plan(task('order_details', [1003, 1001])),
        call('get_order_details_batch', {'order_ids': [1003, 1001]}), '已列出。'])
    service.chat(principal, 'one', '按1003、1001顺序列订单详情')
    install_script(monkeypatch, proposal_steps('propose_payment', 1003))
    previous = service.chat(principal, 'one', '用银行卡支付1003')
    old_id = previous['pending_actions'][0]['action_id']
    service.cancel_action(principal, old_id)
    group = service.snapshot(principal, 'one')['issue']['order_groups'][0]
    selection = {'kind': 'reference', 'group_id': group['group_id'], 'positions': [1]}
    current = task('propose_payment', selection=selection)
    current['instruction'] = '用银行卡重新支付列表里第一笔'
    checked = verdict([1003]); checked['selections'][0]['orders'] = selection
    calls = install_script(monkeypatch, [plan(current), checked,
        call('get_payment_status', {'order_id': 1003}),
        call('process_payment', {'order_id': 1003, 'method': 'card'}), '待确认。'])
    result = service.chat(principal, 'one', '用银行卡重新支付列表里第一笔')
    assert result['status'] == 'pending_confirmation' and result['pending_actions'][0]['action_id'] != old_id
    assert old_id in calls[0][-1].text  # The planner retains cancellation semantics.
    assert old_id not in calls[2][0].text  # The executor sees a fresh, resolved task.
    assert '1003' in calls[2][0].text and '必须调用当前查询工具' in calls[2][0].text
    assert [entry['tool'] for entry in result['trace']] == ['get_payment_status', 'process_payment']


def test_selected_target_is_current_focus_while_mentioned_candidates_stay_in_source_list(service, monkeypatch):
    from src.auth import Principal
    from tests.online_support import call, install_script
    principal = Principal('customer_a')
    install_script(monkeypatch, [plan(task('order_details', [1003, 1001])),
        call('get_order_details_batch', {'order_ids': [1003, 1001]}), '已列出。'])
    service.chat(principal, 'one', '按1003、1001的顺序查询详情')
    source = service.snapshot(principal, 'one')['issue']['order_groups'][0]
    current = task('order_details', selection={'kind': 'reference', 'group_id': source['group_id'], 'positions': [1]})
    current['context_use'] = 'topic'
    install_script(monkeypatch, [plan(current, mentioned=[1003, 1001]),
        call('get_order_details', {'order_id': 1003}), '已查询第一笔。'])
    service.chat(principal, 'one', '只查询第一笔详情')
    issue = service.snapshot(principal, 'one')['issue']
    assert issue['active_order_id'] == 1003 and issue['candidate_order_ids'] == [1003]
    assert issue['order_groups'][0]['order_ids'] == [1003, 1001]
    followup = task('refund_eligibility', [1003]); followup['context_use'] = 'order'
    calls = install_script(monkeypatch, [plan(followup),
        call('check_refund_eligibility', {'order_id': 1003}), '尚未送达。'])
    result = service.chat(principal, 'one', '我可以退款吗')
    assert json.loads(calls[0][-1].text)['issue']['active_order_id'] == 1003
    assert result['trace'][0]['args']['order_id'] == 1003


def test_lost_order_does_not_renumber_a_persisted_list(service, monkeypatch):
    from src import tools
    from src.auth import Principal
    from src.backend.db import Order
    principal = Principal('customer_a')
    payload = service.store.load_session('customer_a', 'online', 'one')
    payload['issue']['order_groups'] = [{'group_id': 'list_1', 'source_request_id': 'old', 'order_ids': [1001, 1006]}]
    service.store.save_session_if_version('customer_a', 'online', 'one', payload, 0)
    with tools._session() as db:
        db.get(Order, 1001).customer_id = 'customer_b'
        db.commit()
    group = service.snapshot(principal, 'one')['issue']['order_groups'][0]
    assert group['order_ids'] == [1001, 1006] and group['unavailable_order_ids'] == [1001]
    from tests.online_support import call, install_script
    install_script(monkeypatch, [plan(task('order_details', selection={'kind': 'reference', 'group_id': 'list_1', 'positions': [1]})),
        call('get_order_details', {'order_id': 1001}), '不可访问。'])
    result = service.chat(principal, 'one', '查询第一笔详情')
    assert 'Alice' not in result['reply'] and not result['pending_actions']
    assert result['trace'][0]['result'].get('error')
