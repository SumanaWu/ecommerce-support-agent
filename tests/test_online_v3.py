"""Regressions for the real multi-turn failures, with explicit model tool calls."""
import json

import pytest
from sqlalchemy import select

from src import context, tools
from src.agents import base
from src.auth import Principal
from src.backend.db import Order, PendingAction, get_engine, get_session_factory
from src.backend.seed import seed
from src.planning import compile_task_plan, task_evidence_complete
from src.service import SupportService
from src.understanding import UnderstandingResult
from tests.online_support import call, install_script

A = Principal('customer_a')


def task(operation, order_id=1001, **extra):
    return {'task_id': 'task_1', 'operation': operation, 'context_use': 'none', 'answer_mode': 'result',
            'orders': {'kind': 'single', 'ids': [order_id]},
            'instruction': '按用户要求办理，原因：个人不需要了', **extra}


def plan(*tasks, **extra):
    return {'kind': 'tasks', 'tasks': list(tasks), **extra}


@pytest.fixture
def service(tmp_path):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    yield SupportService(tmp_path / 'memory.sqlite')
    engine.dispose()


def test_compact_contract_derives_expert_and_intent():
    value = plan(task('shipment', 1002))
    assert 'expert' not in UnderstandingResult.model_json_schema()['$defs']['UnderstoodTask']['properties']
    compiled = compile_task_plan(value, '查询1002物流').tasks[0]
    assert compiled['route'] == compiled['expert'] == 'ORDER'
    assert compiled['intent'] == 'read' and not compiled['allow_write']


def test_prerequisite_validation_error_is_not_evidence():
    compiled = compile_task_plan(plan(task('propose_refund')), '退款').tasks[0]
    trace = [{'tool': 'initiate_refund', 'args': {'order_id': 1001},
              'result': {'error': '必须先查资格', 'executed': False,
                         'error_code': 'missing_prerequisite', 'recoverable': True}}]
    assert not task_evidence_complete(compiled, trace)


def test_context_retains_recent_dialogue_and_eight_historical_facts():
    turns = [{'request_id': str(n), 'user': '查资格', 'reply': '尚未送达',
              'trace': [{'tool': 'check_refund_eligibility', 'args': {'order_id': n},
                         'result': {'order_id': n, 'eligible': False, 'reason_code': 'not_delivered'}}]}
             for n in range(1001, 1021)]
    packet = context.planning_context(turns)
    assert len(packet['recent_turns']) == 6 and len(packet['recent_facts']) == 8
    assert packet['recent_turns'][-1]['user'] == '查资格'
    assert packet['recent_facts'][-1]['source_request_id'] == '1020'
    assert all(fact['historical'] for fact in packet['recent_facts'])
    assert packet['recent_facts'][-1]['result']['reason_code'] == 'not_delivered'


def test_refund_recovers_wrong_phase_and_retries_same_proposal(service, monkeypatch):
    bindings = []
    install_script(monkeypatch, [plan(task('propose_refund')),
        call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='early'),
        call('check_refund_eligibility', {'order_id': 1001}, call_id='read'),
        call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='retry'), '等待确认。'])
    original = base.get_llm
    def model():
        instance = original()
        bind = instance.bind_tools
        def bind_tools(tool_list, **kwargs):
            bindings.append([tool.name for tool in tool_list])
            return bind(tool_list, **kwargs)
        instance.bind_tools = bind_tools
        return instance
    monkeypatch.setattr(base, 'get_llm', model)
    result = service.chat(A, 'one', '帮我申请1001退款，原因是个人不需要了')
    assert result['status'] == 'pending_confirmation'
    assert bindings[:3] == [['check_refund_eligibility'], ['check_refund_eligibility'], ['initiate_refund']]
    assert result['trace'][0]['result']['error_code'] == 'missing_prerequisite'
    assert len(result['pending_actions']) == 1 and tools.check_refund_eligibility(1001)['eligible']
    with tools._session() as db:
        assert len(db.scalars(select(PendingAction)).all()) == 1


def test_why_uses_context_and_renders_verified_explanation(service, monkeypatch):
    install_script(monkeypatch, [plan(task('refund_eligibility', 1007)),
        call('check_refund_eligibility', {'order_id': 1007}), '尚未送达。'])
    service.chat(A, 'one', '只查1007能不能退款，不要申请')
    calls = install_script(monkeypatch, [plan(task('refund_eligibility', 1007, context_use='topic', answer_mode='explain')),
        call('check_refund_eligibility', {'order_id': 1007}), '请提供不能退款的原因。'])
    result = service.chat(A, 'one', '为什么')
    sent = json.loads(calls[0][-1].content)
    assert sent['recent_turns'][-1]['user'] == '只查1007能不能退款，不要申请'
    assert sent['recent_facts'][-1]['result']['reason_code'] == 'not_delivered'
    assert '运输中' in result['reply'] and '已送达' in result['reply']
    assert '请提供' not in result['reply'] and result['status'] == 'completed'


def test_order_issue_retains_topic_for_next_orders_followup(service, monkeypatch):
    install_script(monkeypatch, [plan(task('refund_eligibility', 1004)),
        call('check_refund_eligibility', {'order_id': 1004}), '已有退款记录。'])
    service.chat(A, 'one', '1004可以退款吗')
    assert service.snapshot(A, 'one')['issue']['operation'] == 'refund_eligibility'
    calls = install_script(monkeypatch, [plan(task('refund_eligibility', 1005, context_use='topic')),
        call('check_refund_eligibility', {'order_id': 1005}), '订单已取消。'])
    result = service.chat(A, 'one', '1005呢')
    assert json.loads(calls[0][-1].text)['issue']['operation'] == 'refund_eligibility'
    assert result['tasks'][0]['order_ids'] == [1005] and not result['pending_actions']


def test_previous_proposal_topic_does_not_authorize_next_order():
    from src.understanding import understand_request
    class Model:
        def invoke_structured(self, messages, schema, **kwargs):
            return {'parsed': schema.model_validate({'kind': 'clarify', 'message': '要办理什么事项？',
                'mentioned_order_ids': [1006], 'missing_fields': ['purpose']}), 'usage': {}}
    result = understand_request('1006', issue={'operation': 'propose_refund', 'active_order_id': 1001}, model=Model())
    assert result['understanding']['kind'] == 'clarify' and result['understanding']['tasks'] == []


def test_generic_order_query_does_not_silently_reuse_previous_order(service, monkeypatch):
    install_script(monkeypatch, [plan(task('refund_eligibility', 1007)),
        call('check_refund_eligibility', {'order_id': 1007}), '尚未送达。'])
    service.chat(A, 'one', '只查1007能不能退款，不要申请')
    install_script(monkeypatch, [plan(task('order_details', 1007)),
        {'kind': 'clarify', 'message': '想查刚才的1007，还是其他订单？请提供订单编号。',
         'missing_fields': ['orders'], 'pending_operation': 'order_details'}])
    result = service.chat(A, 'one', '查订单')
    assert result['status'] == 'needs_clarification' and not result['trace']
    assert service.snapshot(A, 'one')['pending_clarification']['operation'] == 'order_details'


@pytest.mark.parametrize('context_use,previous,wrong,correct,message', [
    ('order', 'order_details', ('refund_eligibility', 1007), ('refund_eligibility', 1004), '我可以退款吗'),
])
def test_reference_binding_repairs_stale_order_and_wrong_topic(context_use, previous, wrong, correct, message):
    from src.understanding import understand_request
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            selection = wrong if self.calls == 0 else correct
            self.calls += 1
            return {'parsed': schema.model_validate(plan(task(*selection, context_use=context_use))), 'usage': {}}
    model = Model()
    result = understand_request(message, issue={'operation': previous, 'active_order_id': 1004,
        'candidate_order_ids': [1004]}, context_order_ids=[1007, 1004], model=model)
    assert model.calls == 2
    chosen = result['understanding']['tasks'][0]
    assert (chosen['operation'], chosen['orders']['ids']) == (correct[0], [correct[1]])
    assert len([d for d in result['model_diagnostics'] if d['stage'] == 'understanding']) == 1


def test_new_explicit_operation_is_not_replaced_by_previous_topic():
    from src.understanding import understand_request
    class Model:
        def invoke_structured(self, messages, schema, **kwargs):
            sent = json.loads(messages[-1].text)
            assert sent['issue']['operation'] == 'refund_eligibility'
            return {'parsed': schema.model_validate(plan(task('payment_status', 1003, context_use='topic'))), 'usage': {}}
    result = understand_request('查询1003付款状态', issue={'operation': 'refund_eligibility',
        'active_order_id': 1007, 'candidate_order_ids': [1007]}, model=Model())
    assert result['understanding']['tasks'][0]['operation'] == 'payment_status'
    assert result['model_diagnostics'] == []


def test_explicit_older_order_can_override_current_order():
    from src.understanding import understand_request
    class Model:
        def invoke_structured(self, messages, schema, **kwargs):
            return {'parsed': schema.model_validate(plan(task('refund_eligibility', 1007))), 'usage': {}}
    result = understand_request('查1007能不能退款', issue={'operation': 'order_details', 'active_order_id': 1004,
        'candidate_order_ids': [1004]}, context_order_ids=[1007, 1004], model=Model())
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1007]


def test_missing_pending_operation_repair_requires_model_to_choose_operation():
    from src.understanding import understand_request
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            self.calls += 1
            value = {'kind': 'clarify', 'message': '请提供订单编号。', 'missing_fields': ['orders']}
            if self.calls == 2:
                assert 'pending_operation' in schema.model_json_schema()['required']
                assert 'anyOf' not in schema.model_json_schema()['properties']['pending_operation']
                value['pending_operation'] = 'order_details'
            return {'parsed': schema.model_validate(value), 'usage': {}}
    result = understand_request('查订单', model=Model())
    assert result['understanding']['pending_operation'] == 'order_details'
    assert len([d for d in result['model_diagnostics'] if d['stage'] == 'understanding']) == 1


def test_new_query_clarification_repair_does_not_reuse_previous_refund_topic():
    from src.understanding import understand_request
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            self.calls += 1
            if self.calls == 1:
                value = {'kind': 'clarify', 'message': '想查哪笔订单？', 'mentioned_order_ids': [1007],
                         'missing_fields': ['purpose']}
            else:
                sent = json.loads(messages[-1].text)
                assert sent['user_input'] == '帮我查一下订单'
                assert sent['issue']['operation'] == 'refund_eligibility'
                assert not sent['recent_turns'] and not sent['recent_facts']
                value = {'kind': 'clarify', 'message': '请提供订单编号。', 'missing_fields': ['orders'],
                         'pending_operation': 'order_details'}
            return {'parsed': schema.model_validate(value), 'usage': {}}
    result = understand_request('帮我查一下订单', issue={'operation': 'refund_eligibility',
        'active_order_id': 1007, 'candidate_order_ids': [1007]}, model=Model())
    assert result['understanding']['pending_operation'] == 'order_details'


def test_task_metadata_repair_keeps_validated_target_in_model_context():
    from src.understanding import understand_request
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            self.calls += 1
            value = plan(task('order_details', 1004, context_use='pending'))
            if self.calls == 1:
                value['missing_fields'] = ['orders']
            else:
                sent = json.loads(messages[-1].text)
                assert sent['task_draft'][0]['orders']['ids'] == [1004]
                assert sent['task_draft'][0]['operation'] == 'order_details'
                assert sent['issue'] == {} and not sent['recent_facts']
            return {'parsed': schema.model_validate(value), 'usage': {}}
    result = understand_request('1004', pending={'operation': 'order_details', 'candidate_order_ids': [],
        'messages': ['查订单']}, model=Model())
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1004]


@pytest.mark.parametrize('continues', [False, True])
def test_pending_context_is_available_but_does_not_force_new_explicit_business(continues):
    from src.understanding import understand_request
    class Model:
        def invoke_structured(self, messages, schema, **kwargs):
            sent = json.loads(messages[-1].text)
            assert sent['pending_clarification']['operation'] == 'order_details'
            return {'parsed': schema.model_validate(plan(task('refund_eligibility', 1004, context_use='pending'),
                                                        continues_pending=continues)), 'usage': {}}
    result = understand_request('查询1004能否退款', pending={'operation': 'order_details', 'candidate_order_ids': [],
        'messages': ['查订单']}, model=Model())
    assert result['understanding']['tasks'][0]['operation'] == 'refund_eligibility'
    assert result['model_diagnostics'] == []


def test_reply_metadata_repair_cannot_force_a_business_task():
    from src.understanding import understand_request
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            self.calls += 1
            value = {'kind': 'reply', 'message': '你好。'}
            if self.calls == 1:
                value['missing_fields'] = ['orders']
            return {'parsed': schema.model_validate(value), 'usage': {}}
    result = understand_request('你好', model=Model())
    assert result['understanding']['kind'] == 'reply' and not result['understanding']['tasks']


def test_suggested_old_order_is_not_the_target_of_a_new_pending_query(service, monkeypatch):
    install_script(monkeypatch, [plan(task('refund_eligibility', 1007)),
        call('check_refund_eligibility', {'order_id': 1007}), '尚未送达。'])
    service.chat(A, 'one', '只查1007能不能退款，不要申请')
    install_script(monkeypatch, [{'kind': 'clarify', 'message': '想查1007还是其他订单？',
        'mentioned_order_ids': [1007], 'missing_fields': ['orders'], 'pending_operation': 'order_details'}])
    service.chat(A, 'one', '查订单')
    pending = service.snapshot(A, 'one')['pending_clarification']
    assert pending['candidate_order_ids'] == [1007] and pending['operation'] == 'order_details'
    calls = install_script(monkeypatch, [plan(task('order_details', 1004, context_use='pending')),
        call('get_order_details', {'order_id': 1004}), '已查询。'])
    result = service.chat(A, 'one', '1004')
    assert json.loads(calls[0][-1].text)['pending_clarification']['candidate_order_ids'] == []
    assert result['trace'][0]['args']['order_id'] == 1004


def test_suggested_order_source_still_protects_pending_context_after_ownership_change(service, monkeypatch):
    install_script(monkeypatch, [plan(task('refund_eligibility', 1007)),
        call('check_refund_eligibility', {'order_id': 1007}), '尚未送达。'])
    service.chat(A, 'one', '只查1007能不能退款，不要申请')
    install_script(monkeypatch, [{'kind': 'clarify', 'message': '想查1007还是其他订单？',
        'mentioned_order_ids': [1007], 'missing_fields': ['orders'], 'pending_operation': 'order_details'}])
    service.chat(A, 'one', '查订单')
    with tools._session() as db:
        db.get(Order, 1007).customer_id = 'customer_b'
        db.commit()
    assert service.snapshot(A, 'one')['pending_clarification'] is None


def test_duplicate_unreferenced_reads_are_merged_before_tools_execute(service, monkeypatch):
    repeated = plan(task('order_details', 1004), task('order_details', 1004, task_id='task_2'))
    install_script(monkeypatch, [repeated, call('get_order_details', {'order_id': 1004}), '已查询。'])
    result = service.chat(A, 'one', '查询1004详情')
    assert result['status'] == 'completed' and len(result['tasks']) == len(result['trace']) == 1
    assert result['model_diagnostics'][0]['reason'] == 'duplicate_read_tasks_merged'


def test_referenced_duplicate_read_is_not_removed_from_the_dependency_graph():
    from src.understanding import merge_duplicate_reads
    condition = {'task_id': 'task_2', 'order_id': 1004, 'field': 'eligible', 'equals': True}
    value = UnderstandingResult.model_validate(plan(task('refund_eligibility', 1004),
        task('refund_eligibility', 1004, task_id='task_2'), task('propose_refund', 1004, task_id='task_3',
             depends_on=['task_2'], conditions=[condition])))
    assert len(merge_duplicate_reads(value).tasks) == 3


def test_different_policy_questions_are_not_merged_as_duplicate_reads():
    from src.understanding import merge_duplicate_reads
    value = UnderstandingResult.model_validate(plan(
        {**task('policy'), 'orders': {'kind': 'none'}, 'instruction': '查询退款政策'},
        {**task('policy', task_id='task_2'), 'orders': {'kind': 'none'}, 'instruction': '查询物流政策'}))
    assert len(merge_duplicate_reads(value).tasks) == 2


def test_mixed_dialogue_keeps_current_order_and_refund_topic(service, monkeypatch):
    turns = [
        ('只查1007能不能退款，不要申请', plan(task('refund_eligibility', 1007)), 'check_refund_eligibility', 1007),
        ('为什么', plan(task('refund_eligibility', 1007, context_use='topic', answer_mode='explain')), 'check_refund_eligibility', 1007),
        ('查订单', {'kind': 'clarify', 'message': '想查1007还是其他订单？请提供编号。',
                    'missing_fields': ['orders'], 'pending_operation': 'order_details'}, None, None),
        ('你好', {'kind': 'reply', 'message': '你好。'}, None, None),
        ('1004', plan(task('order_details', 1004, context_use='pending')), 'get_order_details', 1004),
        ('我可以退款吗', plan(task('refund_eligibility', 1004, context_use='order')), 'check_refund_eligibility', 1004),
        ('1005呢', plan(task('refund_eligibility', 1005, context_use='topic')), 'check_refund_eligibility', 1005),
    ]
    for message, understanding, tool, number in turns:
        responses = [understanding] if not tool else [understanding, call(tool, {'order_id': number}), '已查询。']
        install_script(monkeypatch, responses)
        result = service.chat(A, 'one', message)
        assert result['status'] != 'failed'
        if tool:
            assert result['trace'][0]['args']['order_id'] == number
            assert service.snapshot(A, 'one')['issue']['active_order_id'] == number
        assert not result['pending_actions']
    assert service.snapshot(A, 'one')['issue']['operation'] == 'refund_eligibility'


def test_planning_failure_keeps_diagnostics_without_raw_text(service, monkeypatch):
    from src import understanding
    from src.errors import ModelOutputError
    class Model:
        def invoke_structured(self, *args, **kwargs):
            return {'parsing_error': ModelOutputError('结构无效', diagnostics={
                'reason': 'schema_validation', 'issues': [{'path': ['tasks', 0, 'operation'],
                'type': 'literal_error', 'allowed_values': ['shipment']}]}),
                'parsed': None, 'usage': {}, 'context_info': {'budget_tokens': 6000}}
    monkeypatch.setattr(understanding, 'get_llm', Model)
    result = service.chat(A, 'one', '查物流')
    assert result['status'] == 'failed' and len(result['model_diagnostics']) == 2
    assert result['context_info']['budget_tokens'] == 6000
    saved = service.snapshot(A, 'one')['turns'][-1]
    assert saved['model_diagnostics'] == result['model_diagnostics']
    assert saved['error_type'] == 'model_output'
    assert all('previous_output' not in item for item in result['model_diagnostics'])


def test_each_proposal_stage_has_its_own_bounded_correction(service, monkeypatch):
    install_script(monkeypatch, [plan(task('propose_refund')), '请先提供资格。',
        call('check_refund_eligibility', {'order_id': 1001}), '请再次申请。',
        call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}), '等待确认。'])
    result = service.chat(A, 'one', '帮我申请1001退款，原因是个人不需要了')
    assert result['status'] == 'pending_confirmation'
    assert {row['stage'] for row in result['model_diagnostics']} == {'prerequisite', 'propose'}
    assert all(row['result'].get('error') is None for row in result['trace'])


def test_payment_argument_correction_preserves_pending_business_state(service, monkeypatch):
    install_script(monkeypatch, [plan(task('propose_payment', 1003)),
        call('get_payment_status', {'order_id': 1003}),
        call('process_payment', {'order_id': 1003, 'method': '银行卡'}),
        call('process_payment', {'order_id': 1003, 'method': 'card'}), '待确认。'])
    result = service.chat(A, 'one', '用银行卡支付1003')
    assert result['status'] == 'pending_confirmation'
    assert result['pending_actions'][0]['args']['method'] == 'card'
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert result['trace'][1]['result']['executed'] is False


def test_model_cannot_forecast_write_in_same_batch_as_prerequisite(service, monkeypatch):
    first = call('check_refund_eligibility', {'order_id': 1001}, call_id='read')
    first['tool_calls'] += call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='early')['tool_calls']
    install_script(monkeypatch, [plan(task('propose_refund')), first,
        call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='after-result'), '待确认。'])
    result = service.chat(A, 'one', '申请1001退款，原因是个人不需要了')
    assert result['status'] == 'pending_confirmation' and len(result['pending_actions']) == 1
    assert result['trace'][1]['result']['error_code'] == 'invalid_stage'


def test_recovered_read_does_not_consume_next_stages_correction(service, monkeypatch):
    first = call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='early')
    first['tool_calls'] += call('check_refund_eligibility', {'order_id': 1001}, call_id='read')['tool_calls']
    install_script(monkeypatch, [plan(task('propose_refund')), first,
        call('initiate_refund', {'order_id': 1001, 'reason': ''}, call_id='invalid-reason'),
        call('initiate_refund', {'order_id': 1001, 'reason': '个人不需要了'}, call_id='correct-reason'), '待确认。'])
    result = service.chat(A, 'one', '申请1001退款，原因个人不需要了')
    assert result['status'] == 'pending_confirmation' and len(result['pending_actions']) == 1
    assert [row['stage'] for row in result['model_diagnostics']] == ['prerequisite', 'propose']


def test_greeting_preserves_pending_need_even_without_continues_flag(service, monkeypatch):
    install_script(monkeypatch, [{'kind': 'clarify', 'message': '请提供订单编号。', 'missing_fields': ['orders'],
                                 'pending_operation': 'order_details'}])
    service.chat(A, 'one', '查订单')
    install_script(monkeypatch, [{'kind': 'reply', 'message': '你好。'}])
    service.chat(A, 'one', '你好')
    assert service.snapshot(A, 'one')['pending_clarification']['messages'] == ['查订单']
    assert service.snapshot(A, 'one')['pending_clarification']['operation'] == 'order_details'
    install_script(monkeypatch, [plan(task('order_details', 1007), continues_pending=True),
        call('get_order_details', {'order_id': 1007}), '已查询。'])
    result = service.chat(A, 'one', '1007')
    assert result['status'] == 'completed' and service.snapshot(A, 'one')['pending_clarification'] is None


@pytest.mark.parametrize('count', [20, 50])
def test_long_history_evaluation_fixture_is_read_only(service, count):
    from eval.run_eval import business_snapshot, seed_history
    before = business_snapshot(tools._engine)
    seed_history(service, A, count)
    snapshot = service.snapshot(A, 'eval')
    packet = context.planning_context(snapshot['turns'])
    assert len(snapshot['turns']) == count
    assert len(packet['recent_turns']) == 6 and len(packet['recent_facts']) == 8
    assert business_snapshot(tools._engine) == before


def test_plan_repair_preserves_dialogue_pending_and_current_input():
    from src.errors import ModelOutputError
    from src.understanding import understand_request
    pending = {'operation': 'propose_refund', 'candidate_order_ids': [1001, 1006],
               'messages': ['给那笔申请退款'], 'question': '哪一笔？'}
    class Model:
        calls = 0
        def invoke_structured(self, messages, schema, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'parsed': None, 'parsing_error': ModelOutputError('条件错误'), 'usage': {}}
            sent = json.loads(messages[-1].text)
            assert sent['user_input'] == '第二笔，原因个人不需要了'
            assert sent['pending_clarification'] == pending
            assert len(sent['recent_turns']) == 4
            if schema.__name__ != 'ReferenceCheck':
                assert len(sent['recent_facts']) == 2
            if schema.__name__ == 'ReferenceCheck':
                return {'parsed': schema.model_validate({'selections': [{'task_id': 'task_1', 'orders': {'kind': 'single', 'ids': [1006]}, 'message': '客户选择第二笔。'}]}), 'usage': {}}
            return {'parsed': schema.model_validate(plan(task('propose_refund', 1006, context_use='pending'))), 'usage': {}}
    result = understand_request('第二笔，原因个人不需要了', pending=pending,
        recent_turns=[{'user': str(n), 'reply': '已查询'} for n in range(4)],
        recent_facts=[{'order_id': 1001, 'historical': True} for _ in range(8)], model=Model())
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1006]
    assert len([d for d in result['model_diagnostics'] if d['stage'] == 'understanding']) == 1
