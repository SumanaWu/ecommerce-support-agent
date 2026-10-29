from contextlib import contextmanager
from copy import deepcopy
import json

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver
import pytest
from sqlalchemy import select

import api
from src import orchestrator, tools
from src.auth import Principal
from src.backend.db import ActionRecord, Order, PendingAction, get_engine, get_session_factory
from src.backend.seed import seed
from src.service import SupportService
from src.understanding import validate_understanding
from tests.online_support import call, plan, task

A = Principal('customer_a')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    with get_session_factory(engine)() as db:
        seed(db)
    service = SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(api, 'service', service)
    for key, value in {'LLM_PROVIDER':'deepseek', 'DEEPSEEK_API_KEY':'unit-key', 'DEEPSEEK_MODEL':'deepseek-flash',
                       'DEEPSEEK_BASE_URL':'https://model.invalid', 'LLM_FALLBACKS':'[]'}.items():
        monkeypatch.setenv(key,value)
    yield service, engine
    engine.dispose()


@pytest.fixture
def scripted(monkeypatch):
    import langchain_openai
    original = langchain_openai.ChatOpenAI
    @contextmanager
    def install(sequence):
        responses, requests = list(sequence), []
        def handle(request):
            body = json.loads(request.content); requests.append(body)
            assert responses, 'unexpected model call'
            value = responses.pop(0)
            if isinstance(value, BaseException):
                raise value
            if callable(value):
                value = value(body)
            if isinstance(value, httpx.Response):
                return value
            if isinstance(value, dict) and ('kind' in value or 'selections' in value):
                value = {'role':'assistant', 'content':json.dumps(value,ensure_ascii=False)}
            elif isinstance(value,str):
                value = {'role':'assistant', 'content':value}
            return httpx.Response(200,json={'id':'test','object':'chat.completion','created':1,'model':'deepseek-flash',
                'choices':[{'index':0,'message':value,'finish_reason':'tool_calls' if value.get('tool_calls') else 'stop'}],
                'usage':{'prompt_tokens':12,'completion_tokens':8,'total_tokens':20}})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client, monkeypatch.context() as patch:
            patch.setattr(langchain_openai,'ChatOpenAI',lambda **kwargs: original(**kwargs,http_client=client))
            yield requests
    return install


def chat(service, message='查询付款状态', *, request_id=None):
    return service.chat(A,'one',message,request_id=request_id)


def checkpoint(service, result):
    run = service.store.get_run(A.customer_id,result['run_id'])
    with SqliteSaver.from_conn_string(run['checkpoint_path']) as saver:
        return orchestrator.build_graph(saver).get_state({'configurable':{'thread_id':run['run_id']}}).values


def test_complete_range_input_reaches_model_before_any_old_semantic_rules(setup, scripted, monkeypatch):
    service, _ = setup
    service.set_active_order(A,'one',1003,0)
    from src import memory, planning
    assert not hasattr(memory, 'resolve_order_reference')
    assert not hasattr(planning, 'split_task_clauses')
    assert not hasattr(planning, 'allows_write')
    value = plan(task('payment_status',selection={'kind':'range','start':1001,'end':1006}))
    with scripted([value,call('get_payment_status_batch',{'order_ids':list(range(1001,1007))}),'六笔付款结果已查询。']) as requests:
        result = chat(service,'展示1001到1006的付款状态')
    assert result['status']=='completed',result
    assert json.loads(requests[0]['messages'][-1]['content'])['user_input']=='展示1001到1006的付款状态'
    assert len(result['tasks'])==1 and result['tasks'][0]['route']=='PAYMENT'
    assert result['tasks'][0]['order_ids']==list(range(1001,1007))
    assert '"active_order_id": null' in requests[1]['messages'][0]['content']
    assert [item['order_id'] for item in result['batch_results'][0]['items']]==list(range(1001,1007))
    assert len(requests)==result['execution_budget']['real_model_calls']==3
    assert service.snapshot(A,'one')['issue']['active_order_id'] is None


@pytest.mark.parametrize('end,status', [(1006, 'completed'), (1010, 'partial')])
def test_general_order_range_requests_one_full_detail_table(setup, scripted, end, status):
    service, _ = setup
    message = f'查询1001到{end}所有订单'
    ids = list(range(1001, end + 1))
    def understand(body):
        assert json.loads(body['messages'][-1]['content'])['user_input'] == message
        return plan(task('order_details', selection={'kind': 'range', 'start': 1001, 'end': end}))
    with scripted([understand, call('get_order_details_batch', {'order_ids': ids}), '订单详情已查询。']) as requests:
        result = chat(service, message)
    assert result['status'] == status
    assert result['reply'].count('| 订单 |') == 1
    assert '| 订单 | 订单状态 | 金额 | 商品 | 下单日期 | 收件人 |' in result['reply']
    assert '| 1001 | 已送达 | 79.99 CNY | Wireless Headphones | 2026-06-01 | Alice Chen |' in result['reply']
    assert '| 1003 | 已下单 | 34.50 CNY | USB-C Hub |' in result['reply']
    if end == 1010:
        for number in (1008, 1009, 1010):
            assert f'| {number} | 未找到可访问的订单 | — | — | — | — |' in result['reply']
        assert 'Webcam' not in result['reply'] and 'Hassan' not in result['reply']
    assert len(result['tasks']) == 1 and not result['pending_actions']
    assert [entry['tool'] for entry in result['trace']] == ['get_order_details_batch']
    assert len(requests) == 3


@pytest.mark.parametrize('message',['1006付了吗','1006付款了吗','1005付款了吗'])
def test_colloquial_query_uses_model_plan_and_true_tool_request(setup,scripted,message):
    service,_=setup
    number=1005 if message.startswith('1005') else 1006
    with scripted([plan(task('payment_status',[number])),call('get_payment_status',{'order_id':number}),'根据工具结果说明状态。']):
        result=chat(service,message)
    assert result['status']=='completed' and result['tasks'][0]['route']=='PAYMENT'
    assert result['trace'][0]['result']['status']==('refunded' if number==1005 else 'paid')
    assert 'source' not in result['trace'][0] and not result['pending_actions']


def test_clarification_and_short_answer_survive_restart(setup,scripted):
    service,_=setup
    with scripted([plan(kind='clarify',message='要给1001还是1006申请退款？',mentioned=[1001,1006])]):
        first=chat(service,'1001和1006，退掉那笔')
    assert first['status']=='needs_clarification' and first['agent']=='clarify'
    restored=SupportService(service.store.path)
    assert restored.snapshot(A,'one')['pending_clarification']['messages']==['1001和1006，退掉那笔']
    continued=plan(task('propose_refund',[1006],selection={'kind':'context','ids':[1006]}),continues=True)
    continued['tasks'][0]['context_use'] = 'pending'
    with scripted([continued,{'selections': [{'task_id': 'task_1', 'orders': {'kind': 'single', 'ids': [1006]}, 'message': '第二笔1006。'}]},call('check_refund_eligibility',{'order_id':1006}),
                   call('initiate_refund',{'order_id':1006,'reason':'用户申请'},call_id='propose'), '已生成提案，等待确认。']) as requests:
        result=chat(restored,'第二笔')
    context=json.loads(requests[0]['messages'][-1]['content'])
    assert context['user_input']=='第二笔' and context['pending_clarification']['question']=='要给1001还是1006申请退款？'
    assert result['status']=='pending_confirmation' and result['pending_actions'][0]['order_id']==1006
    assert restored.snapshot(A,'one')['pending_clarification'] is None
    assert tools.check_refund_eligibility(1006)['eligible']


def test_model_clarifies_and_normal_reply_does_not_call_specialist(setup,scripted):
    service,_=setup
    with scripted([plan(kind='reply',message='你好，请告诉我需要什么帮助。')]) as requests:
        result=chat(service,'你好')
    assert result['status']=='completed' and result['agent']=='conversation' and result['trace']==[]
    assert len(requests)==1


@pytest.mark.parametrize('selection,expected',[
    ({'kind':'list','ids':[1006,1001,1006,1005]},[1006,1001,1005]),
    ({'kind':'range','start':1006,'end':1003},[1006,1005,1004,1003]),
])
def test_batch_keeps_specified_order_and_deduplicates(setup,scripted,selection,expected):
    service,_=setup
    with scripted([plan(task('payment_status',selection=selection)),call('get_payment_status_batch',{'order_ids':expected}),'查询完成。']):
        result=chat(service,'按照我给的次序列出付款情况')
    assert result['status']=='completed'
    assert result['tasks'][0]['order_ids']==expected
    assert [item['order_id'] for item in result['batch_results'][0]['items']]==expected


def test_mixed_batch_returns_owned_facts_and_identical_unavailable_items(setup,scripted):
    service,_=setup
    ids=[1001,1008,9999]
    with scripted([plan(task('payment_status',ids)),call('get_payment_status_batch',{'order_ids':ids}),'部分订单不可访问。']):
        result=chat(service,'查这些订单')
    rows=result['batch_results'][0]['items']
    assert result['status']=='partial' and rows[0]['result']['status']=='paid'
    assert rows[1]['result']==rows[2]['result']==tools.ORDER_UNAVAILABLE
    assert all('amount_minor' not in row['result'] for row in rows[1:])


@pytest.mark.parametrize('name,args', [('get_payment_status',{'order_id':1008}),
    ('get_payment_status_batch',{'order_ids':[1001,1008]}),('get_order_details',{'order_id':1001}),
    ('process_payment',{'order_id':1001,'method':'card'}),('get_payment_status',{'order_id':1001,'customer_id':'customer_b'})])
def test_tool_and_order_scope_prevents_model_from_expanding_task(setup,scripted,name,args):
    service,engine=setup
    with scripted([plan(task('payment_status',[1001])),call(name,args),'我已经完成了所有办理。','我已经完成了所有办理。']):
        result=chat(service,'只查这一笔')
    assert result['status'] in {'failed','partial'} and not result['pending_actions']
    assert result['trace'][0]['result'].get('error') and '我已经完成' not in result['reply']
    with get_session_factory(engine)() as db:
        assert not db.scalars(select(PendingAction)).all() and not db.scalars(select(ActionRecord)).all()


def test_missing_tool_once_uses_required_then_returns_to_auto_selection(setup,scripted):
    service,_=setup
    with scripted([plan(task('payment_status',[1006])),'请再给编号。',call('get_payment_status',{'order_id':1006}),'已付款。']) as requests:
        result=chat(service,'1006付了吗')
    assert result['status']=='completed'
    assert requests[2]['tool_choice']=='required' and requests[3].get('tool_choice','auto')=='auto'
    assert requests[3]['messages'][-1]['tool_call_id']=='tool-1'
    assert all(entry.get('source')!='server_read_recovery' for entry in result['trace'])


def test_understanding_format_failure_has_one_correction_and_keeps_usage(setup,scripted):
    service,_=setup
    with scripted(['not json','still not json']) as requests:
        result=chat(service,'查订单')
    assert len(requests)==2 and result['status']=='failed' and result['error_type']=='model_output'
    assert result['usage']['total_tokens']==40 and result['trace']==[]
    assert result['status']!='needs_clarification'


def test_refund_plan_correction_receives_failed_json_and_exact_field_error(setup, scripted):
    service, _ = setup
    message = '符合条件就帮我申请1002退款'
    bad = plan(task('propose_refund', [1002]))
    bad['tasks'][0]['operation'] = 'refund'
    def correct(body):
        feedback = json.loads(body['messages'][-1]['content'])['format_correction']
        assert json.loads(feedback['previous_output']) == bad
        assert {'path': ['tasks', 0, 'operation'], 'type': 'literal_error'} in feedback['diagnostics']['issues']
        assert json.loads(body['messages'][-1]['content'])['user_input'] == message
        return plan(task('propose_refund', [1002]))
    with scripted([bad, correct, call('check_refund_eligibility', {'order_id': 1002}), '订单尚未送达，不能退款。']) as requests:
        result = chat(service, message)
    assert result['status'] == 'rejected' and not result['pending_actions']
    assert '尚未送达' in result['reply'] and len(requests) == 4
    assert [entry['tool'] for entry in result['trace']] == ['check_refund_eligibility']
    assert result['planning_usage']['total_tokens'] == 40


def test_correction_bounds_previous_output_and_does_not_replace_current_input(setup, scripted):
    service, _ = setup
    def correct(body):
        feedback = json.loads(body['messages'][-1]['content'])['format_correction']
        assert feedback['previous_output_truncated'] is True and len(feedback['previous_output']) == 2000
        assert feedback['diagnostics']['reason'] == 'invalid_json'
        assert json.loads(body['messages'][-1]['content'])['user_input'] == '1006付款了吗'
        return plan(task('payment_status', [1006]))
    with scripted(['x' * 5000, correct, call('get_payment_status', {'order_id': 1006}), '已付款。']) as requests:
        result = chat(service, '1006付款了吗')
    assert result['status'] == 'completed' and len(requests) == 4


def test_self_referencing_refund_condition_is_corrected_before_tools_execute(setup, scripted):
    service, engine = setup
    bad = plan(task('propose_refund', [1001], conditions=[
        {'task_id': 'task_1', 'order_id': 1001, 'field': 'eligible', 'equals': True}]))
    def correct(body):
        feedback = json.loads(body['messages'][-1]['content'])['format_correction']
        assert {'path': ['tasks', 0, 'conditions', 0, 'task_id'], 'type': 'invalid_dependency'} in feedback['diagnostics']['issues']
        return plan(task('propose_refund', [1001]))
    reason = '用户申请退款（未提供具体原因）'
    with scripted([bad, correct, call('check_refund_eligibility', {'order_id': 1001}),
                   call('initiate_refund', {'order_id': 1001, 'reason': reason}), '等待确认。']) as requests:
        result = chat(service, '符合条件就帮我申请1001退款')
    assert result['status'] == 'pending_confirmation' and len(requests) == 5
    assert result['pending_actions'][0]['args']['reason'] == reason
    assert len(result['trace']) == 2 and result['planning_usage']['total_tokens'] == 40
    from src.backend.db import Refund
    with get_session_factory(engine)() as db:
        assert db.get(Refund, 1).status == 'none'


def test_understanding_default_example_has_no_unnecessary_conditional_steps():
    from src.understanding import EXAMPLE
    value = validate_understanding(EXAMPLE)
    assert len(value.tasks) == 1 and value.tasks[0].operation == 'shipment'
    assert value.tasks[0].depends_on == value.tasks[0].conditions == []


def test_unverified_task_message_is_corrected_and_not_logged(setup, scripted, monkeypatch):
    service, _ = setup
    from src import understanding
    events = []
    monkeypatch.setattr(understanding, 'record_event', lambda name, **fields: events.append(fields))
    bad = plan(task('propose_refund', [1002]), message='private-value-unverified-business-reply')
    def correct(body):
        current = json.loads(body['messages'][-1]['content'])
        assert current['user_input'] == '符合条件就帮我申请1002退款'
        feedback = current['format_correction']
        assert {'path': ['message'], 'type': 'must_be_empty'} in feedback['diagnostics']['issues']
        assert json.loads(feedback['previous_output']) == bad
        return plan(task('propose_refund', [1002]))
    with scripted([bad, correct, call('check_refund_eligibility', {'order_id': 1002}), '尚未送达，不能退款。']):
        result = chat(service, '符合条件就帮我申请1002退款')
    assert result['status'] == 'rejected' and not result['pending_actions']
    assert 'private-value' not in json.dumps(events) and 'private-value' not in result['reply']


def test_new_explicit_refund_does_not_replan_previous_confirmed_order(setup, scripted):
    service, _ = setup
    from tests.online_support import proposal_steps
    with scripted(proposal_steps()) as requests:
        first = chat(service, '符合条件就帮我申请1001退款')
    service.confirm_action(A, first['pending_actions'][0]['action_id'])
    def understand(body):
        assert json.loads(body['messages'][-1]['content'])['user_input'] == '符合条件就帮我申请1002退款'
        assert [message['role'] for message in body['messages']] == ['system', 'user']
        assert '1001' in json.dumps(json.loads(body['messages'][-1]['content'])['recent_turns'])
        return plan(task('propose_refund', [1002]))
    with scripted([understand, call('check_refund_eligibility', {'order_id': 1002}), '尚未送达，不能退款。']):
        second = chat(service, '符合条件就帮我申请1002退款')
    assert second['status'] == 'rejected' and not second['pending_actions']
    assert [entry['args']['order_id'] for entry in second['trace']] == [1002]


@pytest.mark.parametrize('message', ['1004', '1006付了吗', '1006付款了吗'])
def test_long_understanding_history_is_reference_data_not_chat_messages(setup, scripted, message):
    from src.context import planning_context
    from src.memory import empty_issue
    from src.understanding import understand_request
    history = [{'user': f'历史问题{index}', 'reply': f'历史客服回答{index}', 'tasks': [
        {'operation': 'order_status', 'order_ids': [1001 + index], 'status': 'completed'}]} for index in range(22)]
    packet = planning_context(history)
    references = packet['recent_turns']
    expected = plan(kind='clarify', message='请问要查询订单1004的什么信息？', mentioned=[1004]) if message == '1004' else plan(task('payment_status', [1006]))
    def understand(body):
        assert [entry['role'] for entry in body['messages']] == ['system', 'user']
        current = json.loads(body['messages'][-1]['content'])
        assert current['user_input'] == message and current['issue'] == empty_issue()
        assert list(current)[-1] == 'user_input'
        assert current['profile'] == {'tone': '简短'}
        assert current['recent_turns'] == references and len(references) == 6
        assert '历史问题21' in body['messages'][-1]['content'] and '历史客服回答21' in body['messages'][-1]['content']
        assert '历史问题0' not in body['messages'][-1]['content']
        return expected
    with scripted([understand]) as requests:
        result = understand_request(message, recent_turns=references, issue=empty_issue(), profile={'tone': '简短'})
    assert result['understanding']['kind'] == expected['kind'] and len(requests) == 1
    assert result['context_info']['history_format'] == 'conversation_facts'


def test_understanding_correction_preserves_reference_history_and_pending_request(setup, scripted):
    from src.memory import empty_issue
    from src.understanding import understand_request
    references = [{'user': '查询1002', 'reply': '已查询', 'tasks': [{'operation': 'order_status', 'order_ids': [1002]}]}]
    pending = {'messages': ['查付款状态'], 'question': '请提供订单编号。', 'candidate_order_ids': []}
    def correct(body):
        assert [entry['role'] for entry in body['messages']] == ['system', 'user']
        current = json.loads(body['messages'][-1]['content'])
        assert current['user_input'] == '1006' and current['pending_clarification'] == pending
        assert list(current)[-1] == 'user_input'
        assert current['recent_turns'] == references
        assert current['format_correction']['previous_output'] == '请提供订单编号。'
        return plan(task('payment_status', [1006]), continues=True)
    with scripted(['请提供订单编号。', correct]) as requests:
        result = understand_request('1006', recent_turns=references, pending=pending, issue=empty_issue())
    assert len(requests) == 2 and result['understanding']['continues_pending']
    assert result['understanding']['tasks'][0]['orders']['ids'] == [1006]


def test_long_session_payment_uses_current_tool_fact_not_historical_answer(setup, scripted):
    from src.backend.db import Payment, Refund
    service, engine = setup
    payload = service.store.load_session(A.customer_id, 'online', 'one')
    payload['turns'] = [{'user': '1006付款了吗', 'reply': '订单1006已付款。', 'agent': 'payment',
        'status': 'completed', 'context_order_ids': [1006], 'trace': [], 'tasks': [
            {'operation': 'payment_status', 'order_ids': [1006], 'status': 'completed'}]} for _ in range(22)]
    service.store.save_session_if_version(A.customer_id, 'online', 'one', payload, 0)
    with get_session_factory(engine).begin() as db:
        db.get(Payment, 6).status = 'refunded'
        db.get(Refund, 6).status = 'completed'
    def understand(body):
        assert [message['role'] for message in body['messages']] == ['system', 'user']
        current = json.loads(body['messages'][-1]['content'])
        assert current['user_input'] == '1006付了吗' and len(current['recent_turns']) == 6
        assert '已付款' in body['messages'][-1]['content']  # Historical prose is data; current tools still decide.
        return plan(task('payment_status', [1006]))
    with scripted([understand, call('get_payment_status', {'order_id': 1006}), '订单1006已付款。']) as requests:
        result = chat(service, '1006付了吗')
    assert result['status'] == 'completed' and '当前已退款' in result['reply']
    assert [entry['tool'] for entry in result['trace']] == ['get_payment_status']
    assert len(requests) == 3 and not result['pending_actions']


def test_invalid_expert_is_not_silently_rerouted_and_model_must_correct(setup,scripted):
    service,_=setup
    bad=plan(task('payment_status',[1006]));bad['tasks'][0]['expert']='ORDER'
    with scripted([bad,plan(task('payment_status',[1006])),call('get_payment_status',{'order_id':1006}),'已付款。']) as requests:
        result=chat(service,'看看这笔付款')
    assert result['status']=='completed' and result['tasks'][0]['route']=='PAYMENT'
    assert '上次结构校验失败' in requests[1]['messages'][0]['content']
    assert result['planning_usage']['total_tokens']==40 and result['usage']['total_tokens']==80


@pytest.mark.parametrize('value',[
    plan(task('payment_status',selection={'kind':'range','start':1001,'end':1021})),
    plan(*[task('payment_status',[1001],task_id='task_'+str(i)) for i in range(7)]),
    plan(task('payment_status',[1001]),task('payment_status',[1002])),
    plan(task('payment_status',[True])),
    plan(task('payment_status',[2**63])),
    plan(task('payment_status',[1001],selection={'kind':'context','ids':[1001]})),
    plan(task('propose_refund',selection={'kind':'range','start':1001,'end':1006})),
    plan(task('propose_payment',[1001,1003])),
    plan(task('payment_status',[1001],depends_on=['missing'])),
    plan(task('propose_payment',[1003]),task('payment_status',[1003],task_id='next',depends_on=['task_1'])),
])
def test_invalid_plans_are_rejected_by_structure_and_resource_constraints(value):
    with pytest.raises(ValueError):validate_understanding(value)


def test_range_write_is_clarified_by_model_not_expanded_into_proposals(setup,scripted):
    service,engine=setup
    with scripted([plan(kind='clarify',message='范围退款需要逐笔明确，请告诉我先办理哪一笔。',mentioned=list(range(1001,1007)))]) as requests:
        result=chat(service,'把1001到1006都退款')
    assert len(requests)==1 and result['status']=='needs_clarification' and result['trace']==[]
    with get_session_factory(engine)() as db:assert not db.scalars(select(PendingAction)).all()


@pytest.mark.parametrize('number,expected',[(1001,'pending_confirmation'),(1002,'partial')])
def test_condition_is_decided_from_tools_not_model_prose(setup,scripted,number,expected):
    service,engine=setup
    first=task('refund_eligibility',[number])
    condition={'task_id':'task_1','order_id':number,'field':'eligible','equals':True}
    second=task('propose_refund',[number],task_id='task_2',depends_on=['task_1'],conditions=[condition])
    sequence=[plan(first,second),call('check_refund_eligibility',{'order_id':number}),'符合资格，可以退款。']
    if number==1001:
        sequence += [call('initiate_refund',{'order_id':number,'reason':'用户申请'}),'等待确认。']
    with scripted(sequence):result=chat(service,f'符合条件就申请{number}退款')
    assert result['status']==expected,result
    assert result['tasks'][0]['status']=='completed'
    assert result['tasks'][1]['condition_result']==('true' if number==1001 else 'false')
    assert tools.check_refund_eligibility(number)['eligible']==(number==1001)
    with get_session_factory(engine)() as db:assert len(db.scalars(select(PendingAction)).all())==(number==1001)


def test_unknown_condition_and_future_event_do_not_create_proposal(setup,scripted):
    service,engine=setup
    condition={'task_id':'task_1','order_id':1008,'field':'eligible','equals':True}
    value=plan(task('refund_eligibility',[1008]),task('propose_refund',[1008],task_id='task_2',depends_on=['task_1'],conditions=[condition]))
    with scripted([value,call('check_refund_eligibility',{'order_id':1008}),'不可访问。']):
        result=chat(service,'1008能退就退')
    assert result['status']=='partial' and result['tasks'][1]['status']=='blocked'
    with scripted([plan(kind='clarify',message='我无法等待未来到账后自动办理，请到账后再发起付款。')]) as requests:
        future=chat(service,'等到账再付')
    assert len(requests)==1 and future['status']=='needs_clarification'
    with get_session_factory(engine)() as db:assert not db.scalars(select(PendingAction)).all()


def test_conditions_cannot_reference_model_text_or_wrong_fact_type():
    first=task('payment_status',[1001])
    for field,equals in [('eligible',True),('payment_status','任意字符串'),('payment_status',True)]:
        value=plan(first,task('propose_payment',[1003],task_id='task_2',depends_on=['task_1'],
            conditions=[{'task_id':'task_1','order_id':1001,'field':field,'equals':equals}]))
        with pytest.raises(ValueError):validate_understanding(value)


def test_explicit_writes_need_prerequisite_reads_and_each_independent_confirmation(setup,scripted):
    service,engine=setup
    value=plan(task('propose_refund',[1001]),task('propose_payment',[1003],task_id='task_2'))
    with scripted([value,call('check_refund_eligibility',{'order_id':1001}),
        call('initiate_refund',{'order_id':1001,'reason':'用户申请'},call_id='refund'),'退款提案待确认。',
        call('get_payment_status',{'order_id':1003}),call('process_payment',{'order_id':1003,'method':'card'},call_id='payment'),'付款提案待确认。']):
        result=chat(service,'给1001退款，并替1003付款',request_id='two-proposals')
    pending={row['order_id']:row['action_id'] for row in result['pending_actions']}
    assert len(pending)==2 and tools.get_payment_status(1003)['status']=='pending'
    assert service.confirm_action(A,pending[1001])['status']=='completed'
    assert tools.get_payment_status(1003)['status']=='pending'
    assert service.confirm_action(A,pending[1003])['status']=='completed'
    assert service.confirm_action(A,pending[1003])['replayed']
    replay=chat(service,'给1001退款，并替1003付款',request_id='two-proposals')
    assert replay['replayed'] and not replay['pending_actions']
    with get_session_factory(engine)() as db:assert len(db.scalars(select(ActionRecord)).all())==4


def test_writing_before_eligibility_read_is_blocked(setup,scripted):
    service,engine=setup
    with scripted([plan(task('propose_refund',[1001])),call('initiate_refund',{'order_id':1001,'reason':'跳过查询'}),'已提交。','已提交。']):
        result=chat(service,'申请1001退款')
    assert not result['pending_actions'] and result['trace'][0]['result'].get('error')
    assert '已提交。' not in result['reply']
    with get_session_factory(engine)() as db:assert not db.scalars(select(PendingAction)).all()


def test_read_negation_never_authorizes_proposal(setup,scripted):
    service,engine=setup
    with scripted([plan(task('refund_eligibility',[1001])),call('initiate_refund',{'order_id':1001,'reason':'模型越权'}),'退款已办理。','退款已办理。']):
        result=chat(service,'不要退款，只看看能不能退')
    assert result['status']=='failed' and not result['pending_actions'] and '退款已办理。' not in result['reply']
    assert tools.check_refund_eligibility(1001)['eligible']


def test_model_answer_failure_after_proposal_keeps_pending_fact_and_replay(setup,scripted,monkeypatch):
    service,engine=setup
    monkeypatch.setattr('src.errors.time.sleep',lambda _:None)
    with scripted([plan(task('propose_refund',[1001])),call('check_refund_eligibility',{'order_id':1001}),
        call('initiate_refund',{'order_id':1001,'reason':'申请退款'},call_id='propose'),
        httpx.Response(503,json={}),httpx.Response(503,json={})]):
        result=chat(service,'申请1001退款',request_id='answer-failed')
    assert result['status']=='pending_confirmation' and len(result['pending_actions'])==1
    replay=chat(service,'申请1001退款',request_id='answer-failed')
    assert replay['replayed'] and replay['pending_actions'][0]['action_id']==result['pending_actions'][0]['action_id']
    assert tools.check_refund_eligibility(1001)['eligible']
    with get_session_factory(engine)() as db:assert len(db.scalars(select(PendingAction)).all())==1


def test_budget_counts_understanding_and_stops_before_specialist(setup,scripted,monkeypatch):
    service,_=setup
    monkeypatch.setenv('SUPPORT_REQUEST_MAX_MODEL_CALLS','1')
    with scripted([plan(task('payment_status',[1006]))]) as requests:
        result=chat(service,'1006付了吗')
    assert len(requests)==1 and result['stop_reason']=='budget_exhausted'
    assert result['execution_budget']['real_model_calls']==1 and result['trace']==[]


def test_saved_batch_result_cannot_be_replayed_after_ownership_changes(setup,scripted):
    service,engine=setup
    with scripted([plan(task('payment_status',[1001,1006])),call('get_payment_status_batch',{'order_ids':[1001,1006]}),'结果已返回。']):
        first=chat(service,'查两笔',request_id='owned-batch')
    with get_session_factory(engine).begin() as db:db.get(Order,1006).customer_id='customer_b'
    replay=chat(service,'查两笔',request_id='owned-batch')
    assert replay['replayed'] and replay['trace']==[] and replay['batch_results']==[]
    assert replay['understanding']=={} and not service.snapshot(A,'one')['turns']


def test_new_checkpoint_reuses_model_plan_and_real_tool_id_after_restart(setup,scripted,monkeypatch):
    service,_=setup
    with scripted([plan(task('payment_status',[1006])),call('get_payment_status',{'order_id':1006},call_id='saved-model-id')]):
        with monkeypatch.context() as patch:
            patch.setattr(orchestrator,'tool_step',lambda state: (_ for _ in ()).throw(SystemExit('before tool')))
            with pytest.raises(SystemExit):chat(service,'1006付了吗',request_id='interrupted')
    run=service.list_runs(A,'one')[0]
    def resume_reply(body):
        assert body['messages'][-1]['tool_call_id']=='saved-model-id'
        return '订单已付款。'
    with scripted([resume_reply]) as requests:
        result=SupportService(service.store.path).resume_run(A,run['run_id'])
    assert len(requests)==1 and result['replayed'] and result['status']=='completed'
    assert result['execution_budget']['real_model_calls']==3
    state=checkpoint(service,result)
    assert state['pipeline_version']=='online-v4' and state['understanding']['tasks'][0]['expert']=='PAYMENT'


@pytest.mark.parametrize('version', [None, 'online-v2', 'online-v3'])
def test_incompatible_old_online_run_only_returns_saved_actions(setup,scripted,monkeypatch,version):
    service,_=setup
    with scripted([plan(task('payment_status',[1006])),call('get_payment_status',{'order_id':1006})]):
        with monkeypatch.context() as patch:
            patch.setattr(orchestrator,'tool_step',lambda state: (_ for _ in ()).throw(SystemExit('stop')))
            with pytest.raises(SystemExit):chat(service,'查付款',request_id='old-version')
    run=service.store.list_runs(A.customer_id,'online','one')[0]
    saved=deepcopy(run['input']);saved['pipeline_version']=version
    with service.store.connect() as db:db.execute('UPDATE graph_runs SET input=? WHERE run_id=?',(json.dumps(saved),run['run_id']))
    with scripted([]) as requests:result=service.resume_run(A,run['run_id'])
    assert requests==[] and result['stop_reason']=='pipeline_incompatible' and result['trace']==[]


def test_incomplete_batch_cannot_be_claimed_as_completed_by_model(setup,scripted):
    service,_=setup
    with scripted([plan(task('payment_status',[1001,1006])),call('get_payment_status',{'order_id':1001}),'两笔都已付款。','两笔都已付款。']):
        result=chat(service,'查两笔付款')
    assert result['status']=='partial' and result['stop_reason']=='missing_tool_call'
    assert len(result['trace'])==1 and '两笔都已付款' not in result['reply']


def test_oversized_range_can_only_repair_to_valid_model_output(setup,scripted):
    service,_=setup
    invalid=plan(task('payment_status',selection={'kind':'range','start':1,'end':100000000}))
    with scripted([invalid,plan(kind='clarify',message='范围超过20笔，请缩小查询范围。')]) as requests:
        result=chat(service,'查1到一亿的付款情况')
    assert len(requests)==2 and result['status']=='needs_clarification' and result['trace']==[]


def test_no_missing_user_information_is_inferred_from_model_http_failure(setup,scripted,monkeypatch):
    service,_=setup
    monkeypatch.setattr('src.errors.time.sleep',lambda _:None)
    with scripted([httpx.Response(503,json={}),httpx.Response(503,json={})]):
        result=chat(service,'1006付了吗')
    assert result['status']=='failed' and result['error_type']=='service' and result['status']!='needs_clarification'
    assert result['trace']==[]


def test_partial_failure_keeps_independent_tasks_and_completed_facts(setup,scripted,monkeypatch):
    service,_=setup
    monkeypatch.setattr('src.errors.time.sleep',lambda _:None)
    with scripted([plan(task('order_status',[1001]),task('payment_status',[1006],task_id='task_2')),
        httpx.Response(503,json={}),httpx.Response(503,json={}),call('get_payment_status',{'order_id':1006}),'已付款。']):
        result=chat(service,'查订单状态，同时查另一笔付款')
    assert result['status']=='partial' and result['error_type']=='service' and result['stop_reason']=='model_error'
    assert [item['status'] for item in result['tasks']]==['failed','completed']
    assert result['trace'][0]['result']['order_id']==1006


def test_malformed_tool_ids_do_not_execute_anything(setup,scripted):
    service,_=setup
    request=call('get_payment_status',{'order_id':1006})
    request['tool_calls'] += request['tool_calls']
    with scripted([plan(task('payment_status',[1006])),request]):
        result=chat(service,'查付款')
    assert result['status']=='failed' and result['stop_reason']=='invalid_tool_call' and result['trace']==[]


def test_pending_clarification_memory_can_be_repaired_without_repeating_model(setup,scripted,monkeypatch):
    service,_=setup
    with scripted([plan(kind='clarify',message='要处理哪笔订单？')]) as requests:
        with monkeypatch.context() as patch:
            patch.setattr(service.store,'append_turn_once',lambda *a,**k: (_ for _ in ()).throw(OSError('disk error')))
            first=chat(service,'给那笔退款',request_id='memory-repair')
    assert not first['memory_saved'] and len(requests)==1
    with scripted([]) as requests:replay=chat(service,'给那笔退款',request_id='memory-repair')
    assert replay['replayed'] and replay['memory_saved'] and requests==[]
    snapshot=service.snapshot(A,'one')
    assert len(snapshot['turns'])==1 and snapshot['pending_clarification']['messages']==['给那笔退款']


def test_same_request_concurrently_does_not_start_a_second_plan(setup,scripted):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    service,_=setup
    entered,release=Event(),Event()
    def waiting_plan(body):
        entered.set()
        assert release.wait(5)
        return plan(task('payment_status',[1006]))
    with scripted([waiting_plan,call('get_payment_status',{'order_id':1006}),'已付款。']) as requests:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running=executor.submit(chat,service,'1006付了吗',request_id='concurrent')
            try:
                assert entered.wait(5)
                duplicate=chat(service,'1006付了吗',request_id='concurrent')
                assert duplicate['status']=='in_progress' and len(requests)==1
            finally:release.set()
            result=running.result(timeout=10)
    assert result['status']=='completed' and len(requests)==3


def test_client_cannot_inject_a_trusted_understanding_plan(setup):
    from fastapi.testclient import TestClient
    client=TestClient(api.app,headers={'Authorization':'Bearer test-token-a'})
    response=client.post('/chat',json={'message':'查付款','understanding':plan(task('propose_payment',[1003]))})
    assert response.status_code==422


def test_independent_process_commit_checkpoint_gap_replays_proposal_once(setup,scripted):
    import os,subprocess,sys
    service,engine=setup
    value=plan(task('propose_refund',[1001]))
    child='''
import os
from langchain_core.messages import AIMessage
from src import orchestrator
from src.agents import base
from src.auth import Principal
from src.budget import check_execution_budget
from src.service import SupportService
value=VALUE

def understand(*a,**k):
    check_execution_budget('model',consume=True)
    return {'understanding':value,'usage':{}}
orchestrator.understand_request=understand
class Model:
    def bind_tools(self,tools):return self
    def invoke(self,messages):
        if any(getattr(m,'type',None)=='tool' for m in messages):
            return AIMessage(content='',tool_calls=[{'name':'initiate_refund','args':{'order_id':1001,'reason':'用户申请'},'id':'proposal-from-model','type':'tool_call'}])
        return AIMessage(content='',tool_calls=[{'name':'check_refund_eligibility','args':{'order_id':1001},'id':'read-from-model','type':'tool_call'}])
base.get_llm=Model
original=orchestrator.tool_step

def stop(state):
    result=original(state)
    if result['trace'][-1]['tool']=='initiate_refund':os._exit(23)
    return result
orchestrator.tool_step=stop
SupportService().chat(Principal('customer_a'),'one','申请1001退款',request_id='commit-gap')
'''.replace('VALUE',repr(value))
    directory=str(__import__('pathlib').Path(service.store.path).parent)
    env=dict(os.environ,SUPPORT_DATA_DIR=directory,SUPPORT_AGENT_DB_URL=str(engine.url))
    process=subprocess.run([sys.executable,'-c',child],env=env,capture_output=True,text=True,timeout=20)
    assert process.returncode==23,process.stderr
    run=service.list_runs(A,'one')[0]
    with scripted(['已生成提案，等待确认。']) as requests:result=service.resume_run(A,run['run_id'])
    assert len(requests)==1 and result['status']=='pending_confirmation'
    assert result['execution_budget']['tool_calls']==2
    with get_session_factory(engine)() as db:
        assert len(db.scalars(select(ActionRecord)).all())==len(db.scalars(select(PendingAction)).all())==1
    action=result['pending_actions'][0]['action_id']
    assert service.confirm_action(A,action)['status']=='completed' and service.confirm_action(A,action)['replayed']


def test_payment_condition_uses_owned_item_in_partial_batch_without_waiting_for_foreign_item(setup,scripted):
    service,_=setup
    condition={'task_id':'task_1','order_id':1003,'field':'payment_status','equals':'pending'}
    value=plan(task('payment_status',[1003,1008]),task('propose_payment',[1003],task_id='task_2',depends_on=['task_1'],conditions=[condition]))
    with scripted([value,call('get_payment_status_batch',{'order_ids':[1003,1008]}),'1003待付款，另一笔不可访问。',
        call('process_payment',{'order_id':1003,'method':'card'}),'已创建付款提案。']):
        result=chat(service,'查两笔付款，1003待付就申请付款')
    assert result['status']=='partial' and result['tasks'][1]['condition_result']=='true'
    assert result['pending_actions'][0]['order_id']==1003 and tools.get_payment_status(1003)['status']=='pending'


def test_order_status_condition_uses_order_fact_not_payment_status(setup,scripted):
    service,_=setup
    condition={'task_id':'task_1','order_id':1001,'field':'order_status','equals':'delivered'}
    value=plan(task('order_status',[1001]),task('propose_refund',[1001],task_id='task_2',depends_on=['task_1'],conditions=[condition]))
    with scripted([value,call('get_order_status',{'order_id':1001}),'已送达。',call('check_refund_eligibility',{'order_id':1001}),
        call('initiate_refund',{'order_id':1001,'reason':'用户申请'}),'等待确认。']):
        result=chat(service,'1001送达就申请退款')
    assert result['tasks'][1]['condition_result']=='true' and result['status']=='pending_confirmation'
