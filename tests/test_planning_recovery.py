"""Formatting repair preserves independently checked targets and capability replies."""
import json

from langchain_core.messages import AIMessage
import pytest
from sqlalchemy import select

from eval.run_eval import seed_history
from src import tools, understanding
from src.agents import base
from src.auth import Principal
from src.backend.db import PendingAction, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.budget import check_execution_budget
from src.errors import ModelOutputError
from src.service import SupportService
from tests.online_support import call, plan, task


class Scripted:
    def __init__(self, structured, expert=()):
        self.structured, self.expert, self.calls = list(structured), list(expert), []

    def invoke_structured(self, messages, schema, **kwargs):
        check_execution_budget('model', consume=True)
        self.calls.append((schema.__name__, json.loads(messages[-1].text)))
        value = self.structured.pop(0)
        return {'parsed': schema.model_validate(value), 'parsing_error': None,
                'raw': AIMessage(content=json.dumps(value)),
                'usage': {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5},
                'context_info': {'source': 'scripted-recovery'}}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(('expert', messages))
        value = self.expert.pop(0)
        return AIMessage(content=value.get('content') or '', tool_calls=[
            {'id': item['id'], 'name': item['function']['name'], 'args': json.loads(item['function']['arguments'])}
            for item in value.get('tool_calls', [])],
            usage_metadata={'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6})


@pytest.mark.parametrize('invalid_kind', ['single', 'context'])
def test_empty_reference_selection_replans_once_then_creates_only_correct_proposal(tmp_path, monkeypatch, invalid_kind):
    previous = tools._engine, tools._SessionLocal
    engine = get_engine('sqlite:///' + str(tmp_path / 'business.sqlite'))
    tools.configure_db(engine)
    try:
        with get_session_factory(engine)() as db:
            seed(db)
        service, customer = SupportService(tmp_path / 'memory.sqlite'), Principal('customer_a')
        seed_history(service, customer, 1)  # A real read establishes current order 1001.
        reason = '声音断断续续的'
        draft = plan({**task('propose_refund', [1001], refund_reason=reason), 'context_use': 'order'})
        bad_check = {'selections': [{'task_id': 'task_1', 'orders': {'kind': invalid_kind, 'ids': []},
                                    'message': '当前订单唯一。'}]}
        good_check = {'selections': [{'task_id': 'task_1', 'orders': {'kind': 'single', 'ids': [1001]},
                                     'message': '沿用当前订单1001。'}]}
        model = Scripted([draft, bad_check, draft, good_check], [
            call('check_refund_eligibility', {'order_id': 1001}),
            call('initiate_refund', {'order_id': 1001, 'reason': reason}),
            {'content': '请核对卡片。'}])
        monkeypatch.setattr(understanding, 'get_llm', lambda: model)
        monkeypatch.setattr(base, 'get_llm', lambda: model)
        result = service.chat(customer, 'eval', '给它申请退款，原因是声音断断续续的', request_id='reference-repair')
        structured = [(name, packet) for name, packet in model.calls if name != 'expert']
        assert [name for name, _ in structured] == ['UnderstandingResult', 'ReferenceCheck'] * 2
        for _, packet in structured[2:]:
            correction = packet['reference_format_correction']
            selection = correction['previous_selection']['selections'][0]['orders']
            assert selection['kind'] == invalid_kind and selection['ids'] == []
            assert correction['diagnostics']['reason'] == 'invalid_reference_check'
            assert 'ids' in correction['recovery'] and '唯一编号' in correction['recovery']
            assert packet['user_input'] == '给它申请退款，原因是声音断断续续的'
        assert result['status'] == 'pending_confirmation' and len(result['pending_actions']) == 1
        view = result['pending_actions'][0]
        assert view['order_id'] == 1001 and view['args'] == {'order_id': 1001, 'reason': reason}
        assert result['planning_usage']['total_tokens'] == 20
        assert result['usage']['total_tokens'] >= 20
        assert result['execution_budget']['model_calls'] == len(model.calls)
        assert result['execution_budget']['tool_calls'] == 2
        diagnostics = result['model_diagnostics']
        assert [(item['attempt'], item['reason']) for item in diagnostics] == [
            (1, 'invalid_reference_check'), (2, 'checked')]
        assert diagnostics[0]['context_info'] == {'source': 'scripted-recovery'}
        with get_session_factory(engine)() as db:
            assert len(db.scalars(select(PendingAction)).all()) == 1
            assert db.scalars(select(Refund).where(Refund.order_id == 1001)).one().status == 'none'
    finally:
        engine.dispose()
        tools._engine, tools._SessionLocal = previous


def test_capability_reply_keeps_discussed_orders_but_conversation_cannot_bypass_tools():
    capability = {**plan(kind='reply', message='当前不支持退款到账后自动付款。', mentioned=[1001, 1003]),
                  'reply_kind': 'capability'}
    assert understanding.validate_understanding(capability).mentioned_order_ids == [1001, 1003]
    conversation = {**capability, 'reply_kind': 'conversation'}
    with pytest.raises(ModelOutputError) as failure:
        understanding.validate_understanding(conversation)
    assert failure.value.diagnostics['issues'][0]['type'] == 'business_reply_without_tools'
    model = Scripted([conversation, capability])
    result = understanding.understand_request('退款后自动拿1001的钱去付1003，到账时再办。', model=model)
    assert result['understanding']['reply_kind'] == 'capability'
    assert result['understanding']['mentioned_order_ids'] == [1001, 1003]
    assert [name for name, _ in model.calls] == ['UnderstandingResult', 'UnderstandingResult']
    assert result['usage']['total_tokens'] == 10
    assert result['model_diagnostics'][0]['issues'][0]['type'] == 'business_reply_without_tools'
