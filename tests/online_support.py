"""Explicit scripted model outputs for tests, never a natural-language simulator."""
from copy import deepcopy

from src import orchestrator
from src.budget import check_execution_budget


def task(operation, ids=(), *, task_id='task_1', selection=None, depends_on=(), conditions=(), answer_mode='result',
         refund_reason=None, payment_method=None):
    return {'task_id': task_id, 'operation': operation, 'context_use': 'none', 'answer_mode': answer_mode,
            'orders': selection or {'kind': 'none' if not ids else 'single' if len(ids) == 1 else 'list', 'ids': list(ids)},
            'instruction': '执行指定的' + operation, 'depends_on': list(depends_on), 'conditions': list(conditions),
            **({'refund_reason': refund_reason} if refund_reason is not None else {}),
            **({'payment_method': payment_method} if payment_method is not None else {})}


def plan(*tasks, kind='tasks', message='', mentioned=(), continues=False, missing=None, clarification_parameter=None):
    return {'kind': kind, 'message': message, 'tasks': list(tasks), 'mentioned_order_ids': list(mentioned), 'continues_pending': continues,
            'missing_fields': (['purpose'] if kind == 'clarify' else []) if missing is None else missing,
            **({'clarification_parameter': clarification_parameter} if clarification_parameter is not None else {})}


def install_plan(monkeypatch, value):
    def understand(message, **kwargs):
        check_execution_budget('model', consume=True)
        return {'understanding': deepcopy(value), 'usage': {}, 'context_info': {}}
    monkeypatch.setattr(orchestrator, 'understand_request', understand)


def call(name, args, *, call_id='tool-1'):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': __import__('json').dumps(args)}}]}


def reference_verdict(order_id, task_id='task_1'):
    return {'selections': [{'task_id': task_id, 'orders': {'kind': 'single', 'ids': [order_id]},
                            'message': '测试中明确指定的上下文对象。'}]}


def install_script(monkeypatch, responses):
    """Drive the actual online graph using explicitly authored model responses."""
    import json
    from langchain_core.messages import AIMessage
    from src import understanding
    from src.agents import base
    remaining, calls = list(responses), []
    class Model:
        def bind_tools(self, tools, **kwargs):
            return self
        def invoke_structured(self, messages, schema, **kwargs):
            check_execution_budget('model', consume=True)
            calls.append(messages)
            value = remaining.pop(0)
            return {'parsed': schema.model_validate(value), 'parsing_error': None, 'usage': {}, 'context_info': {}}
        def invoke(self, messages):
            calls.append(messages)
            value = remaining.pop(0)
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, str):
                return AIMessage(content=value)
            return AIMessage(content=value.get('content') or '', tool_calls=[
                {'id': item['id'], 'name': item['function']['name'], 'args': json.loads(item['function']['arguments'])}
                for item in value.get('tool_calls', [])])
    monkeypatch.setattr(understanding, 'get_llm', Model)
    monkeypatch.setattr(base, 'get_llm', Model)
    return calls


def proposal_steps(operation='propose_refund', order_id=1001):
    if operation == 'propose_refund':
        read, write, args = 'check_refund_eligibility', 'initiate_refund', {'order_id': order_id, 'reason': '用户申请退款'}
    else:
        read, write, args = 'get_payment_status', 'process_payment', {'order_id': order_id, 'method': 'card'}
    return [plan(task(operation, [order_id])), call(read, {'order_id': order_id}, call_id='read'),
            call(write, args, call_id='propose'), '请确认办理。']
