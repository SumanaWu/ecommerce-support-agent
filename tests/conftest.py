import pytest
from contextvars import ContextVar
from src.runtime_context import active_customer

stub_runner = ContextVar('stub_runner', default=None)


@pytest.fixture(autouse=True)
def offline_planner(monkeypatch):
    """Keep old scripted fixtures as planner inputs, executing the actual new graph."""
    from src import orchestrator as workflow, dialogue
    from src.service import SupportService
    from tests.fake_model import DemoChatModel, plan_query
    from src.planning import Plan
    original_chat = SupportService.chat

    def plan(message, history=None, profile=None):
        runner = stub_runner.get()
        if not runner:
            return Plan.model_validate(plan_query(message, history)).model_dump(), {}
        result = runner(message, history=history)
        tasks = []
        for entry in result.get('trace', []):
            proposal = entry.get('result', {}).get('pending_action') if isinstance(entry.get('result'), dict) else None
            operation = proposal['type'] + '.prepare' if proposal else {'get_order_details': 'order.details', 'get_order_status': 'order.status', 'track_shipment': 'order.shipment', 'get_payment_status': 'payment.status'}.get(entry.get('tool'), 'order.details')
            oid = (proposal or entry.get('args', {})).get('order_id')
            if oid:
                tasks.append({'task_id': f't{len(tasks)+1}', 'operation': operation, 'targets': [oid],
                    'source_text': message, 'intent': 'prepare' if proposal else 'query'})
        output = Plan.model_validate({'action': 'PLAN', 'tasks': tasks}).model_dump() if tasks else plan_query(message, history)
        return output, result.get('usage', {})

    def chat(self, user_id, session_id, message, runner=None, request_id=None):
        # A scripted provider is test-only; production has one execution graph.
        if runner is None:
            import src.service as service_module
            runner = getattr(service_module, 'run_orchestrator', None)
            if runner is None:
                import sys
                runner = getattr(sys.modules.get('api'), 'run_orchestrator', None)
        token = stub_runner.set(runner)
        try:
            return original_chat(self, user_id, session_id, message, request_id=request_id)
        finally:
            stub_runner.reset(token)

    monkeypatch.setattr(workflow, 'make_plan', plan)
    monkeypatch.setattr(dialogue, 'get_llm', DemoChatModel)
    monkeypatch.setattr(SupportService, 'chat', chat)


@pytest.fixture(autouse=True)
def customer_scope():
    import streamlit as st
    st.cache_resource.clear()
    token = active_customer.set('customer_a')
    yield
    active_customer.reset(token)
