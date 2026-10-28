"""Isolated routing/workflow evaluation. Online mode must be explicitly selected."""
import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.planning import SPECIALIST_ROUTES
from src.runtime_context import active_budget, active_mode

ROUTES = SPECIALIST_ROUTES | {'CLARIFY', 'ESCALATE'}


def route_node(state):
    # Delay imports until the CLI selects temporary paths, including dry-run.
    from src.orchestrator import classify_route
    return classify_route(state['user_message'])


def load_dataset(filename='eval_dataset.json'):
    return json.loads((Path(__file__).parent / filename).read_text(encoding='utf-8'))


def classify(message, mode='demo'):
    token = active_mode.set(mode)
    try:
        return route_node({'user_message': message})['route']
    finally:
        active_mode.reset(token)


@contextmanager
def isolated_environment(directory):
    values = {'SUPPORT_DATA_DIR': str(directory), 'SUPPORT_AGENT_DB_URL': 'sqlite:///' + str(Path(directory) / 'business.sqlite'),
              'SUPPORT_POLICY_DIR': str(ROOT / 'policies')}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def evaluate_routing(dataset, mode, delay):
    from src import tools
    from src.budget import RequestBudget
    from src.errors import classify_error
    from src.requests import RequestContext
    cases = []
    for index, item in enumerate(dataset):
        budget = RequestBudget(RequestContext('eval', mode, uuid4().hex, 'routing'), tools._SessionLocal)
        token = active_budget.set(budget)
        try:
            actual, error = classify(item['message'], mode), None
        except Exception as exc:
            actual, error = 'ERROR', classify_error(exc)
        finally:
            active_budget.reset(token)
        cases.append({**item, 'actual_route': actual, 'passed': actual == item['expected_route'],
                      'error_type': error, 'execution_budget': budget.snapshot()})
        if delay and index < len(dataset) - 1:
            time.sleep(delay)
    return {'cases': cases, 'accuracy': sum(item['passed'] for item in cases) / len(cases)}


def business_snapshot(engine):
    from sqlalchemy import select
    from src.backend.db import ActionRecord, Order, Payment, PendingAction, Refund, Ticket, get_session_factory
    with get_session_factory(engine)() as db:
        return {'orders': [(row.order_id, row.status, row.version) for row in db.scalars(select(Order).order_by(Order.order_id))],
                'payments': [(row.order_id, row.status, row.amount_minor, row.currency) for row in db.scalars(select(Payment).order_by(Payment.order_id))],
                'refunds': [(row.order_id, row.status, row.reason) for row in db.scalars(select(Refund).order_by(Refund.order_id))],
                'pending': len(db.scalars(select(PendingAction)).all()), 'tickets': len(db.scalars(select(Ticket)).all()),
                'receipts': len(db.scalars(select(ActionRecord)).all())}


def score_workflow(item, result, before, after):
    trace = result.get('trace', [])
    actual_routes = [task['route'] for task in result.get('tasks', []) if task.get('route')]
    if not actual_routes:
        actual_routes = [{'system': 'CLARIFY', 'escalate': 'ESCALATE'}.get(result.get('agent'), str(result.get('agent')).upper())]
    parameters = all(any(entry['tool'] == expected['tool'] and all(entry['args'].get(key) == value for key, value in expected.items() if key != 'tool')
                         for entry in trace) for expected in item['calls'])
    order_ids = {call['order_id'] for call in item['calls'] if 'order_id' in call}
    parameters = parameters and all('order_id' not in entry['args'] or entry['args']['order_id'] in order_ids for entry in trace)
    if not item['calls']:
        parameters = parameters and not trace
    facts = all(any(entry['tool'] == expected['tool'] and isinstance(entry.get('result'), dict) and
                    all(entry['result'].get(key) == value for key, value in expected.items() if key != 'tool')
                    for entry in trace) for expected in item['facts'])
    facts = facts and all(any(fragment in result['reply'] for fragment in alternatives) for alternatives in item.get('reply_any', []))
    effects = (all(before[key] == after[key] for key in ('orders', 'payments', 'refunds')) and
               after['pending'] - before['pending'] == item['pending'] and
               after['tickets'] - before['tickets'] == item.get('tickets', 0) and
               after['receipts'] - before['receipts'] == item['pending'] + item.get('tickets', 0))
    return {'routing': actual_routes == item['routes'], 'parameters': parameters, 'fact_faithfulness': facts,
            'task_completion': result['status'] == item['status'] and
                (not item.get('task_statuses') or [task['status'] for task in result.get('tasks', [])] == item['task_statuses']),
            'side_effects': effects}


def evaluate_workflows(dataset, mode, directory, delay):
    from src import tools
    from src.auth import Principal
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src.service import SupportService
    cases = []
    for index, item in enumerate(dataset):
        path = Path(directory) / ('case_' + str(index))
        path.mkdir()
        engine = get_engine('sqlite:///' + str(path / 'business.sqlite'))
        try:
            tools.configure_db(engine)
            with get_session_factory(engine)() as db:
                seed(db)
            before = business_snapshot(engine)
            result = SupportService(path / 'memory.sqlite').chat(Principal('customer_a'), 'eval', mode, item['message'])
            metrics = score_workflow(item, result, before, business_snapshot(engine))
            cases.append({'id': item['id'], 'message': item['message'], 'metrics': metrics, 'passed': all(metrics.values()),
                          'status': result['status'], 'stop_reason': result.get('stop_reason'), 'reply': result['reply'],
                          'tasks': result.get('tasks', []), 'error_type': result.get('error_type'),
                          'execution_budget': result.get('execution_budget', {})})
        finally:
            engine.dispose()
        if delay and index < len(dataset) - 1:
            time.sleep(delay)
    names = ('routing', 'parameters', 'fact_faithfulness', 'task_completion', 'side_effects')
    return {'cases': cases, 'metrics': {name: sum(case['metrics'][name] for case in cases) / len(cases) for name in names},
            'limitations': '事实指标只检查样例工具事实与回复必要短语，不是通用语义判分；未执行真实资金操作。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'online'], default='demo')
    parser.add_argument('--suite', choices=['routing', 'workflow', 'all'], default='routing')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    routing, workflows = load_dataset(), load_dataset('workflow_dataset.json')
    if args.sleep < 0 or args.limit is not None and args.limit <= 0:
        parser.error('sleep 不能为负，limit 必须为正整数。')
    assert all(item['expected_route'] in ROUTES for item in routing), 'bad routing dataset'
    assert all(set(item) >= {'id', 'message', 'routes', 'calls', 'facts', 'status', 'pending'} and
               set(item['routes']) <= ROUTES for item in workflows), 'bad workflow dataset'
    if args.limit:
        routing, workflows = routing[:args.limit], workflows[:args.limit]
    if args.dry_run:
        print(f'Dataset OK: routing={len(routing)}, workflow={len(workflows)}; 未调用模型或打开业务库。')
        return 0
    report = {'mode': args.mode, 'suite': args.suite,
              'evaluation': 'offline_rules' if args.mode == 'demo' else 'online_model',
              'configured_models': {}, 'data_isolation': 'temporary_databases'}
    with TemporaryDirectory(prefix='support-agent-eval-') as directory, isolated_environment(directory):
        from src.logging_config import configure_logging
        configure_logging(logging.ERROR)
        from src import tools
        initial_engine = tools._engine
        for name, default in [('ZHIPU_MODEL', 'glm-4.7-flash'), ('GEMINI_MODEL', 'gemini-2.5-flash'), ('GROQ_MODEL', 'llama-3.3-70b-versatile')]:
            report['configured_models'][name] = os.environ.get(name) or default
        report['configured_provider'] = 'demo-rules' if args.mode == 'demo' else os.environ.get('LLM_PROVIDER', 'auto')
        if args.suite in {'routing', 'all'}:
            report['routing'] = evaluate_routing(routing, args.mode, args.sleep)
        if args.suite in {'workflow', 'all'}:
            report['workflow'] = evaluate_workflows(workflows, args.mode, directory, args.sleep)
        tools._engine.dispose()
        initial_engine.dispose()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"mode={report['mode']}; evaluation={report['evaluation']}; provider={report['configured_provider']}")
    for suite in ('routing', 'workflow'):
        if suite in report:
            data = report[suite]
            print(suite + ': ' + json.dumps(data.get('metrics', {'accuracy': data.get('accuracy')}), ensure_ascii=False))
            for case in data['cases']:
                if not case['passed']:
                    print('FAIL: ' + case.get('id', case['message']) + ' ' + str(case.get('error_type') or 'metric_assertion'))
    return int(any(not case['passed'] for suite in ('routing', 'workflow') if suite in report for case in report[suite]['cases']))


if __name__ == '__main__':
    raise SystemExit(main())

