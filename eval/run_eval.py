"""Validate online evaluation cases; only --live can call the configured model."""
import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = Path(__file__).with_name('online_cases.json')
METRICS = ('operations', 'expert_selection', 'objects', 'write_intent', 'clarification',
           'tool_facts', 'task_completion', 'side_effects', 'answer_mode', 'action_parameters')


def load_dataset(path=DATASET):
    cases = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('评估案例必须是非空列表。')
    ids = set()
    for case in cases:
        if not isinstance(case.get('id'), str) or case['id'] in ids or not case.get('turns'):
            raise ValueError('案例编号必须唯一，每个案例必须包含对话轮次。')
        ids.add(case['id'])
        if type(case.get('history_turns', 0)) is not int or not 0 <= case.get('history_turns', 0) <= 50:
            raise ValueError('历史测试轮次必须为0至50。')
        for turn in case['turns']:
            expected = turn['expect']
            if not isinstance(turn['message'], str) or not 1 <= len(turn['message']) <= 2000:
                raise ValueError('评估输入必须为1至2000字。')
            if expected['kind'] not in {'tasks', 'clarify', 'reply', 'escalate', 'action'}:
                raise ValueError('评估结果类型无效。')
            if turn.get('action') is not None and (turn['action'] != 'cancel' or type(turn.get('order_id')) is not int):
                raise ValueError('动作步骤只允许按订单选择当前待确认提案并取消。')
            if (expected['kind'] == 'action') != bool(turn.get('action')) or (
                    expected['kind'] == 'action' and expected.get('order_id') != turn['order_id']):
                raise ValueError('动作预期必须匹配取消步骤的订单。')
            for task in expected.get('tasks', []):
                if task['expert'] not in {'ORDER', 'REFUND', 'PAYMENT', 'FAQ'} or task['intent'] not in {'read', 'propose'}:
                    raise ValueError('评估专家或读写意图无效。')
                if any(type(number) is not int or number <= 0 for number in task['order_ids']):
                    raise ValueError('评估订单必须为正整数。')
            if 'status' not in expected:
                raise ValueError('评估案例必须声明任务状态。')
    return cases


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


def business_snapshot(engine):
    from sqlalchemy import select
    from src.backend.db import ActionRecord, Order, Payment, PendingAction, Refund, Ticket, get_session_factory
    with get_session_factory(engine)() as db:
        return {'orders': [(row.order_id, row.status, row.version) for row in db.scalars(select(Order).order_by(Order.order_id))],
                'payments': [(row.order_id, row.status, row.amount_minor, row.currency) for row in db.scalars(select(Payment).order_by(Payment.order_id))],
                'refunds': [(row.order_id, row.status, row.reason) for row in db.scalars(select(Refund).order_by(Refund.order_id))],
                'pending': len(db.scalars(select(PendingAction)).all()), 'tickets': len(db.scalars(select(Ticket)).all()),
                'receipts': len(db.scalars(select(ActionRecord)).all())}


def score_turn(expected, result, before, after):
    from src.planning import iter_order_results
    from src.understanding import expand_order_selection
    if expected['kind'] == 'action':
        unchanged = all(before[key] == after[key] for key in ('orders', 'payments', 'refunds', 'pending', 'tickets'))
        targets = {number for entry in result.get('trace', []) for number, _ in iter_order_results(entry.get('result'))}
        correct = result.get('status') == expected['status'] and targets == {expected['order_id']}
        return {name: unchanged and after['receipts'] == before['receipts'] + 1 if name == 'side_effects' else correct for name in METRICS}
    understanding = result.get('understanding') or {}
    actual = understanding.get('tasks', [])
    wanted = expected.get('tasks', [])
    kind = understanding.get('kind') == expected['kind']
    try:
        objects = [expand_order_selection(task['orders']) for task in actual]
    except (KeyError, ValueError, TypeError):
        objects = None
    facts = [(entry['tool'].removesuffix('_batch').replace('track_shipments', 'track_shipment'), number, fact)
             for entry in result.get('trace', []) for number, fact in iter_order_results(entry.get('result'))]
    factual = all(any(tool == wanted_fact['tool'] and number == wanted_fact['order_id'] and
                     all(fact.get(key) == value for key, value in wanted_fact.items() if key not in {'tool', 'order_id'})
                     for tool, number, fact in facts) for wanted_fact in expected.get('facts', []))
    pending = sorted(view['order_id'] for view in result.get('pending_actions', []))
    expected_pending = sorted(expected.get('pending_orders', []))
    tickets = expected.get('tickets', 0)
    effects = (all(before[key] == after[key] for key in ('orders', 'payments', 'refunds')) and
               after['pending'] - before['pending'] == len(expected_pending) and
               after['tickets'] - before['tickets'] == tickets and
               after['receipts'] - before['receipts'] == len(expected_pending) + tickets)
    return {'operations': kind and [task['operation'] for task in actual] == [task['operation'] for task in wanted],
            'expert_selection': kind and [task['expert'] for task in actual] == [task['expert'] for task in wanted],
            'objects': kind and objects == [task['order_ids'] for task in wanted],
            'write_intent': kind and [task['intent'] for task in actual] == [task['intent'] for task in wanted],
            'clarification': kind and (expected['kind'] != 'clarify' or
                result.get('status') == 'needs_clarification' and bool(result.get('clarification_question')) and
                all(understanding.get(field) == expected[field] for field in ('missing_fields', 'pending_operation') if field in expected)),
            'tool_facts': factual,
            'task_completion': result.get('status') == expected['status'] and pending == expected_pending,
            'answer_mode': kind and len(actual) == len(wanted) and all(
                goal.get('answer_mode') != 'explain' or task.get('answer_mode') == 'explain'
                for task, goal in zip(actual, wanted)),
            'action_parameters': all(any(view['order_id'] == int(number) and all(view['args'].get(key) == value for key, value in args.items())
                for view in result.get('pending_actions', [])) for number, args in expected.get('action_args', {}).items()),
            'side_effects': effects}


def seed_history(service, principal, count):
    """Synthetic past dialogue built from real reads of the isolated fixture DB."""
    from src import tools
    from src.auth import principal_scope
    from src.memory import empty_issue
    from src.results import compose_response
    if not count:
        return
    turns = []
    with principal_scope(principal):
        for index in range(count):
            operation, name, number, message = [
                ('order_details', 'get_order_details', 1001, '查1001详情'),
                ('refund_eligibility', 'check_refund_eligibility', 1007, '只查1007能否退款'),
                ('shipment', 'track_shipment', 1002, '查1002物流'),
                ('order_status', 'get_order_status', 1003, '只查1003状态')][index % 4]
            trace = [{'tool': name, 'args': {'order_id': number}, 'result': getattr(tools, name)(number)}]
            reply = compose_response({'trace': trace, 'status': 'completed', 'reply': '', 'reply_source': 'facts'})['reply']
            turns.append({'request_id': 'history_' + str(index), 'user': message, 'reply': reply,
                'agent': 'fixture', 'status': 'completed', 'trace': trace,
                'tasks': [{'task_id': 'task_1', 'message': message, 'operation': operation, 'order_ids': [number],
                           'status': 'completed', 'reply': reply, 'trace': trace}]})
    payload = service.store.load_session(principal.customer_id, 'online', 'eval')
    issue = {**empty_issue(), 'active_order_id': number, 'candidate_order_ids': [number], 'revision': 1}
    service.store.save_session_if_version(principal.customer_id, 'online', 'eval',
        {**payload, 'turns': turns, 'issue': issue, 'last_order_id': number}, payload['version'])


def evaluate_cases(dataset, directory, delay=0, *, progress=None):
    from src import tools
    from src.auth import Principal
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src.service import SupportService
    previous = tools._engine, tools._SessionLocal
    cases = []
    try:
        for index, case in enumerate(dataset):
            path = Path(directory) / ('case_' + str(index))
            path.mkdir()
            engine = get_engine('sqlite:///' + str(path / 'business.sqlite'))
            turns = []
            try:
                tools.configure_db(engine)
                with get_session_factory(engine)() as db:
                    seed(db)
                service = SupportService(path / 'memory.sqlite')
                principal = Principal(case.get('customer_id', 'customer_a'))
                seed_history(service, principal, case.get('history_turns', 0))
                for turn in case['turns']:
                    before = business_snapshot(engine)
                    started = time.monotonic()
                    if turn.get('action') == 'cancel':
                        pending = [action for action in service.list_pending_actions(principal, 'eval')
                                   if action['order_id'] == turn['order_id']]
                        if len(pending) == 1:
                            result = service.cancel_action(principal, pending[0]['action_id'])
                        else:
                            result = {'status': 'failed', 'reply': '评估步骤未取得唯一待取消提案。'}
                    else:
                        result = service.chat(principal, 'eval', turn['message'])
                    after = business_snapshot(engine)
                    metrics = score_turn(turn['expect'], result, before, after)
                    turns.append({'message': turn['message'], 'expected': turn['expect'], 'metrics': metrics,
                        'passed': all(metrics.values()), 'status': result['status'], 'reply': result['reply'],
                        'understanding': result.get('understanding'), 'error_type': result.get('error_type'),
                        'stop_reason': result.get('stop_reason'), 'usage': result.get('usage', {}),
                        'model_diagnostics': result.get('model_diagnostics', []), 'trace': result.get('trace', []),
                        'business_state_unchanged': all(before[key] == after[key] for key in ('orders', 'payments', 'refunds')),
                        'usage_status': 'reported_positive' if result.get('usage', {}).get('total_tokens') else 'unknown_or_zero',
                        'elapsed_seconds': round(time.monotonic() - started, 3), 'execution_budget': result.get('execution_budget', {})})
                    if delay:
                        time.sleep(delay)
            finally:
                engine.dispose()
            cases.append({'id': case['id'], 'tags': case.get('tags', []), 'turns': turns,
                          'passed': all(turn['passed'] for turn in turns)})
            print('CASE ' + case['id'] + ': ' + ('PASS' if cases[-1]['passed'] else 'FAIL'), flush=True)
            if progress:
                progress(cases)
    finally:
        tools._engine, tools._SessionLocal = previous
    turns = [turn for case in cases for turn in case['turns']]
    return {'cases': cases, 'actual_cases': len(cases), 'actual_turns': len(turns),
            'unconfirmed_business_changes': sum(not turn['business_state_unchanged'] for turn in turns),
            'metrics': {name: sum(turn['metrics'][name] for turn in turns) / len(turns) for name in METRICS},
            'case_pass_rate': sum(case['passed'] for case in cases) / len(cases),
            'reported_usage': {key: sum(turn['usage'].get(key) or 0 for turn in turns)
                               for key in ('input_tokens', 'output_tokens', 'total_tokens')},
            'usage_complete': all(turn['usage_status'] == 'reported_positive' for turn in turns),
            'limitations': '结构化指标验证声明的预期与本地副作用；追问合理性、替代的有效任务拆分需人工复核。缺失或零用量不被认定为已知零消耗。'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    option = parser.add_mutually_exclusive_group()
    option.add_argument('--live', action='store_true', help='显式允许调用配置的真实模型')
    option.add_argument('--dry-run', action='store_true', help='仅检查数据；默认行为')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.limit <= 0 or args.repeat <= 0 or not 0 <= args.sleep <= 60:
        parser.error('limit和repeat必须为正整数，sleep必须为0至60秒。')
    try:
        dataset = load_dataset(args.dataset)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error('评估数据格式错误：' + str(exc))
    selected = dataset[:args.limit]
    if args.repeat > 1:
        selected = [{**case, 'id': case['id'] + '_repeat_' + str(repetition + 1)}
                    for repetition in range(args.repeat) for case in selected]
    if not args.live:
        print(f'Dataset OK: total={len(dataset)}, selected={len(selected)}; 未调用模型或打开业务库。真实评估需 --live。')
        return 0
    report = {'evaluation': 'online_model', 'data_isolation': 'temporary_databases', 'dataset': str(args.dataset),
              'dataset_cases': len(dataset), 'requested_limit': args.limit, 'repetitions': args.repeat, 'actual_cases': 0, 'actual_turns': 0}
    with TemporaryDirectory(prefix='support-agent-eval-') as directory, isolated_environment(directory):
        from src.logging_config import configure_logging
        configure_logging(logging.ERROR)
        from src import tools
        from src.errors import classify_error
        from src.llm import create_chat_model
        initial_engine = tools._engine
        try:
            model = create_chat_model()
            from src.understanding import PIPELINE_VERSION
            from src.context import context_budget
            sources = ['src/context.py', 'src/memory.py', 'src/understanding.py', 'src/orchestrator.py',
                       'src/service.py', 'src/storage.py', 'src/agents/base.py']
            digest = hashlib.sha256(b''.join(name.encode() + (ROOT / name).read_bytes() for name in sources)).hexdigest()
            report.update(provider=model.provider, model=model.model_name, pipeline_version=PIPELINE_VERSION,
                          implementation_sha256=digest, context_budget_tokens=context_budget())
        except Exception as exc:
            report.update(error_type=classify_error(exc), status='configuration_failed')
        else:
            def save_progress(cases):
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({**report, 'status': 'running', 'cases': cases,
                        'actual_cases': len(cases), 'actual_turns': sum(len(case['turns']) for case in cases)},
                        ensure_ascii=False, indent=2), encoding='utf-8')
            report.update(evaluate_cases(selected, directory, args.sleep, progress=save_progress), status='measured')
        finally:
            initial_engine.dispose()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, ensure_ascii=False))
    for case in report.get('cases', []):
        if not case['passed']:
            print('FAIL: ' + case['id'])
    return 2 if report['status'] != 'measured' else int(any(not case['passed'] for case in report['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
