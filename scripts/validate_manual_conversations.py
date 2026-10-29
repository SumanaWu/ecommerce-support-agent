"""Run the three documented conversations; real API calls require --live."""
import argparse
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.run_eval import business_snapshot, isolated_environment, load_dataset, score_turn

DATASET = ROOT / 'eval/manual_conversation_cases.json'
BUSINESS_FIELDS = ('orders', 'payments', 'refunds', 'payment_methods', 'business_rows')


def matches_action_args(actual, expected):
    """Ignore sentence-ending punctuation in reason expectations, never word changes."""
    return all(actual.get(key) == value or key == 'reason' and
               isinstance(actual.get(key), str) and isinstance(value, str) and
               actual[key].strip().rstrip('。.') == value.strip().rstrip('。.')
               for key, value in expected.items())


def load_cases():
    cases = load_dataset(DATASET)
    if len(cases) != 3 or any(case.get('customer_id') != 'customer_a' or len(case['turns']) != 25 for case in cases):
        raise ValueError('手工验证必须是customer_a的三轮，每轮25条聊天输入。')
    for case in cases:
        for turn in case['turns']:
            if not turn.get('expected_text'):
                raise ValueError('每条聊天必须提供人工语义期望。')
            for action in turn.get('after_actions', []):
                if (action.get('action') not in {'confirm', 'cancel'} or
                        action.get('tool') not in {'initiate_refund', 'process_payment'} or
                        type(action.get('order_id')) is not int or action['order_id'] <= 0 or
                        not isinstance(action.get('args', {}), dict)):
                    raise ValueError('按钮动作必须声明确认/取消、订单和对应工具。')
    if len({tuple(turn['message'] for turn in case['turns']) for case in cases}) != 3:
        raise ValueError('三轮聊天应使用不同的连续对话。')
    return cases


def snapshot(engine):
    from sqlalchemy import select
    from src.backend.db import ActionRecord, Order, Payment, PendingAction, Refund, get_session_factory
    result = business_snapshot(engine)
    with get_session_factory(engine)() as db:
        result['payment_methods'] = [(row.order_id, row.method) for row in db.scalars(select(Payment).order_by(Payment.order_id))]
        # Compare all business columns, including frozen money and unrelated customer rows.
        result['business_rows'] = {model.__tablename__: {
            str(row.order_id): {column.name: (value.isoformat() if hasattr(value, 'isoformat') else value)
                               for column in model.__table__.columns for value in [getattr(row, column.name)]}
            for row in db.scalars(select(model).order_by(model.order_id))} for model in (Order, Payment, Refund)}
        result['action_rows'] = {row.action_id: {column.name: getattr(row, column.name)
                                 for column in PendingAction.__table__.columns}
                                 for row in db.scalars(select(PendingAction))}
        result['receipt_rows'] = {json.dumps([row.user_id, row.mode, row.request_id, row.action_key]):
                                  {'user_id': row.user_id, 'mode': row.mode, 'request_id': row.request_id,
                                   'tool': row.tool, 'args': json.loads(row.args), 'result': json.loads(row.result)}
                                  for row in db.scalars(select(ActionRecord))}
    return result


def score_chat(expected, result, before, after):
    alternatives = [{key: value for key, value in expected.items() if key != 'alternatives'}]
    alternatives.extend({**alternatives[0], **choice} for choice in expected.get('alternatives', []))
    scores = []
    actual_tasks = (result.get('understanding') or {}).get('tasks', [])
    for choice in alternatives:
        score = score_turn(choice, result, before, after)
        score['action_parameters'] = all(any(
            view['order_id'] == int(number) and matches_action_args(view['args'], args)
            for view in result.get('pending_actions', []))
            for number, args in choice.get('action_args', {}).items())
        score['response_purpose'] = all(
            'presentation_intent' not in goal or item.get('presentation_intent') == goal['presentation_intent']
            for item, goal in zip(actual_tasks, choice.get('tasks', [])))
        score['reply_kind'] = ('reply_kind' not in choice or
                               (result.get('understanding') or {}).get('reply_kind') == choice['reply_kind'])
        scores.append(score)
    metrics = max(scores, key=lambda value: sum(value.values()))
    metrics['no_unconfirmed_business_write'] = all(before[key] == after[key] for key in BUSINESS_FIELDS)
    # Fixtures are public to the runner, but customer_b's values must never reach customer_a.
    from src.backend.seed import ORDERS
    from src.money import format_amount, major_to_minor, SAMPLE_CURRENCY
    private = [value for number, name, item, _, _, amount in ORDERS if number >= 1008
               for value in (name, item, format_amount(major_to_minor(amount, SAMPLE_CURRENCY), SAMPLE_CURRENCY))]
    visible = json.dumps({'reply': result.get('reply'), 'trace': result.get('trace', [])}, ensure_ascii=False)
    metrics['no_private_fixture_disclosure'] = not any(value in visible for value in private)
    return metrics


def click(service, principal, session_id, expected, engine):
    before = snapshot(engine)
    pending = [view for view in service.list_pending_actions(principal, session_id)
               if view['order_id'] == expected['order_id'] and view['tool'] == expected['tool']]
    if len(pending) != 1 or not matches_action_args(pending[0]['args'], expected.get('args', {})):
        return {'expected': expected, 'passed': False, 'reason': '未取得唯一且参数匹配的待确认卡；没有点击其他卡片。'}
    view = pending[0]
    started = time.monotonic()
    decision = expected['action']
    decide = service.confirm_action if decision == 'confirm' else service.cancel_action
    result = decide(principal, view['action_id'])
    after = snapshot(engine)
    wanted = deepcopy(before['business_rows'])
    number = str(view['order_id'])
    price = before['business_rows']['payments' if view['tool'] == 'process_payment' else 'orders'][number]
    if decision == 'confirm':
        wanted['orders'][number]['version'] += 1
        if view['tool'] == 'process_payment':
            wanted['payments'][number].update(status='paid', method=view['args']['method'])
        else:
            wanted['refunds'][number].update(status='requested', reason=view['args']['reason'])
    current = service.get_action(principal, view['action_id'])
    wanted_actions = deepcopy(before['action_rows'])
    wanted_actions[view['action_id']].update(status='executed' if decision == 'confirm' else 'cancelled',
                                            result=after['action_rows'][view['action_id']]['result'])
    receipts = [row for key, row in after['receipt_rows'].items() if key not in before['receipt_rows']]
    receipt_tool = view['tool'] if decision == 'confirm' else 'cancel_action'
    receipt_args = view['args'] if decision == 'confirm' else {'order_id': view['order_id'], 'action_id': view['action_id']}
    checks = {'frozen_amount_and_currency': all(view[key] == current[key] == price[key] for key in ('amount_minor', 'currency')),
              'only_expected_business_changes': after['business_rows'] == wanted,
              'only_target_action_changed': after['action_rows'] == wanted_actions,
              'no_unrelated_pending_or_ticket_change': all(after[key] == before[key] for key in ('pending', 'tickets')),
              'independent_receipt': len(receipts) == 1 and receipts[0] == {
                  'user_id': principal.customer_id, 'mode': 'online', 'request_id': decision + '_' + view['action_id'],
                  'tool': receipt_tool, 'args': receipt_args, 'result': current['result']} and
                  after['receipts'] == before['receipts'] + 1 and all(
                      after['receipt_rows'].get(key) == row for key, row in before['receipt_rows'].items()),
              'independent_tool_result': len(result.get('trace', [])) == 1 and
                  result['trace'][0]['tool'] == receipt_tool and result['trace'][0]['result'] == current['result'],
              'button_does_not_call_model': result.get('execution_budget', {}).get('real_model_calls') == 0,
              'successful_decision': current['result'].get('success') is True and
                  result.get('status') == ('completed' if decision == 'confirm' else 'cancelled')}
    return {'expected': expected, 'status': result.get('status'), 'reply': result.get('reply'),
            'action_id': view['action_id'], 'frozen_money': {key: view[key] for key in ('amount_minor', 'currency')},
            'checks': checks, 'passed': all(checks.values()), 'trace': result.get('trace', []),
            'receipts': receipts, 'usage': result.get('usage', {}), 'execution_budget': result.get('execution_budget', {}),
            'changed_business_tables': [key for key in wanted if before['business_rows'][key] != after['business_rows'][key]],
            'mismatched_business_tables': [key for key in wanted if wanted[key] != after['business_rows'][key]],
            'elapsed_seconds': round(time.monotonic() - started, 3)}


def save_report(report, output):
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)


def evaluate(cases, directory, report, output, maximum):
    from src import tools
    from src.auth import Principal
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src.service import SupportService
    previous = tools._engine, tools._SessionLocal
    try:
        for case in cases:
            path = Path(directory) / case['id']
            path.mkdir()
            engine = get_engine('sqlite:///' + str(path / 'business.sqlite'))
            case_report = {'id': case['id'], 'customer_id': 'customer_a', 'turns': [], 'passed': True,
                           'requested_chat_turns': maximum, 'complete_conversation': maximum == 25}
            report['cases'].append(case_report)
            try:
                tools.configure_db(engine)
                with get_session_factory(engine)() as db:
                    seed(db)
                service, principal, session_id = SupportService(path / 'memory.sqlite'), Principal('customer_a'), 'manual'
                for index, turn in enumerate(case['turns'][:maximum], 1):
                    before = snapshot(engine)
                    started = time.monotonic()
                    result = service.chat(principal, session_id, turn['message'])
                    after = snapshot(engine)
                    metrics = score_chat(turn['expect'], result, before, after)
                    elapsed = time.monotonic() - started
                    budget_elapsed = result.get('execution_budget', {}).get('elapsed_seconds', elapsed)
                    measured = {'index': index, 'message': turn['message'], 'expected_text': turn['expected_text'],
                                'expected': turn['expect'], 'metrics': metrics, 'passed': all(metrics.values()),
                                'status': result.get('status'), 'reply': result.get('reply'),
                                'understanding': result.get('understanding'), 'trace': result.get('trace', []),
                                'usage': result.get('usage', {}), 'error_type': result.get('error_type'),
                                'model_diagnostics': result.get('model_diagnostics', []),
                                'stop_reason': result.get('stop_reason'), 'pending_actions': result.get('pending_actions', []),
                                'execution_budget': result.get('execution_budget', {}),
                                'elapsed_seconds': round(elapsed, 3),
                                'clock_discontinuity_seconds': round(budget_elapsed - elapsed, 3), 'button_actions': []}
                    case_report['turns'].append(measured)
                    report['actual_turns'] += 1
                    save_report(report, output)
                    for action in turn.get('after_actions', []):
                        measured['button_actions'].append(click(service, principal, session_id, action, engine))
                        report['actual_button_actions'] += 1
                        save_report(report, output)
                    measured['passed'] &= all(action['passed'] for action in measured['button_actions'])
                    case_report['passed'] &= measured['passed']
                    failed = [key for key, passed in metrics.items() if not passed]
                    print(f"{case['id']} {index}/25 {result.get('status')}: " +
                          ('PASS' if measured['passed'] else 'FAIL ' + ','.join(failed)), flush=True)
                    save_report(report, output)
            finally:
                engine.dispose()
    finally:
        tools._engine, tools._SessionLocal = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--round', type=int, choices=[1, 2, 3])
    parser.add_argument('--max-turns', type=int, default=25)
    parser.add_argument('--provider', choices=['zhipu', 'deepseek', 'auto'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_turns <= 25:
        parser.error('max-turns必须为1至25，不允许跳过前置上下文。')
    try:
        dataset = load_cases()
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    selected = [dataset[args.round - 1]] if args.round else dataset
    if not args.live:
        save_report({'evaluation': 'manual_continuous_conversations', 'status': 'dataset_validated',
                     'dataset_rounds': len(dataset), 'dataset_chat_turns': sum(len(case['turns']) for case in dataset),
                     'selected_rounds': [case['id'] for case in selected],
                     'requested_turns_per_round': args.max_turns, 'actual_turns': 0, 'actual_button_actions': 0}, args.output)
        print(f'Dataset OK: rounds=3, chat_turns=75; selected={len(selected)}, selected_chat_turns={len(selected) * args.max_turns}; 未调用模型或打开业务库。')
        return 0
    report = {'evaluation': 'manual_continuous_conversations', 'status': 'running',
              'data_isolation': 'fresh_temporary_business_and_memory_databases_per_round',
              'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
              'requested_rounds': len(selected), 'requested_turns_per_round': args.max_turns,
              'actual_turns': 0, 'actual_button_actions': 0, 'cases': [],
              'limitations': '机器检查结构化期望和数据库变化；自然表达、澄清时机、未列出的合理任务拆分仍需人工复核。部分执行不代表完整75条通过。'}
    previous_provider = os.environ.get('LLM_PROVIDER')
    if args.provider:
        os.environ['LLM_PROVIDER'] = args.provider
    try:
        with TemporaryDirectory(prefix='support-manual-') as directory, isolated_environment(directory):
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
                sources = ['src/llm.py', 'src/context.py', 'src/memory.py', 'src/understanding.py',
                           'src/orchestrator.py', 'src/service.py', 'src/storage.py', 'src/results.py',
                           'src/rules.py', 'src/agents/base.py', 'src/agents/payment_agent.py',
                           'scripts/validate_manual_conversations.py', *[str(path.relative_to(ROOT))
                               for path in sorted((ROOT / 'policies').glob('*.md'))]]
                report.update(provider=getattr(model, 'provider', args.provider), model=getattr(model, 'model_name', None),
                              pipeline_version=PIPELINE_VERSION, context_budget_tokens=context_budget(),
                              implementation_sha256=hashlib.sha256(b''.join(
                                  name.encode() + (ROOT / name).read_bytes() for name in sources)).hexdigest())
                evaluate(selected, directory, report, args.output, args.max_turns)
                report['status'] = 'measured' if args.max_turns == 25 else 'measured_partial'
            except Exception as exc:
                report.update(status='failed', error_type=classify_error(exc))
            finally:
                initial_engine.dispose()
    finally:
        if previous_provider is None:
            os.environ.pop('LLM_PROVIDER', None)
        else:
            os.environ['LLM_PROVIDER'] = previous_provider
    turns = [turn for case in report['cases'] for turn in case['turns']]
    report['passed_turns'] = sum(turn['passed'] for turn in turns)
    report['reported_usage'] = {key: sum(turn['usage'].get(key) or 0 for turn in turns)
                                for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    report['usage_complete'] = bool(turns) and all(turn['usage'].get('total_tokens') for turn in turns)
    save_report(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, ensure_ascii=False))
    return 2 if report['status'] == 'failed' else int(any(not turn['passed'] for turn in turns))


if __name__ == '__main__':
    raise SystemExit(main())
