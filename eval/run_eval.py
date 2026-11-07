"""Real-model workflow evaluation using only disposable synthetic databases."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--case', type=int, nargs='+')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--report', default='eval/live-results.json')
    args = parser.parse_args()
    scenarios = json.loads(Path(__file__).with_name('scenarios.json').read_text())
    if args.case:
        scenarios = [s for s in scenarios if s['id'] in args.case]
    if args.dry_run:
        from src.planning import CAPABILITIES
        assert all(set(s['operations']) <= CAPABILITIES.keys() for s in scenarios)
        print(f'{len(scenarios)} scenarios validated; no API calls.')
        return
    root = Path(tempfile.mkdtemp(prefix='support-live-'))
    os.environ['SUPPORT_DATA_DIR'] = str(root)
    os.environ['SUPPORT_AGENT_DB_URL'] = 'sqlite:///' + str(root / 'bootstrap.sqlite')
    from src import tools
    from src.backend.db import get_engine, get_session_factory, init_db, Ticket
    from src.backend.seed import seed
    from src.runtime_context import active_customer
    from src.service import SupportService
    records = []
    for repeat in range(args.repeats):
        for scenario in scenarios:
            folder = root / f"case-{scenario['id']}-{repeat}"
            folder.mkdir()
            engine = get_engine('sqlite:///' + str(folder / 'business.sqlite'))
            tools.configure_db(engine)
            with get_session_factory(engine)() as db:
                seed(db)
            service = SupportService(folder / 'memory.sqlite')
            item = {'case': scenario['id'], 'repeat': repeat + 1, 'turns': [], 'passed': False}
            started = time.monotonic()
            token = active_customer.set('customer_a')
            try:
                if scenario['id'] == 10:
                    service.store.save_profile('customer_a', '简洁，不使用Emoji', '站内文字')
                operations = set()
                for n, message in enumerate(scenario['messages']):
                    r = service.chat('customer_a', 'one', message, request_id=f'chat{n}')
                    item['turns'].append(r)
                    operations.update(t['operation'] for t in r['tasks'])
                    if scenario['id'] == 9 and n < 2:
                        with tools._session() as db:
                            assert db.query(Ticket).count() == 0, 'Unrequested escalation'
                    if scenario['id'] == 8:
                        foreign = service.chat('customer_b', 'one', '查询订单1006详情', request_id='foreign')
                        item['foreign'] = foreign
                        assert all(not f.get('amount') for t in foreign['tasks'] for f in t.get('facts', [])), 'Ownership leak'
                    while r['status'] != 'completed':
                        assert r['status'] != 'processing', 'Incomplete execution'
                        if r['pending_inputs']:
                            entry = r['pending_inputs'][0]
                            decision = 'cancel' if entry['type'] == 'orders' else 'submit'
                            if entry['type'] == 'method':
                                r = service.select_payment_method('customer_a', 'one', f'chat{n}', f'method{n}{len(item["turns"])}', decision, 'wechat_pay' if scenario['id'] == 17 else 'paypal', entry['input_id'])
                            else:
                                r = service.select_orders('customer_a', 'one', f'chat{n}', f'orders{n}', decision, [], entry['input_id'])
                        else:
                            a = r['current_action']
                            assert a is not None, 'Missing current action'
                            decision = 'cancel' if scenario['id'] in {5, 8} or (scenario['id'] == 7 and a['order_id'] == 1006) else 'confirm'
                            r = service.resume('customer_a', 'one', f'chat{n}', 'decide' + a['action_id'], a['action_id'], decision)
                        item['turns'].append(r)
                assert set(scenario['operations']) <= operations, f'Missing operations: {set(scenario["operations"]) - operations}'
                if scenario['id'] in {11, 12, 16}:
                    assert not r['actions'], 'Read request prepared a financial action'
                if scenario['id'] == 15:
                    assert tools.get_payment_status(1003)['status'] == 'paid'
                    assert any(t['operation'] == 'refund.check' and any(f.get('eligible') for f in t['facts']) for t in r['tasks'])
                    assert all(a['type'] == 'payment' for a in r['actions'])
                if scenario['id'] in {17, 20}:
                    assert tools.get_payment_status(1003)['status'] == 'paid'
                    assert '待付款' not in r['reply'] and '待支付' not in r['reply']
                if scenario['id'] == 18:
                    assert {a['order_id'] for a in r['actions']} == {1001, 1006}
                if scenario['id'] in {21, 22}:
                    assert not r['actions']
                if scenario['id'] == 16:
                    assert any(word in r['reply'] for word in ['无法', '不能确认', '不能确定', '不足以', '缺少'])
                if scenario['id'] in {2, 16}:
                    assert r['sources'], 'Search returned no sources'
                assert all(t['status'] not in {'failed', 'pending'} for t in r['tasks']), 'Failed or unexecuted task'
                item['passed'] = True
            except Exception as exc:
                item['error'] = str(exc)
            finally:
                active_customer.reset(token)
                service.close()
                engine.dispose()
            item['seconds'] = round(time.monotonic() - started, 2)
            records.append(item)
            Path(args.report).write_text(json.dumps(records, ensure_ascii=False, indent=2))
            print(json.dumps({k: item[k] for k in ['case', 'repeat', 'passed', 'seconds']} | {'error': item.get('error')}, ensure_ascii=False), flush=True)
    print(f'Report: {Path(args.report).resolve()} ({sum(r["passed"] for r in records)}/{len(records)} passed)')


if __name__ == '__main__':
    main()
