"""Persistent coordinator: scoped domain work, inputs and transaction decisions."""
import copy
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from src import tools
from src.actions import decide_action, prepare_action
from src.agents.faq_agent import get_policies
from src.payment_methods import PAYMENT_METHODS
from src.planning import CAPABILITIES, make_plan
from src.runtime_context import active_customer, active_profile, active_request


def merge_results(left, right):
    return {**left, **right}


class State(TypedDict, total=False):
    user_id: str
    session_id: str
    request_id: str
    message: str
    history: list
    profile: dict
    knowledge_root: str
    plan: dict
    results: Annotated[dict, merge_results]
    work: list
    task: dict
    inputs: list
    actions: list
    event: dict
    usage: dict
    response: dict


TERMINAL = {'succeeded', 'failed', 'rejected', 'skipped', 'cancelled'}
LABELS = {'placed': '已下单，尚未发货', 'shipped': '已发货，运输中', 'delivered': '已签收', 'cancelled': '已取消',
          'pending': '待付款', 'paid': '已付款', 'refunded': '已退款'}


def scope(state):
    return [(active_customer, active_customer.set(state['user_id'])),
            (active_profile, active_profile.set(state.get('profile', {}))),
            (active_request, active_request.set(state['request_id']))]


def reset(tokens):
    for var, token in reversed(tokens):
        var.reset(token)


def plan_node(state):
    tokens = scope(state)
    try:
        plan, usage = make_plan(state['message'], state.get('history'), state.get('profile'))
    finally:
        reset(tokens)
    return {'plan': plan, 'usage': usage, 'results': {}, 'actions': [], 'inputs': []}


def facts_for(result, oid):
    return next((r for r in result.get('facts', []) if r.get('order_id') == oid), {})


def matches(condition, row, result):
    value = result.get('status') if condition['field'] == 'status' else row.get(condition['field'])
    if value is None:
        return None
    other = condition['value']
    op = condition['operator']
    if op == 'eq':
        return value == other
    if op == 'ne':
        return value != other
    if not isinstance(other, list):
        raise ValueError('集合条件必须使用列表')
    return value in other if op == 'in' else value not in other


def schedule(state):
    results = dict(state.get('results', {}))
    updates, work = {}, []
    for task in state['plan']['tasks']:
        tid = task['task_id']
        if tid in results and results[tid]['status'] != 'selected':
            continue
        deps = [results.get(i) for i in task['depends_on']]
        if any(r is None or r['status'] not in TERMINAL for r in deps):
            continue
        if any(r['status'] != 'succeeded' for r in deps):
            updates[tid] = {'status': 'skipped', 'facts': [], 'trace': [], 'reason': '前置任务未成功，未执行'}
            results[tid] = updates[tid]
            continue
        targets = task['targets']
        if task.get('target_from'):
            targets = [r['order_id'] for r in results[task['target_from']].get('facts', []) if r.get('order_id') and not r.get('error')]
        kept, excluded, reasons = [], [], []
        for oid in targets:
            if any(facts_for(r, oid).get('error') for r in deps):
                excluded.append(oid)
                continue
            checks = [matches(c, facts_for(results[c['task']], oid), results[c['task']]) for c in task['conditions']]
            good = not checks or (all(x is True for x in checks) if task['condition_mode'] == 'all' else any(x is True for x in checks))
            (kept if good else excluded).append(oid)
            if not good:
                payment = next((facts_for(r, oid).get('payment_status') for r in deps if facts_for(r, oid).get('payment_status')), None)
                reason = ('已付款，无需再次支付' if payment == 'paid' else '已退款，不能再次支付') if task['operation'] == 'payment.prepare' and payment in {'paid', 'refunded'} else '缺少判断依据，未执行' if any(x is None for x in checks) else '条件未满足，未执行'
                reasons.append(f'订单 {oid}：{reason}')
        if targets and not kept:
            updates[tid] = {'status': 'skipped', 'facts': [], 'trace': [], 'reason': '；'.join(reasons) or '前置订单查询失败，未执行', 'excluded': excluded}
            results[tid] = updates[tid]
        elif task.get('target_from') and not targets:
            updates[tid] = {'status': 'skipped', 'facts': [], 'trace': [], 'reason': '上游没有符合条件的订单'}
            results[tid] = updates[tid]
        else:
            work.append({**task, 'targets': kept, 'excluded': excluded})
    return {'results': updates, 'work': work}


def dispatch(state):
    # Finish one read wave before serial preparation or any human interaction.
    reads = [t for t in state['work'] if t['intent'] == 'query']
    if reads:
        return [Send('expert', {**state, 'task': task}) for task in reads]
    if state['inputs'] or any(a['status'] == 'pending' for a in state['actions']):
        return 'review'
    if state['work']:
        return 'prepare'
    return 'review'


def expert(state):
    from src import dialogue
    task = state['task']
    op, tid = task['operation'], task['task_id']
    facts, trace = [], []
    tokens = scope(state)
    try:
        def call(name, args, fn):
            value = fn(**args)
            trace.append({'task_id': tid, 'agent': CAPABILITIES[op], 'tool': name, 'args': args, 'result': value})
            return value
        if op == 'order.list':
            with tools._session() as db:
                ids = [o.order_id for o in db.query(tools.Order).filter_by(customer_id=state['user_id']).order_by(tools.Order.order_id)]
            for oid in ids:
                row = call('get_order_details', {'order_id': oid}, tools.get_order_details)
                facts.append({**row, 'order_status': row['status']})
        elif op == 'knowledge.search':
            from src.knowledge import get_knowledge
            ref = task.get('product_ref','')
            if not ref:
                names = list(dict.fromkeys(r['item'] for dep in task['depends_on'] for r in state['results'][dep].get('facts',[]) if r.get('item')))
                if len(names) == 1:
                    ref = names[0]
            facts = [call('knowledge_search', {'query':task['query'],'kind':task.get('knowledge_kind','all'), 'product_ref':ref,'category':task.get('category','')}, get_knowledge(state.get('knowledge_root')).search)]
        elif op == 'faq.policies':
            topics = task['topics']
            if 'refund' in topics and 'return' not in topics:
                topics = ['return', *topics]
            facts = [call('get_policies', {'topics': topics}, get_policies)]
        elif op == 'search.web':
            facts = [call('search_web', {'query': task['query']}, dialogue.search_web)]
        elif op == 'answer.explain':
            evidence = {tid: copy.deepcopy({key:value for key,value in state['results'][tid].items() if key in {'status','facts','answer','reason'}}) for tid in task['depends_on']}
            groups = [fact for result in evidence.values() for fact in result.get('facts',[]) if fact.get('knowledge')]
            if any(fact.get('sources') for fact in groups):
                from src.knowledge import get_knowledge
                tokenizer = get_knowledge(state.get('knowledge_root')).encoder().tokenizer
                budget, seen = 6000, set()
                originals = [fact.get('sources',[]) for fact in groups]
                for fact in groups:
                    fact['sources'] = []
                for position in range(max(map(len,originals))):
                    for fact, candidates in zip(groups, originals):
                        if position >= len(candidates):
                            continue
                        source = candidates[position]
                        size = len(tokenizer.encode(source['content'],add_special_tokens=False))
                        if source['id'] not in seen and size <= budget:
                            fact['sources'].append(source)
                            seen.add(source['id'])
                            budget -= size
            answer, usage = dialogue.compose(state['message'], evidence, state.get('history'), task['source_text'])
            kb_sources = [source for result in evidence.values() for fact in result.get('facts',[]) if fact.get('knowledge') for source in fact.get('sources',[]) if source['id'] in answer]
            return {'results': {tid: {'status': 'succeeded', 'facts': [], 'answer': answer, 'usage': usage, 'sources': kb_sources,
                'trace': [{'task_id': tid, 'agent': 'assistant', 'tool': 'compose', 'args': {'depends_on': task['depends_on']}, 'result': {'answer': answer}}]}}}
        else:
            handlers = {'order.details': tools.get_order_details, 'order.status': tools.get_order_status,
                        'order.shipment': tools.track_shipment, 'payment.status': tools.get_payment_status,
                        'refund.check': tools.check_refund_eligibility}
            if not task['targets']:
                raise ValueError('缺少明确订单，请提供订单号')
            for oid in task['targets']:
                row = call(handlers[op].__name__, {'order_id': oid}, handlers[op])
                if 'status' in row:
                    row = {**row, ('payment_status' if op == 'payment.status' else 'order_status'): row['status']}
                facts.append(row)
        result = {'status': 'succeeded', 'facts': facts, 'trace': trace, 'excluded': task.get('excluded', [])}
        if facts and all(r.get('error') for r in facts):
            result['status'] = 'failed'
        if op in {'faq.policies', 'search.web'} and result['status'] == 'succeeded':
            result['answer'], result['usage'] = dialogue.compose(state['message'], facts[0], state.get('history'))
        return {'results': {tid: result}}
    except Exception as exc:
        return {'results': {tid: {'status': 'failed', 'facts': facts, 'trace': trace, 'reason': str(exc)}}}
    finally:
        reset(tokens)


def prepare(state):
    task = state['work'][0]
    tid, op = task['task_id'], task['operation']
    tokens = scope(state)
    try:
        if op == 'escalate.ticket':
            token = active_request.set(state['request_id'] + ':' + tid)
            try:
                result = tools.escalate_to_human(task['reason'], task['source_text'])
            finally:
                active_request.reset(token)
            return {'results': {tid: {'status': 'succeeded', 'facts': [result], 'trace': [
                {'task_id': tid, 'agent': 'escalate', 'tool': 'escalate_to_human', 'args': {}, 'result': result}]}}}
        kind = 'payment' if op == 'payment.prepare' else 'refund'
        targets = task['targets']
        if not targets and kind == 'refund':
            with tools._session() as db:
                candidates = [{**tools.get_order_details(o.order_id), **tools.refund_eligibility(db, o.order_id)}
                    for o in db.query(tools.Order).filter_by(customer_id=state['user_id']).order_by(tools.Order.order_id)]
            candidates.sort(key=lambda o: (not o.get('eligible', False), o['order_id']))
            item = {'input_id': tid + '_orders', 'task_id': tid, 'type': 'orders', 'candidates': candidates}
            trace = [{'task_id': tid, 'agent': 'refund', 'tool': 'list_customer_orders', 'args': {}, 'result': candidates}]
            return {'results': {tid: {'status': 'awaiting_input', 'facts': [], 'trace': trace, 'kind': kind}}, 'inputs': [*state['inputs'], item]}
        if not targets:
            raise ValueError('缺少付款订单，请提供订单号')
        facts, trace, actions, inputs = [], [], list(state['actions']), list(state['inputs'])
        for oid in targets:
            try:
                row = tools.get_payment_status(oid) if kind == 'payment' else tools.check_refund_eligibility(oid)
                trace.append({'task_id': tid, 'agent': kind, 'tool': 'get_payment_status' if kind == 'payment' else 'check_refund_eligibility', 'args': {'order_id': oid}, 'result': row})
                if kind == 'payment' and 'status' in row:
                    row = {**row, 'payment_status': row['status']}
                if kind == 'payment' and row.get('status') == 'paid':
                    facts.append({**row, 'payment_status': 'paid', 'reason': '已付款，无需再次支付'})
                elif row.get('error') or (kind == 'refund' and not row.get('eligible')) or (kind == 'payment' and row.get('status') != 'pending'):
                    facts.append({**row, 'error': row.get('error') or row.get('reason') or ('已退款，不能再次支付' if row.get('status') == 'refunded' else '订单不符合支付条件')})
                elif kind == 'payment':
                    inputs.append({'input_id': tid + '_method_' + str(oid), 'task_id': tid, 'type': 'method',
                        'orders': [{'order_id': oid, 'amount': format(row['amount'], '.2f')}], 'methods': PAYMENT_METHODS})
                    facts.append({**row, 'payment_status': row['status']})
                else:
                    action = prepare_action(state['user_id'], state['session_id'], state['request_id'], tid + ':' + str(oid) + ':refund',
                        {'type': 'refund', 'order_id': oid, 'reason': task['reason']})
                    actions.append({**action, 'task_id': tid})
                    trace.append({'task_id': tid, 'agent': 'refund', 'tool': 'prepare_refund',
                        'args': {'order_id': oid, 'reason': task['reason']}, 'result': {'pending_action': action}})
                    facts.append(row)
            except Exception as exc:
                facts.append({'order_id': oid, 'error': str(exc), 'reason_code': 'PREPARATION_FAILED'})
        status = 'awaiting_input' if any(i['task_id'] == tid for i in inputs) else 'awaiting_confirmation' if any(a['task_id'] == tid and a['status'] == 'pending' for a in actions) else 'rejected' if any(f.get('error') for f in facts) else 'succeeded'
        return {'results': {tid: {'status': status, 'facts': facts, 'trace': trace, 'kind': kind, 'excluded': task.get('excluded', [])}}, 'actions': actions, 'inputs': inputs}
    except ValueError as exc:
        return {'results': {tid: {'status': 'rejected', 'facts': [], 'trace': [], 'reason': str(exc)}}}
    finally:
        reset(tokens)


def review(state):
    return {'response': response(state)}


def after_review(state):
    return 'wait' if state['inputs'] or any(a['status'] == 'pending' for a in state['actions']) else END


def wait(state):
    # Exactly one interrupt; preparation and execution live in separate nodes.
    return {'event': interrupt(state['response'])}


def apply_input(state):
    event = state['event']
    results, actions, inputs = copy.deepcopy(state['results']), copy.deepcopy(state['actions']), copy.deepcopy(state['inputs'])
    if event['type'] == 'decision':
        current = next(a for a in actions if a['status'] == 'pending')
        if current['action_id'] != event['action_id'] or inputs:
            raise ValueError('确认操作不匹配')
        done = decide_action(state['user_id'], state['session_id'], current['action_id'], event['decision'])
        current.update(done)
        tid = current['task_id']
        related = [a for a in actions if a['task_id'] == tid]
        if all(a['status'] != 'pending' for a in related):
            results[tid]['status'] = 'succeeded' if any(a['status'] == 'completed' for a in related) else 'cancelled' if any(a['status'] == 'cancelled' for a in related) else 'rejected'
        row = facts_for(results[tid], current['order_id'])
        if current['status'] == 'completed':
            if current['type'] == 'payment':
                row.update(payment_status='paid', method=current['parameters']['method'])
            else:
                row.update(refund_status='requested', eligible=False, reason='退款申请已记录，尚不代表到账')
        else:
            row['error'] = (current.get('result') or {}).get('reason', '用户取消，未办理')
    else:
        item = next((i for i in inputs if i['input_id'] == event['input_id']), None)
        if item is None:
            raise ValueError('输入已失效')
        tid = item['task_id']
        if event['decision'] == 'cancel':
            inputs.remove(item)
            if item['type'] == 'method':
                facts_for(results[tid], item['orders'][0]['order_id'])['error'] = '用户取消，未办理'
            results[tid]['status'] = 'awaiting_input' if any(i['task_id'] == tid for i in inputs) else 'awaiting_confirmation' if any(a['task_id'] == tid and a['status'] == 'pending' for a in actions) else 'cancelled'
        elif item['type'] == 'method':
            method = event.get('method')
            if method not in PAYMENT_METHODS:
                raise ValueError('请选择列表中的支付方式')
            oid = item['orders'][0]['order_id']
            try:
                action = prepare_action(state['user_id'], state['session_id'], state['request_id'], tid + ':' + str(oid) + ':payment',
                    {'type': 'payment', 'order_id': oid, 'method': method})
                actions.append({**action, 'task_id': tid})
            except ValueError as exc:
                payment = tools.get_payment_status(oid)
                if payment.get('status') == 'paid':
                    facts_for(results[tid], oid).update(payment_status='paid', method=payment['method'], reason='已付款，无需再次支付')
                else:
                    results[tid]['facts'].append({'order_id': oid, 'error': str(exc)})
            inputs.remove(item)
            results[tid]['status'] = 'awaiting_input' if any(i['task_id'] == tid for i in inputs) else 'awaiting_confirmation' if any(a['task_id'] == tid and a['status'] == 'pending' for a in actions) else 'succeeded' if facts_for(results[tid], oid).get('payment_status') == 'paid' else 'rejected'
        else:
            ids = list(dict.fromkeys(event['order_ids']))
            allowed = {i['order_id'] for i in item['candidates']}
            if not ids or not set(ids) <= allowed:
                raise ValueError('所选订单不在候选集合中')
            # Store selection into the existing immutable task identity, not a new plan.
            plan = copy.deepcopy(state['plan'])
            next(t for t in plan['tasks'] if t['task_id'] == tid)['targets'] = ids
            results.pop(tid)
            inputs.remove(item)
            return {'plan': plan, 'results': {tid: {'status': 'selected', 'facts': [], 'trace': []}}, 'inputs': inputs}
    positions = {(task['task_id'], row.get('order_id')): (i, j)
        for i, task in enumerate(state['plan']['tasks'])
        for j, row in enumerate(results.get(task['task_id'], {}).get('facts', []))}
    actions.sort(key=lambda a: positions[(a['task_id'], a['order_id'])])
    return {'results': results, 'actions': actions, 'inputs': inputs}


def response(state):
    from src.agents.base import sum_usage
    from src.dialogue import checked_reply
    tasks, trace, sources, lines, agents, order_results = [], [], [], [], [], []
    orders = {}
    usage = state.get('usage', {})
    for task in state['plan']['tasks']:
        result = state['results'].get(task['task_id'], {'status': 'pending', 'facts': [], 'trace': []})
        agent = CAPABILITIES[task['operation']]
        if result.get('trace') and agent not in agents:
            agents.append(agent)
        tasks.append({**task, **result, 'agent': agent})
        trace.extend(result.get('trace', []))
        usage = sum_usage(usage, result.get('usage'))
        sources.extend(result.get('sources',[]))
        if task['operation'] == 'knowledge.search':
            continue
        explained = task['operation'] in {'faq.policies', 'search.web'} and any(
            other['operation'] == 'answer.explain' and task['task_id'] in other['depends_on']
            and state['results'].get(other['task_id'], {}).get('answer')
            and state['results'][other['task_id']]['status'] == 'succeeded'
            for other in state['plan']['tasks'])
        if result.get('answer') and not explained:
            lines.append(('### 政策说明\n\n' if task['operation'] == 'faq.policies' else '### 综合说明\n\n' if task['operation'] == 'answer.explain' else '### 联网资料\n\n') + result['answer'])
        for oid in result.get('excluded', []):
            row = orders.setdefault(oid, {})
            for dep in task['depends_on']:
                row.update({k: v for k, v in facts_for(state['results'][dep], oid).items() if k not in {'status', 'order_id'}})
            checks = [matches(c, facts_for(state['results'][c['task']], oid), state['results'][c['task']]) for c in task['conditions']]
            if task['operation'] == 'payment.prepare' and row.get('payment_status') in {'paid', 'refunded'}:
                row['reason'] = '已付款，无需再次支付' if row['payment_status'] == 'paid' else '已退款，不能再次支付'
            else:
                row['reason'] = '缺少判断依据，未执行' if any(x is None for x in checks) else '条件未满足，未执行'
        if result.get('reason') and not result.get('excluded'):
            lines.append(task['source_text'] + '：' + result['reason'])
        for row in result.get('facts', []):
            paid = next((a for a in state['actions'] if a['type'] == 'payment' and a['status'] == 'completed' and a['order_id'] == row.get('order_id')), None)
            if paid and 'payment_status' in row:
                row = {**row, 'payment_status': 'paid', 'method': paid['parameters']['method']}
            if task['operation'].startswith('refund.') and row.get('order_id'):
                action = next((a for a in state['actions'] if a['task_id'] == task['task_id'] and a['order_id'] == row['order_id']), None)
                item = {'task_id': task['task_id'], 'order_id': row['order_id'], 'reason': row.get('reason', row.get('error', '')), 'reason_code': row.get('reason_code'),
                    'stage': '准备失败' if row.get('reason_code') == 'PREPARATION_FAILED' else '符合条件' if row.get('eligible') else '不符合条件'}
                if action:
                    item.update(action_id=action['action_id'], amount=action['amount'],
                        stage={'pending': '待确认', 'completed': '已提交申请', 'cancelled': '用户取消', 'rejected': '未办理'}[action['status']])
                    if (action.get('result') or {}).get('reason'):
                        item.update(reason=action['result']['reason'], reason_code=action['result'].get('reason_code'))
                order_results.append(item)
            if row.get('sources') is not None:
                sources.extend(row['sources'])
                if not result.get('answer') and not explained:
                    lines.append(row.get('error') or '联网资料已获取，见来源。')
            elif row.get('message'):
                lines.append(row['message'])
            elif 'order_id' in row:
                summary = orders.setdefault(row['order_id'], {})
                summary.update({k: v for k, v in row.items() if k not in {'status', 'order_id'}})
            elif not result.get('answer'):
                lines.extend(row.values())
    for action in state['actions']:
        row = orders.setdefault(action['order_id'], {})
        kind = '支付' if action['type'] == 'payment' else '退款申请'
        labels = {'pending': '已准备，等待确认，尚未执行', 'completed': '支付已完成' if action['type'] == 'payment' else '退款申请已记录，尚不代表到账', 'cancelled': '用户取消，未办理', 'rejected': (action.get('result') or {}).get('reason', '未办理')}
        row[kind] = labels[action['status']]
        if action['type'] == 'payment':
            row['method'] = action['parameters']['method']
        elif action['status'] == 'pending':
            row[kind] = '符合退款条件，已准备申请，等待确认，尚未执行'
            row.pop('refund_status', None)
        row.pop('reason', None)
        if action['status'] == 'cancelled' and row.get('error') == '用户取消，未办理':
            row.pop('error')
        if action['status'] == 'completed':
            if action['type'] == 'payment':
                row.update(payment_status='paid', method=action['parameters']['method'])
            else:
                row.update(refund_status='requested')
    if orders:
        lines.append('### 订单结果')
        rendered = []
        for oid, row in orders.items():
            refund_context = any(t['operation'].startswith('refund.') and any(f.get('order_id') == oid for f in t.get('facts', [])) for t in tasks) and not any(t['operation'] == 'order.details' and oid in t['targets'] and '详情' in t['source_text'] for t in tasks)
            details = []
            fields = {'item': '商品', 'order_date': '下单日期', 'order_status': '订单状态', 'payment_status': '支付状态', 'amount': '金额', 'refundable_amount': '可退金额', 'refund_status': '退款状态', 'reason': '说明', 'error': '未办理', '支付': '支付办理', '退款申请': '退款办理'}
            for key, label in fields.items():
                if key not in row or (refund_context and key in {'order_date', 'amount'}):
                    continue
                value = format(float(row[key]), '.2f') if key in {'amount', 'refundable_amount'} else LABELS.get(str(row[key]), str(row[key]))
                value = {'none': '无申请', 'requested': '申请已记录', 'approved': '已批准', 'completed': '已完成'}.get(value, value) if key == 'refund_status' else value
                details.append(f'{label}：{value}')
            if row.get('method') and not refund_context:
                details.append(('支付方式' if row.get('payment_status') == 'paid' or '支付' in row else '记录方式') + '：' + PAYMENT_METHODS.get(row['method'], row['method']))
            rendered.append((oid, details))
        if len(rendered) < 4:
            lines.extend(f'**订单 {oid}**\n\n' + '\n'.join('- ' + d for d in details) for oid, details in rendered)
        else:
            lines.append('| 订单 | 详情 |\n| --- | --- |\n' + '\n'.join(f'| **{oid}** | ' + '；'.join(d.replace('|', '\\|').replace('\n', ' ') for d in details) + ' |' for oid, details in rendered))
    current = next((a for a in state['actions'] if a['status'] == 'pending'), None)
    status = 'awaiting_selection' if any(i['type'] == 'orders' for i in state['inputs']) else 'awaiting_payment_method' if state['inputs'] else 'awaiting_confirmation' if current else 'completed'
    return {'status': status, 'reply': '\n\n'.join(str(x) for x in lines) or checked_reply(state['plan'].get('answer', '')),
        'agent': 'collaboration' if len(agents) > 1 else (agents[0] if agents and agents[0] != 'search' else 'assistant'),
        'agents': agents, 'tasks': tasks, 'trace': trace, 'sources': sources, 'actions': state['actions'],
        'current_action': current if not state['inputs'] else None, 'pending_inputs': state['inputs'], 'order_results': order_results, 'usage': usage}


def build_workflow(checkpointer):
    graph = StateGraph(State)
    for name, fn in [('plan', plan_node), ('schedule', schedule), ('expert', expert), ('prepare', prepare), ('review', review), ('wait', wait), ('apply_input', apply_input)]:
        graph.add_node(name, fn)
    graph.add_edge(START, 'plan')
    graph.add_edge('plan', 'schedule')
    graph.add_conditional_edges('schedule', dispatch, ['expert', 'prepare', 'review'])
    graph.add_edge('expert', 'schedule')
    graph.add_edge('prepare', 'schedule')
    graph.add_conditional_edges('review', after_review, ['wait', END])
    graph.add_edge('wait', 'apply_input')
    graph.add_edge('apply_input', 'schedule')
    return graph.compile(checkpointer=checkpointer)
