"""Execution facts are independent of whether a model produced a final answer."""
import json

from src.money import format_amount, major_to_minor
from src.planning import iter_order_results


PAYMENT_STATUS = {'paid': '已付款', 'pending': '待付款', 'refunded': '已退款'}
ORDER_STATUS = {'placed': '已下单', 'shipped': '运输中', 'delivered': '已送达', 'cancelled': '已取消'}
TASK_LABELS = {'order_status': '订单状态', 'order_details': '订单详情', 'shipment': '物流查询',
               'payment_status': '付款状态', 'refund_eligibility': '退款资格',
               'propose_refund': '退款申请', 'propose_payment': '付款办理', 'policy': '政策查询'}


def _text(value):
    """Keep business strings in a single Markdown cell, never executable markup."""
    import html
    value = html.escape(str(value)).replace('\n', ' ').replace('\r', ' ')
    for character in ('\\', '`', '*', '_', '[', ']', '|'):
        value = value.replace(character, '\\' + character)
    return value


def _amount(result):
    currency = result.get('currency')
    try:
        minor = result.get('amount_minor')
        if minor is None:
            minor = major_to_minor(result.get('amount_display', result.get('amount')), currency)
        return format_amount(minor, currency) + ' ' + currency
    except ValueError:
        return ''  # Legacy results without a currency do not acquire an inferred one.


def _fact(tool, result, order_id=None, answer_mode='result'):
    if not isinstance(result, dict):
        return '本项尚未取得可核实的业务结果。'
    order_id = result.get('order_id', order_id)
    prefix = f'订单 {order_id}：' if order_id is not None else ''
    if result.get('error'):
        return prefix + ('未找到可访问的订单。' if result['error'] == '未找到可访问的订单。' else '本项执行未完成，请核对后重试。')
    amount = _amount(result)
    price = '，金额 ' + amount if amount else ''
    if result.get('requires_confirmation') or result.get('status') == 'pending_confirmation':
        label = '退款申请' if tool == 'initiate_refund' else '付款'
        parameters = []
        if tool == 'process_payment' and result.get('method') in {'card', 'paypal'}:
            parameters.append('支付方式：' + {'card': '银行卡', 'paypal': 'PayPal'}[result['method']])
        if tool == 'initiate_refund' and result.get('reason'):
            parameters.append('申请原因：' + _text(result['reason']))
        return ('⏳ ' + prefix + label + '待确认' + price + '。' +
                ('\n\n' + '\n'.join('- ' + value for value in parameters) if parameters else '') +
                '\n\n请核对后点击确认；当前尚未办理。')
    if result.get('status') == 'cancelled' and 'action_id' in result:
        return prefix + ('操作已取消，不能确认执行。' if result.get('error_code') == 'cancelled' else '操作已取消，未执行业务办理。')
    if result.get('error_code') == 'already_executed':
        return prefix + '操作已执行，不能取消。'
    if result.get('error_code') == 'stale_action':
        return prefix + '订单数据或业务状态已变化，本次操作未执行；请重新查询并创建待确认操作。'
    if tool == 'get_payment_status':
        status = PAYMENT_STATUS.get(result.get('status'), '状态未知')
        method = {'card': '银行卡', 'paypal': 'PayPal'}.get(result.get('method'))
        return prefix + '当前' + status + price + ('，支付方式' + method if method else '') + '。'
    if tool in {'get_order_status', 'get_order_details', 'track_shipment'}:
        status = ORDER_STATUS.get(result.get('status'), '状态未知')
        if tool == 'track_shipment':
            status = {'placed': '已下单，尚未发货', 'cancelled': '已取消，没有配送中的包裹'}.get(result.get('status'), status)
        label = '物流状态' if tool == 'track_shipment' else '订单状态'
        text = prefix + label + '为' + status + price + '。'
        if tool == 'get_order_details':
            details = [('商品', 'item'), ('下单日期', 'order_date'), ('收件人', 'customer_name')]
            fields = ['- ' + label + '：' + _text(result[key]) for label, key in details if key in result]
            if fields:
                text += '\n\n' + '\n'.join(fields)
        return text
    if result.get('success') and tool == 'initiate_refund' and result.get('status') == 'requested':
        return '✅ ' + prefix + '退款申请已提交，等待处理' + price + '。'
    if result.get('success') and tool == 'process_payment' and result.get('status') == 'paid':
        return '✅ ' + prefix + '付款成功，当前已付款' + price + '。'
    if 'eligible' in result:
        existing = {'requested': '已有退款申请，正在等待处理，不能重复申请；申请不代表款项已到账',
                    'approved': '退款申请已获批准，不能重复申请；当前没有银行到账凭据',
                    'completed': '已有退款完成记录，不能重复申请'}.get(result.get('refund_status'), '已有退款记录，不能重复申请')
        if answer_mode == 'explain':
            status = ORDER_STATUS.get(result.get('order_status'), '尚未送达')
            explanations = {
                'not_delivered': f'当前订单状态为{status}。退款规则要求订单已送达且没有已有退款记录，因此暂不能申请退款。',
                'existing_refund': existing + '。退款规则不允许重复申请。'}
            return prefix + ('订单已送达，且没有已有退款记录，符合当前退款申请规则；申请仍需独立确认。' if result['eligible'] else
                             explanations.get(result.get('reason_code'), '当前工具确认订单不符合退款条件，暂不能申请。'))
        reason = {'not_delivered': '订单已取消，不满足已送达条件' if result.get('order_status') == 'cancelled' else '订单尚未送达',
                  'existing_refund': existing}.get(result.get('reason_code'), '当前不符合退款条件')
        return prefix + ('符合退款申请条件；尚未提交申请。' if result['eligible'] else '暂不能申请退款：' + reason + '。')
    if result.get('success') is False:
        return prefix + '本次操作未办理，请重新核对当前业务状态。'
    if tool == 'answer_faq' and isinstance(result.get('answer'), str):
        sources = result.get('sources', [])
        if sources:
            if len(sources) == 1:
                return sources[0]['excerpt']
            return '\n\n'.join('- **' + _text(source['section']) + '**：' +
                               source['excerpt'].replace('\n', ' ') for source in sources)
        return result['answer']
    if result.get('escalated'):
        return '已生成待处理工单 #' + _text(result['ticket_id']) + '。'
    return prefix + '本项尚无可展示的业务结果。'


def _table(tool, items):
    single = {'get_order_status_batch': 'get_order_status', 'get_order_details_batch': 'get_order_details',
              'track_shipments_batch': 'track_shipment', 'get_payment_status_batch': 'get_payment_status'}[tool]
    payment = single == 'get_payment_status'
    detail = single == 'get_order_details'
    headers = ['订单', '付款状态' if payment else '物流状态' if single == 'track_shipment' else '订单状态']
    if detail or payment:
        headers.append('金额')
    if detail:
        headers += ['商品', '下单日期', '收件人']
    rows = []
    for item in items:
        fact = item['result']
        unavailable = bool(fact.get('error'))
        status = ('未找到可访问的订单' if fact.get('error') == '未找到可访问的订单。' else '未完成') if unavailable else (
            PAYMENT_STATUS if payment else ORDER_STATUS).get(fact.get('status'), '状态未知')
        row = [item['order_id'], status]
        if detail or payment:
            row.append('—' if unavailable else _amount(fact) or '—')
        if detail:
            row += [fact.get(key, '—') if not unavailable else '—' for key in ('item', 'order_date', 'customer_name')]
        rows.append('| ' + ' | '.join(_text(value) for value in row) + ' |')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |', *rows])


def _without_covered_order_reads(trace, details):
    """Hide status rows already shown by details; retain original execution records."""
    def covered(number, fact):
        return (number, fact.get('status'), fact.get('error')) in details

    for entry in trace:
        fact = entry.get('result')
        if entry['tool'] not in {'get_order_status', 'get_order_status_batch', 'track_shipment', 'track_shipments_batch'} or not isinstance(fact, dict):
            yield entry
        elif isinstance(fact.get('items'), list):
            items = [item for item in fact['items'] if not covered(item['order_id'], item['result'])]
            if items:
                yield {**entry, 'result': {**fact, 'items': items}}
        elif not covered(fact.get('order_id', (entry.get('args') or {}).get('order_id')), fact):
            yield entry


def _presentation_note(intent, trace):
    facts = [(entry['tool'], fact) for entry in trace
             for _, fact in iter_order_results(entry.get('result'))
             if not fact.get('error') and fact.get('executed') is not False]
    order_tools = {'get_order_status', 'get_order_status_batch', 'get_order_details',
                   'get_order_details_batch', 'track_shipment', 'track_shipments_batch'}
    has_order_status = any(tool in order_tools and fact.get('status') in ORDER_STATUS for tool, fact in facts)
    if intent == 'delivery_timing' and has_order_status:
        return '本次查询未提供预计送达日期，无法保证具体哪天送达。'
    if intent == 'proposal_effect' and has_order_status:
        return '取消待确认操作或创建工单，都不等于取消订单；订单状态以本次查询结果为准。'
    if intent == 'refund_progress' and any(tool == 'check_refund_eligibility' and
            fact.get('refund_status') in {'none', 'requested', 'approved', 'completed'} for tool, fact in facts):
        return '当前工具未提供银行到账凭据或预计到账日期，无法据此确认实际到账或承诺到账时间。'
    return ''


def compose_response(result):
    """Render tool facts without interpreting the user's language or trusting model numbers.

    The model's conversation/clarification stays intact. Business prose uses the
    already planned task order and recorded results; no extra model call is made.
    """
    views = {view['action_id']: view for view in result.get('actions', [])}
    details = {(number, fact.get('status'), fact.get('error'))
               for entry in result.get('trace', []) if entry['tool'] in {'get_order_details', 'get_order_details_batch'}
               for number, fact in iter_order_results(entry.get('result'))
               if fact.get('status') or fact.get('error') == '未找到可访问的订单。'}
    def render(trace, *, omit_covered=False, answer_mode='result', presentation_intent='result', recovered=False,
               operation=None, status=None):
        lines = []
        proposed = {entry['result']['order_id'] for entry in trace if isinstance(entry.get('result'), dict)
                    and entry['result'].get('action_id') and entry['tool'] in {'initiate_refund', 'process_payment'}}
        for entry in (_without_covered_order_reads(trace, details) if omit_covered else trace):
            fact = entry.get('result')
            if recovered and isinstance(fact, dict) and fact.get('executed') is False:
                continue
            if entry['tool'] in {'check_refund_eligibility', 'get_payment_status'} and isinstance(fact, dict) and fact.get('order_id') in proposed:
                continue
            action = views.get(fact.get('action_id')) if isinstance(fact, dict) else None
            if action:
                proposal = fact.get('requires_confirmation') or fact.get('status') == 'pending_confirmation'
                fact = {**(action.get('result') or {} if proposal and action['status'] != 'pending' else fact),
                        'order_id': action['order_id'], 'action_id': action['action_id'],
                        **{key: action[key] for key in ('amount_minor', 'currency')},
                        **{key: value for key, value in action.get('args', {}).items() if key in {'method', 'reason'}}}
                fact['requires_confirmation'] = bool(proposal and action['status'] == 'pending')
            if isinstance(fact, dict) and isinstance(fact.get('items'), list) and entry['tool'] in {
                    'get_order_status_batch', 'get_order_details_batch', 'track_shipments_batch', 'get_payment_status_batch'}:
                lines.append(_table(entry['tool'], fact['items']))
            else:
                lines.append(_fact(entry['tool'], fact, (entry.get('args') or {}).get('order_id'), answer_mode))
        note = _presentation_note(presentation_intent, trace)
        if note:
            lines.append(note)
        if operation == 'propose_payment' and status == 'rejected' and any(
                entry['tool'] == 'get_payment_status' and isinstance(entry.get('result'), dict) and
                entry['result'].get('status') in {'paid', 'refunded'} for entry in trace):
            lines.append('该订单不能再次付款；本次未创建新的付款操作。')
        return '\n\n'.join(lines)
    if not result.get('trace'):
        return result
    result = {**result}
    if result.get('reply_source') == 'model':
        result.setdefault('model_reply', result.get('reply', ''))
    tasks = []
    for task in result.get('tasks', []):
        tasks.append({**task, 'reply': render(task['trace'], answer_mode=task.get('answer_mode', 'result'),
                      presentation_intent=task.get('presentation_intent', 'result'),
                      operation=task.get('operation'), status=task['status'],
                      recovered=task['status'] in {'completed', 'rejected', 'pending_confirmation'}) if task.get('trace') else task['reply']})
    result['tasks'] = tasks
    sections = [(TASK_LABELS.get(task.get('operation'), '任务结果'),
                 render(task['trace'], omit_covered=True, answer_mode=task.get('answer_mode', 'result'),
                        presentation_intent=task.get('presentation_intent', 'result'),
                        operation=task.get('operation'), status=task['status'],
                        recovered=task['status'] in {'completed', 'rejected', 'pending_confirmation'}) if task.get('trace') else task['reply']) for task in tasks]
    sections = [(label, body) for label, body in sections if body]
    reply = ('\n\n'.join('- **' + label + '**：' + ('\n\n  ' if '\n' in body else '') +
                         body.replace('\n', '\n  ') for label, body in sections)
             if len(sections) > 1 else sections[0][1] if sections else render(result['trace'], omit_covered=True,
                 presentation_intent=result.get('presentation_intent', 'result')))
    business_outcomes = bool(tasks) and all(task['status'] in {'completed', 'rejected', 'pending_confirmation', 'cancelled'} for task in tasks)
    if result.get('status') in {'partial', 'failed', 'blocked'} and not business_outcomes:
        reply += '\n\n本次请求尚未全部完成；以上仅为已取得的结果。'
    if result.get('error_type') in {'configuration', 'authentication', 'balance', 'rate_limit', 'service', 'network', 'program', 'budget'}:
        from src.errors import error_message
        reply += '\n\n' + error_message(result['error_type'])
    if result.get('stop_reason') == 'pipeline_incompatible':
        reply += '\n\n旧流程与当前版本不兼容，未自动重跑；请重新发起剩余需求。'
    if result.get('error_type') == 'order_reference_changed':
        reply += '\n\n当前事项已更正，原运行已停止；已有动作请单独核对或取消。'
    return {**result, 'reply': reply, 'reply_source': 'policy' if result.get('reply_source') == 'policy' else 'facts', 'response_version': 1}


def execution_status(trace, *, readonly=False):
    results = []
    for entry in trace:
        result = entry.get('result')
        results.extend([item['result'] for item in result['items']] if isinstance(result, dict) and 'items' in result else [result])
    if any(isinstance(r, dict) and r.get('requires_confirmation') for r in results):
        return 'pending_confirmation'
    if results and all(isinstance(r, dict) and r.get('status') == 'cancelled' and r.get('success') for r in results):
        return 'cancelled'
    succeeded = any(isinstance(r, dict) and (r.get('success') is True or r.get('escalated')) for r in results)
    errors = any(isinstance(r, dict) and r.get('error') for r in results)
    refused = any(isinstance(r, dict) and (r.get('eligible') is False or r.get('success') is False) for r in results)
    if succeeded and (errors or refused):
        return 'partial'
    if errors:
        return 'partial' if any(isinstance(r, dict) and not r.get('error') for r in results) else 'failed'
    if readonly:
        return 'completed'
    return 'rejected' if refused else 'completed'


def fallback_result(trace, *, agent=None, usage=None, interrupted=False):
    lines = []
    for entry in trace:
        result = entry['result']
        if entry['tool'] == 'initiate_refund' and result.get('success') and result.get('status') == 'requested':
            lines.append(f"订单 {result['order_id']} 的退款申请已提交，状态为 requested。")
        elif entry['tool'] == 'process_payment' and result.get('success') and result.get('status') == 'paid':
            lines.append(f"订单 {result['order_id']} 付款成功，支付状态为 paid。")
        elif entry['tool'] == 'get_payment_status' and isinstance(result, dict) and 'order_id' in result and 'status' in result:
            status = {'paid': '已付款', 'pending': '待付款', 'refunded': '已退款'}.get(result['status'], result['status'])
            detail = f"，金额 {result['amount_display']} {result['currency']}" if 'amount_display' in result and 'currency' in result else ''
            lines.append(f"订单 {result['order_id']} 的付款状态：{status}{detail}。")
        elif isinstance(result, dict) and (result.get('escalated') or result.get('requires_confirmation') or result.get('status') == 'cancelled'):
            lines.append(result['message'])
        else:
            lines.append(json.dumps(result, ensure_ascii=False))
    if interrupted:
        lines.append('请求执行曾中断，以上是已保存的动作或提案结果；未自动重跑业务流程，请核对其余需求。')
    return {'reply': '\n\n'.join(lines), 'agent': agent or trace[-1]['agent'],
            'trace': trace, 'usage': usage or {}, 'cached': False,
            'status': 'partial' if interrupted else execution_status(trace), 'reply_source': 'fallback'}
