"""Execution facts are independent of whether a model produced a final answer."""
import json


def execution_status(trace):
    results = [entry.get('result') for entry in trace]
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
        return 'failed'
    return 'rejected' if refused else 'completed'


def fallback_result(trace, *, agent=None, usage=None, interrupted=False):
    lines = []
    for entry in trace:
        result = entry['result']
        if entry['tool'] == 'initiate_refund' and result.get('success') and result.get('status') == 'requested':
            lines.append(f"订单 {result['order_id']} 的本地退款申请已保存，状态为 requested；不代表退款到账。")
        elif entry['tool'] == 'process_payment' and result.get('success') and result.get('status') == 'paid':
            lines.append(f"订单 {result['order_id']} 的本地支付状态已更新为 paid；没有真实扣款。")
        elif isinstance(result, dict) and (result.get('escalated') or result.get('requires_confirmation') or result.get('status') == 'cancelled'):
            lines.append(result['message'])
        else:
            lines.append(json.dumps(result, ensure_ascii=False))
    if interrupted:
        lines.append('请求执行曾中断，以上是已保存的动作或提案结果；未自动重跑业务流程，请核对其余需求。')
    return {'reply': '\n\n'.join(lines), 'agent': agent or trace[-1]['agent'],
            'trace': trace, 'usage': usage or {}, 'cached': False,
            'status': 'partial' if interrupted else execution_status(trace), 'reply_source': 'fallback'}
