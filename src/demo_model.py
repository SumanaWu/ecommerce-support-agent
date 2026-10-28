"""无密钥规则模拟器。产生标准工具请求，实际图与工具照常执行。"""
import json
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.memory import extract_order_ids
from src.runtime_context import active_order_reference
from src.planning import allows_write, explicit_escalation, negative_action


def route_query(text):
    if text.startswith('(Context:'):
        text = text.split('\n', 1)[-1]
    q = text.lower()
    if explicit_escalation(q):
        return 'ESCALATE'
    # 通用规则问题与具体业务操作分开；并非语义分类模型。
    if any(x in q for x in ['政策', '规则', 'policy', '多久', '退货条件', 'shipping time', 'how long', 'internationally', 'support email']):
        return 'FAQ'
    if not negative_action(q, 'REFUND') and any(x in q for x in ['退款', '退钱', '退掉', '退那笔', '退这笔', 'refund', 'money back', 'return order']):
        return 'REFUND'
    if not negative_action(q, 'PAYMENT') and any(x in q for x in ['支付', '付款', 'payment', 'pay ', 'card get charged']):
        return 'PAYMENT'
    if any(x in q for x in ['订单', 'order', '物流', '快递', 'tracking', 'package', '详情']):
        return 'ORDER'
    return 'CLARIFY'


class DemoChatModel:
    def __init__(self, tools=None):
        self.tools = {t.name for t in (tools or [])}

    def bind_tools(self, tools):
        return DemoChatModel(tools)

    def _call(self, name, args):
        return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': uuid4().hex, 'type': 'tool_call'}])

    def invoke(self, messages):
        if isinstance(messages, str):
            q = messages.split('User message: ', 1)[-1].rsplit('Answer with one word only.', 1)[0]
            return AIMessage(content=route_query(q))
        last_user = max(i for i,m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[last_user].content)
        # 上下文备注在前，用户显式输入在后；明确编号优先于记住的旧编号。
        reference = active_order_reference.get()
        ids = extract_order_ids(query)
        order_id = reference['order_id'] if reference is not None else ids[0] if len(ids) == 1 else None
        results = [m for m in messages[last_user+1:] if isinstance(m, ToolMessage)]
        used = {m.name for m in results}
        q = query.lower()
        if 'answer_faq' in self.tools:
            if not results:
                return self._call('answer_faq', {'question': query})
        elif order_id is None:
            return AIMessage(content='请先提供订单编号，例如“查询订单 1001”。')
        elif 'initiate_refund' in self.tools:
            if 'check_refund_eligibility' not in used:
                return self._call('check_refund_eligibility', {'order_id': order_id})
            data = json.loads(results[-1].content)
            wants_write = allows_write(query, 'REFUND')
            if data.get('eligible') and wants_write and 'initiate_refund' not in used:
                return self._call('initiate_refund', {'order_id': order_id, 'reason': '客户申请：'+query[-150:]})
        elif 'process_payment' in self.tools:
            if 'get_payment_status' not in used:
                return self._call('get_payment_status', {'order_id': order_id})
            data = json.loads(results[-1].content)
            wants_write = allows_write(query, 'PAYMENT')
            if data.get('status') == 'pending' and wants_write and 'process_payment' not in used:
                method = 'paypal' if 'paypal' in q else 'card'
                return self._call('process_payment', {'order_id': order_id, 'method': method})
        elif not results:
            name = 'track_shipment' if any(x in q for x in ['物流', '快递', 'tracking']) else 'get_order_details'
            return self._call(name, {'order_id': order_id})
        outputs = '\n\n'.join(m.content for m in results)
        return AIMessage(content='【离线规则演示】以下为本地样例系统的实际工具结果，不代表真实电商平台或资金流水。\n\n'+outputs)
