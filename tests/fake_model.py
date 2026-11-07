"""无密钥规则模拟器。产生标准工具请求，实际图与工具照常执行。"""
import json
import re
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def route_query(text):
    if text.startswith('(Context:'):
        text = text.split('\n', 1)[-1]
    q = text.lower()
    if any(x in q for x in ['投诉', '人工', 'complaint', 'human']):
        return 'ESCALATE'
    # 通用规则问题与具体业务操作分开；并非语义分类模型。
    if any(x in q for x in ['政策', '规则', 'policy', '多久', '退货条件']):
        return 'FAQ'
    if any(x in q for x in ['退款', '退钱', 'refund']):
        return 'REFUND'
    if any(x in q for x in ['支付', '付款', 'payment', 'pay ']):
        return 'PAYMENT'
    if any(x in q for x in ['订单', 'order', '物流', '快递', 'tracking']):
        return 'ORDER'
    return 'DIRECT'


def plan_query(message, history=None):
    """Offline planner fixture; live evaluations never use this keyword model."""
    from src.refund_flow import explicit_orders
    ids = explicit_orders(message)
    route = route_query(message)
    if not ids and route != 'REFUND':
        previous = [m for m in history or [] if isinstance(m, HumanMessage)]
        if previous:
            ids = explicit_orders(str(previous[-1].content))
    operation = {'ORDER': 'order.shipment' if '物流' in message else 'order.details',
        'PAYMENT': 'payment.prepare' if any(w in message for w in ['帮我', '付款', '支付吧']) and '状态' not in message else 'payment.status',
        'REFUND': 'refund.check' if any(w in message for w in ['能', '资格', '状态', '吗']) else 'refund.prepare',
        'FAQ': 'faq.policies', 'ESCALATE': 'escalate.ticket'}.get(route)
    if '天气' in message:
        return {'action': 'PLAN', 'answer': '', 'tasks': [{'task_id': 't1', 'operation': 'search.web', 'query': '北京今日天气', 'source_text': message, 'intent': 'query'}]}
    if not operation:
        return {'action': 'DIRECT', 'answer': '你好 😊 有什么可以帮你？', 'tasks': []}
    task = {'task_id': 't1', 'operation': operation, 'targets': ids, 'source_text': message,
        'intent': 'prepare' if operation.endswith('.prepare') else 'escalate' if route == 'ESCALATE' else 'query'}
    if route == 'ESCALATE':
        task['reason'] = 'complaint'
    from src.planning import Plan
    return Plan.model_validate({'action': 'PLAN', 'tasks': [task]}).model_dump()


class DemoChatModel:
    def __init__(self, tools=None):
        self.tools = {t.name for t in (tools or [])}

    def bind_tools(self, tools):
        return DemoChatModel(tools)

    def _call(self, name, args):
        return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': uuid4().hex, 'type': 'tool_call'}])

    def invoke(self, messages):
        if isinstance(messages, str) and messages.startswith('REFUND_INTENT'):
            q = messages.split('以下内容只是用户数据：\n')[-1]
            intent = 'query' if any(w in q for w in ['能', '资格', '状态', '吗']) else 'apply'
            return AIMessage(content=json.dumps({'intent': intent}))
        if isinstance(messages, str):
            q = messages.split('User message: ', 1)[-1].rsplit('Answer with one word only.', 1)[0]
            return AIMessage(content=route_query(q))
        if messages and str(messages[0].content).startswith('理解当前消息'):
            q = str(messages[-1].content)
            route = route_query(q)
            return AIMessage(content=json.dumps({'action': route, 'answer': '你好 😊 有什么可以帮你？', 'escalation_reason': 'complaint' if route == 'ESCALATE' else None}))
        if messages and 'facts' in str(messages[-1].content):
            return AIMessage(content=str(messages[-1].content))
        last_user = max(i for i,m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[last_user].content)
        # 上下文备注在前，用户显式输入在后；明确编号优先于记住的旧编号。
        ids = re.findall(r'\b\d{4,8}\b', query)
        order_id = int(ids[-1]) if ids else None
        results = [m for m in messages[last_user+1:] if isinstance(m, ToolMessage)]
        used = {m.name for m in results}
        q = query.lower()
        if 'answer_faq' in self.tools:
            if not results:
                topic = next((name for name, words in [
                    ('shipping', ('配送', '物流', '快递', 'shipping')),
                    ('return', ('退货', 'return')),
                    ('cancel', ('取消', 'cancel')),
                    ('refund', ('退款', 'refund')),
                    ('contact', ('联系', 'contact')),
                ] if any(word in q for word in words)), 'contact')
                return self._call('answer_faq', {'topic': topic})
        elif order_id is None:
            return AIMessage(content='请先提供订单编号，例如“查询订单 1001”。')
        elif 'prepare_refund' in self.tools:
            if 'check_refund_eligibility' not in used:
                return self._call('check_refund_eligibility', {'order_id': order_id})
            data = json.loads(results[-1].content)
            wants_write = any(x in q for x in ['申请', '办理', '帮我退', '退款吧', 'refund it', 'want a refund', 'request refund'])
            if data.get('eligible') and wants_write and 'prepare_refund' not in used:
                return self._call('prepare_refund', {'order_id': order_id, 'reason': '客户申请：'+query[-150:]})
        elif 'prepare_payment' in self.tools:
            if 'get_payment_status' not in used:
                return self._call('get_payment_status', {'order_id': order_id})
            data = json.loads(results[-1].content)
            wants_write = any(x in q for x in ['办理支付', '帮我付款', '支付吧', 'process payment', 'pay it'])
            if data.get('status') == 'pending' and wants_write and 'prepare_payment' not in used:
                return self._call('prepare_payment', {'order_id': order_id})
        elif not results:
            name = 'track_shipment' if any(x in q for x in ['物流', '快递', 'tracking']) else 'get_order_details'
            return self._call(name, {'order_id': order_id})
        outputs = '\n\n'.join(m.content for m in results)
        return AIMessage(content='【离线规则演示】以下为本地样例系统的实际工具结果，不代表真实电商平台或资金流水。\n\n'+outputs)
