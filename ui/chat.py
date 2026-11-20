from ui.i18n import tr, table_rows, localize_format
from uuid import uuid4
import streamlit as st
from src.payment_methods import PAYMENT_METHODS
OPERATIONS = {'order.list': '查询订单列表', 'order.details': '查询订单详情', 'order.status': '查询订单状态', 'order.shipment': '查询物流', 'payment.status': '查询支付', 'payment.prepare': '准备支付', 'refund.check': '检查退款资格', 'refund.prepare': '准备退款', 'knowledge.search': '检索平台资料', 'answer.explain': '综合资料', 'faq.policies': '系统帮助', 'search.web': '联网搜索', 'escalate.ticket': '创建本地工单'}
STATUSES = {'pending': '等待执行', 'selected': '已选择', 'succeeded': '已完成', 'failed': '处理失败', 'rejected': '未办理', 'skipped': '已跳过', 'cancelled': '已取消', 'awaiting_input': '等待选择', 'awaiting_confirmation': '等待确认'}
AGENTS = {'order': '订单专家', 'payment': '支付专家', 'refund': '退款专家', 'knowledge': '知识检索', 'assistant': '主对话层', 'faq': '系统帮助', 'search': '联网搜索', 'escalate': '本地工单'}

def render(svc, user_id, session_id, payload):
    if not payload['turns']:
        st.subheader(tr('你好，需要什么帮助？'))
        st.caption(tr('可以查询订单、了解平台政策，或告诉我你的长期购物偏好。'))
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            st.caption('协作处理：' + '、'.join(t['agents']) if t.get('agent') == 'collaboration' else '主对话层' if t['agent'] == 'assistant' else '处理专家：' + t['agent'])
            for source in t.get('sources') or []:
                if source.get('url'):
                    st.markdown('[{0}]({1})'.format(source['title'], source['url']))
                else:
                    with st.expander('{0} · {1} · {2}'.format(source['title'], source.get('location', ''), source.get('doc_id', ''))):
                        st.write(source.get('content', ''))
            if t.get('status') == 'processing' and st.button(tr('继续处理'), key='retry_' + t['request_id']):
                try:
                    svc.chat(user_id, session_id, t['user'], request_id=t['request_id'])
                    st.rerun()
                except Exception as exc:
                    st.error(tr('处理失败：' + str(exc)))
            memory_result = t.get('memory') or {}
            if t is payload['turns'][-1] and memory_result.get('status') == 'applied':
                current_memories = {m['key']: m for m in svc.store.memories(user_id)}
                for item in memory_result.get('candidates', []):
                    current_memory = current_memories.get(item['key'])
                    if item['key'].startswith('shopping_') and item['origin'] == 'observed' and current_memory and (current_memory['status'] == 'active') and (current_memory['value'] != item['value']):
                        st.caption(tr('已保留原有购物偏好；如需修改，请明确告诉我。'))
            prefix = '{0}_{1}_{2}'.format(user_id, session_id, t['request_id'])
            if t.get('status') == 'awaiting_selection':
                selection = t['selection']
                prefix += '_' + selection.get('input_id', 'orders')
                all_orders = st.checkbox(tr('查看我的全部订单'), key=prefix + '_all')
                candidates = sorted(selection['candidates'], key=lambda order: (not order.get('eligible', False), order['order_id']))
                visible_orders = candidates if all_orders else candidates[:3]
                drafts = st.session_state.setdefault('draft_' + prefix, {})
                eligible_ids = {order['order_id'] for order in candidates if order.get('eligible')}
                for oid in list(drafts):
                    if oid not in eligible_ids:
                        drafts.pop(oid)
                for order in visible_orders:
                    oid = order['order_id']
                    label = '订单 {0} · {1} · {2} · {3}'.format(oid, order['item'], order['amount'], order['status'])
                    if order.get('eligible'):
                        key = prefix + '_order_' + str(oid)
                        drafts[oid] = st.toggle(tr(label), value=drafts.get(oid, False), key=key)
                    else:
                        st.write(tr(label))
                    code = order.get('reason_code')
                    if code in {'ELIGIBLE', 'REFUND_IN_PROGRESS'}:
                        color = 'green'
                    elif code in {'ORDER_NOT_ACCESSIBLE', 'UNPAID', 'NO_REFUNDABLE_BALANCE', 'STATUS_NOT_ELIGIBLE'}:
                        color = 'red'
                    else:
                        color = 'blue'
                    st.caption(':{0}[{1}]'.format(color, order.get('reason', '')))
                selected = [oid for oid, chosen in drafts.items() if chosen]
                st.caption(tr('已选 {0} 笔'.format(len(selected))))
                for decision, label in [('submit', '准备所选订单退款'), ('cancel', '取消选单')]:
                    if st.button(tr(label), key=prefix + decision, disabled=decision == 'submit' and (not selected)):
                        operation_key = prefix + decision + str(selected)
                        operation = st.session_state.setdefault(operation_key, uuid4().hex)
                        try:
                            svc.select_orders(user_id, session_id, t['request_id'], operation, decision, selected, selection.get('input_id'))
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
            slots = {}
            for task in t.get('tasks', []):
                if task.get('operation') not in {'payment.prepare', 'refund.prepare'}:
                    continue
                tid = task['task_id']
                ids = list(dict.fromkeys(([*task.get('targets', [])] if task.get('status') == 'pending' else []) + [a['order_id'] for a in t.get('actions', []) if a['task_id'] == tid] + [o['order_id'] for entry in t.get('pending_inputs', []) if entry.get('task_id') == tid for o in entry.get('orders', [])] + [f['order_id'] for f in task.get('facts', []) if f.get('order_id') and f.get('error') == '用户取消，未办理']))
                for oid in ids:
                    slots[tid, oid] = st.container()
                    if task.get('status') == 'pending':
                        with slots[tid, oid], st.container(border=True):
                            kind = '支付' if task['operation'] == 'payment.prepare' else '退款申请'
                            st.markdown(tr('**订单 {0} · {1}**'.format(oid, kind)))
                            st.caption(tr('状态：等待前一项处理，尚未核实'))
                if not ids and task.get('status') == 'pending':
                    with st.container(border=True):
                        kind = '支付' if task['operation'] == 'payment.prepare' else '退款申请'
                        st.markdown('**{0}**'.format(kind))
                        st.caption(tr('状态：等待前置任务确定订单和处理条件'))
            for entry in t.get('pending_inputs', []):
                if entry['type'] != 'method':
                    continue
                with slots[entry['task_id'], entry['orders'][0]['order_id']], st.container(border=True, key=prefix + '_payment_' + str(entry['orders'][0]['order_id'])):
                    order = entry['orders'][0]
                    st.markdown(tr('**订单 {0} · 支付**'.format(order['order_id'])))
                    st.write(tr('待支付金额：{0}'.format(order['amount'])))
                    key = prefix + '_' + entry['input_id']
                    method = st.selectbox(tr('支付方式'), list(PAYMENT_METHODS), index=None, format_func=localize_format(PAYMENT_METHODS.get), placeholder=tr('请选择支付方式'), key=key)
                    st.caption(tr('选择方式仅准备方案；点击确认付款后才执行。'))
                    if method is not None:
                        operation = st.session_state.setdefault(key + '_submit_' + method, uuid4().hex)
                        try:
                            svc.select_payment_method(user_id, session_id, t['request_id'], operation, 'submit', method, entry['input_id'])
                            st.rerun()
                        except Exception as exc:
                            st.error(tr('处理失败：' + str(exc)))
                    st.button(tr('确认付款'), key=key + '_confirm', disabled=True)
                    if st.button(tr('取消支付'), key=key + '_cancel'):
                        operation = st.session_state.setdefault(key + '_cancel_request', uuid4().hex)
                        svc.select_payment_method(user_id, session_id, t['request_id'], operation, 'cancel', input_id=entry['input_id'])
                        st.rerun()
            current = t.get('current_action')
            for action in t.get('actions', []):
                with slots[action['task_id'], action['order_id']], st.container(border=True, key=prefix + '_' + action['type'] + '_' + str(action['order_id'])):
                    st.markdown(tr('**订单 {0} · 金额 {1} · {2}**'.format(action['order_id'], action['amount'], '退款申请' if action['type'] == 'refund' else '支付')))
                    st.caption(tr('状态：' + {'pending': '待确认，尚未执行', 'completed': '已完成', 'cancelled': '已取消', 'rejected': '未办理'}[action['status']]))
                    if action['type'] == 'payment':
                        method = action['parameters']['method']
                        st.write(tr('支付方式：' + PAYMENT_METHODS.get(method, method)))
                    if action['status'] == 'pending' and (not current):
                        st.caption(tr('请先完成其他必要选择，再按顺序确认。'))
                    elif action['status'] == 'pending' and current['action_id'] != action['action_id']:
                        st.caption(tr('等待前一项操作确认或取消。'))
                    if current and action['action_id'] == current['action_id']:
                        for decision, label in [('confirm', '确认付款' if action['type'] == 'payment' else '确认办理'), ('cancel', '取消本项')]:
                            key = '{0}_{1}_{2}_{3}'.format(user_id, session_id, action['action_id'], decision)
                            if st.button(tr(label), key=key):
                                operation = st.session_state.setdefault('op_' + key, uuid4().hex)
                                try:
                                    svc.resume(user_id, session_id, t['request_id'], operation, action['action_id'], decision)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(tr('处理失败：' + str(exc)))
            for task in t.get('tasks', []):
                if task.get('operation') != 'payment.prepare':
                    continue
                for fact in task.get('facts', []):
                    oid = fact.get('order_id')
                    if fact.get('error') != '用户取消，未办理' or any((a['task_id'] == task['task_id'] and a['order_id'] == oid for a in t.get('actions', []))):
                        continue
                    with slots[task['task_id'], oid], st.container(border=True, key=prefix + '_payment_cancelled_' + task['task_id'] + '_' + str(oid)):
                        st.markdown(tr('**订单 {0} · 金额 {1:.2f} · 支付**'.format(oid, float(fact['amount']))))
                        st.caption(tr('状态：已取消，未支付'))
            if t.get('tasks'):
                st.caption(tr('协作任务 · ' + '\u3000|\u3000'.join(('{0} {1}（{2}）'.format(task['task_id'], OPERATIONS.get(task['operation'], task['operation']), STATUSES.get(task['status'], task['status'])) for task in t['tasks']))))
                with st.expander(tr('任务进度')):
                    st.dataframe(table_rows([{'任务': x['task_id'], '操作': OPERATIONS.get(x['operation'], x['operation']), '订单': x['targets'], '状态': STATUSES.get(x['status'], x['status']), '依赖': ', '.join(x.get('depends_on', [])), '条件': str(x.get('conditions', []))} for x in t['tasks']]), hide_index=True)
            with st.expander(tr('工具执行轨迹')):
                for entry in t.get('trace', []):
                    st.caption(tr('任务 {0} · {1} · {2}'.format(entry.get('task_id', '—'), AGENTS.get(entry.get('agent'), entry.get('agent', '—')), entry.get('tool', '—'))))
                st.json(t.get('trace', []))
            if t.get('order_results'):
                with st.expander(tr('办理明细')):
                    st.dataframe(t['order_results'], hide_index=True)
    waiting = any((t.get('status') in {'processing', 'awaiting_confirmation', 'awaiting_selection', 'awaiting_payment_method'} for t in payload['turns']))
    prompt = st.chat_input('请先完成确认或取消' if waiting else '例如：查询订单 1001 的详情', max_chars=2000, disabled=waiting)
    if prompt:
        with st.chat_message('user'):
            st.write(prompt)
        with st.spinner(tr('正在处理')):
            try:
                svc.chat(user_id, session_id, prompt, request_id=uuid4().hex)
                st.rerun()
            except Exception as exc:
                st.error(tr('执行失败：' + str(exc)))
