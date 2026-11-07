from uuid import uuid4
import json
from pathlib import Path
import streamlit as st
from src.service import SupportService
from src.payment_methods import PAYMENT_METHODS

st.set_page_config(page_title='电商订单与售后助手', layout='wide')
st.title('电商订单与售后助手')
st.caption('LangGraph 专家路由 · 订单与售后工具 · 持久化 Memory')
st.warning('全部订单和资金状态均为本地样例。办理支付、退款会修改样例数据库，不连接真实电商或资金渠道。')

@st.cache_resource
def service():
    return SupportService()


with st.sidebar:
    user_id = st.text_input('客户编号', 'customer_a')
    svc = service()
    try:
        svc.validate(user_id)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    session_key = 'session_' + user_id
    st.session_state.setdefault(session_key, svc.store.current_session(user_id))
    next_key = 'next_' + user_id
    if next_key in st.session_state:
        st.session_state[session_key] = st.session_state.pop(next_key)
    session_id = st.text_input('会话编号', key=session_key)
    try:
        svc.validate(session_id)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    svc.store.current_session(user_id, session_id)
    if st.button('新建会话', key='new_' + user_id):
        st.session_state[next_key] = 'case_' + uuid4().hex[:8]
        st.rerun()
    st.caption('当前为客户独立对话页面；仅可访问本人订单。客户编号用于本地演示身份切换，并非登录认证。')

try:
    svc = service()
    payload = svc.snapshot(user_id, session_id)
except (ValueError, RuntimeError) as exc:
    st.error(str(exc))
    st.stop()

chat, memories, guide = st.tabs(['客服对话', '记忆与客户偏好', '样例场景'])
with memories:
    profile = svc.store.profile(user_id)
    with st.form('profile_' + user_id):
        tone = st.text_input('回复偏好', profile.get('tone', '简洁，先结论后解释'), max_chars=50, key='tone_' + user_id)
        contact = st.text_input('联系偏好', profile.get('contact_preference', '优先站内文字联系'), max_chars=100, key='contact_' + user_id)
        if st.form_submit_button('保存客户偏好'):
            svc.store.save_profile(user_id, tone, contact)
            st.success('偏好已保存，可跨会话读取。')
    st.write('跨会话档案')
    st.json(svc.store.profile(user_id))
    st.write('当前会话最近讨论订单', payload['recent_order_ids'])
    st.write('保留的完整轮次数', len(payload['turns']))
    st.metric('当前会话累计模型 Token', payload['total_tokens'])
with guide:
    st.info('默认使用customer_a和初始样例。不同组建议新建会话，同组依次输入；新建会话不会重置订单状态。资金操作先选择、再逐项按钮确认。')
    scenarios = json.loads((Path(__file__).parent / 'eval' / 'scenarios.json').read_text())
    for scenario in scenarios:
        if scenario['id'] > 16:
            continue
        with st.expander(f"{scenario['id']}. {scenario['title']}"):
            for message in scenario['messages']:
                st.code(message, language=None)
            st.write('预期表现：' + scenario['expected'])
    st.info('样例数据会被写操作改变。重新检查前请停止服务并手动删除全部数据库；正常重启和客户切换会保留状态。')
with chat:
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            st.caption('协作处理：' + '、'.join(t['agents']) if t.get('agent') == 'collaboration' else '主对话层' if t['agent'] == 'assistant' else '处理专家：' + t['agent'])
            for source in t.get('sources') or []:
                st.markdown(f"[{source['title']}]({source['url']})")
            if t.get('status') == 'processing' and st.button('继续处理', key='retry_' + t['request_id']):
                try:
                    svc.chat(user_id, session_id, t['user'], request_id=t['request_id'])
                    st.rerun()
                except Exception as exc:
                    st.error('处理失败：' + str(exc))
            prefix = f"{user_id}_{session_id}_{t['request_id']}"
            if t.get('status') == 'awaiting_selection':
                selection = t['selection']
                prefix += '_' + selection.get('input_id', 'orders')
                all_orders = st.checkbox('查看我的全部订单', key=prefix + '_all')
                candidates = sorted(selection['candidates'], key=lambda order: (not order.get('eligible', False), order['order_id']))
                visible_orders = candidates if all_orders else candidates[:3]
                drafts = st.session_state.setdefault('draft_' + prefix, {})
                eligible_ids = {order['order_id'] for order in candidates if order.get('eligible')}
                for oid in list(drafts):
                    if oid not in eligible_ids:
                        drafts.pop(oid)
                for order in visible_orders:
                    oid = order['order_id']
                    label = f"订单 {oid} · {order['item']} · {order['amount']} · {order['status']}"
                    if order.get('eligible'):
                        key = prefix + '_order_' + str(oid)
                        drafts[oid] = st.toggle(label, value=drafts.get(oid, False), key=key)
                    else:
                        st.write(label)
                    code = order.get('reason_code')
                    if code in {'ELIGIBLE', 'REFUND_IN_PROGRESS'}:
                        color = 'green'
                    elif code in {'ORDER_NOT_ACCESSIBLE', 'UNPAID', 'NO_REFUNDABLE_BALANCE', 'STATUS_NOT_ELIGIBLE'}:
                        color = 'red'
                    else:
                        color = 'blue'
                    st.caption(f":{color}[{order.get('reason', '')}]")
                selected = [oid for oid, chosen in drafts.items() if chosen]
                st.caption(f'已选 {len(selected)} 笔')
                for decision, label in [('submit', '准备所选订单退款'), ('cancel', '取消选单')]:
                    if st.button(label, key=prefix + decision, disabled=decision == 'submit' and not selected):
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
                ids = list(dict.fromkeys(
                    ([*task.get('targets', [])] if task.get('status') == 'pending' else [])
                    + [a['order_id'] for a in t.get('actions', []) if a['task_id'] == tid]
                    + [o['order_id'] for entry in t.get('pending_inputs', []) if entry.get('task_id') == tid for o in entry.get('orders', [])]
                    + [f['order_id'] for f in task.get('facts', []) if f.get('order_id') and f.get('error') == '用户取消，未办理']
                ))
                for oid in ids:
                    slots[(tid, oid)] = st.container()
                    if task.get('status') == 'pending':
                        with slots[(tid, oid)], st.container(border=True):
                            kind = '支付' if task['operation'] == 'payment.prepare' else '退款申请'
                            st.markdown(f'**订单 {oid} · {kind}**')
                            st.caption('状态：等待前一项处理，尚未核实')
                if not ids and task.get('status') == 'pending':
                    with st.container(border=True):
                        kind = '支付' if task['operation'] == 'payment.prepare' else '退款申请'
                        st.markdown(f'**{kind}**')
                        st.caption('状态：等待前置任务确定订单和处理条件')
            for entry in t.get('pending_inputs', []):
                if entry['type'] != 'method':
                    continue
                with slots[(entry['task_id'], entry['orders'][0]['order_id'])], st.container(border=True, key=prefix + '_payment_' + str(entry['orders'][0]['order_id'])):
                    order = entry['orders'][0]
                    st.markdown(f"**订单 {order['order_id']} · 支付**")
                    st.write(f"待支付金额：{order['amount']}")
                    key = prefix + '_' + entry['input_id']
                    method = st.selectbox('支付方式', list(PAYMENT_METHODS), index=None,
                        format_func=PAYMENT_METHODS.get, placeholder='请选择支付方式', key=key)
                    st.caption('选择方式仅准备方案；点击确认付款后才执行。')
                    if method is not None:
                        operation = st.session_state.setdefault(key + '_submit_' + method, uuid4().hex)
                        try:
                            svc.select_payment_method(user_id, session_id, t['request_id'], operation, 'submit', method, entry['input_id'])
                            st.rerun()
                        except Exception as exc:
                            st.error('处理失败：' + str(exc))
                    st.button('确认付款', key=key + '_confirm', disabled=True)
                    if st.button('取消支付', key=key + '_cancel'):
                        operation = st.session_state.setdefault(key + '_cancel_request', uuid4().hex)
                        svc.select_payment_method(user_id, session_id, t['request_id'], operation, 'cancel', input_id=entry['input_id'])
                        st.rerun()
            current = t.get('current_action')
            for action in t.get('actions', []):
                with slots[(action['task_id'], action['order_id'])], st.container(border=True, key=prefix + '_' + action['type'] + '_' + str(action['order_id'])):
                    st.markdown(f"**订单 {action['order_id']} · 金额 {action['amount']} · "
                             f"{'退款申请' if action['type'] == 'refund' else '支付'}**")
                    st.caption('状态：' + {'pending': '待确认，尚未执行', 'completed': '已完成', 'cancelled': '已取消', 'rejected': '未办理'}[action['status']])
                    if action['type'] == 'payment':
                        method = action['parameters']['method']
                        st.write('支付方式：' + PAYMENT_METHODS.get(method, method))
                    if action['status'] == 'pending' and not current:
                        st.caption('请先完成其他必要选择，再按顺序确认。')
                    elif action['status'] == 'pending' and current['action_id'] != action['action_id']:
                        st.caption('等待前一项操作确认或取消。')
                    if current and action['action_id'] == current['action_id']:
                        for decision, label in [('confirm', '确认付款' if action['type'] == 'payment' else '确认办理'), ('cancel', '取消本项')]:
                            key = f"{user_id}_{session_id}_{action['action_id']}_{decision}"
                            if st.button(label, key=key):
                                operation = st.session_state.setdefault('op_' + key, uuid4().hex)
                                try:
                                    svc.resume(user_id, session_id, t['request_id'], operation,
                                               action['action_id'], decision)
                                    st.rerun()
                                except Exception as exc:
                                    st.error('处理失败：' + str(exc))
            for task in t.get('tasks', []):
                if task.get('operation') != 'payment.prepare':
                    continue
                for fact in task.get('facts', []):
                    oid = fact.get('order_id')
                    if fact.get('error') != '用户取消，未办理' or any(
                        a['task_id'] == task['task_id'] and a['order_id'] == oid for a in t.get('actions', [])
                    ):
                        continue
                    with slots[(task['task_id'], oid)], st.container(border=True, key=prefix + '_payment_cancelled_' + task['task_id'] + '_' + str(oid)):
                        st.markdown(f"**订单 {oid} · 金额 {float(fact['amount']):.2f} · 支付**")
                        st.caption('状态：已取消，未支付')
            if t.get('tasks'):
                with st.expander('任务进度'):
                    st.dataframe([{'任务': x['task_id'], '操作': x['operation'], '订单': x['targets'], '状态': x['status']} for x in t['tasks']], hide_index=True)
            with st.expander('工具执行轨迹'):
                st.json(t.get('trace', []))
            if t.get('order_results'):
                with st.expander('办理明细'):
                    st.dataframe(t['order_results'], hide_index=True)
    waiting = any(t.get('status') in {'processing', 'awaiting_confirmation', 'awaiting_selection', 'awaiting_payment_method'} for t in payload['turns'])
    prompt = st.chat_input('请先完成确认或取消' if waiting else '例如：查询订单 1001 的详情',
                           max_chars=2000, disabled=waiting)
    if prompt:
        with st.chat_message('user'):
            st.write(prompt)
        with st.spinner('正在处理'):
            try:
                svc.chat(user_id, session_id, prompt, request_id=uuid4().hex)
                st.rerun()
            except Exception as exc:
                st.error('执行失败：' + str(exc))
