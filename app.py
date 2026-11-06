from uuid import uuid4
import streamlit as st
from src.service import SupportService

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
    st.markdown('''1. 查询订单 1001 的详情。
2. 查询订单 1002 的物流。
3. 接着输入“帮我申请退款”，展示本人订单，可退款优先，默认前三笔；不可退款订单没有选择开关。
4. 输入“帮我申请订单 1001 的退款”，检查方案后选择确认或取消。
5. 输入“查询订单 1003 的支付状态”，再输入“帮我付款”，系统会准备方案。
6. 输入“退货政策是什么”，或“我要投诉这个问题”，观察政策工具与本地工单。''')
    st.info('样例数据会被写操作改变。需要重新演示时，请停止服务后按 README 在独立目录初始化，不要在运行中重置。')
with chat:
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            st.caption('处理专家：' + t['agent'])
            with st.expander('工具执行轨迹'):
                st.json(t.get('trace', []))
            if t.get('status') == 'processing' and st.button('继续处理', key='retry_' + t['request_id']):
                try:
                    svc.chat(user_id, session_id, t['user'], request_id=t['request_id'])
                    st.rerun()
                except Exception as exc:
                    st.error('处理失败：' + str(exc))
            prefix = f"{user_id}_{session_id}_{t['request_id']}"
            if t.get('order_results'):
                st.dataframe(t['order_results'], hide_index=True)
            if t.get('status') == 'awaiting_selection':
                selection = t['selection']
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
                            svc.select_orders(user_id, session_id, t['request_id'], operation, decision, selected)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
            current = t.get('current_action')
            for action in t.get('actions', []):
                st.write(f"订单 {action['order_id']} · 金额 {action['amount']} · "
                         f"{'退款申请' if action['type'] == 'refund' else '支付'}")
                st.caption('状态：' + action['status'])
                if current and action['action_id'] == current['action_id']:
                    for decision, label in [('confirm', '确认办理'), ('cancel', '取消本项')]:
                        key = f"{user_id}_{session_id}_{action['action_id']}_{decision}"
                        if st.button(label, key=key):
                            operation = st.session_state.setdefault('op_' + key, uuid4().hex)
                            try:
                                svc.resume(user_id, session_id, t['request_id'], operation,
                                           action['action_id'], decision)
                                st.rerun()
                            except Exception as exc:
                                st.error('处理失败：' + str(exc))
    waiting = any(t.get('status') in {'processing', 'awaiting_confirmation', 'awaiting_selection'} for t in payload['turns'])
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
