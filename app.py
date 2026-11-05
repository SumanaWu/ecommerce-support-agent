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

if 'next_session' in st.session_state:
    st.session_state.session_id = st.session_state.pop('next_session')
with st.sidebar:
    mode = st.selectbox('运行模式', ['demo', 'online'])
    user_id = st.text_input('客户编号', 'customer_a')
    if 'session_id' not in st.session_state:
        st.session_state.session_id = 'case_1'
    session_id = st.text_input('会话编号', key='session_id')
    if st.button('新建会话'):
        st.session_state.next_session = 'case_' + uuid4().hex[:8]
        st.rerun()
    st.caption('客户编号用于会话和偏好隔离，并非登录认证；订单工具尚未实现客户归属鉴权。')

try:
    svc = service()
    payload = svc.snapshot(user_id, session_id, mode)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

chat, memories, guide = st.tabs(['客服对话', '记忆与客户偏好', '样例场景'])
with memories:
    profile = svc.store.profile(user_id)
    with st.form('profile'):
        tone = st.text_input('回复偏好', profile.get('tone', '简洁，先结论后解释'), max_chars=50)
        contact = st.text_input('联系偏好', profile.get('contact_preference', '优先站内文字联系'), max_chars=100)
        if st.form_submit_button('保存客户偏好'):
            svc.store.save_profile(user_id, tone, contact)
            st.success('偏好已保存，可跨会话读取。在线模型会读取这些偏好，离线规则不模拟文风变化。')
    st.write('跨会话档案')
    st.json(svc.store.profile(user_id))
    st.write('当前会话最后讨论订单', payload['last_order_id'])
    st.write('保留的完整轮次数', len(payload['turns']))
    st.metric('当前会话累计模型 Token', payload['total_tokens'])
    st.caption('离线模式没有模型调用；Token 为零不是在线成本测量结果。')
with guide:
    st.markdown('''1. 查询订单 1001 的详情。
2. 查询订单 1002 的物流。
3. 接着输入“帮我申请退款”，观察系统沿用 1002 并拒绝未送达退款。
4. 输入“帮我申请订单 1001 的退款”，观察本地退款状态变成 requested。
5. 输入“查询订单 1003 的支付状态”，再输入“帮我付款”。
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
                st.json(t['trace'])
    prompt = st.chat_input('例如：查询订单 1001 的详情', max_chars=2000)
    if prompt:
        with st.spinner('正在处理'):
            try:
                svc.chat(user_id, session_id, mode, prompt)
                st.rerun()
            except Exception as exc:
                st.error('执行失败：' + str(exc))
