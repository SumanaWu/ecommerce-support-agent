from ui.i18n import tr
from uuid import uuid4
import streamlit as st
from src.service import SupportService
from ui import chat as chat_ui, memory as memory_ui, guide as guide_ui, knowledge as knowledge_ui
st.set_page_config(page_title=tr('电商订单与售后助手'), layout='wide')
st.sidebar.radio('Language / 界面语言', ['中文', 'English'], key='ui_language')
st.markdown('<style>.block-container {max-width: 1200px; padding-top: 2rem;} [data-testid=stSidebar] {background: #f5f7fb;} h1,h2,h3 {letter-spacing: -0.02em;} [data-testid=stChatMessage] {border-radius: 14px;}</style>', unsafe_allow_html=True)
st.title(tr('电商订单与售后助手'))
st.caption(tr('订单与售后 · 多专家协作 · 客户记忆'))
st.info(tr('本地模拟环境：支付与退款仅修改样例数据。'))

@st.cache_resource
def service():
    return SupportService()
with st.sidebar:
    user_id = st.text_input(tr('客户编号'), 'customer_a', key='customer_id')
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
    session_id = st.text_input(tr('会话编号'), key=session_key)
    try:
        svc.validate(session_id)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    svc.store.current_session(user_id, session_id)
    if st.button(tr('新建会话'), key='new_' + user_id):
        st.session_state[next_key] = 'case_' + uuid4().hex[:8]
        st.rerun()
    st.caption(tr('当前为客户独立对话页面；仅可访问本人订单。客户编号用于本地演示身份切换，并非登录认证。'))
try:
    svc = service()
    payload = svc.snapshot(user_id, session_id)
except (ValueError, RuntimeError) as exc:
    st.error(str(exc))
    st.stop()
with st.sidebar.expander(tr('会话详情')):
    st.write(tr('最近讨论订单'), payload['recent_order_ids'])
    st.write(tr('展示轮次'), len(payload['turns']))
    st.metric(tr('累计模型 Token'), payload['total_tokens'])
chat, memories, knowledge_page, guide = st.tabs([tr(label) for label in ['客服对话', '记忆与客户偏好', '平台资料详情', '使用说明与多轮测试']])
with chat:
    chat_ui.render(svc, user_id, session_id, payload)
with memories:
    memory_ui.render(svc, user_id, payload)
with knowledge_page:
    knowledge_ui.render(svc)
with guide:
    guide_ui.render(svc)
