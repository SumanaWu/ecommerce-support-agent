from uuid import uuid4
import streamlit as st
from src.service import SupportService
from src.auth import AccessError, resolve_principal

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
    account_token = st.text_input('测试账户令牌', type='password', key='account_token')
    try:
        principal = resolve_principal(account_token)
    except AccessError as exc:
        st.info(str(exc))
        st.caption('在 .env 配置 SUPPORT_ACCOUNT_TOKENS 后重启服务。这里只验证本地账户，不是生产登录系统。')
        st.stop()
    user_id = principal.customer_id
    st.caption('当前客户：' + user_id)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = 'case_1'
    session_id = st.text_input('会话编号', key='session_id')
    if st.button('新建会话'):
        st.session_state.next_session = 'case_' + uuid4().hex[:8]
        st.rerun()
    st.caption('订单按当前账户校验归属。退款和支付需在待确认操作中点击确认。')

try:
    svc = service()
    payload = svc.snapshot(principal, session_id, mode)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

chat, memories, guide = st.tabs(['客服对话', '记忆与客户偏好', '样例场景'])
with memories:
    profile = svc.profile(principal)
    with st.form('profile'):
        tone = st.text_input('回复偏好', profile.get('tone', '简洁，先结论后解释'), max_chars=50)
        contact = st.text_input('联系偏好', profile.get('contact_preference', '优先站内文字联系'), max_chars=100)
        if st.form_submit_button('保存客户偏好'):
            svc.save_profile(principal, tone, contact)
            st.success('偏好已保存，可跨会话读取。在线模型会读取这些偏好，离线规则不模拟文风变化。')
    st.write('跨会话档案')
    st.json(svc.profile(principal))
    st.write('当前会话最后讨论订单', payload['last_order_id'])
    st.write('保留的完整轮次数', len(payload['turns']))
    st.metric('当前会话累计模型 Token', payload['total_tokens'])
    st.caption('离线模式没有模型调用；Token 为零不是在线成本测量结果。')
with guide:
    st.markdown('''1. 查询订单 1001 的详情。
2. 查询订单 1002 的物流。
3. 接着输入“帮我申请退款”，观察系统沿用 1002 并拒绝未送达退款。
4. 输入“帮我申请订单 1001 的退款”，检查待确认操作，再点击确认申请。
5. 输入“查询订单 1003 的支付状态”，再输入“帮我付款”，确认后才更新支付状态。
6. 输入“退货政策是什么”，或“我要投诉这个问题”，观察政策工具与本地工单。''')
    st.info('样例数据会被写操作改变。需要重新演示时，请停止服务后按 README 在独立目录初始化，不要在运行中重置。')
with chat:
    for run in svc.list_runs(principal, session_id, mode):
        with st.container(border=True):
            st.write('运行待恢复' if run['resumable'] else '运行结果待补写到会话')
            st.caption('请求编号：' + run['request_id'])
            st.caption('从已保存步骤继续，已提交动作会查重；恢复不会代替退款或付款确认。')
            if st.button('恢复执行' if run['resumable'] else '补写运行结果', key='resume_' + run['run_id']):
                try:
                    result = svc.resume_run(principal, run['run_id'])
                    if result.get('memory_saved'):
                        for key, pending in list(st.session_state.get('pending_requests', {}).items()):
                            if key == (user_id, mode, session_id) and pending['request_id'] == run['request_id']:
                                del st.session_state.pending_requests[key]
                    st.rerun()
                except AccessError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error('恢复尚未确认完成，请使用原运行重试。')
    if 'pending_decisions' not in st.session_state:
        st.session_state.pending_decisions = {}

    def decide(action_id, confirm):
        key = (user_id, mode, session_id, action_id)
        st.session_state.pending_decisions[key] = confirm
        try:
            result = svc.confirm_action(principal, action_id) if confirm else svc.cancel_action(principal, action_id)
            if result['status'] != 'in_progress' and result['memory_saved']:
                del st.session_state.pending_decisions[key]
            st.rerun()
        except AccessError:
            del st.session_state.pending_decisions[key]
            st.error('未找到可访问的动作，请刷新页面。')
        except Exception:
            st.error('动作结果尚未确认，可使用下方按钮重试；已完成的动作不会重复办理。')

    for key, confirm in list(st.session_state.pending_decisions.items()):
        if key[:3] == (user_id, mode, session_id):
            if st.button('重试动作结果', key='retry_' + key[3]):
                decide(key[3], confirm)

    for action in svc.list_pending_actions(principal, session_id, mode):
        with st.container(border=True):
            label = '退款申请' if action['tool'] == 'initiate_refund' else '付款'
            st.write(f"待确认：订单 {action['order_id']} · 本地{label}")
            st.write(f"本地样例金额：{action['amount']:.2f}")
            if 'method' in action['args']:
                st.write('支付方式：' + action['args']['method'])
            if 'reason' in action['args']:
                st.write('申请原因：' + action['args']['reason'])
            st.caption('当前尚未办理；确认时会重新检查订单归属和业务状态。')
            if st.button('确认申请' if label == '退款申请' else '确认付款', key='confirm_' + action['action_id']):
                decide(action['action_id'], True)
            if st.button('取消操作', key='cancel_' + action['action_id']):
                decide(action['action_id'], False)
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            st.caption('处理专家：' + t['agent'])
            with st.expander('工具执行轨迹'):
                st.json(t['trace'])
    if 'pending_requests' not in st.session_state:
        st.session_state.pending_requests = {}
    request_key = (user_id, mode, session_id)
    pending = st.session_state.pending_requests.get(request_key)
    retry = False
    if pending:
        st.info(pending.get('notice') or '上次提交尚未确认完成。点击重试会复用请求编号，查询已有结果并补写记忆；中断运行请使用上方“恢复执行”。')
        st.caption('请求编号：' + pending['request_id'])
        retry = st.button('重试同一请求')
    prompt = st.chat_input('例如：查询订单 1001 的详情', max_chars=2000, disabled=bool(pending))
    if prompt or retry:
        if prompt:
            pending = {'message': prompt, 'request_id': uuid4().hex}
            st.session_state.pending_requests[request_key] = pending
        with st.spinner('正在处理'):
            try:
                result = svc.chat(principal, session_id, mode, pending['message'], request_id=pending['request_id'])
                if result['status'] != 'in_progress' and result['memory_saved']:
                    del st.session_state.pending_requests[request_key]
                else:
                    pending['notice'] = result.get('memory_message') or result.get('reply')
                st.rerun()
            except Exception:
                st.error('本次提交尚未确认完成，请保留当前页面并重试同一请求。详细信息按请求编号检查日志。')
