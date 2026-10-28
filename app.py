from uuid import uuid4
import streamlit as st
from src.service import SupportService
from src.auth import AccessError, resolve_principal
from src.context import context_budget

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
    details = svc.profile_details(principal)
    profile = details['values']
    profile_key = f"{user_id}_{details['version']}"
    with st.form('profile_' + profile_key):
        tone = st.text_input('回复偏好', profile.get('tone', '简洁，先结论后解释'), max_chars=50, key='tone_' + profile_key)
        contact = st.text_input('联系偏好', profile.get('contact_preference', '优先站内文字联系'), max_chars=100, key='contact_' + profile_key)
        if st.form_submit_button('保存客户偏好'):
            try:
                svc.update_profile(principal, {'tone': tone, 'contact_preference': contact}, details['version'])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.write('跨会话档案')
    st.json(profile)
    st.caption('偏好版本：' + str(details['version']) + '。仅保存你明确提交的内容，聊天推测不会写入档案。')
    sources = {'user_form': '页面明确保存', 'user_api': 'API 明确保存', 'legacy_user_saved': '旧版用户保存'}
    for field, label in [('tone', '回复偏好'), ('contact_preference', '联系偏好')]:
        meta = details['fields'].get(field)
        if meta:
            st.caption(f"{label}：{'已删除' if meta.get('deleted') else sources.get(meta['source'], meta['source'])}；更新时间 {meta['updated_at']}；版本 {meta['version']}")
        if field in profile and st.button('删除' + label, key='delete_' + field + '_' + profile_key):
            try:
                svc.delete_profile_field(principal, field, details['version'])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.caption('删除偏好只影响后续模型输入；清理聊天、待确认操作和业务审计分别处理。')
    issue = payload['issue']
    st.write('候选订单', issue['candidate_order_ids'])
    st.write('当前明确选择的订单', issue['active_order_id'] or '尚未明确选择')
    issue_key = f"{user_id}_{mode}_{session_id}_{payload['version']}"
    with st.form('order_' + issue_key):
        current_order = st.text_input('更正当前订单编号', value=str(issue['active_order_id'] or ''), key='current_order_' + issue_key)
        if st.form_submit_button('设置当前订单'):
            try:
                if not current_order.isascii() or not current_order.isdigit():
                    raise ValueError('请填写有效的数字订单编号。')
                svc.set_active_order(principal, session_id, mode, int(current_order), payload['version'])
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.caption('设置当前订单只更正指代，不代表确认付款或退款；已有待确认动作请单独核对或取消。')
    if st.button('清除当前订单与候选', key='clear_order_' + issue_key):
        try:
            svc.set_active_order(principal, session_id, mode, None, payload['version'])
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.button('清除当前会话聊天', key='clear_history_' + issue_key):
        try:
            svc.clear_history(principal, session_id, mode, payload['version'])
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.caption('清除聊天保留当前事项、累计用量、请求去重和业务记录；已有待确认动作需要单独取消。')
    st.write('保留的完整轮次数', len(payload['turns']))
    st.metric('当前会话累计模型 Token', payload['total_tokens'])
    st.caption('离线模式没有模型调用；Token 为零不是在线成本测量结果。')
    st.caption(f'模型输入上下文预算：{context_budget()} Token，使用 UTF-8 字节估算，不是模型服务的精确计费值。')
    if payload['turns'] and payload['turns'][-1].get('context_info'):
        st.json(payload['turns'][-1]['context_info'])
    if payload['turns'] and payload['turns'][-1].get('execution_budget'):
        with st.expander('最近一轮的执行预算'):
            st.json(payload['turns'][-1]['execution_budget'])
            st.caption('demo 的模型调用数是规则模拟次数，real_model_calls 为零。预算包含重试；重放结果不会新增调用。')
with guide:
    st.markdown('''1. 查询订单 1001 的详情。
2. 查询订单 1002 的物流。
3. 接着输入“帮我申请退款”，观察系统沿用 1002 并拒绝未送达退款。
4. 输入“帮我申请订单 1001 的退款”，检查待确认操作，再点击确认申请。
5. 输入“查询订单 1003 的支付状态”，再输入“帮我付款”，确认后才更新支付状态。
6. 输入“退货政策是什么”，或“我要投诉这个问题”，观察政策工具与本地工单。''')
    st.write('多任务示例：“查订单1001的物流，再看能否退款”；或“帮我申请订单1001的退款，并帮我付款订单1003”。后一种会产生两个需要分别确认的动作。')
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

    def render_pending_action(action):
        with st.container(border=True):
            label = '退款申请' if action['tool'] == 'initiate_refund' else '付款'
            st.write(f"待确认：订单 {action['order_id']} · 本地{label}")
            st.write(f"本地样例金额：{action['amount_display']} {action['currency']}")
            if 'method' in action['args']:
                st.write('支付方式：' + action['args']['method'])
            if 'reason' in action['args']:
                st.write('申请原因：' + action['args']['reason'])
            st.caption('当前尚未办理；确认时会重新检查订单归属和业务状态。')
            if st.button('确认申请' if label == '退款申请' else '确认付款', key='confirm_' + action['action_id']):
                decide(action['action_id'], True)
            if st.button('取消操作', key='cancel_' + action['action_id']):
                decide(action['action_id'], False)

    pending_actions = {action['action_id']: action for action in svc.list_pending_actions(principal, session_id, mode)}
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            for entry in t.get('trace', []):
                result = entry.get('result')
                if isinstance(result, dict):
                    action = pending_actions.pop(result.get('action_id'), None)
                    if action:
                        render_pending_action(action)
            st.caption('处理专家：' + t['agent'])
            if len(t.get('tasks', [])) > 1:
                labels = {'completed': '完成', 'pending_confirmation': '等待独立确认', 'rejected': '业务拒绝',
                          'partial': '部分完成', 'failed': '失败', 'needs_clarification': '需要澄清',
                          'not_started': '尚未执行', 'blocked': '前置事项未解决', 'cancelled': '已取消'}
                with st.expander('逐项任务结果', expanded=True):
                    for index, task in enumerate(t['tasks'], 1):
                        st.write(f"{index}. {task['message']}：{labels.get(task['status'], task['status'])}")
                    st.caption('这里记录本次请求执行时的结果。动作的当前状态请查看确认卡片和后续确认回复。')
            reasons = {'budget_exhausted': '本轮调用或耗时预算已用完，剩余任务尚未完成。',
                       'round_limit': '专家达到工具轮数上限，请核对部分结果。',
                       'repeated_tool_call': '专家重复请求相同工具，已停止继续调用。',
                       'batch_limit': '单轮工具数量超过限制，未执行该批请求。',
                       'missing_action': '本项只完成查询，尚未创建你请求的待确认操作。'}
            if t.get('stop_reason') in reasons:
                st.warning(reasons[t['stop_reason']])
            for entry in t.get('trace', []):
                if entry['tool'] == 'answer_faq' and isinstance(entry.get('result'), dict):
                    with st.expander('政策来源与版本'):
                        for source in entry['result'].get('sources', []):
                            st.text(f"{source['title']} · {source['source']} · {source['section']}\n版本 {source['version']} · 生效 {source['effective_from']}")
                            st.text(source['excerpt'])
            with st.expander('工具执行轨迹'):
                st.json(t['trace'])

    # Keep actions accessible if their chat turn was not saved or has aged out.
    for action in pending_actions.values():
        render_pending_action(action)
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
