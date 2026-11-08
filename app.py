from uuid import uuid4
from pathlib import Path
import streamlit as st
from src.service import SupportService
from src.payment_methods import PAYMENT_METHODS
from src.memory import MEMORY_OBSERVATIONS_REQUIRED

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

chat, memories, guide, knowledge_page = st.tabs(['客服对话', '记忆与客户偏好', '使用说明与多轮测试', '平台资料详情'])
with memories:
    st.subheader('短期记忆 → 长期记忆')
    st.caption('从已完成的用户对话提取回复习惯；同一偏好的证据按不同请求计数，刷新或重试不增加次数。')
    st.info(f'观察到的习惯累计 {MEMORY_OBSERVATIONS_REQUIRED} 次才转为长期记忆；“以后都……”“请记住……”等明确要求可直接保存。')
    memory_items = svc.store.memories(user_id)
    memory_labels = {'detail': '回答详略', 'format': '组织方式', 'language': '回复语言', 'tone': '语气风格'}
    for column, status, label in zip(st.columns(3), ['candidate', 'active', 'revoked'], ['待转换', '长期记忆', '已撤销']):
        column.metric(label, sum(item['status'] == status for item in memory_items))
    for status, title in [('candidate', '待转换的习惯'), ('active', '已保存的长期记忆'), ('revoked', '已撤销的记忆')]:
        st.subheader(title)
        items = [item for item in memory_items if item['status'] == status]
        if not items:
            st.caption({'candidate': '暂无待转换候选。普通聊天、仅本次要求和手动保存表单不会自动累加为习惯证据。',
                        'active': '暂无自动提取的长期记忆。', 'revoked': '暂无撤销记录。'}[status])
        for item in items:
            count = len(item['evidence'])
            with st.container(border=True):
                st.write(f"{memory_labels.get(item['key'], item['key'])}：{item['value']}")
                if status == 'candidate':
                    remaining = max(0, MEMORY_OBSERVATIONS_REQUIRED - count)
                    st.progress(min(count / MEMORY_OBSERVATIONS_REQUIRED, 1.0),
                        text=f'已出现 {count} 次 / {MEMORY_OBSERVATIONS_REQUIRED} 次 · 还差 {remaining} 次转为长期记忆')
                    st.caption('当前仍是候选，不作为已生效的长期回复偏好。')
                elif status == 'revoked':
                    st.caption('已停止使用；旧证据不会自动恢复这条记忆。')
                elif item['origin'] == 'explicit':
                    st.caption('用户明确要求，已直接保存，无需累计三次。')
                else:
                    st.caption(f'已累计 {count} 次证据，已转为长期记忆。')
                with st.expander(f'来源原话（{count} 条）'):
                    for index, evidence in enumerate(item['evidence'], 1):
                        st.caption(f"第 {index} 次 · {evidence['created_at']}")
                        st.code(evidence['quote'], language=None)
                if status != 'revoked' and st.button('忘记此偏好', key='forget_' + user_id + item['key']):
                    svc.store.forget_memory(user_id, item['key'])
                    st.rerun()
    with st.expander('转换规则与可复制的表达', expanded=True):
        st.write('观察路径：出现一次 → 待转换；出现两次 → 继续观察；累计三次 → 转为长期记忆。仅统计模型已提取并保存的同类、同值证据，不是把所有重复文字都计数。')
        st.caption('已有明确设置或已撤销的偏好，不会被观察记录自动覆盖；涉及业务确认的对话在本轮操作结束后才写入记忆。')
        st.caption('直接保存长期偏好：')
        st.code('以后请先给结论再解释，请记住这个回复偏好。', language=None)
        st.caption('只对当前回答生效，不应形成永久偏好：')
        st.code('这一次请详细解释，不要把它保存为长期习惯。', language=None)
        st.caption('撤销已经保存的偏好：')
        st.code('请忘记关于回答详细程度的长期偏好。', language=None)
    with st.expander('最近对话的提取与转换记录'):
        records = []
        for turn in reversed(payload['turns'][-6:]):
            memory_result = turn.get('memory') or {}
            candidates = memory_result.get('candidates') or []
            state_label = {'applied': '已检查（实际状态见上方）', 'pending': '待写入，可重试原请求'}.get(memory_result.get('status'),
                '本轮未完成，尚未写入' if turn.get('status') != 'completed' else '无可用提取记录')
            for candidate in candidates:
                records.append({'用户原话': candidate['evidence'], '偏好类别': memory_labels.get(candidate['key'], candidate['key']),
                    '提炼内容': candidate['value'], '识别方式': '明确要求' if candidate['origin'] == 'explicit' else '观察习惯',
                    '操作': '撤销记忆' if candidate['operation'] == 'forget' else '保存或累计证据', '处理结果': state_label})
            if not candidates:
                records.append({'用户原话': turn['user'], '提炼内容': '本轮没有长期记忆候选', '处理结果': state_label})
        if records:
            st.dataframe(records, hide_index=True)
        else:
            st.caption('当前会话暂无对话。转换记录来自实际请求，不展示虚构的候选。')
    st.subheader('手动保存的客户偏好')
    profile = svc.store.profile(user_id)
    with st.form('profile_' + user_id):
        tone = st.text_input('回复偏好', profile.get('tone', '简洁，先结论后解释'), max_chars=50, key='tone_' + user_id)
        contact = st.text_input('联系偏好', profile.get('contact_preference', '优先站内文字联系'), max_chars=100, key='contact_' + user_id)
        if st.form_submit_button('保存客户偏好'):
            svc.store.save_profile(user_id, tone, contact)
            st.success('偏好已保存，可跨会话读取。')
    with st.expander('跨会话档案原始数据'):
        st.json(svc.store.profile(user_id))
    st.write('当前会话最近讨论订单', payload['recent_order_ids'])
    st.write('保留的完整轮次数', len(payload['turns']))
    st.metric('当前会话累计模型 Token', payload['total_tokens'])
with guide:
    guide_text = (Path(__file__).parent / 'docs' / 'usage_and_scenarios.md').read_text(encoding='utf-8')
    guide_text = guide_text.replace('__PROJECT_ROOT__', str(Path(__file__).resolve().parent))
    guide_text = guide_text.replace('__DATA_ROOT__', str(Path(svc.store.path).resolve().parent))
    introduction, *sections = guide_text.split('\n## ')
    st.markdown(introduction)
    for section in sections:
        title, _, body = section.partition('\n')
        with st.expander(title):
            st.markdown(body)
with chat:
    for t in payload['turns']:
        with st.chat_message('user'):
            st.write(t['user'])
        with st.chat_message('assistant'):
            st.write(t['reply'])
            st.caption('协作处理：' + '、'.join(t['agents']) if t.get('agent') == 'collaboration' else '主对话层' if t['agent'] == 'assistant' else '处理专家：' + t['agent'])
            for source in t.get('sources') or []:
                if source.get('url'):
                    st.markdown(f"[{source['title']}]({source['url']})")
                else:
                    with st.expander(f"{source['title']} · {source.get('location','')} · {source.get('doc_id','')}"):
                        st.write(source.get('content',''))
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


with knowledge_page:
    st.caption('虚构平台资料；规则解释不等于本地工具已具备对应办理能力。')
    kind_label = st.radio('资料类型', ['全部','平台政策','商品资料'])
    query = st.text_input('按标题、商品名称或编号查找资料')
    category = st.text_input('分类筛选（完整分类名称，可留空）')
    page = int(st.number_input('页码',min_value=1,value=1,step=1))
    catalog = svc.knowledge.documents({'全部':'','平台政策':'policy','商品资料':'product'}[kind_label],category=category,q=query,offset=(page-1)*20)
    st.write(f"资料总数：{catalog['total']}；索引状态：{catalog['index_status']}")
    if catalog['items']:
        selected = st.selectbox('查看文档',catalog['items'],format_func=lambda r:f"{r['id']} · {r['title']}")
        doc = svc.knowledge.document(selected['id'])
        st.write('分类：',doc['category'],'版本：',doc['version'])
        for section in doc['sections']:
            st.markdown(section['text'])
    else:
        st.info('尚无匹配资料。请停止应用后通过目录导入命令导入知识库。')
