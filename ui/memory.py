from ui.i18n import tr, table_rows, localize_format
import streamlit as st
from src.memory import MEMORY_OBSERVATIONS_REQUIRED

def render(svc, user_id, payload):
    st.subheader(tr('短期记忆 → 长期记忆'))
    st.caption(tr('从已完成的用户对话提取回复与购物偏好；同一偏好的证据按不同请求计数，刷新或重试不增加次数。'))
    st.info(tr('观察到的习惯累计 {0} 次才转为长期记忆；“以后都……”“请记住……”等明确要求可直接保存。'.format(MEMORY_OBSERVATIONS_REQUIRED)))
    memory_items = svc.store.memories(user_id)
    memory_labels = {'detail': '回答详略', 'format': '组织方式', 'language': '回复语言', 'tone': '语气风格', 'shopping_brand': '购物 · 品牌', 'shopping_category': '购物 · 品类', 'shopping_budget': '购物 · 预算'}
    for column, status, label in zip(st.columns(3), ['candidate', 'active', 'revoked'], ['待转换', '长期记忆', '已撤销']):
        column.metric(tr(label), sum((item['status'] == status for item in memory_items)))
    group = st.radio(tr('偏好分类'), ['全部', '回复偏好', '购物偏好'], key='memory_group', horizontal=True, format_func=localize_format())
    status_filter = st.radio(tr('记忆状态'), ['当前记忆', '已撤销'], key='memory_status', horizontal=True, format_func=localize_format())
    statuses = [('candidate', '待转换的习惯'), ('active', '已保存的长期记忆')] if status_filter == '当前记忆' else [('revoked', '已撤销的记忆')]
    for status, title in statuses:
        st.subheader(tr(title))
        items = [item for item in memory_items if item['status'] == status and (group == '全部' or item['key'].startswith('shopping_') == (group == '购物偏好'))]
        if not items:
            st.caption(tr({'candidate': '暂无待转换候选。普通聊天、仅本次要求和手动保存表单不会自动累加为习惯证据。', 'active': '暂无自动提取的长期记忆。', 'revoked': '暂无撤销记录。'}[status]))
        for item in items:
            count = len(item['evidence'])
            with st.container(border=True):
                st.write('{0}：{1}'.format(tr(memory_labels.get(item['key'], item['key'])), item['value']))
                if status == 'candidate':
                    remaining = max(0, MEMORY_OBSERVATIONS_REQUIRED - count)
                    st.progress(min(count / MEMORY_OBSERVATIONS_REQUIRED, 1.0), text=tr('已出现 {0} 次 / {1} 次 · 还差 {2} 次转为长期记忆'.format(count, MEMORY_OBSERVATIONS_REQUIRED, remaining)))
                    st.caption(tr('当前仍是候选，不作为已生效的长期偏好。'))
                elif status == 'revoked':
                    st.caption(tr('已停止使用；旧证据不会自动恢复这条记忆。'))
                elif item['origin'] == 'explicit':
                    st.caption(tr('用户明确要求，已直接保存，无需累计三次。'))
                else:
                    st.caption(tr('已累计 {0} 次证据，已转为长期记忆。'.format(count)))
                with st.expander(tr('来源原话（{0} 条）'.format(count))):
                    for index, evidence in enumerate(item['evidence'], 1):
                        st.caption(tr('第 {0} 次 · {1}'.format(index, evidence['created_at'])))
                        st.code(evidence['quote'], language=None)
                if status != 'revoked' and st.button(tr('忘记此偏好'), key='forget_' + user_id + item['key']):
                    svc.store.forget_memory(user_id, item['key'])
                    st.rerun()
    with st.expander(tr('转换规则与可复制的表达')):
        st.write(tr('观察路径：出现一次 → 待转换；出现两次 → 继续观察；累计三次 → 转为长期记忆。仅统计模型已提取并保存的同类、同值证据，不是把所有重复文字都计数。'))
        st.caption(tr('已有明确设置或已撤销的偏好，不会被观察记录自动覆盖；涉及业务确认的对话在本轮操作结束后才写入记忆。'))
        st.caption(tr('购物偏好生效后，需明确要求修改；临时购买需求不会自动覆盖。'))
        st.code(tr('请记住，我买耳机通常预算在500元以内。'), language=None)
        st.caption(tr('直接保存长期偏好：'))
        st.code(tr('以后请先给结论再解释，请记住这个回复偏好。'), language=None)
        st.caption(tr('只对当前回答生效，不应形成永久偏好：'))
        st.code(tr('这一次请详细解释，不要把它保存为长期习惯。'), language=None)
        st.caption(tr('撤销已经保存的偏好：'))
        st.code(tr('请忘记关于回答详细程度的长期偏好。'), language=None)
    with st.expander(tr('最近对话的提取与转换记录')):
        records = []
        for turn in reversed(payload['turns'][-6:]):
            memory_result = turn.get('memory') or {}
            candidates = memory_result.get('candidates') or []
            state_label = {'applied': '已检查（实际状态见上方）', 'pending': '待写入，可重试原请求'}.get(memory_result.get('status'), '本轮未完成，尚未写入' if turn.get('status') != 'completed' else '无可用提取记录')
            for candidate in candidates:
                records.append({'用户原话': candidate['evidence'], '偏好类别': memory_labels.get(candidate['key'], candidate['key']), '提炼内容': candidate['value'], '识别方式': '明确要求' if candidate['origin'] == 'explicit' else '观察习惯', '操作': '撤销记忆' if candidate['operation'] == 'forget' else '保存或累计证据', '处理结果': state_label})
            if not candidates:
                records.append({'用户原话': turn['user'], '提炼内容': '本轮没有长期记忆候选', '处理结果': state_label})
        if records:
            st.dataframe(table_rows(records), hide_index=True)
        else:
            st.caption(tr('当前会话暂无对话。转换记录来自实际请求，不展示虚构的候选。'))
    st.subheader(tr('手动保存的客户偏好'))
    profile = svc.store.profile(user_id)
    with st.form('profile_' + user_id):
        tone = st.text_input(tr('回复偏好'), profile.get('tone', '简洁，先结论后解释'), max_chars=50, key='tone_' + user_id)
        contact = st.text_input(tr('联系偏好'), profile.get('contact_preference', '优先站内文字联系'), max_chars=100, key='contact_' + user_id)
        if st.form_submit_button(tr('保存客户偏好')):
            svc.store.save_profile(user_id, tone, contact)
            st.success(tr('偏好已保存，可跨会话读取。'))
    with st.expander(tr('跨会话档案原始数据')):
        st.json(svc.store.profile(user_id))
