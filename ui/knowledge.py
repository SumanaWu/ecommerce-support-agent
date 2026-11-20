from ui.i18n import tr, localize_format
import streamlit as st

def clear_filters():
    st.session_state['knowledge_query'] = ''
    for kind in ('all', 'policy', 'product'):
        st.session_state['knowledge_category_' + kind] = '全部'
    st.session_state['knowledge_page'] = 1

def render(svc):
    st.subheader(tr('平台资料'))
    st.caption(tr('虚构平台资料；规则解释不等于本地工具已具备对应办理能力。'))
    kind_label = st.radio(tr('资料类型'), ['全部', '平台政策', '商品资料'], key='knowledge_kind', horizontal=True, format_func=localize_format())
    kind = {'全部': '', '平台政策': 'policy', '商品资料': 'product'}[kind_label]
    directory, content = st.columns([1, 2], gap='large')
    with directory:
        query = st.text_input(tr('按标题、商品名称或编号查找资料'), key='knowledge_query')
        categories = svc.knowledge.categories(kind)
        category_key = 'knowledge_category_' + (kind or 'all')
        if st.session_state.get(category_key) not in [None, '全部', *categories]:
            st.session_state[category_key] = '全部'
        category = st.selectbox(tr('分类筛选'), ['全部'] + categories, key=category_key, format_func=localize_format()) if categories else '全部'
        st.button(tr('清除筛选'), on_click=clear_filters)
        filters = (kind, category, query)
        if st.session_state.get('knowledge_filters') != filters:
            st.session_state['knowledge_filters'] = filters
            st.session_state['knowledge_page'] = 1
        page = st.session_state.setdefault('knowledge_page', 1)
        catalog = svc.knowledge.documents(kind, category='' if category == '全部' else category, q=query, offset=(page - 1) * 20)
        st.caption(tr('共 {0} 份 · 第 {1} 页 · 索引状态：{2}'.format(catalog['total'], page, catalog['index_status'])))
        previous, next_page = st.columns(2)
        if previous.button(tr('上一页'), disabled=page <= 1):
            st.session_state['knowledge_page'] = page - 1
            st.rerun()
        if next_page.button(tr('下一页'), disabled=page * 20 >= catalog['total']):
            st.session_state['knowledge_page'] = page + 1
            st.rerun()
        if catalog['items']:
            selected = st.selectbox(tr('查看文档'), catalog['items'], format_func=localize_format(lambda r: '{0} · {1}'.format(r['id'], r['title'])), key='knowledge_document_{0}_{1}_{2}_{3}'.format(kind, category, query, page))
        else:
            selected = None
    with content:
        if selected:
            doc = svc.knowledge.document(selected['id'])
            st.markdown('### {0}'.format(doc['title']))
            st.caption(tr('分类：{0} · 版本：{1}'.format(doc['category'], doc['version'])))
            for section in doc['sections']:
                st.markdown(section['text'])
        elif not svc.knowledge.documents(limit=1)['total']:
            st.info(tr('知识库为空，请停止应用后通过目录导入命令导入资料。'))
        else:
            st.info(tr('没有符合筛选条件的资料，请调整或清除筛选。'))
