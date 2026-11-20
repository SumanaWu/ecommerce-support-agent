from pathlib import Path
import streamlit as st


def render(svc):
    filename = 'usage_and_scenarios.en.md' if st.session_state.get('ui_language') == 'English' else 'usage_and_scenarios.md'
    guide_text = (Path(__file__).resolve().parent.parent / 'docs' / filename).read_text(encoding='utf-8')
    guide_text = guide_text.replace('__PROJECT_ROOT__', str(Path(__file__).resolve().parent.parent))
    guide_text = guide_text.replace('__DATA_ROOT__', str(Path(svc.store.path).resolve().parent))
    introduction, *sections = guide_text.split('\n## ')
    st.markdown(introduction)
    for section in sections:
        title, _, body = section.partition('\n')
        with st.expander(title):
            st.markdown(body)
