import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_customer_refunds import svc
from tests.test_actions import business


def test_catalog_filters_pagination_and_clear(svc, monkeypatch):
    with svc.knowledge.connect() as db:
        for number in range(25):
            item = {'id': f'P{number:02}', 'title': f'商品{number:02}', 'kind': 'product',
                    'category': '耳机', 'version': '1', 'sections': [{'text': f'正文{number:02}'}]}
            db.execute('INSERT INTO documents VALUES (?,?)', (item['id'], json.dumps(item, ensure_ascii=False)))
        item = {'id': 'Q01', 'title': '退款政策', 'kind': 'policy', 'category': '退款',
                'version': '1', 'sections': [{'text': '政策正文'}]}
        db.execute('INSERT INTO documents VALUES (?,?)', (item['id'], json.dumps(item, ensure_ascii=False)))
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    assert any('共 26 份 · 第 1 页' in c.value for c in app.caption)
    next(b for b in app.button if b.label == '下一页').click().run()
    assert not app.exception
    assert any('第 2 页' in c.value for c in app.caption)
    next(t for t in app.text_input if t.key == 'knowledge_query').set_value('退款').run()
    assert not app.exception
    assert any('共 1 份 · 第 1 页' in c.value for c in app.caption)
    assert any('政策正文' in m.value for m in app.markdown)
    next(t for t in app.text_input if t.key == 'knowledge_query').set_value('无结果').run()
    assert any('没有符合筛选条件' in i.value for i in app.info)
    next(b for b in app.button if b.label == '清除筛选').click().run()
    assert not app.exception
    assert next(t for t in app.text_input if t.key == 'knowledge_query').value == ''
    assert any('共 26 份 · 第 1 页' in c.value for c in app.caption)
