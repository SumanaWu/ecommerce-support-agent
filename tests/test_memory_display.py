from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_customer_refunds import svc
from tests.test_actions import business


def test_memory_page_shows_real_evidence_progress(svc, monkeypatch):
    for key, value, count, origin in [('detail', '详细', 1, 'observed'),
                                       ('format', '先结论', 2, 'observed'),
                                       ('language', '中文', 1, 'explicit')]:
        candidate = dict(key=key, value=value, origin=origin, operation='set', evidence=value)
        for index in range(count):
            svc.store.apply_memories('customer_a', f'{key}{index}', [candidate], f'2026-01-0{index+1}')
        svc.store.apply_memories('customer_a', f'{key}0', [candidate], '2026-01-01')
    svc.store.forget_memory('customer_a', 'tone')
    other = dict(key='detail', value='不要泄露其他客户', origin='observed', operation='set', evidence='其他客户的原话')
    svc.store.apply_memories('customer_b', 'foreign', [other], '2026-01-01')
    monkeypatch.setattr('src.service.SupportService', lambda: svc)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics['待转换'] == '2'
    assert metrics['长期记忆'] == '1'
    assert metrics['已撤销'] == '1'
    progress = [item.proto for item in app.get('progress')]
    assert {item.value for item in progress} == {33, 66}
    assert any('已出现 1 次 / 3 次' in item.text for item in progress)
    assert any('已出现 2 次 / 3 次' in item.text for item in progress)
    assert any('已直接保存' in c.value for c in app.caption)
    assert not any('不要泄露其他客户' in m.value for m in app.markdown)
    assert any('以后请先给结论' in c.value for c in app.code)

    next(b for b in app.button if b.key == 'forget_customer_aformat').click().run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}['待转换'] == '1'
    assert next(r for r in svc.store.memories('customer_a') if r['key'] == 'format')['status'] == 'revoked'


def test_third_distinct_request_promotes_candidate(svc):
    candidate = dict(key='detail', value='详细', origin='observed', operation='set', evidence='详细一点')
    for count in range(1, 4):
        svc.store.apply_memories('customer_a', str(count), [candidate], f'2026-01-0{count}')
        item = svc.store.memories('customer_a')[0]
        assert len(item['evidence']) == count
        assert item['status'] == ('candidate' if count < 3 else 'active')
