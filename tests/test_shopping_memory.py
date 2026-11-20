from src.memory import memory_candidates
from src.storage import SupportStore


def candidate(key, value, origin='observed', operation='set'):
    return dict(key=key, value=value, origin=origin, operation=operation, evidence=value)


def test_shopping_candidates_and_validation():
    items = [candidate('shopping_brand', '品牌A'), candidate('shopping_category', '耳机'), candidate('shopping_budget', '耳机500元以内')]
    assert memory_candidates(items, '品牌A 耳机 耳机500元以内') == items
    assert not memory_candidates([items[0]], '品牌B')
    assert not memory_candidates([candidate('shopping_brand', '品牌A', operation='forget')], '品牌A')


def test_shopping_lifecycle_and_customer_isolation(tmp_path):
    store = SupportStore(tmp_path / 'memory.sqlite')
    original = candidate('shopping_budget', '耳机500元以内')
    for i in range(1, 4):
        store.apply_memories('a', str(i), [original], f'2026-01-0{i}')
        assert store.memories('a')[0]['status'] == ('active' if i == 3 else 'candidate')
    store.apply_memories('a', '3', [original], '2026-01-03')
    assert len(store.memories('a')[0]['evidence']) == 3
    changed = candidate('shopping_budget', '耳机800元以内')
    for i in range(4, 7):
        store.apply_memories('a', str(i), [changed], f'2026-01-0{i}')
    assert store.memories('a')[0]['value'] == original['value']
    store.apply_memories('a', '7', [candidate('shopping_budget', '耳机800元以内', 'explicit')], '2026-01-07')
    assert store.profile('a')['learned_preferences'][0]['value'] == changed['value']
    assert not store.memories('b')
    store.forget_memory('a', 'shopping_budget')
    store.apply_memories('a', '8', [changed], '2099-01-01')
    assert store.memories('a')[0]['status'] == 'revoked'
    store.apply_memories('a', '9', [candidate('shopping_budget', '耳机1000元以内', 'explicit')], '2099-01-02')
    assert store.memories('a')[0]['status'] == 'active'
