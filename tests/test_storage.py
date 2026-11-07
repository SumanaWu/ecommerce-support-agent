from src.storage import SupportStore


def test_new_databases_are_independent(tmp_path):
    first = SupportStore(tmp_path / 'first' / 'memory.sqlite')
    second = SupportStore(tmp_path / 'second' / 'memory.sqlite')
    first.save_session('a', 'one', {'turns': [{'user': 'hello'}]})
    assert second.load_session('a', 'one')['turns'] == []
