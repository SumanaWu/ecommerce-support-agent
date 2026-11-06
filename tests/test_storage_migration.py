import sqlite3

import pytest

from src.storage import SupportStore


def old_database(path, *, nonempty=False):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE sessions (user_id TEXT NOT NULL, mode TEXT NOT NULL, '
                   'session_id TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL, '
                   'PRIMARY KEY(user_id, mode, session_id))')
        db.execute('CREATE TABLE profiles (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, '
                   'updated_at TEXT NOT NULL)')
        db.execute("INSERT INTO profiles VALUES('a', '{}', 'now')")
        if nonempty:
            db.execute("INSERT INTO sessions VALUES('a', 'online', 'one', '{}', 'now')")


def test_empty_old_sessions_are_rebuilt_without_losing_profiles(tmp_path):
    path = tmp_path / 'memory.sqlite'
    old_database(path)
    store = SupportStore(path)
    assert store.profile('a') == {}
    with sqlite3.connect(path) as db:
        assert [r[1] for r in db.execute('PRAGMA table_info(sessions)')] == [
            'user_id', 'session_id', 'payload', 'updated_at']
        assert db.execute('SELECT count(*) FROM profiles').fetchone()[0] == 1


def test_nonempty_old_sessions_are_not_changed(tmp_path):
    path = tmp_path / 'memory.sqlite'
    old_database(path, nonempty=True)
    with pytest.raises(RuntimeError, match='旧会话表非空'):
        SupportStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT mode FROM sessions').fetchone()[0] == 'online'


def test_new_databases_are_independent(tmp_path):
    first = SupportStore(tmp_path / 'first' / 'memory.sqlite')
    second = SupportStore(tmp_path / 'second' / 'memory.sqlite')
    first.save_session('a', 'one', {'turns': [{'user': 'hello'}]})
    assert second.load_session('a', 'one')['turns'] == []
