"""应用级会话与跨会话档案。不是 LangGraph checkpointer。"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from src.memory import empty_issue, session_issue


class SessionConflict(ValueError):
    def __init__(self, expected, current):
        super().__init__('会话已被其他请求更新；请重读后重试同一请求以补写结果，不会重新执行业务。')
        self.expected = expected
        self.current = current


class ProfileConflict(ValueError):
    pass


class SupportStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Back up an existing v1 memory database before adding version metadata.
        with self.connect() as db:
            columns = {row[1] for row in db.execute('PRAGMA table_info(sessions)')}
            if columns and 'version' not in columns:
                with sqlite3.connect(self.path + '.before-v2-' + uuid4().hex + '.bak') as backup:
                    db.backup(backup)
            if columns and 'version' in columns and not db.execute('SELECT 1 FROM schema_migrations WHERE version=3').fetchone():
                with sqlite3.connect(self.path + '.before-v3-' + uuid4().hex + '.bak') as backup:
                    db.backup(backup)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT NOT NULL, mode TEXT NOT NULL, session_id TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, mode, session_id));
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS session_requests (
                user_id TEXT NOT NULL, mode TEXT NOT NULL, session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                PRIMARY KEY(user_id, mode, session_id, request_id));
            CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO schema_migrations VALUES(1);
            ''')
            db.execute('BEGIN IMMEDIATE')
            columns = {row[1] for row in db.execute('PRAGMA table_info(sessions)')}
            if 'version' not in columns:
                db.execute('ALTER TABLE sessions ADD COLUMN version INTEGER NOT NULL DEFAULT 0')
            db.execute('''CREATE TABLE IF NOT EXISTS graph_runs (
                run_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, mode TEXT NOT NULL,
                session_id TEXT NOT NULL, request_id TEXT NOT NULL,
                session_version INTEGER NOT NULL, input TEXT NOT NULL,
                UNIQUE(user_id,mode,request_id))''')
            db.execute('INSERT OR IGNORE INTO schema_migrations VALUES(2)')
            if not db.execute('SELECT 1 FROM schema_migrations WHERE version=3').fetchone():
                columns = {row[1] for row in db.execute('PRAGMA table_info(profiles)')}
                if 'version' not in columns:
                    db.execute('ALTER TABLE profiles ADD COLUMN version INTEGER NOT NULL DEFAULT 0')
                if 'metadata' not in columns:
                    db.execute("ALTER TABLE profiles ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
                for user, mode, session, raw in db.execute('SELECT user_id,mode,session_id,payload FROM sessions').fetchall():
                    payload = json.loads(raw)
                    payload.setdefault('issue', session_issue(payload))
                    db.execute('UPDATE sessions SET payload=? WHERE user_id=? AND mode=? AND session_id=?',
                               (json.dumps(payload, ensure_ascii=False), user, mode, session))
                for user, raw, updated in db.execute('SELECT user_id,payload,updated_at FROM profiles').fetchall():
                    metadata = {key: {'source': 'legacy_user_saved', 'updated_at': updated, 'version': 0}
                                for key in json.loads(raw)}
                    db.execute('UPDATE profiles SET metadata=? WHERE user_id=?', (json.dumps(metadata), user))
                db.execute('INSERT INTO schema_migrations VALUES(3)')

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def load_session(self, user_id, mode, session_id):
        with self.connect() as db:
            return self._load_session(db, user_id, mode, session_id)

    def _load_session(self, db, user_id, mode, session_id):
        row = db.execute('SELECT payload,version FROM sessions WHERE user_id=? AND mode=? AND session_id=?', (user_id, mode, session_id)).fetchone()
        payload = json.loads(row[0]) if row else {'turns': [], 'last_order_id': None, 'total_tokens': 0}
        return {**payload, 'issue': session_issue(payload), 'version': row[1] if row else 0}

    def save_session(self, user_id, mode, session_id, value):
        return self.save_session_if_version(user_id, mode, session_id, value, value.get('version', 0))

    def save_session_if_version(self, user_id, mode, session_id, value, expected_version):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = self._load_session(db, user_id, mode, session_id)['version']
            if current != expected_version:
                raise SessionConflict(expected_version, current)
            self._write_session(db, user_id, mode, session_id, value)
            return current + 1

    def _write_session(self, db, user_id, mode, session_id, value):
        payload = json.dumps({k: v for k, v in value.items() if k != 'version'}, ensure_ascii=False)
        db.execute('''INSERT INTO sessions(user_id,mode,session_id,payload,updated_at,version)
                      VALUES(?,?,?,?,?,1) ON CONFLICT(user_id,mode,session_id) DO UPDATE SET
                      payload=excluded.payload,updated_at=excluded.updated_at,version=sessions.version+1''',
                   (user_id, mode, session_id, payload, datetime.now(timezone.utc).isoformat()))

    def append_turn_once(self, user_id, mode, session_id, request_id, message, result, *, expected_version, preserve_context=False):
        from src.memory import ConversationMemory
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            key = (user_id, mode, session_id, request_id)
            if db.execute('SELECT 1 FROM session_requests WHERE user_id=? AND mode=? AND session_id=? AND request_id=?', key).fetchone():
                return False
            payload = self._load_session(db, *key[:3])
            if payload['version'] != expected_version:
                raise SessionConflict(expected_version, payload['version'])
            memory = ConversationMemory(last_order_id=payload['last_order_id'])
            memory.update_last_order_id(result.get('trace', []))
            turn = {'request_id': request_id, 'user': message, 'reply': result['reply'],
                    'agent': result['agent'], 'trace': result.get('trace', []),
                    'cached': result.get('cached', False), 'status': result['status'],
                    'reply_source': result['reply_source'], 'context_info': result.get('context_info', {})}
            turn.update(tasks=result.get('tasks', []), stop_reason=result.get('stop_reason'),
                        execution_budget=result.get('execution_budget', {}), clarification_question=result.get('clarification_question'),
                        planning_usage=result.get('planning_usage', {}))
            payload['turns'] = (payload['turns'] + [turn])[-50:]
            if not preserve_context:
                reference = result.get('order_reference')
                if reference:
                    payload['issue'] = reference['issue']
                    payload['last_order_id'] = reference.get('order_id') or reference['issue']['active_order_id']
                elif not request_id.startswith(('confirm_', 'cancel_')):
                    payload['last_order_id'] = memory.last_order_id
            payload['total_tokens'] += (result.get('usage') or {}).get('total_tokens') or 0
            self._write_session(db, user_id, mode, session_id, payload)
            db.execute('INSERT INTO session_requests VALUES(?,?,?,?)', key)
            return True

    def set_active_order(self, user_id, mode, session_id, order_id, expected_version):
        payload = self.load_session(user_id, mode, session_id)
        if payload['version'] != expected_version:
            raise SessionConflict(expected_version, payload['version'])
        issue = empty_issue()
        issue['revision'] = payload['issue']['revision'] + 1
        if order_id is not None:
            issue.update(candidate_order_ids=list(dict.fromkeys(payload['issue']['candidate_order_ids'] + [order_id])),
                         active_order_id=order_id, selection_source='user_selection',
                         updated_at=datetime.now(timezone.utc).isoformat())
        else:
            issue.update(selection_source='user_selection', updated_at=datetime.now(timezone.utc).isoformat())
        payload.update(issue=issue, last_order_id=order_id)
        self.save_session_if_version(user_id, mode, session_id, payload, expected_version)
        return self.load_session(user_id, mode, session_id)

    def clear_history(self, user_id, mode, session_id, expected_version):
        payload = self.load_session(user_id, mode, session_id)
        if payload['version'] != expected_version:
            raise SessionConflict(expected_version, payload['version'])
        payload['turns'] = []
        self.save_session_if_version(user_id, mode, session_id, payload, expected_version)
        return self.load_session(user_id, mode, session_id)

    def create_run(self, context, session_version, value):
        with self.connect() as db:
            db.execute('INSERT INTO graph_runs VALUES(?,?,?,?,?,?,?)',
                       (uuid4().hex, context.user_id, context.mode, context.session_id,
                        context.request_id, session_version, json.dumps(value, ensure_ascii=False)))
        return self.find_run(context)

    def find_run(self, context):
        with self.connect() as db:
            row = db.execute('SELECT * FROM graph_runs WHERE user_id=? AND mode=? AND request_id=?', context.key).fetchone()
        return self._run_view(row) if row else None

    def get_run(self, user_id, run_id):
        from src.auth import AccessError
        with self.connect() as db:
            row = db.execute('SELECT * FROM graph_runs WHERE user_id=? AND run_id=?', (user_id, run_id)).fetchone()
        if row is None:
            raise AccessError('未找到可访问的运行。', 404)
        return self._run_view(row)

    def list_runs(self, user_id, mode, session_id):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM graph_runs WHERE user_id=? AND mode=? AND session_id=? ORDER BY rowid DESC',
                              (user_id, mode, session_id)).fetchall()
        return [self._run_view(row) for row in rows]

    def _run_view(self, row):
        keys = ('run_id', 'user_id', 'mode', 'session_id', 'request_id', 'session_version', 'input')
        value = dict(zip(keys, row))
        value['input'] = json.loads(value['input'])
        value['checkpoint_path'] = str(Path(self.path).with_suffix('.checkpoints.sqlite'))
        return value

    def profile(self, user_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM profiles WHERE user_id=?', (user_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def profile_details(self, user_id):
        with self.connect() as db:
            row = db.execute('SELECT payload,metadata,version FROM profiles WHERE user_id=?', (user_id,)).fetchone()
        return {'values': json.loads(row[0]), 'fields': json.loads(row[1]), 'version': row[2]} if row else {
            'values': {}, 'fields': {}, 'version': 0}

    def has_turn(self, user_id, mode, session_id, request_id):
        with self.connect() as db:
            return db.execute('SELECT 1 FROM session_requests WHERE user_id=? AND mode=? AND session_id=? AND request_id=?',
                              (user_id, mode, session_id, request_id)).fetchone() is not None

    def update_profile(self, user_id, fields, expected_version, *, source='user_form', delete=None):
        limits = {'tone': 50, 'contact_preference': 100}
        if (set(fields) - limits.keys() or delete is not None and delete not in limits or
                any(not isinstance(value, str) or len(value) > limits[key] for key, value in fields.items())):
            raise ValueError('仅支持回复偏好（最多 50 字）与联系偏好（最多 100 字）。')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload,metadata,version FROM profiles WHERE user_id=?', (user_id,)).fetchone()
            value, metadata, version = (json.loads(row[0]), json.loads(row[1]), row[2]) if row else ({}, {}, 0)
            if version != expected_version:
                raise ProfileConflict('客户偏好已被其他请求更新，请刷新后重试。')
            now = datetime.now(timezone.utc).isoformat()
            value.update(fields)
            for field in list(fields) + ([delete] if delete else []):
                metadata[field] = {'source': source, 'updated_at': now, 'version': version + 1, 'deleted': field == delete}
            if delete:
                value.pop(delete, None)
            db.execute('''INSERT INTO profiles(user_id,payload,updated_at,metadata,version) VALUES(?,?,?,?,?)
                          ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at,
                          metadata=excluded.metadata,version=excluded.version''',
                       (user_id, json.dumps(value, ensure_ascii=False), now, json.dumps(metadata), version + 1))
        return self.profile_details(user_id)

    def delete_profile_field(self, user_id, field, expected_version, *, source='user_form'):
        return self.update_profile(user_id, {}, expected_version, source=source, delete=field)

    def save_profile(self, user_id, tone, contact):
        return self.update_profile(user_id, {'tone': tone[:50], 'contact_preference': contact[:100]},
                                   self.profile_details(user_id)['version'])['values']
