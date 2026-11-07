"""应用级会话与跨会话档案。不是 LangGraph checkpointer。"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class SupportStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, session_id))''')
            db.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)
            ''')

    def current_session(self, user_id, session_id=None):
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS current_sessions (user_id TEXT PRIMARY KEY, session_id TEXT NOT NULL)')
            row = db.execute('SELECT session_id FROM current_sessions WHERE user_id=?', (user_id,)).fetchone()
            value = session_id or (row[0] if row else 'case_1')
            db.execute('INSERT INTO current_sessions VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET session_id=excluded.session_id', (user_id, value))
        return value

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def load_session(self, user_id, session_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM sessions WHERE user_id=? AND session_id=?', (user_id, session_id)).fetchone()
        return json.loads(row[0]) if row else {'turns': [], 'last_order_id': None, 'total_tokens': 0}

    def save_session(self, user_id, session_id, value):
        payload = json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(user_id,session_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, session_id, payload, datetime.now(timezone.utc).isoformat()))

    def profile(self, user_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM profiles WHERE user_id=?', (user_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_profile(self, user_id, tone, contact):
        value = {'tone': tone[:50], 'contact_preference': contact[:100]}
        with self.connect() as db:
            db.execute('INSERT INTO profiles VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return value
