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
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT NOT NULL, mode TEXT NOT NULL, session_id TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, mode, session_id));
            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def load_session(self, user_id, mode, session_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM sessions WHERE user_id=? AND mode=? AND session_id=?', (user_id, mode, session_id)).fetchone()
        return json.loads(row[0]) if row else {'turns': [], 'last_order_id': None, 'total_tokens': 0}

    def save_session(self, user_id, mode, session_id, value):
        payload = json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?,?) ON CONFLICT(user_id,mode,session_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, mode, session_id, payload, datetime.now(timezone.utc).isoformat()))

    def profile(self, user_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM profiles WHERE user_id=?', (user_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_profile(self, user_id, tone, contact):
        value = {'tone': tone[:50], 'contact_preference': contact[:100]}
        with self.connect() as db:
            db.execute('INSERT INTO profiles VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return value
