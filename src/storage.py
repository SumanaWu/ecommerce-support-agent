"""应用级会话与跨会话档案。不是 LangGraph checkpointer。"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from src.memory import MEMORY_OBSERVATIONS_REQUIRED


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

            db.execute("CREATE TABLE IF NOT EXISTS memories (user_id TEXT, key TEXT, value TEXT, origin TEXT, status TEXT, updated_at TEXT, PRIMARY KEY(user_id,key))")
            db.execute("CREATE TABLE IF NOT EXISTS memory_evidence (user_id TEXT, request_id TEXT, key TEXT, value TEXT, quote TEXT, created_at TEXT, PRIMARY KEY(user_id,request_id,key))")

    def memories(self, user_id):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute('SELECT * FROM memories WHERE user_id=?', (user_id,))]
            for row in rows:
                row['evidence'] = [dict(r) for r in db.execute('SELECT request_id,quote,created_at FROM memory_evidence WHERE user_id=? AND key=? AND value=? ORDER BY created_at', (user_id, row['key'], row['value']))]
        return rows

    def forget_memory(self, user_id, key):
        from src.memory import MEMORY_KEYS
        if key not in MEMORY_KEYS:
            raise ValueError('不支持的偏好类别')
        with self.connect() as db:
            db.execute("INSERT INTO memories VALUES (?,?,?,'explicit','revoked',?) ON CONFLICT(user_id,key) DO UPDATE SET status='revoked',updated_at=excluded.updated_at", (user_id,key,'',datetime.now(timezone.utc).isoformat()))

    def apply_memories(self, user_id, request_id, candidates, created_at):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for item in candidates:
                key, value = item['key'], item['value']
                inserted = db.execute('INSERT OR IGNORE INTO memory_evidence VALUES (?,?,?,?,?,?)', (user_id, request_id, key, value, item['evidence'], created_at)).rowcount
                if not inserted:
                    continue
                old = db.execute('SELECT value,origin,status,updated_at FROM memories WHERE user_id=? AND key=?', (user_id,key)).fetchone()
                if old and (old[3] > created_at or (item['origin'] == 'observed' and (old[1] == 'explicit' or old[2] == 'revoked'))):
                    continue
                count = db.execute('SELECT count(*) FROM memory_evidence WHERE user_id=? AND key=? AND value=?', (user_id,key,value)).fetchone()[0]
                if old and old[2] == 'active' and item['origin'] == 'observed' and count < MEMORY_OBSERVATIONS_REQUIRED:
                    continue
                status = 'revoked' if item['operation'] == 'forget' else 'active' if item['origin'] == 'explicit' or count >= MEMORY_OBSERVATIONS_REQUIRED else 'candidate'
                db.execute('INSERT INTO memories VALUES (?,?,?,?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value,origin=excluded.origin,status=excluded.status,updated_at=excluded.updated_at', (user_id,key,value,item['origin'],status,created_at))

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
            row = db.execute('SELECT payload,updated_at FROM profiles WHERE user_id=?', (user_id,)).fetchone()
        value = json.loads(row[0]) if row else {}
        memories = self.memories(user_id)
        learned = [r for r in memories if r['status'] == 'active']
        candidates = [r for r in memories if r['status'] == 'candidate']
        if candidates:
            value['candidate_preferences'] = candidates
        if row:
            value['manual_updated_at'] = row[1]
        if learned:
            value['learned_preferences'] = learned
        return value

    def save_profile(self, user_id, tone, contact):
        value = {'tone': tone[:50], 'contact_preference': contact[:100]}
        with self.connect() as db:
            db.execute('INSERT INTO profiles VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at', (user_id, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return value
