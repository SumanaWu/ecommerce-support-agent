"""Local request locks and business-transaction action receipts.

The OS releases a request lock when a worker exits. Reclaiming that lock permits
result recovery; continuing an interrupted graph requires explicit resume.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from sqlalchemy import select
from src.backend.db import ActionRecord, RequestRecord
from src.runtime_context import record_event


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


class RequestConflict(ValueError):
    pass


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    mode: str
    request_id: str
    session_id: str

    @property
    def key(self):
        return self.user_id, self.mode, self.request_id


class RequestJournal:
    def __init__(self, session_factory, engine):
        self.sessions = session_factory
        database = engine.url.database
        if database and database != ':memory:':
            self.database_key = str(Path(database).resolve())
            self.lock_dir = Path(database).resolve().parent / '.request-locks'
        else:
            self.database_key = str(id(engine))
            self.lock_dir = Path(tempfile.gettempdir()) / f'support-agent-locks-{os.getpid()}-{id(engine)}'

    @contextmanager
    def request_guard(self, user_id, mode, request_id):
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(canonical([self.database_key, user_id, mode, request_id]).encode()).hexdigest()
        # Keep lock files: unlinking a held file would allow a second lock inode.
        with (self.lock_dir / name).open('a+b') as lock:
            if os.name == 'nt':
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b'\0')
                    lock.flush()
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    yield False
                    return
            else:
                import fcntl
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    yield False
                    return
            try:
                yield True
            finally:
                if os.name == 'nt':
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def begin_request(self, user_id, mode, request_id, session_id, message, *, create=True):
        fingerprint = hashlib.sha256(canonical([session_id, message]).encode()).hexdigest()
        with self.sessions.begin() as db:
            row = db.get(RequestRecord, (user_id, mode, request_id))
            if row is not None:
                if row.fingerprint != fingerprint:
                    raise RequestConflict('同一 request_id 的消息和会话参数必须保持一致。')
                context = RequestContext(user_id, mode, request_id, row.session_id)
                return context, json.loads(row.result) if row.result else None, False
            context = RequestContext(user_id, mode, request_id, session_id or uuid4().hex)
            if create:
                db.add(RequestRecord(user_id=user_id, mode=mode, request_id=request_id,
                                     fingerprint=fingerprint, session_id=context.session_id,
                                     message=message))
            return context, None, True

    def finish_request(self, context, result):
        with self.sessions.begin() as db:
            row = db.get(RequestRecord, context.key)
            row.result = canonical(result)
        record_event('request_finished', status=result['status'])

    def request_input(self, user_id, mode, request_id):
        from src.auth import AccessError
        with self.sessions() as db:
            row = db.get(RequestRecord, (user_id, mode, request_id))
            if row is None:
                raise AccessError('未找到可访问的运行。', 404)
            auto_session = hashlib.sha256(canonical([None, row.message]).encode()).hexdigest() == row.fingerprint
            return {'session_id': row.session_id, 'requested_session': None if auto_session else row.session_id,
                    'message': row.message, 'result': json.loads(row.result) if row.result else None}

    def get_action_result(self, context, action_key, args):
        with self.sessions() as db:
            row = db.get(ActionRecord, (*context.key, action_key))
            if row is None:
                return None
            if row.args != canonical(args):
                raise RequestConflict('同一动作编号不能更换参数。')
            return json.loads(row.result)

    def action_trace(self, context):
        with self.sessions() as db:
            rows = db.scalars(select(ActionRecord).where(
                ActionRecord.user_id == context.user_id, ActionRecord.mode == context.mode,
                ActionRecord.request_id == context.request_id).order_by(ActionRecord.created_at)).all()
            agents = {'initiate_refund': 'refund', 'process_payment': 'payment',
                      'escalate_to_human': 'escalate', 'cancel_action': 'system'}
            return [{'agent': agents[row.tool], 'tool': row.tool, 'action_id': row.action_key,
                     'args': json.loads(row.args), 'result': json.loads(row.result)} for row in rows]

    def execute_action_once(self, context, tool, args, apply):
        if context is None:
            with self.sessions.begin() as db:
                return apply(db)
        action_key = tool + ':' + str(args.get('order_id', 'ticket'))
        previous = self.get_action_result(context, action_key, args)
        if previous is not None:
            record_event('action_replayed', action_id=action_key)
            return previous
        with self.sessions.begin() as db:
            receipt = ActionRecord(user_id=context.user_id, mode=context.mode,
                                   request_id=context.request_id, action_key=action_key,
                                   tool=tool, args=canonical(args), result='null', created_at=time.time())
            db.add(receipt)
            db.flush()  # Reserve before writing; result and business commit together.
            result = apply(db)
            receipt.result = canonical(result)
        record_event('action_committed', action_id=action_key)
        return result
