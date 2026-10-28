"""Request-wide, durable attempt limits; checkpoint recovery never resets counters."""
import os
import time

from src.backend.db import ExecutionBudgetRecord
from src.runtime_context import active_budget, active_mode, record_event


class ExecutionBudgetExceeded(RuntimeError):
    pass


def _setting(name, default, maximum, *, integer=False):
    raw = os.environ.get(name) or str(default)
    try:
        value = int(raw) if integer else float(raw)
        if not 1 <= value <= maximum:
            raise ValueError()
        return value
    except ValueError:
        raise ValueError(f'{name} 必须在 1 至 {maximum} 之间。') from None


class RequestBudget:
    def __init__(self, context, session_factory):
        self.context, self.sessions = context, session_factory
        with self.sessions.begin() as db:
            db.connection().exec_driver_sql('BEGIN IMMEDIATE')
            if db.get(ExecutionBudgetRecord, context.key) is None:
                seconds = _setting('SUPPORT_REQUEST_TIMEOUT_SECONDS', 120, 3600)
                started = time.time()
                db.add(ExecutionBudgetRecord(user_id=context.user_id, mode=context.mode, request_id=context.request_id,
                    max_model_calls=_setting('SUPPORT_REQUEST_MAX_MODEL_CALLS', 24, 256, integer=True),
                    max_tool_calls=_setting('SUPPORT_REQUEST_MAX_TOOL_CALLS', 24, 256, integer=True),
                    started_at=started, deadline_at=started + seconds))

    def check(self, kind=None, *, consume=False):
        with self.sessions.begin() as db:
            db.connection().exec_driver_sql('BEGIN IMMEDIATE')
            row = db.get(ExecutionBudgetRecord, self.context.key)
            remaining = row.deadline_at - time.time()
            if remaining <= 0:
                raise ExecutionBudgetExceeded('请求耗时预算已用完；已保存的业务结果仍然有效。')
            if kind in {'model', 'tool'}:
                field, maximum = kind + '_calls', getattr(row, 'max_' + kind + '_calls')
                if getattr(row, field) >= maximum:
                    raise ExecutionBudgetExceeded(f"请求的{'模型' if kind == 'model' else '工具'}调用预算已用完；未继续执行剩余任务。")
                if consume:
                    setattr(row, field, getattr(row, field) + 1)
            return remaining

    def snapshot(self):
        with self.sessions() as db:
            row = db.get(ExecutionBudgetRecord, self.context.key)
            return {'model_calls': row.model_calls, 'tool_calls': row.tool_calls,
                    'real_model_calls': row.model_calls if self.context.mode == 'online' else 0,
                    'max_model_calls': row.max_model_calls, 'max_tool_calls': row.max_tool_calls,
                    'timeout_seconds': row.deadline_at - row.started_at,
                    'elapsed_seconds': round(max(0, time.time() - row.started_at), 3),
                    'remaining_seconds': round(max(0, row.deadline_at - time.time()), 3)}


def check_execution_budget(kind=None, *, consume=False):
    budget = active_budget.get()
    return budget.check(kind, consume=consume) if budget else None


def invoke_model(model, messages):
    # A pool counts each actual client attempt itself; mocks/demo count here.
    if not getattr(model, '_counts_attempts', False):
        check_execution_budget('model', consume=True)
        record_event('model_call_started', simulated=active_mode.get() == 'demo')
    return model.invoke(messages)
