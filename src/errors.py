from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time

import httpx
from src.runtime_context import record_event


class ModelConfigurationError(ValueError):
    pass


class ModelOutputError(ValueError):
    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def classify_error(exc):
    from src.budget import ExecutionBudgetExceeded
    if isinstance(exc, ExecutionBudgetExceeded):
        return 'budget'
    if isinstance(exc, ModelOutputError):
        return 'model_output'
    code = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if code in (401, 403):
        return 'authentication'
    if code == 402:
        return 'balance'
    if code == 404 or isinstance(exc, (ModelConfigurationError, ImportError)):
        return 'configuration'
    if code == 429 or 'RESOURCE_EXHAUSTED' in str(exc):
        return 'rate_limit'
    if isinstance(code, int) and code >= 500:
        return 'service'
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)) or type(exc).__name__ in {
        'APITimeoutError', 'APIConnectionError', 'OpenAITimeoutError', 'OpenAIConnectionError',
        'ConnectError', 'ReadError', 'ReadTimeout',
    }:
        return 'network'
    if code in (400, 422) or isinstance(exc, (ValueError, TypeError)):
        return 'validation'
    return 'program'


def error_message(kind):
    return {'configuration': '模型配置不可用，请检查服务商、模型名称和依赖。',
            'authentication': '模型服务认证或访问权限不可用，请检查 API Key 和账户权限。',
            'balance': '模型账户余额不足，请充值后使用新的请求编号重试。',
            'rate_limit': '模型请求频率受限，请稍后使用新的请求编号重试。',
            'quota': '模型配额或请求频率受限，请稍后使用新的请求编号重试。',
            'service': '模型服务暂时异常，请稍后使用新的请求编号重试。',
            'network': '模型服务暂时不可达，请稍后使用新的请求编号重试。',
            'validation': '请求或模型工具参数无效，请检查后重新提交。',
            'model_output': '模型未返回有效的结构化结果，本次理解未完成。',
            'program': '处理过程出现内部错误，请根据请求编号检查日志。',
            'budget': '请求调用或耗时预算已用完；已保存结果保留，请核对尚未完成的任务。'}[kind]


def _retry_delay(exc, default):
    response = getattr(exc, 'response', None)
    value = response.headers.get('Retry-After') if response is not None else None
    if value is not None:
        try:
            seconds = float(value)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                seconds = (when - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return default
        if math.isfinite(seconds):
            return max(0, seconds)
    return default


def invoke_with_retry(fn, *, max_attempts=2, backoff_seconds=1):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
        raise ValueError('模型调用最多允许两次尝试。')
    if not math.isfinite(backoff_seconds) or backoff_seconds < 0:
        raise ValueError('重试等待时间必须是非负有限数。')
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            kind = classify_error(exc)
            record_event('model_call_failed', attempt=attempt, error_type=kind, exception_type=type(exc).__name__)
            if kind not in {'rate_limit', 'network', 'service'}:
                raise
            from src.budget import check_execution_budget
            remaining = check_execution_budget()
            if attempt == max_attempts:
                raise
            delay = _retry_delay(exc, backoff_seconds)
            if remaining is not None and remaining <= delay:
                from src.budget import ExecutionBudgetExceeded
                raise ExecutionBudgetExceeded('剩余耗时不足以重试，已停止继续调用。') from None
            record_event('model_retry_scheduled', attempt=attempt, delay_seconds=delay)
            # Long provider delays stay interruptible while we recheck the request deadline.
            while delay > 0:
                step = min(delay, 1)
                time.sleep(step)
                delay -= step
                check_execution_budget()
