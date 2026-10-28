import time

import httpx
from src.runtime_context import record_event


class ModelConfigurationError(ValueError):
    pass


def classify_error(exc):
    from src.budget import ExecutionBudgetExceeded
    if isinstance(exc, ExecutionBudgetExceeded):
        return 'budget'
    code = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if code in (401, 403, 404) or isinstance(exc, (ModelConfigurationError, ImportError)):
        return 'configuration'
    if code == 429 or 'RESOURCE_EXHAUSTED' in str(exc):
        return 'quota'
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)) or type(exc).__name__ in {
        'APITimeoutError', 'APIConnectionError', 'ConnectError', 'ReadError', 'ReadTimeout',
    } or (isinstance(code, int) and code >= 500):
        return 'network'
    if code in (400, 422) or isinstance(exc, (ValueError, TypeError)):
        return 'validation'
    return 'program'


def error_message(kind):
    return {'configuration': '模型配置或认证不可用，请检查服务商、Key、模型名称和依赖。',
            'quota': '模型配额或请求频率受限，请稍后使用新的请求编号重试。',
            'network': '模型服务暂时不可达，请稍后使用新的请求编号重试。',
            'validation': '请求或模型工具参数无效，请检查后重新提交。',
            'program': '处理过程出现内部错误，请根据请求编号检查日志。',
            'budget': '请求调用或耗时预算已用完；已保存结果保留，请核对尚未完成的任务。'}[kind]


def invoke_with_retry(fn, *, max_attempts=2, backoff_seconds=0.25):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            kind = classify_error(exc)
            record_event('model_call_failed', attempt=attempt, error_type=kind, exception_type=type(exc).__name__)
            if kind not in {'quota', 'network'}:
                raise
            from src.budget import check_execution_budget
            remaining = check_execution_budget()
            if attempt == max_attempts:
                raise
            if remaining is not None and remaining <= backoff_seconds:
                from src.budget import ExecutionBudgetExceeded
                raise ExecutionBudgetExceeded('剩余耗时不足以重试，已停止继续调用。') from None
            time.sleep(backoff_seconds)


def call_with_retry(fn, *, max_attempts=2, backoff_seconds=0.25, escalate_reason, escalate_context):
    """Compatibility helper for callers; never wrap a business graph in this."""
    try:
        return invoke_with_retry(fn, max_attempts=max_attempts, backoff_seconds=backoff_seconds), None
    except Exception as exc:
        from src.tools import escalate_to_human
        return None, escalate_to_human(reason=escalate_reason, context=escalate_context + ' | ' + classify_error(exc))

