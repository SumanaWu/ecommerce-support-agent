"""Local retries for technical failures; never create business escalations."""
import logging

logger = logging.getLogger('support_agent.errors')


def call_with_retry(fn, *, max_attempts=2):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            logger.warning('Attempt %d/%d failed: %s', attempt + 1, max_attempts, type(exc).__name__)
            if attempt + 1 == max_attempts:
                raise RuntimeError('本次处理暂未完成，请稍后重试。') from exc
