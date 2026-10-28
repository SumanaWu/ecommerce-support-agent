import json
import os

from dotenv import load_dotenv
from urllib.parse import urlsplit
from src.errors import ModelConfigurationError, classify_error
from src.runtime_context import active_mode
from src.budget import check_execution_budget
from src.runtime_context import record_event

load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Sticky index into the model pool, shared across all get_llm() calls in this
# process so a working model keeps getting used instead of re-trying exhausted
# ones from the start every turn.
_rotation_state = {"index": 0}


def _load_gemini_keys() -> list[str]:
    multi = os.environ.get("GEMINI_API_KEYS")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GEMINI_API_KEY")
    if single:
        return [single]
    return []


def _is_quota_error(exc: Exception) -> bool:
    return classify_error(exc) == 'quota'


class RotatingChatModel:
    """Wraps multiple chat model clients (across API keys and/or providers)
    and rotates to the next one when the current one hits a quota/rate-limit error."""

    _counts_attempts = True

    def __init__(self, models: list, state: dict, labels=None):
        self._models = models
        self._state = state
        self._labels = labels or ['configured-model'] * len(models)

    def bind_tools(self, tools):
        return RotatingChatModel([m.bind_tools(tools) for m in self._models], state=self._state, labels=self._labels)

    def invoke(self, messages):
        n = len(self._models)
        start = self._state['index']
        last_exc: Exception | None = None
        for offset in range(n):
            idx = (start + offset) % n
            remaining = check_execution_budget('model', consume=True)
            record_event('model_call_started', model=self._labels[idx], pool_offset=offset, simulated=False)
            try:
                options = {'timeout': min(60, remaining)} if remaining is not None else {}
                result = self._models[idx].invoke(messages, **options)
                self._state["index"] = idx
                return result
            except Exception as exc:  # noqa: BLE001 - re-raised below if not a quota error
                last_exc = exc
                record_event('model_attempt_failed', model=self._labels[idx], error_type=classify_error(exc), exception_type=type(exc).__name__)
                if not _is_quota_error(exc):
                    raise
        raise last_exc


def create_chat_model(model: str | None = None, temperature: float = 0.0):
    if active_mode.get() == "demo":
        from src.demo_model import DemoChatModel
        return DemoChatModel()

    provider = os.environ.get('LLM_PROVIDER', 'auto').strip().lower()
    if provider not in {'auto', 'zhipu', 'gemini', 'groq'}:
        raise ModelConfigurationError('LLM_PROVIDER 必须为 auto、zhipu、gemini 或 groq。')
    if provider == 'auto':
        provider = ('zhipu' if os.environ.get('ZHIPU_API_KEY', '').strip() else
                    'gemini' if _load_gemini_keys() else 'groq')
    try:
        fallbacks = json.loads(os.environ.get('LLM_FALLBACKS') or '[]')
        if not isinstance(fallbacks, list) or len(fallbacks) > 8 or any(
                not isinstance(item, dict) or set(item) != {'provider', 'model'} or
                item['provider'] not in {'zhipu', 'gemini', 'groq'} or
                not isinstance(item['model'], str) or not item['model'].strip() for item in fallbacks):
            raise ValueError()
    except (ValueError, TypeError):
        raise ModelConfigurationError('LLM_FALLBACKS 应为最多八项的 provider/model JSON 列表；密钥仍从各服务环境变量读取。') from None
    specs = [{'provider': provider, 'model': model}, *fallbacks]
    models, labels = [], []
    for spec in specs:
        for client in _provider_clients(spec['provider'], spec['model'], temperature):
            models.append(client)
            labels.append(spec['provider'] + ':' + str(spec['model'] or _default_model(spec['provider'])))
    return RotatingChatModel(models, state=_rotation_state, labels=labels)


def _default_model(provider):
    return os.environ.get({'zhipu': 'ZHIPU_MODEL', 'gemini': 'GEMINI_MODEL', 'groq': 'GROQ_MODEL'}[provider]) or {
        'zhipu': 'glm-4.7-flash', 'gemini': DEFAULT_MODEL, 'groq': GROQ_MODEL}[provider]


def _provider_clients(provider, model, temperature):
    if provider == 'zhipu':
        zhipu_key = os.environ.get('ZHIPU_API_KEY', '').strip()
        if not zhipu_key:
            raise ModelConfigurationError('请配置 ZHIPU_API_KEY。')
        from langchain_openai import ChatOpenAI

        base_url = os.environ.get('ZHIPU_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/').strip().rstrip('/') + '/'
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment or parsed.path.rstrip('/').endswith('/chat/completions'):
            raise ModelConfigurationError('ZHIPU_BASE_URL 应为基础 URL，不能附加 chat/completions。')
        zhipu_model = ChatOpenAI(
            model=model or _default_model(provider),
            api_key=zhipu_key,
            base_url=base_url,
            temperature=1.0,
            max_retries=0,
            timeout=60,
            use_responses_api=False,
            disabled_params={'parallel_tool_calls': None},
            extra_body={"thinking": {"type": "disabled"}},
        )
        return [zhipu_model]

    model_name = model or _default_model(provider)
    if provider == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        models = [
            ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=temperature, max_retries=0, timeout=60)
            for key in _load_gemini_keys()
        ]

    else:
        models = []
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and provider == 'groq':
        from langchain_groq import ChatGroq

        models.append(ChatGroq(model=model_name, api_key=groq_key, temperature=temperature, max_retries=0, timeout=60))

    if not models:
        raise ModelConfigurationError("所选服务未配置 Key。请检查 LLM_PROVIDER 和 ZHIPU_API_KEY、GEMINI_API_KEY 或 GROQ_API_KEY。")

    return models


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Compatibility entrypoint; construction and explicit fallback selection live above."""
    return create_chat_model(model, temperature)


def extract_usage(ai_message) -> dict:
    """Pull token usage out of a LangChain AIMessage, if present."""
    meta = getattr(ai_message, "usage_metadata", None)
    if not meta:
        tokens = (getattr(ai_message, 'response_metadata', None) or {}).get('token_usage') or {}
        if not tokens:
            return {}
        return {'input_tokens': tokens.get('prompt_tokens'), 'output_tokens': tokens.get('completion_tokens'),
                'total_tokens': tokens.get('total_tokens')}
    return {
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
    }

