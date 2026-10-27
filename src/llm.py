import os

from dotenv import load_dotenv
from urllib.parse import urlsplit
from src.errors import ModelConfigurationError, classify_error
from src.runtime_context import active_mode

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

    def __init__(self, models: list, state: dict):
        self._models = models
        self._state = state

    def bind_tools(self, tools):
        return RotatingChatModel([m.bind_tools(tools) for m in self._models], state=self._state)

    def invoke(self, messages):
        n = len(self._models)
        last_exc: Exception | None = None
        for offset in range(n):
            idx = (self._state["index"] + offset) % n
            try:
                result = self._models[idx].invoke(messages)
                self._state["index"] = idx
                return result
            except Exception as exc:  # noqa: BLE001 - re-raised below if not a quota error
                last_exc = exc
                if not _is_quota_error(exc):
                    raise
        raise last_exc


def get_llm(model: str | None = None, temperature: float = 0.0) -> RotatingChatModel:
    if active_mode.get() == "demo":
        from src.demo_model import DemoChatModel
        return DemoChatModel()

    provider = os.environ.get('LLM_PROVIDER', 'auto').strip().lower()
    if provider not in {'auto', 'zhipu', 'gemini', 'groq'}:
        raise ModelConfigurationError('LLM_PROVIDER 必须为 auto、zhipu、gemini 或 groq。')
    zhipu_key = os.environ.get("ZHIPU_API_KEY", '').strip()
    if provider == 'zhipu' or (provider == 'auto' and zhipu_key):
        if not zhipu_key:
            raise ModelConfigurationError('请配置 ZHIPU_API_KEY。')
        from langchain_openai import ChatOpenAI

        base_url = os.environ.get('ZHIPU_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/').strip().rstrip('/') + '/'
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment or parsed.path.rstrip('/').endswith('/chat/completions'):
            raise ModelConfigurationError('ZHIPU_BASE_URL 应为基础 URL，不能附加 chat/completions。')
        zhipu_model = ChatOpenAI(
            model=model or os.environ.get("ZHIPU_MODEL", "glm-4.7-flash"),
            api_key=zhipu_key,
            base_url=base_url,
            temperature=1.0,
            max_retries=0,
            timeout=60,
            use_responses_api=False,
            disabled_params={'parallel_tool_calls': None},
            extra_body={"thinking": {"type": "disabled"}},
        )
        return RotatingChatModel([zhipu_model], state=_rotation_state)

    model_name = model or DEFAULT_MODEL
    models = []
    if provider in {'auto', 'gemini'}:
        from langchain_google_genai import ChatGoogleGenerativeAI
        models = [
            ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=temperature, max_retries=0)
            for key in _load_gemini_keys()
        ]

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and provider in {'auto', 'groq'}:
        from langchain_groq import ChatGroq

        models.append(ChatGroq(model=GROQ_MODEL, api_key=groq_key, temperature=temperature, max_retries=0))

    if not models:
        raise ModelConfigurationError("所选服务未配置 Key。请检查 LLM_PROVIDER 和 ZHIPU_API_KEY、GEMINI_API_KEY 或 GROQ_API_KEY。")

    return RotatingChatModel(models, state=_rotation_state)


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

