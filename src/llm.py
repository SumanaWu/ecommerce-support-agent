import json
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from pydantic import BaseModel, ValidationError

from src.budget import check_execution_budget
from src.errors import ModelConfigurationError, ModelOutputError, classify_error
from src.runtime_context import active_budget, record_event

load_dotenv()

SUPPORTED_PROVIDERS = ('deepseek', 'zhipu', 'gemini', 'groq')
AUTO_PROVIDERS = ('deepseek', 'zhipu')
FALLBACK_ERRORS = {'rate_limit', 'network', 'service', 'authentication', 'balance', 'configuration'}


class OnlineChatModel:
    """One service; each SDK attempt consumes the shared request budget."""

    _counts_attempts = True

    def __init__(self, client, provider, model_name):
        self.client, self.provider, self.model_name = client, provider, model_name

    @property
    def supports_required_tool_choice(self):
        return self.provider == 'deepseek'

    def bind_tools(self, tools, **kwargs):
        try:
            client = self.client.bind_tools(tools, **kwargs)
        except (AttributeError, NotImplementedError):
            raise ModelConfigurationError('所选模型适配器不支持工具调用。') from None
        return OnlineChatModel(client, self.provider, self.model_name)

    def invoke(self, messages):
        remaining = check_execution_budget('model', consume=True)
        record_event('model_call_started', model=self.provider + ':' + self.model_name, simulated=False)
        try:
            return self.client.invoke(messages, timeout=min(60, remaining) if remaining is not None else 60)
        except Exception as exc:
            record_event('model_attempt_failed', model=self.provider + ':' + self.model_name,
                         error_type=classify_error(exc), exception_type=type(exc).__name__)
            raise

    def invoke_structured(self, messages, schema: type[BaseModel], *, example: dict, max_tokens=2048):
        """JSON Output + strict local validation, retaining raw response/usage even on failure.

        This performs one attempt. The caller owns transport retry and, later, format correction.
        """
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise ModelConfigurationError('结构化输出需要 Pydantic 模型。')
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ModelConfigurationError('结构化输出长度必须为正整数。')
        schema.model_validate(example, strict=True)
        prompt = ('只返回一个完整的 json 对象，不使用 Markdown。遵守以下 JSON Schema：\n' +
                  json.dumps(schema.model_json_schema(), ensure_ascii=False) + '\nJSON 格式示例：\n' +
                  json.dumps(example, ensure_ascii=False))
        messages = [HumanMessage(content=messages)] if isinstance(messages, str) else convert_to_messages(messages)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=messages[0].text + '\n' + prompt)
        else:
            messages.insert(0, SystemMessage(content=prompt))
        from src.context import fit_structured_context
        messages, info = fit_structured_context(messages)
        if self.provider == 'gemini':
            options = {'response_mime_type': 'application/json', 'max_output_tokens': max_tokens}
        elif self.provider in {'deepseek', 'zhipu'}:
            # ChatOpenAI renames top-level max_tokens; these APIs use the original name.
            # Keep JSON mode in the wire body: LC otherwise selects SDK parse(),
            # which raises on truncation before we can retain the raw usage.
            # Reason only for structured planning/reference checks. Tool calls keep
            # thinking disabled so their history needs no provider-specific reasoning fields.
            thinking = 'enabled' if self.provider == 'deepseek' else 'disabled'
            token_limit = max(8192, max_tokens) if self.provider == 'deepseek' else max_tokens
            options = {'extra_body': {'thinking': {'type': thinking}, 'max_tokens': token_limit,
                                      'response_format': {'type': 'json_object'}}}
        else:
            options = {'response_format': {'type': 'json_object'}, 'max_tokens': max_tokens}
        try:
            client = self.client.bind(**options)
        except (AttributeError, NotImplementedError):
            raise ModelConfigurationError('所选模型适配器不支持 JSON 输出。') from None
        raw = OnlineChatModel(client, self.provider, self.model_name).invoke(messages)
        parsed, error = None, None
        finish = str(raw.response_metadata.get('finish_reason', 'stop')).lower()
        diagnostics = {'finish_reason': finish[:64], 'output_chars': len(raw.text), 'issues': []}
        try:
            if finish != 'stop' or raw.tool_calls or raw.invalid_tool_calls:
                diagnostics['reason'] = 'incomplete_output'
                raise ModelOutputError('模型未返回完整的 JSON 文本。')
            if not raw.text.strip():
                diagnostics['reason'] = 'empty_output'
                raise ModelOutputError('模型 JSON 输出为空。')
            parsed = schema.model_validate_json(raw.text, strict=True)
        except ValidationError as exc:
            diagnostics['issues'] = [{'path': list(issue['loc']), 'type': issue['type']}
                                     for issue in exc.errors(include_input=False, include_context=False, include_url=False)[:6]]
            diagnostics['field_constraints'] = [
                {'path': list(issue['loc']), 'allowed_values': issue['ctx']['expected']}
                for issue in exc.errors(include_input=False, include_url=False)[:6]
                if issue['type'] == 'literal_error' and 'expected' in issue.get('ctx', {})]
            diagnostics['reason'] = 'invalid_json' if any(issue['type'] == 'json_invalid' for issue in diagnostics['issues']) else 'schema_validation'
            error = ModelOutputError('模型 JSON 输出不符合约定结构。', diagnostics=diagnostics)
        except ModelOutputError as exc:
            exc.diagnostics = diagnostics
            error = exc
        if error:
            record_event('structured_output_invalid', model=self.provider + ':' + self.model_name, diagnostics=diagnostics)
        return {'raw': raw, 'parsed': parsed, 'parsing_error': error,
                'usage': extract_usage(raw), 'context_info': info}


class ProviderFallbackModel:
    """Try each configured service once; keep a working service for later bound calls."""

    _counts_attempts = True

    def __init__(self, models, selected=None):
        self.models = models
        preferred = getattr(active_budget.get(), '_model_provider', None)
        initial = next((index for index, model in enumerate(models) if model.provider == preferred), 0)
        self._selected = selected if selected is not None else [initial]

    @property
    def provider(self):
        return self.models[self._selected[0]].provider

    @property
    def model_name(self):
        return self.models[self._selected[0]].model_name

    @property
    def supports_required_tool_choice(self):
        return all(model.supports_required_tool_choice for model in self.models)

    def bind_tools(self, tools, **kwargs):
        return ProviderFallbackModel([model.bind_tools(tools, **kwargs) for model in self.models], self._selected)

    def _invoke(self, method, *args, **kwargs):
        order = [self._selected[0]] + [index for index in range(len(self.models)) if index != self._selected[0]]
        for position, index in enumerate(order):
            model = self.models[index]
            try:
                result = getattr(model, method)(*args, **kwargs)
            except Exception as exc:
                kind = classify_error(exc)
                if kind not in FALLBACK_ERRORS or position == len(order) - 1:
                    raise
                next_model = self.models[order[position + 1]]
                record_event('model_provider_fallback', from_model=model.provider + ':' + model.model_name,
                             to_model=next_model.provider + ':' + next_model.model_name, error_type=kind)
            else:
                self._selected[0] = index
                budget = active_budget.get()
                if budget is not None:
                    budget._model_provider = model.provider
                return result

    def invoke(self, messages):
        return self._invoke('invoke', messages)

    def invoke_structured(self, messages, schema: type[BaseModel], *, example: dict, max_tokens=2048):
        return self._invoke('invoke_structured', messages, schema, example=example, max_tokens=max_tokens)


def _fallback_specs():
    try:
        fallbacks = json.loads(os.environ.get('LLM_FALLBACKS') or '[]')
    except (ValueError, TypeError):
        raise ModelConfigurationError('LLM_FALLBACKS 必须为智谱或 DeepSeek 服务组成的 JSON 列表。') from None
    if not isinstance(fallbacks, list):
        raise ModelConfigurationError('LLM_FALLBACKS 必须为 JSON 列表。')
    specs = []
    for item in fallbacks:
        item = {'provider': item} if isinstance(item, str) else item
        if (not isinstance(item, dict) or set(item) - {'provider', 'model'} or
                item.get('provider') not in AUTO_PROVIDERS or
                ('model' in item and (not isinstance(item['model'], str) or not item['model'].strip()))):
            raise ModelConfigurationError('LLM_FALLBACKS 只支持 deepseek、zhipu 和可选的非空 model，不接受密钥等字段。')
        specs.append((item['provider'], item.get('model')))
    return specs


def create_chat_model(model: str | None = None, temperature: float = 0.0):
    provider = os.environ.get('LLM_PROVIDER', '').strip().lower()
    if provider not in (*SUPPORTED_PROVIDERS, 'auto'):
        raise ModelConfigurationError('LLM_PROVIDER 必须选择 auto、deepseek、zhipu、gemini 或 groq。')
    fallbacks = _fallback_specs()
    if provider == 'auto':
        if model is not None or fallbacks:
            raise ModelConfigurationError('auto 使用各服务的环境配置；请保持 LLM_FALLBACKS=[] 且不指定公共模型名称。')
        specs = [(name, None) for name in AUTO_PROVIDERS
                 if os.environ.get(name.upper() + '_API_KEY', '').strip() and
                 os.environ.get(name.upper() + '_MODEL', '').strip()]
        if not specs:
            raise ModelConfigurationError('auto 至少需要完整配置一家智谱或 DeepSeek 的 API_KEY 和 MODEL。')
    else:
        specs = [(provider, model), *fallbacks]
    if len({name for name, _ in specs}) != len(specs):
        raise ModelConfigurationError('主服务与 LLM_FALLBACKS 中的服务不能重复。')
    models = []
    for name, configured_model in specs:
        model_variable = name.upper() + '_MODEL'
        model_name = (configured_model or os.environ.get(model_variable, '')).strip()
        if not model_name:
            raise ModelConfigurationError('请配置 ' + model_variable + '，模型名称不能为空。')
        models.append(OnlineChatModel(_provider_client(name, model_name, temperature), name, model_name))
    return ProviderFallbackModel(models) if len(models) > 1 else models[0]


def _provider_client(provider, model, temperature):
    key = os.environ.get(provider.upper() + '_API_KEY', '').strip()
    if provider == 'gemini':
        multi = os.environ.get('GEMINI_API_KEYS', '').strip()
        keys = [value.strip() for value in multi.split(',') if value.strip()] if multi else ([key] if key else [])
        if len(keys) > 1:
            raise ModelConfigurationError('Gemini 不再自动轮换密钥；请只配置一个密钥。')
        key = keys[0] if keys else ''
    if not key:
        raise ModelConfigurationError('请配置 ' + provider.upper() + '_API_KEY。')

    if provider in {'deepseek', 'zhipu'}:
        from langchain_openai import ChatOpenAI
        default_url = 'https://api.deepseek.com' if provider == 'deepseek' else 'https://open.bigmodel.cn/api/paas/v4/'
        base_url = os.environ.get(provider.upper() + '_BASE_URL', default_url).strip().rstrip('/') + '/'
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment or
                parsed.username or parsed.password or any(c.isspace() for c in base_url) or
                parsed.path.rstrip('/').endswith(('/chat/completions', '/responses'))):
            raise ModelConfigurationError(provider.upper() + '_BASE_URL 应为基础 URL，不包含接口路径、凭据或查询参数。')
        return ChatOpenAI(model=model, api_key=key, base_url=base_url,
                          temperature=temperature,
                          max_retries=0, timeout=60, use_responses_api=False,
                          disabled_params={'parallel_tool_calls': None},
                          extra_body={'thinking': {'type': 'disabled'}})
    if provider == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature, max_retries=0, timeout=60)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=key, temperature=temperature, max_retries=0, timeout=60)


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Compatibility entrypoint for the existing graph."""
    return create_chat_model(model, temperature)


def extract_usage(ai_message) -> dict:
    """Pull token usage out of a LangChain AIMessage; missing usage remains unknown."""
    meta = getattr(ai_message, 'usage_metadata', None)
    if not meta:
        tokens = (getattr(ai_message, 'response_metadata', None) or {}).get('token_usage') or {}
        if not tokens:
            return {}
        return {'input_tokens': tokens.get('prompt_tokens'), 'output_tokens': tokens.get('completion_tokens'),
                'total_tokens': tokens.get('total_tokens')}
    return {'input_tokens': meta.get('input_tokens'), 'output_tokens': meta.get('output_tokens'),
            'total_tokens': meta.get('total_tokens')}
