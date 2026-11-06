import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0):
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('未配置 DEEPSEEK_API_KEY，请在本地 .env 中填写。')
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'),
        api_key=key,
        base_url=os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
        temperature=temperature,
        max_retries=0,
        timeout=60,
        use_responses_api=False,
        extra_body={'thinking': {'type': 'disabled'}},
    )


def extract_usage(ai_message) -> dict:
    """Pull token usage out of a LangChain AIMessage, if present."""
    meta = getattr(ai_message, 'usage_metadata', None)
    if not meta:
        return {}
    return {
        'input_tokens': meta.get('input_tokens'),
        'output_tokens': meta.get('output_tokens'),
        'total_tokens': meta.get('total_tokens'),
    }
