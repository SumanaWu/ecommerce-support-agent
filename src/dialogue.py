"""Main conversation decisions, grounded replies and web lookup."""
import json
import re
from datetime import datetime, timezone

import httpx
import os
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm, extract_usage
from src.errors import call_with_retry
from src.agents.base import as_text, sum_usage

STYLE = ('用自然、友好的中文回答，先回答核心问题，简洁，不机械追问。普通交流可用0至2个Emoji，不用“根据本轮资料”等内部措辞或无关免责声明；'
         '投诉、失败和资金操作保持认真。不编造实时信息、订单事实或已办理的操作。'
         '历史工单不代表本轮办理。遵守用户的回复偏好。')


def checked_reply(text):
    if re.search(r'(?:已生成|已创建|已提交|已记录).{0,12}工单|工单\s*[#＃]', text):
        return '请告诉我你希望了解的政策或订单信息；需要人工处理时也可以直接提出。'
    return text


def invoke(messages):
    return call_with_retry(lambda: get_llm().invoke(messages))


def compose(question, facts, history=None):
    from src.runtime_context import active_profile
    timing = '区分预报与实况，注明来源时点，不把查询时间当作观测时间；无法验证实时情况时明确说明。' if isinstance(facts, dict) and 'sources' in facts else ''
    response = invoke([SystemMessage(content=STYLE + timing + json.dumps(active_profile.get(), ensure_ascii=False) + '全部政策/都告诉我应覆盖资料全部主题，不要求先选一类；其他政策结合历史补充未介绍的主题，不作无关追问。仅依据下面本轮资料回答；资料没有的信息明确说明。不得声称创建工单或执行支付退款。'),
        *(history or []), HumanMessage(content=json.dumps({'question': question, 'facts': facts}, ensure_ascii=False))])
    return checked_reply(as_text(response.content)), extract_usage(response)


def search_web(query):
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        return {'error': '未配置 DeepSeek API Key，无法联网查询。', 'sources': []}
    query = query.strip()
    if not query or len(query) > 128:
        return {'error': '搜索问题需为1至128个字符，请简短描述后重试。', 'sources': []}
    base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    for attempt in range(2):
        try:
            response = httpx.post(base_url + '/v1/tools/search', headers={'Authorization': 'Bearer ' + key},
                json={'query': query, 'count': 5, 'need_content': False}, timeout=15)
            response.raise_for_status()
            data = response.json()
            return {'sources': [{'title': r['title'], 'url': r['url'], 'content': r.get('summary') or r.get('snippet') or '',
                                 'published_at': r.get('date'), 'retrieved_at': datetime.now(timezone.utc).isoformat()}
                                for r in data.get('results', [])]}
        except (httpx.HTTPError, ValueError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code == 429 or exc.response.status_code >= 500
            if attempt or not retryable:
                return {'error': '本次未能获取最新资料，请稍后重试。', 'sources': []}
