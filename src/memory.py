from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

from langchain_core.messages import AIMessage, HumanMessage


def empty_issue():
    return {'candidate_order_ids': [], 'active_order_id': None, 'selection_source': None, 'updated_at': None, 'revision': 0}


def session_issue(payload):
    if 'issue' in payload:
        return {**empty_issue(), **payload['issue']}
    issue = empty_issue()
    if payload.get('last_order_id'):
        issue.update(candidate_order_ids=[payload['last_order_id']], selection_source='legacy_last_order')
    return issue


def extract_order_ids(message):
    """Recognize explicit order labels or bare IDs; exclude amounts and negated IDs.

    This deliberately covers a small, documented grammar, not general NLP.
    """
    pattern = r'(?:(?:订单(?:编号|号码|号)?|order\s*(?:id|number|no\.?)?)\s*[:：#]?\s*(\d{1,12})(?![A-Za-z0-9_.])|(?<![A-Za-z0-9_.])([0-9]{4,8})(?![A-Za-z0-9_.]|\s*(?:元|块|美元|人民币|dollars?)))'
    result = []
    for match in re.finditer(pattern, message, re.IGNORECASE):
        prefix = message[:match.start()]
        if re.search(r'(?:不是|不要|不选|排除|not)\s*$', prefix, re.IGNORECASE):
            continue
        if match.group(2) and (re.search(r'(?:不是|不要|不选|排除|not)\s*(?:订单|order)?\s*$', prefix, re.IGNORECASE)
                               or re.search(r'(?:金额|价格|总计|实付|amount|[￥¥$])\s*[:：]?\s*$', prefix, re.IGNORECASE)):
            continue
        order_id = int(match.group(1) or match.group(2))
        if order_id not in result:
            result.append(order_id)
    return result


def resolve_order_reference(message, issue, is_owned, *, force_order_task=False):
    issue = {**empty_issue(), **issue}
    issue['candidate_order_ids'] = [number for number in issue['candidate_order_ids'] if is_owned(number)]
    if issue['active_order_id'] is not None and issue['active_order_id'] not in issue['candidate_order_ids']:
        issue.update(active_order_id=None, selection_source=None)
    ids = extract_order_ids(message)
    order_task = force_order_task or bool(re.search(r'订单|物流|快递|退款|退掉|退那|退这|付款|支付|\b(?:order|tracking|refund|payment|pay)\b', message, re.IGNORECASE))
    policy = bool(re.search(r'政策|规则|policy|多久|退货条件', message, re.IGNORECASE))
    selection_only = bool(ids and (re.fullmatch(r'\s*\d{4,8}\s*[。.!！]?\s*', message)
                                  or re.search(r'不是|不对|改为|换成|我选|选择|当前订单', message)))
    selection_only = selection_only and not re.search(r'退款|退掉|帮我退|付款|支付|查询|物流|快递|申请|办理|refund|payment|tracking', message, re.IGNORECASE)
    labels = re.finditer(r'(?:订单(?:编号|号码|号)?|order\s*(?:id|number|no\.?)?)\s*[:：#]?\s*([A-Za-z0-9_.-]+)', message, re.IGNORECASE)
    malformed = any(not re.fullmatch(r'\d{1,12}', match.group(1)) and match.group(1).lower() not in {
        'status', 'details', 'tracking', 'id', 'number', 'no'}
        for match in labels if not re.search(r'(?:不是|不要|不选|排除|not)\s*$', message[:match.start()], re.IGNORECASE))
    if malformed:
        return {'status': 'invalid', 'order_id': None,
                'issue': {**empty_issue(), 'revision': issue['revision'] + 1},
                'explicit_ids': ids, 'selection_only': False}
    if ids:
        if any(not is_owned(number) for number in ids):
            cleared = {**empty_issue(), 'revision': issue['revision'] + 1}
            return {'status': 'invalid', 'order_id': None, 'issue': cleared, 'explicit_ids': ids, 'selection_only': False}
        issue.update(candidate_order_ids=ids if len(ids) > 1 else list(dict.fromkeys(issue['candidate_order_ids'] + ids)),
                     active_order_id=ids[0] if len(ids) == 1 else None,
                     selection_source=('user_correction' if selection_only else 'user_message') if len(ids) == 1 else None,
                     updated_at=datetime.now(timezone.utc).isoformat())
        if selection_only and len(ids) == 1:
            issue['revision'] += 1
        return {'status': 'resolved' if len(ids) == 1 else 'ambiguous',
                'order_id': ids[0] if len(ids) == 1 else None, 'issue': issue,
                'explicit_ids': ids, 'selection_only': selection_only and len(ids) == 1}
    if not order_task or policy:
        status, order_id = 'irrelevant', None
    elif issue['active_order_id']:
        status, order_id = 'resolved', issue['active_order_id']
    elif len(issue['candidate_order_ids']) == 1:
        status, order_id = 'resolved', issue['candidate_order_ids'][0]
    else:
        status = 'ambiguous' if issue['candidate_order_ids'] else 'missing'
        order_id = None
    return {'status': status, 'order_id': order_id, 'issue': issue, 'explicit_ids': [], 'selection_only': False}


def clarify_order_reference(reference):
    candidates = reference['issue']['candidate_order_ids']
    if candidates:
        return '当前涉及多个订单：' + '、'.join(map(str, candidates)) + '。请明确本次处理哪一笔，例如“帮我申请订单 1001 的退款”，或在“记忆与客户偏好”中选择当前订单；当前未生成写入提案。'
    return '请先提供要处理的订单编号，例如“查询订单 1001”，当前未生成写入提案。'


@dataclass
class ConversationMemory:
    """Cross-turn conversation state: plain text history (safe to hand to any
    specialist agent, regardless of which agent handled earlier turns) plus a
    remembered 'last order discussed' so the customer doesn't need to repeat it."""

    turns: list = field(default_factory=list)
    last_order_id: int | None = None

    def add_turn(self, user_message: str, reply: str) -> None:
        self.turns.append(HumanMessage(content=user_message))
        self.turns.append(AIMessage(content=reply))

    def update_last_order_id(self, trace: list) -> None:
        for entry in trace:
            args = entry.get("args") or {}
            result = entry.get("result") or {}
            order_id = args.get("order_id")
            if order_id is None and isinstance(result, dict):
                order_id = result.get("order_id")
            if isinstance(result, dict) and result.get("error"):
                continue
            if isinstance(order_id, int) and order_id > 0:
                self.last_order_id = order_id

    def context_note(self) -> str:
        if self.last_order_id is not None:
            return f"(Context: the customer was last discussing order {self.last_order_id}.)\n"
        return ""

