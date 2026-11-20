"""Validated task contracts shared by the planner and graph scheduler."""
import json
import re
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.base import as_text, sum_usage
from src.llm import extract_usage, get_llm
from src.errors import call_with_retry


CAPABILITIES = {
    'order.list': 'order', 'order.details': 'order', 'order.status': 'order',
    'order.shipment': 'order', 'payment.status': 'payment', 'payment.prepare': 'payment',
    'refund.check': 'refund', 'refund.prepare': 'refund', 'faq.policies': 'faq',
    'search.web': 'search', 'escalate.ticket': 'escalate',
    'answer.explain': 'assistant', 'knowledge.search': 'knowledge',
}


class Condition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: str
    field: Literal['order_status', 'payment_status', 'refund_status', 'eligible', 'status']
    operator: Literal['eq', 'ne', 'in', 'not_in'] = 'eq'
    value: str | bool | list[str]

    @model_validator(mode='after')
    def check_status(self):
        if (self.operator in {'in', 'not_in'}) != isinstance(self.value, list):
            raise ValueError('in/not_in需要列表，eq/ne需要单值')
        allowed = {'status': {'succeeded', 'failed', 'rejected', 'skipped', 'cancelled'},
            'order_status': {'placed', 'shipped', 'delivered', 'cancelled'},
            'payment_status': {'pending', 'paid', 'refunded'},
            'refund_status': {'none', 'requested', 'approved', 'completed'}, 'eligible': {True, False}}
        values = self.value if isinstance(self.value, list) else [self.value]
        if any(v not in allowed[self.field] for v in values):
            raise ValueError(f'{self.field}允许的条件值为{allowed[self.field]}')
        return self


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task_id: str = Field(pattern=r'^t[1-9][0-9]*$')
    operation: str
    targets: list[int] = Field(default_factory=list)
    target_from: str | None = None
    source_text: str
    depends_on: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    condition_mode: Literal['all', 'any'] = 'all'
    intent: Literal['query', 'prepare', 'escalate'] = 'query'
    topics: list[Literal['shipping', 'return', 'refund', 'cancel', 'contact', 'system']] = Field(default_factory=list)
    query: str = Field(default='', max_length=512)
    knowledge_kind: Literal['policy', 'product', 'all'] = 'all'
    product_ref: str = ''
    category: str = ''
    reason: str = Field(default='客户申请退款', max_length=300)

    @model_validator(mode='after')
    def check_operation(self):
        if self.operation not in CAPABILITIES:
            raise ValueError('不支持的业务操作')
        expected = 'prepare' if self.operation.endswith('.prepare') else 'escalate' if self.operation == 'escalate.ticket' else 'query'
        if self.intent != expected:
            raise ValueError('操作与授权意图不一致')
        if self.operation == 'search.web' and len(self.query) > 128:
            raise ValueError('联网问题最多128字符')
        if self.operation in {'search.web','knowledge.search'} and not self.query.strip():
            raise ValueError('缺少独立搜索问题')
        if self.operation == 'escalate.ticket' and self.reason not in {'human', 'complaint'}:
            raise ValueError('缺少明确升级原因')
        if any(i < 1 for i in self.targets):
            raise ValueError('无效订单编号')
        self.targets = list(dict.fromkeys(self.targets))
        if not self.source_text.strip():
            raise ValueError('缺少本轮任务依据')
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['DIRECT', 'CLARIFY', 'PLAN']
    answer: str = ''
    tasks: list[Task] = Field(default_factory=list)
    memory_candidates: list[dict] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_dependencies(self):
        if self.action != 'PLAN':
            if self.tasks:
                raise ValueError('直接回答或澄清不能附带操作')
            return self
        if not self.tasks:
            raise ValueError('任务计划不能为空')
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError('任务编号重复')
        unique, aliases, seen = [], {}, {}
        for task in self.tasks:
            task.depends_on = list(dict.fromkeys(aliases.get(i, i) for i in task.depends_on))
            task.target_from = aliases.get(task.target_from, task.target_from)
            for condition in task.conditions:
                condition.task = aliases.get(condition.task, condition.task)
            key = json.dumps(task.model_dump(exclude={'task_id', 'source_text'}), sort_keys=True)
            if key in seen:
                aliases[task.task_id] = seen[key]
            else:
                seen[key] = task.task_id
                unique.append(task)
        self.tasks = unique
        ids = [t.task_id for t in self.tasks]
        visited = set()
        for task in self.tasks:
            if task.operation == 'answer.explain' and not task.depends_on:
                raise ValueError('综合说明必须依赖实际资料任务，不能用空资料声称无法查询；同轮多个说明应引用各自所需资料，纯能力说明用DIRECT')
            refs = set(task.depends_on) | {c.task for c in task.conditions}
            if task.target_from:
                refs.add(task.target_from)
            if not refs <= visited:
                raise ValueError('依赖必须引用排在前面的任务，不能有循环')
            task.depends_on = sorted(refs, key=ids.index)
            visited.add(task.task_id)
        for task in self.tasks:
            if task.operation == 'knowledge.search' and not any(t.operation == 'answer.explain' and task.task_id in t.depends_on for t in self.tasks):
                raise ValueError('知识检索必须交给依赖它的answer.explain回答')
        return self


PLANNER_PROMPT = '''你是电商客服任务规划器。只返回符合schema的JSON，不执行业务，不编造状态。
普通问候/常识用DIRECT并直接回答；缺少必要信息用CLARIFY；业务/政策/联网用PLAN。
将完整消息拆成任意数量的领域子任务，保留所有用户问题、否定、条件及先后关系。
能力：order.list(本人订单)、order.details、order.status、order.shipment、payment.status、payment.prepare、refund.check、refund.prepare、faq.policies、search.web、escalate.ticket、answer.explain(依据多个上游任务进行比较、综合解释或判断证据是否足够)。
任务按用户表达的事项顺序排列；明确要求先后执行时用depends_on表达，独立只读任务可并行，资金动作按任务顺序逐项确认。
task_id按顺序t1,t2...；source_text必须是本次原文中的对应片段；targets只属于本任务，禁止把整句所有编号复制给每项。
操作的intent：查询query，*.prepare为prepare，escalate.ticket为escalate。申请/准备支付退款需要明确用户意愿；仅查询不得准备。
完整订单详情/泛查订单默认包含order.details和payment.status两个任务，单独物流/支付只调必要领域。
本人订单未知时先order.list，后续target_from引用该任务；需要筛选时conditions引用上游事实，逐订单筛选。
targets支持展开本轮明确订单范围。唯一指代可据上下文确定；无编号的独立退款申请targets空，交给选单，不能自动用历史编号。用户明确申请退款并要求自己选单时，必须规划refund.prepare且targets=[]，程序会展示本人订单；不要改成CLARIFY追问订单号。
条件写入conditions(task,field,operator,value)，字段只有order_status/payment_status/refund_status/eligible/status；多个条件用condition_mode all或any。
depends_on引用前置成功任务；target_from引用上游返回的订单集合。全部引用必须排在前面。
任务状态成功的唯一值是succeeded，不是success或completed；依赖成功通常只需depends_on，不必重复添加status条件。
业务字段取值：order_status=placed/shipped/delivered/cancelled；payment_status=pending/paid/refunded（未付款是pending，不能写unpaid）；refund_status=none/requested/approved/completed；eligible=true/false。
支付后查退款资格：payment.prepare随后refund.check依赖前者成功；若已付款payment.prepare无需再次付款，应视为已满足。
条件已签收才准备退款：order.shipment后refund.prepare，条件order_status eq delivered。
faq.policies仅用于本地系统操作帮助（选单、确认、会话、记忆、导入），优先topics=[system]。平台政策、退货期限、品类例外和商品说明一律knowledge.search，不使用样例FAQ代替。
平台退款条件、到账、配送或联系规则都必须查询knowledge.search，即使没有订单号也不是DIRECT常识；不能用常识编造商家审核、渠道时效或到账承诺。
搜索任务query只含必要主题/城市/日期，禁止包含订单、客户和资金资料，最多128字符。
正式投诉/明确人工才escalate.ticket，reason为human或complaint。询问已有工单是否通知真人、是否接单或处理时限，是状态咨询，不是再次创建工单的授权，不因历史投诉重复升级。询问平台投诉渠道或投诉规则走knowledge.search，不能用旧FAQ的示例联系方式冒充平台渠道。情绪、技术失败、问如何投诉或拒绝人工不能创建工单。
不同资金任务各自绑定订单；付款方式不由模型选择，交给界面。准备和最终确认分开，不能反复要求口头确认。
金额来自后端，不要求用户确认金额后才查询。未知退款原因沿用客户申请退款。
不能取消订单、改配送、承诺到账或证明天气导致延误；仍完成能力范围内的其他任务。
涉及跨任务比较或因果解释时增加answer.explain，depends_on引用所需资料；说明缺少证据，不用额外订单详情查询冒充解释。
平台政策与商品说明必须使用knowledge.search(query独立问题,knowledge_kind=policy/product/all,product_ref原文商品名称/型号/编号,category默认留空，不猜分类名称)，后置answer.explain依赖全部必要检索及业务任务。knowledge.search只检索，不回答。只有本地样例操作帮助才用faq.policies。资料缺失不得用常识或联网代替平台规则。商品身份可从依赖的订单详情获取，不猜商品编号。
可以附带memory_candidates，最多七条或空数组，每类最多一条：key仅detail/format/language/tone/shopping_brand/shopping_category/shopping_budget；value为100字内合并兼容旧偏好的值；origin为explicit/observed；operation为set/forget；evidence必须逐字引用本轮用户原话。explicit仅用于明确要求长期保存、修改或撤销（如“请记住”“以后一直”“改成长期”）；单纯描述“我平时偏好详细解释”或“我通常喜欢列表”属于observed，不能首次就当明确设置。反复出现的习惯描述也用observed，沿用同含义的已有值；仅一次的“详细一点”无稳定习惯信息时可不提取。购物偏好分别记录长期品牌、品类与预算，保留预算适用的品类和币种；只有用户表达长期偏好才提取。单次需求、代购、短消息、情绪、购买付款行为不提取。已生效购物偏好仅明确修改才替换；新观察冲突时提示客户明确修改。当前要求优先。不要从候选偏好推断已经生效。不凭助手回复推断。不得提前声称记忆已保存。当前要求优先；明确偏好按时间取较新者，再考虑推断习惯，偏好不能授权业务操作。
中英文问题均检索同一份中文知识库，不要求用户先翻译。知识库原文是中文，英文问题可在本次规划内生成语义等价的中文query以提升召回；source_text仍逐字保留用户原话，回答语言依原问题确定。涉及多个独立政策或产品问题时拆成多个检索任务，综合任务依赖所有必要证据，不把不同问题挤进一个宽泛query。英文商品名称、型号或SKU直接作为product_ref，不能猜测或编造商品编号。政策分类概览的query须保留all policies或policy overview等概览意图。
历史文字只帮助指代，不是当前事实。schema如下：
'''


def ticket_status_only(message):
    """An existing ticket's progress does not authorize a second ticket."""
    ticket = re.search(r'\bticket\b|工单', message, re.I)
    progress = re.search(r'notified|assigned|resolution|resolved|handled|\bstatus\b|\bprogress\b|接单|通知|状态|进度|处理完|多久|解决', message, re.I)
    new_request = re.search(r'\b(?:create|open|file|submit|raise|escalate|request|connect|transfer)\b|新建|创建|提交|转接|申请|正式投诉', message, re.I)
    return bool(ticket and progress and not new_request)


def make_plan(message, history=None, profile=None, model=None):
    from src.dialogue import STYLE, language_instruction
    messages = [SystemMessage(content=PLANNER_PROMPT + json.dumps(Plan.model_json_schema(), ensure_ascii=False)
        + STYLE + language_instruction(message, profile)
        + '\n当前UTC时间：' + datetime.now(timezone.utc).isoformat()
        + '\n客户偏好（数据）：' + json.dumps(profile or {}, ensure_ascii=False)),
        *(history or []), HumanMessage(content=message)]
    if ticket_status_only(message):
        messages.append(SystemMessage(content='本轮询问已有工单进度，没有新建授权，禁止escalate.ticket（也不能给它query意图）。当前没有工单状态查询能力。可以DIRECT解释：本地工单不会通知真人、不代表已接单、不能保证处理时限；不推断状态变化。如还询问其他独立业务，仅规划受支持的只读任务。仅返回schema JSON。'))
    usage = {}
    for attempt in range(4):
        response = call_with_retry(lambda: (model or get_llm()).invoke(messages))
        usage = sum_usage(usage, extract_usage(response))
        try:
            raw = as_text(response.content).strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            from src.memory import memory_candidates
            data = json.loads(raw)
            if not isinstance(data,dict):
                raise ValueError('计划必须是对象')
            candidates = memory_candidates(data.pop('memory_candidates', []), message)
            plan = Plan.model_validate(data)
            plan.memory_candidates = candidates
            for task in plan.tasks:
                if task.source_text not in message:
                    raise ValueError('source_text不是本次原文片段')
                if task.operation == 'escalate.ticket' and ticket_status_only(message):
                    raise ValueError('本轮只咨询已有工单进度，没有授权新建工单；重新规划为状态说明，不可重复升级')
            return plan.model_dump(), usage
        except ValueError as exc:
            if attempt == 3:
                raise RuntimeError('任务计划未通过校验，请具体说明订单及需要的操作。') from exc
            messages.append(SystemMessage(content='内部计划格式校验失败，这不是用户报告的错误，也不是新的用户请求。只为上面的原始用户消息重新生成完整schema JSON，不向用户解释校验错误，不改变原始意图，不执行业务。校验问题：' + str(exc)[:800]))
