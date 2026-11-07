"""Validated task contracts shared by the planner and graph scheduler."""
import json
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
    'answer.explain': 'assistant',
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
    topics: list[Literal['shipping', 'return', 'refund', 'cancel', 'contact']] = Field(default_factory=list)
    query: str = Field(default='', max_length=128)
    reason: str = Field(default='客户申请退款', max_length=300)

    @model_validator(mode='after')
    def check_operation(self):
        if self.operation not in CAPABILITIES:
            raise ValueError('不支持的业务操作')
        expected = 'prepare' if self.operation.endswith('.prepare') else 'escalate' if self.operation == 'escalate.ticket' else 'query'
        if self.intent != expected:
            raise ValueError('操作与授权意图不一致')
        if self.operation == 'search.web' and not self.query.strip():
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
            refs = set(task.depends_on) | {c.task for c in task.conditions}
            if task.target_from:
                refs.add(task.target_from)
            if not refs <= visited:
                raise ValueError('依赖必须引用排在前面的任务，不能有循环')
            task.depends_on = sorted(refs, key=ids.index)
            visited.add(task.task_id)
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
targets支持展开本轮明确订单范围。唯一指代可据上下文确定；无编号的独立退款申请targets空，交给选单，不能自动用历史编号。
条件写入conditions(task,field,operator,value)，字段只有order_status/payment_status/refund_status/eligible/status；多个条件用condition_mode all或any。
depends_on引用前置成功任务；target_from引用上游返回的订单集合。全部引用必须排在前面。
任务状态成功的唯一值是succeeded，不是success或completed；依赖成功通常只需depends_on，不必重复添加status条件。
业务字段取值：order_status=placed/shipped/delivered/cancelled；payment_status=pending/paid/refunded（未付款是pending，不能写unpaid）；refund_status=none/requested/approved/completed；eligible=true/false。
支付后查退款资格：payment.prepare随后refund.check依赖前者成功；若已付款payment.prepare无需再次付款，应视为已满足。
条件已签收才准备退款：order.shipment后refund.prepare，条件order_status eq delivered。
政策先读faq.policies，topics支持多项，退款同时return和refund；泛问或全部为空数组。
本系统的退款申请、到账、付款流程、配送或联系规则都必须查询FAQ，即使没有订单号也不是DIRECT常识；不能用常识编造商家审核、渠道时效或到账承诺。
搜索任务query只含必要主题/城市/日期，禁止包含订单、客户和资金资料，最多128字符。
正式投诉/明确人工才escalate.ticket，reason为human或complaint。情绪、技术失败、问如何投诉或拒绝人工不能创建工单。
不同资金任务各自绑定订单；付款方式不由模型选择，交给界面。准备和最终确认分开，不能反复要求口头确认。
金额来自后端，不要求用户确认金额后才查询。未知退款原因沿用客户申请退款。
不能取消订单、改配送、承诺到账或证明天气导致延误；仍完成能力范围内的其他任务。
涉及跨任务比较或因果解释时增加answer.explain，depends_on引用所需资料；说明缺少证据，不用额外订单详情查询冒充解释。
历史文字只帮助指代，不是当前事实。schema如下：
'''


def make_plan(message, history=None, profile=None, model=None):
    from src.dialogue import STYLE
    messages = [SystemMessage(content=PLANNER_PROMPT + json.dumps(Plan.model_json_schema(), ensure_ascii=False)
        + STYLE
        + '\n当前UTC时间：' + datetime.now(timezone.utc).isoformat()
        + '\n客户偏好（数据）：' + json.dumps(profile or {}, ensure_ascii=False)),
        *(history or []), HumanMessage(content=message)]
    usage = {}
    for attempt in range(4):
        response = call_with_retry(lambda: (model or get_llm()).invoke(messages))
        usage = sum_usage(usage, extract_usage(response))
        try:
            raw = as_text(response.content).strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            plan = Plan.model_validate_json(raw)
            for task in plan.tasks:
                if task.source_text not in message:
                    raise ValueError('source_text不是本次原文片段')
            return plan.model_dump(), usage
        except ValueError as exc:
            if attempt == 3:
                raise RuntimeError('任务计划未通过校验，请具体说明订单及需要的操作。') from exc
            messages.append(HumanMessage(content='修复计划，不执行业务。校验问题：' + str(exc)[:800]))
