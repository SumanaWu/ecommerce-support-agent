from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage


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



MEMORY_KEYS = {'detail', 'format', 'language', 'tone'}
MEMORY_OBSERVATIONS_REQUIRED = 3


def memory_candidates(raw, message):
    """Memory errors never invalidate the business plan."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:2]:
        if not isinstance(item, dict):
            continue
        if (item.get('key') not in MEMORY_KEYS or item.get('origin') not in {'explicit','observed'}
                or item.get('operation') not in {'set','forget'}):
            continue
        value, evidence = item.get('value'), item.get('evidence')
        if item['operation'] == 'forget' and value == '':
            value = item['value'] = '已撤销'
        if not isinstance(value,str) or not 0 < len(value) <= 100 or not isinstance(evidence,str) or not evidence or evidence not in message:
            continue
        if item['operation'] == 'forget' and item['origin'] != 'explicit':
            continue
        if item['key'] not in {r['key'] for r in result}:
            result.append({k:item[k] for k in ('key','value','origin','operation','evidence')})
    return result
