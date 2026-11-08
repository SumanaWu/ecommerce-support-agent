"""Policy retrieval precedes grounded natural-language answers."""
from src import tools as backend_tools


def get_policies(topics=None):
    allowed = ['shipping', 'return', 'cancel', 'refund', 'contact', 'system']
    return {topic: backend_tools.answer_faq(topic) for topic in (topics or allowed[:5]) if topic in allowed}


def run_faq_agent(user_message: str, history: list | None = None, topics=None) -> dict:
    from src.dialogue import compose
    facts = get_policies(topics)
    reply, usage = compose(user_message, facts, history)
    return {'reply': reply, 'agent': 'faq', 'trace': [{'agent': 'faq', 'tool': 'get_policies',
            'args': {'topics': list(facts)}, 'result': facts}], 'usage': usage, 'cached': False}
