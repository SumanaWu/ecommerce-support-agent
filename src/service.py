import os
import re
import threading
from pathlib import Path
from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator
from src.runtime_context import active_mode, active_profile
from src.storage import SupportStore


class SupportService:
    def __init__(self, path=None):
        path = path or Path(os.environ.get('SUPPORT_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data'))) / 'memory.sqlite'
        self.store = SupportStore(path)
        self.lock = threading.RLock()

    def validate(self, user_id, session_id, mode):
        if mode not in {'demo', 'online'}:
            raise ValueError('运行模式必须为 demo 或 online')
        for value in [user_id, session_id]:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
                raise ValueError('编号只允许 1 至 40 位字母、数字、下划线、短横线')

    def snapshot(self, user_id, session_id, mode):
        self.validate(user_id, session_id, mode)
        return self.store.load_session(user_id, mode, session_id)

    def chat(self, user_id, session_id, mode, message, runner=None):
        self.validate(user_id, session_id, mode)
        if not message.strip() or len(message) > 2000:
            raise ValueError('消息需为 1 至 2000 个字符')
        runner = runner or run_orchestrator
        with self.lock:
            payload = self.snapshot(user_id, session_id, mode)
            memory = ConversationMemory(last_order_id=payload['last_order_id'])
            # 最多六个完整用户轮次，避免无上限发送历史。
            for turn in payload['turns'][-6:]:
                memory.add_turn(turn['user'], turn['reply'])
            explicit = re.findall(r'\b\d{4,8}\b', message)
            augmented = message if explicit else memory.context_note() + message
            mode_token = active_mode.set(mode)
            profile_token = active_profile.set(self.store.profile(user_id))
            try:
                result = runner(augmented, history=memory.turns)
            finally:
                active_profile.reset(profile_token)
                active_mode.reset(mode_token)
            memory.update_last_order_id(result.get('trace', []))
            turn = {'user': message, 'reply': result['reply'], 'agent': result['agent'],
                    'trace': result.get('trace', []), 'cached': result.get('cached', False)}
            payload['turns'] = (payload['turns'] + [turn])[-50:]
            payload['last_order_id'] = memory.last_order_id
            payload['total_tokens'] += (result.get('usage') or {}).get('total_tokens') or 0
            self.store.save_session(user_id, mode, session_id, payload)
            return result
