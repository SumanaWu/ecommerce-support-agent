"""Isolated online-page regression with explicitly scripted model responses."""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
    patch.setenv('SUPPORT_DATA_DIR', directory)
    patch.setenv('SUPPORT_AGENT_DB_URL', 'sqlite:///' + str(Path(directory) / 'business.sqlite'))
    patch.setenv('SUPPORT_ACCOUNT_TOKENS', json.dumps({'customer_a': 'ui-token-a', 'customer_b': 'ui-token-b'}))
    import httpx
    import httpx2
    def deny(*args, **kwargs):
        raise AssertionError('页面回归禁止真实 HTTP 请求。')
    async def deny_async(*args, **kwargs):
        deny()
    for module in (httpx, httpx2):
        patch.setattr(module.HTTPTransport, 'handle_request', deny)
        patch.setattr(module.AsyncHTTPTransport, 'handle_async_request', deny_async)
    from src import tools
    from tests.online_support import call, install_script, plan, proposal_steps, task, reference_verdict
    install_script(patch, [
        plan(task('shipment', [1002])), call('track_shipment', {'order_id': 1002}), '订单运输中。',
        plan({**task('propose_refund', [1002]), 'context_use': 'order'}), reference_verdict(1002), call('check_refund_eligibility', {'order_id': 1002}), '订单未送达，不能申请退款。',
        *proposal_steps(), plan(kind='escalate', message='用户需要人工处理')])
    try:
        app = AppTest.from_file(str(root / 'app.py'), default_timeout=20).run()
        assert not app.exception and not app.chat_input
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-a').run()
        assert not app.exception and not any(t.label == '运行模式' for t in app.selectbox)
        print('1 令牌验证与在线页面通过（脚本模型，不调用真实 API）')
        app.chat_input[0].set_value('查询订单 1002 的物流').run()
        assert not app.exception and any('运输中' in str(item.value) for item in app.markdown)
        print('2 订单工具查询通过')
        app.chat_input[0].set_value('帮我申请退款').run()
        assert not app.exception and '尚未送达' in app.chat_message[-1].markdown[0].value
        print('3 跨专家指代与退款拒绝通过')
        app.chat_input[0].set_value('帮我申请订单 1001 的退款').run()
        assert [b.label for b in app.chat_message[-1].button] == ['确认申请', '取消操作']
        next(b for b in app.button if b.label == '确认申请').click().run()
        assert not app.exception and '退款申请已提交' in app.chat_message[-1].markdown[0].value
        print('4 回复下方提案与确认后退款写入通过')
        app.chat_input[0].set_value('我要投诉这个问题').run()
        assert not app.exception and '已生成待处理工单' in app.chat_message[-1].markdown[0].value
        print('5 工单生成通过')
        next(b for b in app.button if b.label == '新建会话').click().run()
        assert not app.exception and not app.chat_message
        print('6 新会话切换通过')
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-b').run()
        assert not app.exception and not app.chat_message
        print('7 测试账户切换通过')

        # Observe rendering at the service boundary, before a reply can exist.
        from src.service import SupportService
        message = '先查 1001，再把 1002 的物流一起列出来，我有点搞不清。'
        reply = '查到了 🙂\n\n| 订单 | 状态 |\n| --- | --- |\n| 1001 | 已送达 |\n\n- 1002 的物流已单独列出。'
        rendered, roles, attempts = [], [], []
        original_chat_message, original_write, original_chat = st.chat_message, st.write, SupportService.chat

        @contextmanager
        def observe_chat_message(role, *args, **kwargs):
            with original_chat_message(role, *args, **kwargs) as container:
                roles.append(role)
                try:
                    yield container
                finally:
                    roles.pop()

        def observe_write(*args, **kwargs):
            if roles and roles[-1] == 'user' and args and args[0] == message:
                rendered.append(message)
            return original_write(*args, **kwargs)

        def interrupted_chat(self, principal, session_id, text, *, request_id):
            assert text == message and len(rendered) > len(attempts), '调用服务前必须先绘制用户消息。'
            attempts.append(request_id)
            if len(attempts) == 1:
                raise RuntimeError('private simulated transport error')
            if len(attempts) == 2:
                return {'status': 'in_progress', 'memory_saved': False, 'reply': '同一请求仍在处理中。'}
            return original_chat(self, principal, session_id, text, request_id=request_id,
                runner=lambda *_args, **_kwargs: {'reply': reply, 'agent': 'order', 'trace': [], 'usage': {}, 'status': 'completed'})

        patch.setattr(st, 'chat_message', observe_chat_message)
        patch.setattr(st, 'write', observe_write)
        patch.setattr(SupportService, 'chat', interrupted_chat)
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-a').run()
        next(t for t in app.text_input if t.label == '会话编号').set_value('pending_visibility').run()
        app.chat_input[0].set_value(message).run()
        assert not app.exception and len(app.chat_message) == 1
        assert app.chat_message[0].markdown[0].value == message and app.chat_input[0].disabled
        pending = app.session_state['pending_requests'][('customer_a', 'pending_visibility')].copy()
        assert any(button.label == '重试同一请求' for button in app.button)
        assert not any('private simulated' in str(item.value) for item in app.markdown)
        print('8 服务调用前用户气泡已绘制，异常后原输入保留且立即可重试')

        next(t for t in app.text_input if t.label == '会话编号').set_value('other_session').run()
        assert not app.chat_message and not app.chat_input[0].disabled
        next(t for t in app.text_input if t.label == '会话编号').set_value('pending_visibility').run()
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-b').run()
        assert not app.chat_message and not app.chat_input[0].disabled
        next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-a').run()
        assert len(app.chat_message) == 1 and len(attempts) == 1
        print('9 未完成输入按账户和会话隔离，切回后仍可见且不会自动重复调用')

        next(button for button in app.button if button.label == '重试同一请求').click().run()
        assert not app.exception and len(app.chat_message) == 1 and app.chat_input[0].disabled
        next(button for button in app.button if button.label == '重试同一请求').click().run()
        assert not app.exception and attempts == [pending['request_id']] * 3
        assert app.session_state['pending_requests'] == {} and len(app.chat_message) == 2
        assert app.chat_message[-1].markdown[0].value == reply and not app.chat_input[0].disabled
        app.session_state['pending_requests'] = {('customer_a', 'pending_visibility'): pending}
        app.run()
        assert not app.exception and app.session_state['pending_requests'] == {}
        assert len(app.chat_message) == 2 and len(attempts) == 3
        print('10 处理中及重试保留同一消息/请求，成功与重复 rerun 不重复气泡，表格/列表/emoji Markdown 完整保留')
    finally:
        st.cache_resource.clear()
        tools._engine.dispose()
