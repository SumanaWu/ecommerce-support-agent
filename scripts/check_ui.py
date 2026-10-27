import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)
with TemporaryDirectory() as directory:
    os.environ['SUPPORT_DATA_DIR'] = directory
    os.environ['SUPPORT_AGENT_DB_URL'] = 'sqlite:///' + str(Path(directory) / 'business.sqlite')
    os.environ['SUPPORT_ACCOUNT_TOKENS'] = json.dumps({'customer_a': 'ui-token-a', 'customer_b': 'ui-token-b'})
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=20).run()
    assert not app.exception
    assert len(app.chat_input) == 0
    next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-a').run()
    assert not app.exception
    print('1 令牌验证与初始页面通过')
    app.chat_input[0].set_value('查询订单 1002 的物流').run()
    assert not app.exception
    print('2 订单工具查询通过')
    app.chat_input[0].set_value('帮我申请退款').run()
    assert not app.exception
    assert any('false' in str(m.value) for m in app.markdown)
    print('3 跨专家指代与退款拒绝通过')
    app.chat_input[0].set_value('帮我申请订单 1001 的退款').run()
    assert not app.exception
    next(b for b in app.button if b.label == '确认申请').click().run()
    assert not app.exception
    assert any('requested' in str(m.value) for m in app.markdown)
    print('4 提案与确认后退款写入通过')
    app.chat_input[0].set_value('我要投诉这个问题').run()
    assert not app.exception
    assert any('本地待处理工单' in str(m.value) for m in app.markdown)
    print('5 工单生成通过')
    next(b for b in app.button if b.label == '新建会话').click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    print('6 新会话切换通过')
    next(t for t in app.text_input if t.label == '测试账户令牌').set_value('ui-token-b').run()
    assert not app.exception
    assert len(app.chat_message) == 0
    print('7 测试账户切换通过')
