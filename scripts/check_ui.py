import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
os.chdir(root)
with TemporaryDirectory() as d:
    os.environ['SUPPORT_DATA_DIR']=d
    os.environ.pop('SUPPORT_AGENT_DB_URL',None)
    app=AppTest.from_file(str(root/'app.py'),default_timeout=20).run()
    assert not app.exception,app.exception
    print('1 初始页面通过')
    app.chat_input[0].set_value('查询订单 1002 的物流').run()
    assert not app.exception,app.exception
    print('2 订单工具查询通过')
    app.chat_input[0].set_value('帮我申请退款').run()
    assert not app.exception,app.exception
    assert any('false' in str(m.value) for m in app.markdown)
    print('3 跨专家指代与退款拒绝通过')
    app.chat_input[0].set_value('帮我申请订单 1001 的退款').run()
    assert not app.exception,app.exception
    assert any('尚未执行' in str(m.value) for m in app.markdown)
    assert app.chat_input[0].disabled
    next(b for b in app.button if b.label=='确认办理').click().run()
    assert not app.exception,app.exception
    assert not app.chat_input[0].disabled
    print('4 退款确认通过')
    app.chat_input[0].set_value('我要投诉这个问题').run()
    assert not app.exception,app.exception
    assert any('本地待处理工单' in str(m.value) for m in app.markdown)
    print('5 工单生成通过')
    next(b for b in app.button if b.label=='新建会话').click().run()
    assert not app.exception,app.exception
    assert len(app.chat_message)==0
    print('6 新会话切换通过')
    next(t for t in app.text_input if t.label=='客户编号').set_value('customer_b').run()
    assert not app.exception,app.exception
    assert len(app.chat_message)==0
    print('7 客户会话切换通过')
