"""在临时目录进行真实图与工具演示，不污染正常业务库。"""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

with TemporaryDirectory() as d:
    os.environ['SUPPORT_DATA_DIR']=d
    os.environ.pop('SUPPORT_AGENT_DB_URL',None)
    from src.service import SupportService
    service=SupportService()
    service.store.save_profile('customer_a','简洁，先结论','优先文字')
    service.chat('customer_a','case_1','demo','查询订单 1002 的物流')
    service=SupportService()
    out=service.chat('customer_a','case_1','demo','帮我申请退款')
    assert out['trace'][0]['args']['order_id']==1002
    print('重建服务后沿用订单 1002，未送达退款被拒绝。')
    out=service.chat('customer_a','case_1','demo','帮我申请订单 1001 的退款')
    assert out['trace'][-1]['result']['success']
    print('订单 1001 的本地退款申请已写入。')
    out=service.chat('customer_a','case_1','demo','我要投诉这个问题')
    print(out['reply'])
    assert service.snapshot('customer_a','case_2','demo')['turns']==[]
    assert service.store.profile('customer_a')['tone']=='简洁，先结论'
    print('新会话没有旧消息，客户偏好可跨会话读取。')
