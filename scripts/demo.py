"""在临时目录验证身份、提案、确认与 Memory，不影响正常数据。"""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with TemporaryDirectory() as directory:
    os.environ['SUPPORT_DATA_DIR'] = directory
    os.environ['SUPPORT_AGENT_DB_URL'] = 'sqlite:///' + str(Path(directory) / 'business.sqlite')
    from src.auth import Principal
    from src.service import SupportService
    principal = Principal('customer_a')
    service = SupportService()
    service.save_profile(principal, '简洁，先结论', '优先文字')
    service.chat(principal, 'case_1', 'demo', '查询订单 1002 的物流')
    service = SupportService()
    out = service.chat(principal, 'case_1', 'demo', '帮我申请退款')
    assert out['trace'][0]['args']['order_id'] == 1002
    assert out['trace'][0]['result']['eligible'] is False
    print('重建服务后沿用订单 1002，未送达退款被拒绝。')
    out = service.chat(principal, 'case_1', 'demo', '帮我申请订单 1001 的退款')
    assert out['status'] == 'pending_confirmation'
    print('订单 1001 已生成待确认提案，尚未办理。')
    out = service.confirm_action(principal, out['pending_actions'][0]['action_id'])
    assert out['trace'][-1]['result']['success']
    print('用户确认后，本地退款申请已写入。')
    out = service.chat(principal, 'case_1', 'demo', '我要投诉这个问题')
    print(out['reply'])
    assert service.snapshot(principal, 'case_2', 'demo')['turns'] == []
    assert service.profile(principal)['tone'] == '简洁，先结论'
    assert service.profile(Principal('customer_b')) == {}
    print('新会话没有旧消息；客户偏好跨会话读取且账户隔离。')
