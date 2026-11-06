"""两个独立 Python 进程验证文件持久化，不影响正式数据。"""
import os
import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
root=Path(__file__).resolve().parents[1]
with TemporaryDirectory() as d:
    env=dict(os.environ,SUPPORT_DATA_DIR=d)
    env.pop('SUPPORT_AGENT_DB_URL',None)
    first="from src.service import SupportService;s=SupportService();s.store.save_profile('customer_a','简洁','文字');s.chat('customer_a','one','查询订单 1002 的物流')"
    second="from src.service import SupportService;s=SupportService();assert s.store.profile('customer_a')['tone']=='简洁';r=s.chat('customer_a','one','帮我申请退款');assert r['status']=='awaiting_selection';assert r['selection']['recent_order_ids']==[1002];assert len(r['selection']['candidates'])==7;print('独立进程恢复会话和偏好通过')"
    for code in (first,second):
        subprocess.run([sys.executable,'-c',code],cwd=root,env=env,check=True)
