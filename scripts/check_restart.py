"""独立进程恢复会话、待确认动作与中断图，不影响正式数据。"""
import os
import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

root = Path(__file__).resolve().parents[1]
with TemporaryDirectory() as directory:
    env = dict(os.environ, SUPPORT_DATA_DIR=directory,
               SUPPORT_AGENT_DB_URL='sqlite:///' + str(Path(directory) / 'business.sqlite'))
    first = '''
from src.auth import Principal
from src.service import SupportService
s=SupportService(); p=Principal('customer_a')
s.save_profile(p,'简洁','文字')
s.chat(p,'one','demo','查询订单 1002 的物流')
r=s.chat(p,'pending','demo','帮我申请订单 1001 的退款')
assert r['status']=='pending_confirmation'
'''
    second = '''
from src.auth import Principal
from src.service import SupportService
s=SupportService(); p=Principal('customer_a')
assert s.profile(p)['tone']=='简洁'
r=s.chat(p,'one','demo','帮我申请退款')
assert r['trace'][0]['args']['order_id']==1002
assert r['trace'][0]['result']['eligible'] is False
pending=s.list_pending_actions(p,'pending','demo')
assert len(pending)==1
assert s.confirm_action(p,pending[0]['action_id'])['status']=='completed'
assert s.confirm_action(p,pending[0]['action_id'])['replayed']
print('独立进程恢复会话、偏好与待确认动作通过')
'''
    for code in (first, second):
        subprocess.run([sys.executable, '-c', code], cwd=root, env=env, check=True)
    crash = '''
import os
from src import orchestrator
from src.auth import Principal
from src.service import SupportService
original=orchestrator.tool_step
def stop_after_proposal(state):
    result=original(state)
    if result['trace'][-1]['tool']=='initiate_refund': os._exit(23)
    return result
orchestrator.tool_step=stop_after_proposal
SupportService().chat(Principal('customer_a'),'recover','demo','帮我申请订单 1006 的退款',request_id='restart-graph')
'''
    interrupted = subprocess.run([sys.executable, '-c', crash], cwd=root, env=env)
    assert interrupted.returncode == 23
    recover = '''
from sqlalchemy import select
from src.auth import Principal
from src.service import SupportService
from src import tools
from src.backend.db import ActionRecord
s=SupportService(); p=Principal('customer_a')
runs=s.list_runs(p,'recover','demo'); assert len(runs)==1
r=s.resume_run(p,runs[0]['run_id'])
assert r['status']=='pending_confirmation' and r['memory_saved']
assert len(s.list_pending_actions(p,'recover','demo'))==1
with tools._session() as db:
    records=db.scalars(select(ActionRecord).where(ActionRecord.request_id=='restart-graph')).all()
    assert len(records)==1
assert s.confirm_action(p,r['pending_actions'][0]['action_id'])['status']=='completed'
print('工具提交后强制退出，独立进程检查点恢复与动作查重通过')
'''
    subprocess.run([sys.executable, '-c', recover], cwd=root, env=env, check=True)
