"""与页面共用应用级会话持久化；未提供登录认证。"""
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.service import SupportService
from src.orchestrator import run_orchestrator

app = FastAPI(title='电商订单与售后助手 API', version='1.1.0')
service = SupportService()

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    user_id: str = Field(default='customer_a', pattern=r'^[A-Za-z0-9_-]{1,40}$')
    mode: str = Field(default='demo', pattern=r'^(demo|online)$')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/chat')
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    try:
        result = service.chat(req.user_id, session_id, req.mode, req.message, runner=run_orchestrator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {'session_id': session_id, 'user_id': req.user_id, 'mode': req.mode,
            **{k: result.get(k) for k in ['reply', 'agent', 'trace', 'usage', 'cached']}}
