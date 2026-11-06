"""与页面共用应用级会话持久化；未提供登录认证。"""
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from src.service import ConflictError, SupportService
from src.orchestrator import run_orchestrator

@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        service.close()


app = FastAPI(title='电商订单与售后助手 API', version='1.2.0', lifespan=lifespan)
service = SupportService()

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    user_id: str = Field(default='customer_a', pattern=r'^[A-Za-z0-9_-]{1,40}$')
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r'^[A-Za-z0-9_-]{1,64}$')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/chat')
def chat(req: ChatRequest, response: Response):
    try:
        result = service.chat(req.user_id, req.session_id, req.message, runner=run_orchestrator, request_id=req.request_id)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response.status_code = 202 if result.get('status') == 'processing' else 200
    return result


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str
    session_id: str
    request_id: str
    action_id: str
    version: int = Field(ge=1)
    decision: Literal['confirm', 'cancel']


@app.get('/requests/{request_id}')
def get_request(request_id: str, user_id: str, session_id: str, response: Response):
    try:
        result = service.get_request(user_id, session_id, request_id)
        response.status_code = 202 if result.get('status') == 'processing' else 200
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post('/requests/{chat_request_id}/resume')
def resume(chat_request_id: str, req: ResumeRequest):
    try:
        return service.resume(req.user_id, req.session_id, chat_request_id, req.request_id,
                              req.action_id, req.version, req.decision)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
