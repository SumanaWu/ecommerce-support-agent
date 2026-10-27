"""Local authenticated entrypoint; model tools cannot call confirmation endpoints."""
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from src.auth import AccessError, resolve_principal
from src.service import SupportService
from src.orchestrator import run_orchestrator
from src.requests import RequestConflict

app = FastAPI(title='电商订单与售后助手 API', version='1.3.0')
service = SupportService()
bearer = HTTPBearer(auto_error=False)


def current_principal(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    try:
        if credentials is None:
            raise AccessError()
        return resolve_principal(credentials.credentials)
    except AccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc),
                            headers={'WWW-Authenticate': 'Bearer'} if exc.status_code == 401 else None) from None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    user_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    mode: Literal['demo', 'online'] = 'demo'
    request_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')


def response_for(fn):
    try:
        result = fn()
    except AccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except RequestConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    code = 202 if result.get('status') == 'in_progress' else 200
    if result.get('memory_status') == 'conflict':
        code = 409
    if any(isinstance(entry.get('result'), dict) and entry['result'].get('error_code') in {'cancelled', 'already_executed'} for entry in result.get('trace', [])):
        code = 409
    return JSONResponse(status_code=code, content=result)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/chat', responses={202: {'description': '同一请求处理中'}, 409: {'description': '请求编号与内容冲突'}})
def chat(req: ChatRequest, principal=Depends(current_principal)):
    if req.user_id is not None and req.user_id != principal.customer_id:
        raise HTTPException(status_code=403, detail='客户编号与已认证身份不一致。')
    return response_for(lambda: service.chat(principal, req.session_id, req.mode, req.message,
                                             runner=run_orchestrator, request_id=req.request_id))


@app.get('/actions')
def pending_actions(session_id: str = Query(default='case_1', pattern=r'^[A-Za-z0-9_-]{1,40}$'),
                    mode: Literal['demo', 'online'] = 'demo', principal=Depends(current_principal)):
    return response_for(lambda: {'actions': service.list_pending_actions(principal, session_id, mode)})


@app.get('/actions/{action_id}')
def action(action_id: str, principal=Depends(current_principal)):
    return response_for(lambda: service.get_action(principal, action_id))


@app.post('/actions/{action_id}/confirm')
def confirm(action_id: str, body: ActionDecision | None = None, principal=Depends(current_principal)):
    return response_for(lambda: service.confirm_action(principal, action_id))


@app.post('/actions/{action_id}/cancel')
def cancel(action_id: str, body: ActionDecision | None = None, principal=Depends(current_principal)):
    return response_for(lambda: service.cancel_action(principal, action_id))


@app.get('/runs')
def runs(session_id: str = Query(default='case_1', pattern=r'^[A-Za-z0-9_-]{1,40}$'),
         mode: Literal['demo', 'online'] = 'demo', principal=Depends(current_principal)):
    return response_for(lambda: {'runs': service.list_runs(principal, session_id, mode)})


@app.get('/runs/{run_id}')
def run(run_id: str, principal=Depends(current_principal)):
    return response_for(lambda: service.get_run(principal, run_id))


@app.post('/runs/{run_id}/resume')
def resume(run_id: str, body: ActionDecision | None = None, principal=Depends(current_principal)):
    return response_for(lambda: service.resume_run(principal, run_id))
