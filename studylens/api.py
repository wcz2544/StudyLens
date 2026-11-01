"""带鉴权、限流和有界任务的单工作区知识库API。"""
from collections import deque
import asyncio
from contextlib import contextmanager
from hmac import compare_digest
import logging
from threading import Lock
from time import monotonic
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from studylens.ingestion import extract_upload
from studylens.service_config import Settings, build_services

LOG = logging.getLogger("studylens.service")
MAX_UPLOAD = 12 * 1024 * 1024


async def read_bounded_body(request: Request, limit: int) -> bytes:
    async def collect():
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > limit:
                raise HTTPException(413, "请求内容超过大小限制。")
            raw.extend(chunk)
        return bytes(raw)
    try:
        return await asyncio.wait_for(collect(), timeout=30)
    except asyncio.TimeoutError:
        raise HTTPException(408, "上传超时，请检查网络后重试。") from None


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    consent_external: bool = False


def create_app(settings: Settings | None = None, services=None) -> FastAPI:
    """services参数只用于测试注入；正常入口必须从部署配置建立真实服务。"""
    app = FastAPI(title="StudyLens Knowledge API", version="2.0.0", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.services = services
    app.state.task_lock = Lock()
    app.state.rate_lock = Lock()
    app.state.calls = deque()
    app.state.completed = 0
    app.state.failed = 0

    def config():
        if app.state.settings is None:
            try:
                app.state.settings = Settings.from_env()
            except (ValueError, TypeError):
                raise HTTPException(503, "知识库服务尚未配置，请联系维护者。") from None
        return app.state.settings

    def authenticate(request: Request):
        cfg = config()
        value = request.headers.get("authorization", "")
        token = value[7:] if value.startswith("Bearer ") else ""
        if compare_digest(token.encode("utf-8"), cfg.admin_key.encode("utf-8")):
            return "admin"
        if compare_digest(token.encode("utf-8"), cfg.reader_key.encode("utf-8")):
            return "reader"
        raise HTTPException(401, "访问凭据无效。", headers={"WWW-Authenticate": "Bearer"})

    def authorized(role=Depends(authenticate)):
        # 单工作区共同预算，防止通过不断伪造请求ID绕过限制。
        now = monotonic()
        with app.state.rate_lock:
            calls = app.state.calls
            while calls and now - calls[0] >= 60:
                calls.popleft()
            if len(calls) >= config().requests_per_minute:
                raise HTTPException(429, "请求较多，请一分钟后重试。", headers={"Retry-After": "60"})
            calls.append(now)
        return role

    def administrator(role=Depends(authorized)):
        if role != "admin":
            raise HTTPException(403, "更新资料需要管理凭据。")

    @contextmanager
    def task():
        # 整个问答和版本更新互斥，保证一次回答不会跨越删除或版本切换。
        # 此实现明确要求uvicorn单worker；扩容前需数据库锁/任务队列。
        if not app.state.task_lock.acquire(blocking=False):
            raise HTTPException(429, "知识库正在处理任务，请稍后重试。", headers={"Retry-After": "3"})
        try:
            if app.state.services is None:
                app.state.services = build_services(config())
            yield app.state.services
            app.state.completed += 1
        except HTTPException:
            raise
        except Exception as exc:
            app.state.failed += 1
            # 不记录请求内容、资料和服务商响应，避免密钥或私密笔记进入日志。
            LOG.warning("task_failed error_type=%s", type(exc).__name__)
            raise HTTPException(503, "处理服务暂不可用；请刷新资料列表核对状态后再重试，或联系维护者。") from None
        finally:
            app.state.task_lock.release()

    @app.middleware("http")
    async def envelope(request: Request, call_next):
        request_id = uuid4().hex
        started = monotonic()
        # 对JSON问题也做长度检查；文件入口另按流累计限制，不能只信Content-Length。
        limit = MAX_UPLOAD if request.url.path.startswith("/v1/documents/") else 8192
        size = request.headers.get("content-length")
        if size is not None and (not size.isdecimal() or int(size) > limit):
            return JSONResponse({"detail": "请求内容过大或长度无效。", "request_id": request_id}, status_code=413)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        LOG.info("request id=%s method=%s status=%s ms=%d", request_id, request.method,
                 response.status_code, int((monotonic() - started) * 1000))
        return response

    @app.get("/health")
    def health():
        return {"status": "alive", "version": "2.0.0"}

    @app.get("/v1/status", dependencies=[Depends(authorized)])
    def status():
        with task() as (kb, _):
            docs = kb.list_documents(config().workspace)
            if not kb.vector_store.check_health():
                raise HTTPException(503, "知识库向量服务暂不可用。")
            return {"status": "ready", "documents": len(docs), "completed": app.state.completed,
                    "failed": app.state.failed, "scope": "single_workspace_single_worker"}

    @app.get("/v1/documents", dependencies=[Depends(authorized)])
    def documents():
        with task() as (kb, _):
            return {"documents": kb.list_documents(config().workspace)}

    @app.put("/v1/documents/{source_id}", dependencies=[Depends(administrator)])
    async def upload(request: Request, source_id: str, filename: str):
        validate_source_id(source_id)
        if request.headers.get("x-external-processing") != "true":
            raise HTTPException(400, "请确认资料会发送到配置的Embedding服务。")
        raw = await read_bounded_body(request, MAX_UPLOAD)

        def ingest():
            with task() as (kb, _):
                try:
                    text, metadata = extract_upload(filename, raw)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from None
                result = kb.upsert(config().workspace, source_id, filename, text, metadata)
                return {"result": result, "warnings": metadata.get("warnings", [])}
        return await run_in_threadpool(ingest)

    @app.delete("/v1/documents/{source_id}", dependencies=[Depends(administrator)])
    def delete(source_id: str):
        validate_source_id(source_id)
        with task() as (kb, _):
            return kb.delete(config().workspace, source_id)

    @app.post("/v1/query", dependencies=[Depends(authorized)])
    async def query(request: Request):
        # 先鉴权再读取有上限的请求体，拒绝未声明长度的无限JSON流。
        raw = await read_bounded_body(request, 8192)
        try:
            question = Question.model_validate_json(raw)
        except ValueError:
            raise HTTPException(422, "请输入1到500字符的问题。") from None
        if not question.question.strip():
            raise HTTPException(422, "问题不能为空。")
        if not question.consent_external:
            raise HTTPException(400, "请确认问题和资料片段会发送至配置的模型服务。")

        def answer():
            with task() as (_, agent):
                return agent.answer(config().workspace, question.question.strip())
        return await run_in_threadpool(answer)

    return app


def validate_source_id(source_id: str):
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", source_id):
        raise HTTPException(422, "资料编号仅允许字母、数字、短横线和下划线，最多80字符。")


# 启动不调用模型；健康检查不会暗中产生Embedding费用。
app = create_app()
