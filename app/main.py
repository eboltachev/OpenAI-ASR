from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import create_router
from app.core.config import Settings
from app.core.logging import AsyncLogging
from app.core.security import BearerAuth
from app.services.jobs import JobManager, JobStore
from app.services.pipeline import TranscriptionPipeline

settings = Settings.from_env()
async_logging = AsyncLogging(
    level=settings.log_level,
    file_path=settings.log_file,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count,
)
async_logging.start()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = TranscriptionPipeline(settings)
    jobs = JobManager(JobStore(settings.data_dir), pipeline.process, settings.async_job_workers)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.jobs = jobs
    await pipeline.initialize()
    await jobs.start()
    logger.info("service_started", extra={"port": settings.api_port})
    try:
        yield
    finally:
        await jobs.stop()
        await pipeline.close()
        async_logging.stop()


app = FastAPI(
    title="OpenAI ASR",
    version="1.0.0",
    description=(
        "OpenAI-compatible ASR with multilingual diarization, forced alignment and speaker embeddings"
    ),
    lifespan=lifespan,
)
app.include_router(create_router(BearerAuth(settings)))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "invalid_request_error",
                "details": error.errors(),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, error: HTTPException) -> JSONResponse:
    detail = error.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        content = {"error": {"message": str(detail), "type": "invalid_request_error"}}
    return JSONResponse(status_code=error.status_code, content=content, headers=error.headers)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, error: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(error), "type": "server_error"}},
    )


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid4().hex)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        },
    )
    return response
