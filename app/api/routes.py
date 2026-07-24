from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.security import BearerAuth
from app.domain.languages import normalize_language
from app.domain.models import JobStatus, TranscriptionOptions
from app.services.serialization import to_srt, to_vtt


def create_router(auth: BearerAuth) -> APIRouter:
    router = APIRouter()

    @router.get("/health/live", dependencies=[])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/health/ready", dependencies=[])
    async def ready(request: Request) -> JSONResponse:
        pipeline = request.app.state.pipeline
        if pipeline.ready:
            return JSONResponse({"status": "ok"})
        return JSONResponse(
            {"status": "not_ready", "error": pipeline.initialization_error}, status_code=503
        )

    @router.get("/v1/models", dependencies=[Depends(auth)])
    async def models(request: Request) -> dict[str, Any]:
        try:
            return await request.app.state.pipeline.upstream.list_models()
        except Exception:
            settings = request.app.state.settings
            return {
                "object": "list",
                "data": [
                    {
                        "id": settings.default_asr_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "upstream",
                    }
                ],
            }

    async def parse_and_save(
        request: Request,
        file: UploadFile,
        model: str | None,
        language: str | None,
        prompt: str | None,
        temperature: float,
        response_format: str,
        min_speakers: int | None,
        max_speakers: int | None,
        return_speaker_embeddings: bool,
        merge_same_speaker_gap: float | None,
    ) -> tuple[Path, TranscriptionOptions]:
        settings = request.app.state.settings
        source = await request.app.state.pipeline.audio.save_upload(file)
        options = TranscriptionOptions(
            model=model or settings.default_asr_model,
            language=normalize_language(language),
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
            merge_gap_seconds=(
                merge_same_speaker_gap
                if merge_same_speaker_gap is not None
                else settings.same_speaker_merge_gap_seconds
            ),
        )
        if min_speakers and max_speakers and min_speakers > max_speakers:
            source.unlink(missing_ok=True)
            raise HTTPException(422, "min_speakers cannot exceed max_speakers")
        return source, options

    @router.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
    async def transcriptions(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        temperature: Annotated[float, Form()] = 0.0,
        response_format: Annotated[str, Form()] = "diarized_json",
        async_mode: Annotated[bool, Form(alias="async")] = False,
        min_speakers: Annotated[int | None, Form()] = None,
        max_speakers: Annotated[int | None, Form()] = None,
        return_speaker_embeddings: Annotated[bool, Form()] = True,
        merge_same_speaker_gap: Annotated[float | None, Form()] = None,
        timestamp_granularities: Annotated[
            list[str] | None, Form(alias="timestamp_granularities[]")
        ] = None,
    ) -> Any:
        source, options = await parse_and_save(
            request,
            file,
            model,
            language,
            prompt,
            temperature,
            response_format,
            min_speakers,
            max_speakers,
            return_speaker_embeddings,
            merge_same_speaker_gap,
        )
        if async_mode:
            job = await request.app.state.jobs.submit(source, options)
            job["status_url"] = f"/v1/audio/transcriptions/jobs/{job['id']}"
            job["result_url"] = f"/v1/audio/transcriptions/jobs/{job['id']}/result"
            return JSONResponse(job, status_code=status.HTTP_202_ACCEPTED)
        try:
            result = await request.app.state.pipeline.process(source, options)
            return render_result(result, response_format)
        finally:
            source.unlink(missing_ok=True)

    @router.post("/v1/audio/transcriptions/jobs", dependencies=[Depends(auth)])
    async def create_job(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        temperature: Annotated[float, Form()] = 0.0,
        response_format: Annotated[str, Form()] = "diarized_json",
        min_speakers: Annotated[int | None, Form()] = None,
        max_speakers: Annotated[int | None, Form()] = None,
        return_speaker_embeddings: Annotated[bool, Form()] = True,
        merge_same_speaker_gap: Annotated[float | None, Form()] = None,
        timestamp_granularities: Annotated[
            list[str] | None, Form(alias="timestamp_granularities[]")
        ] = None,
    ) -> JSONResponse:
        source, options = await parse_and_save(
            request,
            file,
            model,
            language,
            prompt,
            temperature,
            response_format,
            min_speakers,
            max_speakers,
            return_speaker_embeddings,
            merge_same_speaker_gap,
        )
        job = await request.app.state.jobs.submit(source, options)
        job["status_url"] = f"/v1/audio/transcriptions/jobs/{job['id']}"
        job["result_url"] = f"/v1/audio/transcriptions/jobs/{job['id']}/result"
        return JSONResponse(job, status_code=202)

    @router.get("/v1/audio/transcriptions/jobs/{job_id}", dependencies=[Depends(auth)])
    async def get_job(request: Request, job_id: str) -> dict[str, Any]:
        try:
            return request.app.state.jobs.public(request.app.state.jobs.store.get(job_id))
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @router.get("/v1/audio/transcriptions/jobs/{job_id}/result", dependencies=[Depends(auth)])
    async def get_job_result(request: Request, job_id: str) -> Any:
        try:
            job = request.app.state.jobs.store.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error
        if job["status"] != JobStatus.SUCCEEDED:
            return JSONResponse(request.app.state.jobs.public(job), status_code=202)
        result = request.app.state.jobs.store.result(job_id)
        return render_result(result, job["options"]["response_format"])

    return router


def render_result(result: dict[str, Any], response_format: str) -> Any:
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "srt":
        return PlainTextResponse(to_srt(result["segments"]), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result["segments"]), media_type="text/vtt")
    if response_format == "json":
        return {"text": result["text"]}
    if response_format not in {"verbose_json", "diarized_json"}:
        raise HTTPException(400, f"Unsupported response_format: {response_format}")
    return result
