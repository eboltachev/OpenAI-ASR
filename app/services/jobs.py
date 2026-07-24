from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.domain.models import JobStatus, TranscriptionOptions

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, source: Path, options: TranscriptionOptions) -> dict[str, Any]:
        job_id = f"transcription_{uuid4().hex}"
        directory = self.root / job_id
        directory.mkdir(parents=True)
        input_path = directory / f"input{source.suffix}"
        source.replace(input_path)
        job = {
            "id": job_id,
            "object": "audio.transcription.job",
            "status": JobStatus.QUEUED,
            "created_at": _now(),
            "updated_at": _now(),
            "input_path": str(input_path),
            "options": asdict(options),
            "error": None,
        }
        self.save(job)
        return job

    def save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = _now()
        path = self.root / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.root / job_id / "job.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def result(self, job_id: str) -> dict[str, Any]:
        path = self.root / job_id / "result.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save_result(self, job_id: str, result: dict[str, Any]) -> None:
        path = self.root / job_id / "result.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def recoverable(self) -> list[str]:
        jobs: list[str] = []
        for path in self.root.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job["status"] = JobStatus.QUEUED
                self.save(job)
                jobs.append(job["id"])
        return jobs


class JobManager:
    def __init__(
        self,
        store: JobStore,
        processor: Callable[[Path, TranscriptionOptions], Awaitable[dict[str, Any]]],
        workers: int,
    ) -> None:
        self.store = store
        self._processor = processor
        self._worker_count = workers
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        for job_id in self.store.recoverable():
            await self._queue.put(job_id)
        self._tasks = [
            asyncio.create_task(self._worker(index))
            for index in range(self._worker_count)
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def submit(
        self, source: Path, options: TranscriptionOptions
    ) -> dict[str, Any]:
        job = self.store.create(source, options)
        await self._queue.put(job["id"])
        return self.public(job)

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self.store.get(job_id)
                if job["status"] == JobStatus.CANCELED:
                    continue
                job["status"] = JobStatus.RUNNING
                job["worker"] = index
                self.store.save(job)
                options = TranscriptionOptions(**job["options"])
                result = await self._processor(Path(job["input_path"]), options)
                self.store.save_result(job_id, result)
                await asyncio.to_thread(
                    Path(job["input_path"]).unlink, missing_ok=True
                )
                job["status"] = JobStatus.SUCCEEDED
                self.store.save(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("async_job_failed", extra={"job_id": job_id})
                job = self.store.get(job_id)
                job["status"] = JobStatus.FAILED
                job["error"] = {"message": str(error), "type": type(error).__name__}
                self.store.save(job)
            finally:
                self._queue.task_done()

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key not in {"input_path", "options", "worker"}
        }
