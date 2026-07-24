from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float
    source_spans: list[TimeSpan] = field(default_factory=list)
    embedding: list[float] | None = None
    embedding_quality: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    model: str
    language: str | None
    prompt: str | None
    temperature: float
    response_format: str
    min_speakers: int | None
    max_speakers: int | None
    return_speaker_embeddings: bool
    merge_gap_seconds: float


JsonDict = dict[str, Any]
