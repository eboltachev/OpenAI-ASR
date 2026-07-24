from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline

from app.core.config import Settings
from app.domain.models import SpeakerTurn, TimeSpan

logger = logging.getLogger(__name__)


class DiarizationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Pipeline | None = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        model = self._resolve_model(self._settings.diarization_model)
        logger.info("loading_diarization_model", extra={"model": model})
        self._pipeline = Pipeline.from_pretrained(model, token=self._settings.hf_token)
        self._pipeline.to(torch.device(self._settings.model_device))

    async def diarize(
        self, path: Path, *, min_speakers: int | None, max_speakers: int | None
    ) -> tuple[list[SpeakerTurn], list[TimeSpan]]:
        self.load()
        assert self._pipeline is not None
        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        output = await asyncio.to_thread(self._pipeline, str(path), **kwargs)
        regular = output.speaker_diarization
        exclusive = getattr(output, "exclusive_speaker_diarization", regular)
        turns = self._annotation_to_turns(exclusive)
        overlap = self._find_overlap(self._annotation_to_turns(regular))
        return turns, overlap

    def _resolve_model(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return str(path)
        mounted = self._settings.models_dir / value
        if mounted.exists():
            return str(mounted)
        return value

    @staticmethod
    def _annotation_to_turns(annotation: Any) -> list[SpeakerTurn]:
        turns: list[SpeakerTurn] = []
        if hasattr(annotation, "itertracks"):
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(
                        speaker=str(speaker),
                        start=float(segment.start),
                        end=float(segment.end),
                        source_spans=[TimeSpan(float(segment.start), float(segment.end))],
                    )
                )
        else:
            for segment, speaker in annotation:
                turns.append(
                    SpeakerTurn(
                        speaker=str(speaker),
                        start=float(segment.start),
                        end=float(segment.end),
                        source_spans=[TimeSpan(float(segment.start), float(segment.end))],
                    )
                )
        return sorted(turns, key=lambda item: (item.start, item.end, item.speaker))

    @staticmethod
    def _find_overlap(turns: list[SpeakerTurn]) -> list[TimeSpan]:
        overlap: list[TimeSpan] = []
        for index, first in enumerate(turns):
            for second in turns[index + 1 :]:
                if second.start >= first.end:
                    break
                if first.speaker == second.speaker:
                    continue
                start = max(first.start, second.start)
                end = min(first.end, second.end)
                if end > start:
                    overlap.append(TimeSpan(start, end))
        return overlap
