from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.core import Segment

from app.core.config import Settings
from app.domain.embeddings import WeightedEmbedding, aggregate_embeddings
from app.domain.models import SpeakerTurn, TimeSpan
from app.domain.turns import intersection_duration

logger = logging.getLogger(__name__)


class SpeakerEmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._inference: Inference | None = None

    def load(self) -> None:
        if self._inference is not None:
            return
        model_ref = self._resolve_model(self._settings.speaker_embedding_model)
        logger.info("loading_speaker_embedding_model", extra={"model": model_ref})
        model = Model.from_pretrained(model_ref, token=self._settings.hf_token)
        self._inference = Inference(model, window="whole")
        self._inference.to(torch.device(self._settings.model_device))

    async def attach_embeddings(
        self,
        path: Path,
        merged_turns: list[SpeakerTurn],
        overlaps: list[TimeSpan],
    ) -> None:
        self.load()
        for turn in merged_turns:
            samples: list[WeightedEmbedding] = []
            for span in turn.source_spans:
                overlap_duration = sum(intersection_duration(span, item) for item in overlaps)
                overlap_ratio = overlap_duration / span.duration if span.duration else 1.0
                if span.duration < self._settings.min_embedding_segment_seconds:
                    continue
                if overlap_ratio > self._settings.max_embedding_overlap_ratio:
                    continue
                for window in self._windows(span):
                    vector = await asyncio.to_thread(self._embed, path, window)
                    if vector is not None:
                        samples.append(
                            WeightedEmbedding(
                                vector=vector,
                                duration=window.duration,
                                quality=max(0.0, 1.0 - overlap_ratio),
                            )
                        )
            if not samples:
                fallback = TimeSpan(turn.start, turn.end)
                vector = await asyncio.to_thread(self._embed, path, fallback)
                if vector is not None:
                    samples.append(WeightedEmbedding(vector=vector, duration=fallback.duration))
            if samples:
                centroid, similarities = aggregate_embeddings(
                    samples, similarity_floor=self._settings.embedding_similarity_floor
                )
                turn.embedding = centroid.tolist()
                turn.embedding_quality = float(np.mean(similarities))

    def _embed(self, path: Path, span: TimeSpan) -> np.ndarray | None:
        assert self._inference is not None
        try:
            value = self._inference.crop(str(path), Segment(span.start, span.end))
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            return vector if vector.size else None
        except Exception:
            logger.exception("speaker_embedding_failed", extra={"start": span.start, "end": span.end})
            return None

    def _windows(self, span: TimeSpan) -> list[TimeSpan]:
        maximum = self._settings.max_embedding_segment_seconds
        if span.duration <= maximum:
            return [span]
        windows: list[TimeSpan] = []
        start = span.start
        while start < span.end:
            end = min(span.end, start + maximum)
            if end - start >= self._settings.min_embedding_segment_seconds:
                windows.append(TimeSpan(start, end))
            start = end
        return windows

    def _resolve_model(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return str(path)
        mounted = self._settings.models_dir / value
        if mounted.exists():
            return str(mounted)
        return value
