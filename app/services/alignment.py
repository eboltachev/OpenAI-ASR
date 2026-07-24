from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisperx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class AlignmentService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def align(
        self,
        waveform: torch.Tensor,
        *,
        sample_rate: int,
        text: str,
        language: str | None,
        duration: float,
    ) -> list[dict[str, Any]]:
        if not text:
            return []
        if not language:
            return [{"word": text, "start": 0.0, "end": duration, "score": None}]
        try:
            model, metadata = await self._get_model(language)
        except Exception:
            if self._settings.alignment_required:
                raise
            logger.exception("alignment_model_unavailable", extra={"language": language})
            return [{"word": text, "start": 0.0, "end": duration, "score": None}]
        audio = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
        transcript = [{"start": 0.0, "end": duration, "text": text}]
        async with self._inference_lock:
            result = await asyncio.to_thread(
                whisperx.align,
                transcript,
                model,
                metadata,
                audio,
                self._settings.model_device,
                return_char_alignments=False,
                print_progress=False,
            )
        words = result.get("word_segments", [])
        return [
            {
                "word": item.get("word", ""),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", duration)),
                "score": item.get("score"),
            }
            for item in words
        ]

    async def _get_model(self, language: str) -> tuple[torch.nn.Module, dict[str, Any]]:
        language = language.lower()
        if language in self._models:
            return self._models[language]
        async with self._lock:
            if language in self._models:
                return self._models[language]
            model_ref = self._settings.alignment_models.get(language)
            if not model_ref:
                raise ValueError(f"No alignment model configured for language '{language}'")
            resolved = self._resolve_model(model_ref)
            model, metadata = await asyncio.to_thread(
                whisperx.load_align_model,
                language,
                self._settings.model_device,
                model_name=resolved,
                model_dir=str(self._settings.models_dir / "alignment-cache"),
            )
            self._models[language] = (model, metadata)
            return model, metadata

    def _resolve_model(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return str(path)
        mounted = self._settings.models_dir / value
        if mounted.exists():
            return str(mounted)
        return value
