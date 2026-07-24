from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.config import Settings
from app.domain.models import JsonDict, SpeakerTurn, TranscriptionOptions
from app.domain.turns import merge_adjacent_same_speaker, padded_turn_bounds
from app.services.alignment import AlignmentService
from app.services.audio import AudioService
from app.services.diarization import DiarizationService
from app.services.embeddings import SpeakerEmbeddingService
from app.services.serialization import build_speaker_profiles
from app.services.upstream import UpstreamASRClient

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio = AudioService(settings.temp_dir, settings.max_upload_bytes)
        self.diarization = DiarizationService(settings)
        self.embeddings = SpeakerEmbeddingService(settings)
        self.alignment = AlignmentService(settings)
        self.upstream = UpstreamASRClient(settings)
        self._local_semaphore = asyncio.Semaphore(settings.local_pipeline_concurrency)
        self.ready = False
        self.initialization_error: str | None = None

    async def initialize(self) -> None:
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.preload_models:
            try:
                await asyncio.to_thread(self.diarization.load)
                await asyncio.to_thread(self.embeddings.load)
            except Exception as error:
                self.initialization_error = str(error)
                logger.exception("model_preload_failed")
                return
        self.ready = True

    async def close(self) -> None:
        await self.upstream.close()

    async def process(self, source: Path, options: TranscriptionOptions) -> JsonDict:
        started = perf_counter()
        normalized: Path | None = None
        async with self._local_semaphore:
            try:
                normalized = await self.audio.normalize(source)
                waveform, sample_rate = await asyncio.to_thread(
                    self.audio.load_waveform, normalized
                )
                audio_duration = waveform.shape[-1] / sample_rate
                raw_turns, overlaps = await self.diarization.diarize(
                    normalized,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                )
                turns = merge_adjacent_same_speaker(
                    raw_turns, max_gap_seconds=options.merge_gap_seconds
                )
                if options.return_speaker_embeddings:
                    await self.embeddings.attach_embeddings(normalized, turns, overlaps)

                segment_results = await asyncio.gather(
                    *[
                        self._process_turn(
                            turn,
                            crop_start=crop_start,
                            crop_end=crop_end,
                            waveform=waveform,
                            sample_rate=sample_rate,
                            options=options,
                        )
                        for index, turn in enumerate(turns)
                        for crop_start, crop_end in [
                            padded_turn_bounds(
                                turns,
                                index,
                                audio_duration=audio_duration,
                                padding_seconds=self.settings.segment_padding_seconds,
                            )
                        ]
                    ]
                )
                segments = sorted(
                    segment_results, key=lambda item: (item["start"], item["end"])
                )
                words = [
                    word for segment in segments for word in segment.get("words", [])
                ]
                languages = list(
                    dict.fromkeys(
                        segment["language"]
                        for segment in segments
                        if segment.get("language")
                    )
                )
                result: JsonDict = {
                    "task": "transcribe",
                    "duration": audio_duration,
                    "language": options.language
                    or (languages[0] if len(languages) == 1 else "multilingual"),
                    "languages": languages,
                    "text": " ".join(
                        segment["text"] for segment in segments if segment["text"]
                    ).strip(),
                    "segments": segments,
                    "words": words,
                    "model": options.model,
                    "processing_seconds": perf_counter() - started,
                }
                if options.return_speaker_embeddings:
                    result["speakers"] = build_speaker_profiles(
                        segments,
                        similarity_floor=self.settings.embedding_similarity_floor,
                        model_name=self.settings.speaker_embedding_model,
                    )
                return result
            finally:
                if normalized is not None:
                    normalized.unlink(missing_ok=True)

    async def _process_turn(
        self,
        turn: SpeakerTurn,
        *,
        crop_start: float,
        crop_end: float,
        waveform: Any,
        sample_rate: int,
        options: TranscriptionOptions,
    ) -> JsonDict:
        crop = self.audio.crop(waveform, sample_rate, crop_start, crop_end)
        wav_bytes = self.audio.to_wav_bytes(crop, sample_rate)
        upstream = await self.upstream.transcribe(
            wav_bytes,
            model=options.model,
            language=options.language,
            prompt=options.prompt,
            temperature=options.temperature,
        )
        language = options.language or upstream.language
        words = await self.alignment.align(
            crop,
            sample_rate=sample_rate,
            text=upstream.text,
            language=language,
            duration=crop_end - crop_start,
        )
        global_words = [
            {
                **word,
                "start": crop_start + float(word["start"]),
                "end": crop_start + float(word["end"]),
                "speaker": turn.speaker,
                "language": language,
            }
            for word in words
        ]
        return {
            "id": f"segment_{int(turn.start * 1000)}_{turn.speaker}",
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.speaker,
            "language": language,
            "text": upstream.text,
            "words": global_words,
            "speaker_embedding": turn.embedding,
            "speaker_embedding_quality": turn.embedding_quality,
            "source_segments": [
                {"start": span.start, "end": span.end} for span in turn.source_spans
            ],
        }
