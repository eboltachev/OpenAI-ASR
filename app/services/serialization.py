from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.domain.embeddings import WeightedEmbedding, aggregate_embeddings


def build_speaker_profiles(
    segments: list[dict[str, Any]], *, similarity_floor: float, model_name: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[WeightedEmbedding]] = defaultdict(list)
    for segment in segments:
        embedding = segment.get("speaker_embedding")
        if embedding:
            grouped[segment["speaker"]].append(
                WeightedEmbedding(
                    vector=np.asarray(embedding, dtype=np.float32),
                    duration=float(segment["end"] - segment["start"]),
                    quality=float(segment.get("speaker_embedding_quality") or 1.0),
                )
            )
    profiles: list[dict[str, Any]] = []
    for speaker, samples in sorted(grouped.items()):
        centroid, similarities = aggregate_embeddings(
            samples, similarity_floor=similarity_floor
        )
        profiles.append(
            {
                "id": speaker,
                "embedding": centroid.tolist(),
                "embedding_model": model_name,
                "embedding_dimension": int(centroid.size),
                "normalized": True,
                "segments_used": len(samples),
                "mean_similarity": sum(similarities) / len(similarities),
            }
        )
    return profiles


def to_srt(segments: list[dict[str, Any]]) -> str:
    def timestamp(value: float) -> str:
        milliseconds = round(value * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{timestamp(segment['start'])} --> {timestamp(segment['end'])}\n"
            f"[{segment['speaker']}] {segment['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: list[dict[str, Any]]) -> str:
    def timestamp(value: float) -> str:
        milliseconds = round(value * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"

    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            f"{timestamp(segment['start'])} --> {timestamp(segment['end'])}\n"
            f"[{segment['speaker']}] {segment['text']}"
        )
    return "\n\n".join(blocks) + "\n"
