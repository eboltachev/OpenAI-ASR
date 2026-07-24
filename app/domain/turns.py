from __future__ import annotations

from app.domain.models import SpeakerTurn, TimeSpan


def merge_adjacent_same_speaker(
    turns: list[SpeakerTurn], *, max_gap_seconds: float
) -> list[SpeakerTurn]:
    if not turns:
        return []
    ordered = sorted(turns, key=lambda item: (item.start, item.end, item.speaker))
    merged: list[SpeakerTurn] = []
    for turn in ordered:
        source_spans = turn.source_spans or [TimeSpan(turn.start, turn.end)]
        current = SpeakerTurn(
            speaker=turn.speaker,
            start=turn.start,
            end=turn.end,
            source_spans=list(source_spans),
            embedding=turn.embedding,
            embedding_quality=turn.embedding_quality,
        )
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        gap = current.start - previous.end
        if previous.speaker == current.speaker and gap <= max_gap_seconds:
            previous.end = max(previous.end, current.end)
            previous.source_spans.extend(current.source_spans)
        else:
            merged.append(current)
    return merged


def intersection_duration(first: TimeSpan, second: TimeSpan) -> float:
    return max(0.0, min(first.end, second.end) - max(first.start, second.start))
