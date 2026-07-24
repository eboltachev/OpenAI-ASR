from __future__ import annotations

from app.domain.models import SpeakerTurn, TimeSpan


def merge_adjacent_same_speaker(turns: list[SpeakerTurn], *, max_gap_seconds: float) -> list[SpeakerTurn]:
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


def padded_turn_bounds(
    turns: list[SpeakerTurn],
    index: int,
    *,
    audio_duration: float,
    padding_seconds: float,
) -> tuple[float, float]:
    turn = turns[index]
    left_limit = 0.0
    right_limit = audio_duration

    if index > 0:
        previous = turns[index - 1]
        left_limit = (previous.end + turn.start) / 2.0 if previous.end <= turn.start else turn.start

    if index + 1 < len(turns):
        following = turns[index + 1]
        right_limit = (turn.end + following.start) / 2.0 if turn.end <= following.start else turn.end

    start = max(0.0, left_limit, turn.start - padding_seconds)
    end = min(audio_duration, right_limit, turn.end + padding_seconds)
    return start, max(start, end)


def intersection_duration(first: TimeSpan, second: TimeSpan) -> float:
    return max(0.0, min(first.end, second.end) - max(first.start, second.start))
