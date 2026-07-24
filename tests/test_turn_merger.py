from app.domain.models import SpeakerTurn
from app.domain.turns import merge_adjacent_same_speaker, padded_turn_bounds


def test_merges_only_adjacent_same_speaker_with_small_gap() -> None:
    turns = [
        SpeakerTurn("A", 0.0, 2.0),
        SpeakerTurn("A", 2.4, 4.0),
        SpeakerTurn("B", 4.2, 5.0),
        SpeakerTurn("A", 5.1, 6.0),
    ]
    merged = merge_adjacent_same_speaker(turns, max_gap_seconds=0.5)
    assert [(item.speaker, item.start, item.end) for item in merged] == [
        ("A", 0.0, 4.0),
        ("B", 4.2, 5.0),
        ("A", 5.1, 6.0),
    ]
    assert len(merged[0].source_spans) == 2


def test_padding_uses_silence_midpoints_without_crossing_speakers() -> None:
    turns = [
        SpeakerTurn("A", 1.0, 3.0),
        SpeakerTurn("B", 3.2, 5.0),
    ]
    first = padded_turn_bounds(
        turns, 0, audio_duration=8.0, padding_seconds=0.5
    )
    second = padded_turn_bounds(
        turns, 1, audio_duration=8.0, padding_seconds=0.5
    )
    assert first == (0.5, 3.1)
    assert second == (3.1, 5.5)
