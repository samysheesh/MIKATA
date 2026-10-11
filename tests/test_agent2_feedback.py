"""Agent v2 (phase 13d) : les corrections de l'utilisateur sont entendues, ou ignorées
hors délai."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from mikata.agent2.learn.feedback import detect
from mikata.agent2.trace import TraceStore


def _last(tmp_path: Path, minutes_ago: float = 0.2) -> object:
    store = TraceStore(tmp_path / "traces")
    trace = store.new("lance le dernier opening de One Piece")
    trace.outcome = "ok"
    trace.started = trace.started - timedelta(minutes=minutes_ago)
    return trace


def test_each_pattern_is_heard(tmp_path: Path) -> None:
    now = datetime.now()
    cases = [
        ("non, sur YouTube", "channel", "YouTube"),
        ("pas sur Spotify, sur YouTube", "channel", "YouTube"),
        ("pas celle-là", "wrong", None),
        ("c'est pas ça", "wrong", None),
        ("je voulais le générique", "item", "le générique"),
    ]
    for command, kind, value in cases:
        correction = detect(command, _last(tmp_path), now)
        assert correction is not None, command
        assert correction.kind == kind and correction.value == value


def test_outside_the_60s_window_nothing_is_heard(tmp_path: Path) -> None:
    correction = detect("non, sur YouTube", _last(tmp_path, minutes_ago=2), datetime.now())
    assert correction is None


def test_a_failed_or_missing_last_trace_is_ignored(tmp_path: Path) -> None:
    trace = _last(tmp_path)
    trace.outcome = "failed"  # type: ignore[misc]
    assert detect("non, sur YouTube", trace, datetime.now()) is None
    assert detect("non, sur YouTube", None, datetime.now()) is None
