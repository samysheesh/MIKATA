"""Agent v2 (phase 13a) : les traces s'écrivent, se relisent et se purgent à la corbeille."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mikata.agent2.trace import TraceStore


def _store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "traces")


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    trace = store.new("lance le dernier opening de One Piece")
    trace.outcome = "ok"
    trace.verified = True
    path = store.save(trace)
    assert path.parent.name == trace.started.strftime("%Y-%m-%d")
    assert path.name == f"{trace.id}.json"
    reloaded = store.load(path)
    assert reloaded == trace


def test_recent_filters_verified_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ok = store.new("a")
    ok.verified = True
    store.save(ok)
    failed = store.new("b")
    failed.verified = False
    store.save(failed)
    assert [trace.id for trace in store.recent(30)] == [failed.id, ok.id]
    assert [trace.id for trace in store.recent(30, verified_only=True)] == [ok.id]


def test_recent_ignores_days_older(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = store.new("ancienne")
    old.verified = True
    path = store.save(old)
    old_time = time.time() - 40 * 86_400
    import os

    os.utime(path, (old_time, old_time))
    assert store.recent(30) == []
    assert store.recent(60)  # encore dans la fenêtre


def test_purge_sends_old_days_to_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    old = store.new("ancienne")
    path = store.save(old)
    old_time = time.time() - 40 * 86_400
    import os

    os.utime(path.parent, (old_time, old_time))  # le dossier du jour est vieux
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", trashed.append)
    assert store.purge(30) == 1
    assert trashed and path.parent.name in trashed[0]
