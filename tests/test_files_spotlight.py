"""Backend Spotlight (phase 12d) : parsing de `mdfind`, observateur FSEvents/watchdog et
branchement de FileIndex sur `config.files_index.backend`. Le watchdog et mdfind sont
remplacés par des faux ; aucune dépendance macOS n'est requise."""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from mikata.config import FilesIndexConfig
from mikata.files import index as fi
from mikata.files import spotlight
from mikata.files.spotlight import SpotlightWatcher, spotlight_query, spotlight_snapshot


def _has_trigram() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


requires_trigram = pytest.mark.skipif(not _has_trigram(), reason="FTS5 trigram indisponible")


# -- faux subprocess (mdfind) -----------------------------------------------------------


class FakeMdfindRun:
    """Substitut de subprocess.run : renvoie les chemins donnés, ou lève FileNotFoundError."""

    def __init__(self, results: list[str] | None, returncode: int = 0) -> None:
        self.results = results
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def run(self, args: list[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append(args)
        if self.results is None:
            raise FileNotFoundError("mdfind")
        return SimpleNamespace(returncode=self.returncode, stdout="\n".join(self.results) + "\n")


# -- faux watchdog (FSEvents) -----------------------------------------------------------


class FakeFileSystemEventHandler:
    """Substitut de watchdog.events.FileSystemEventHandler : base des handlers."""


class FakeObserver:
    """Substitut de watchdog.observers.Observer : enregistre ce qui serait surveillé."""

    instances: ClassVar[list[FakeObserver]] = []

    def __init__(self) -> None:
        self.scheduled: list[tuple[Any, str]] = []
        self.daemon = False
        self.started = False
        self.stopped = False
        self.join_timeout: float | None = None
        FakeObserver.instances.append(self)

    def schedule(self, handler: Any, root: str, recursive: bool) -> None:
        self.scheduled.append((handler, root))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float) -> None:
        self.join_timeout = timeout


@pytest.fixture
def fake_watchdog(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    watchdog = types.ModuleType("watchdog")
    events_mod = types.ModuleType("watchdog.events")
    events_mod.FileSystemEventHandler = FakeFileSystemEventHandler  # type: ignore[attr-defined]
    observers_mod = types.ModuleType("watchdog.observers")
    observers_mod.Observer = FakeObserver  # type: ignore[attr-defined]
    watchdog.events = events_mod  # type: ignore[attr-defined]
    watchdog.observers = observers_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "watchdog", watchdog)
    monkeypatch.setitem(sys.modules, "watchdog.events", events_mod)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers_mod)
    FakeObserver.instances = []
    return events_mod


# -- spotlight_query --------------------------------------------------------------------


def test_spotlight_query_runs_mdfind_and_parses_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMdfindRun(["/Users/a/fic.txt", "", "/Users/b/", "notes.pdf"])
    monkeypatch.setattr(spotlight, "subprocess", SimpleNamespace(run=fake.run))
    monkeypatch.setattr(spotlight.sys, "platform", "darwin")
    assert spotlight_query("fic", max_results=2) == ["/Users/a/fic.txt", "/Users/b/"]
    # lignes vides retirées, limite appliquée
    assert fake.calls == [["mdfind", "-interpret", "fic"]]


def test_spotlight_query_strips_and_skips_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMdfindRun([])
    monkeypatch.setattr(spotlight, "subprocess", SimpleNamespace(run=fake.run))
    monkeypatch.setattr(spotlight.sys, "platform", "darwin")
    assert spotlight_query("   ") == []
    assert fake.calls == []  # mdfind jamais lancé pour une requête vide


def test_spotlight_query_nonzero_returncode_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMdfindRun(["/Users/a/fic.txt"], returncode=1)
    monkeypatch.setattr(spotlight, "subprocess", SimpleNamespace(run=fake.run))
    monkeypatch.setattr(spotlight.sys, "platform", "darwin")
    assert spotlight_query("fic") == []


def test_spotlight_query_raises_outside_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotlight.sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="Spotlight"):
        spotlight_query("fic")


def test_spotlight_query_missing_mdfind_raises_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeMdfindRun(None)
    monkeypatch.setattr(spotlight, "subprocess", SimpleNamespace(run=fake.run))
    monkeypatch.setattr(spotlight.sys, "platform", "darwin")
    with pytest.raises(NotImplementedError, match="mdfind introuvable"):
        spotlight_query("fic")


def test_spotlight_snapshot_pairs_paths_with_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "fic.txt"
    target.write_text("x", encoding="utf-8")
    fake = FakeMdfindRun([str(target)])
    monkeypatch.setattr(spotlight, "subprocess", SimpleNamespace(run=fake.run))
    monkeypatch.setattr(spotlight.sys, "platform", "darwin")
    rows = spotlight_snapshot()
    assert len(rows) == 1 and rows[0][0] == str(target)
    assert rows[0][1] == target.stat().st_mtime


# -- SpotlightWatcher -------------------------------------------------------------------


def test_watcher_schedules_roots_and_dispatches_events(fake_watchdog: types.ModuleType) -> None:
    seen: list[str] = []
    watcher = SpotlightWatcher(["/Users", "/Applications"], on_change=seen.append)
    watcher.start()
    observer = FakeObserver.instances[-1]
    assert observer.daemon and observer.started
    assert [root for _, root in observer.scheduled] == ["/Users", "/Applications"]
    handler = observer.scheduled[0][0]
    handler.on_any_event(SimpleNamespace(is_directory=False, src_path="/Users/x/fic.txt"))
    assert seen == ["/Users/x/fic.txt"]  # les dossiers ne remontent pas
    handler.on_any_event(SimpleNamespace(is_directory=True, src_path="/Users/x"))
    assert seen == ["/Users/x/fic.txt"]
    watcher.stop()
    assert observer.stopped and observer.join_timeout == 2
    watcher.stop()  # idempotent : rien à arrêter


def test_watcher_without_watchdog_logs_and_does_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]
) -> None:
    monkeypatch.setitem(sys.modules, "watchdog", None)  # import forcé à échouer
    watcher = SpotlightWatcher(["/Users"])
    watcher.start()
    watcher.stop()
    assert any("watchdog non installé" in message for message in caplog_loguru)


# -- FileIndex : branchement sur config.files_index.backend -----------------------------


class FakeSpotlightWatcher:
    instances: ClassVar[list[FakeSpotlightWatcher]] = []

    def __init__(self, roots: list[str], on_change: Any = None) -> None:
        self.roots = roots
        self.started = False
        FakeSpotlightWatcher.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def _force_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Le moteur est choisi par la propriété de config, pas par sys.platform."""
    monkeypatch.setattr(FilesIndexConfig, "backend", property(lambda self: backend))


def _raise_assertion(message: str) -> Any:
    """Fabrique un faux qui échoue s'il est appelé (moteur qui ne doit pas servir)."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(message)

    return boom


@requires_trigram
def test_file_index_dispatch_to_spotlight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """backend « spotlight » : construction via mdfind, recherche directe, FSEvents armé —
    sans jamais toucher à l'USN, quelle que soit la plateforme de test."""
    from tool_fakes import make_config

    monkeypatch.setattr(spotlight, "spotlight_snapshot", lambda: [("/Users/x/fic.txt", 5.0)])
    monkeypatch.setattr(
        spotlight, "spotlight_query", lambda query, max_results: ["/Users/x/fic.txt"]
    )
    monkeypatch.setattr(spotlight, "SpotlightWatcher", FakeSpotlightWatcher)
    FakeSpotlightWatcher.instances = []
    monkeypatch.setattr(
        "mikata.files.index.usn.can_read_volumes",
        _raise_assertion("USN ne doit pas être consulté"),
    )

    _force_backend(monkeypatch, "spotlight")
    service = fi.FileIndex(make_config(tmp_path))
    assert service._backend == "spotlight"
    conn = sqlite3.connect(":memory:")
    service._ensure_built(conn)
    assert service._source == "spotlight" and service.ready()
    assert service.stats()["entries"] == 1
    row = conn.execute("SELECT path, name FROM files").fetchone()
    assert row == ("/Users/x/fic.txt", "fic.txt")
    # Recherche : Spotlight en direct, jamais le cache SQLite.
    assert service.search("fic", 10) == [
        {"path": "/Users/x/fic.txt", "name": "fic.txt", "is_dir": False, "mtime": 0.0}
    ]
    # L'observateur FSEvents est armé sur les racines utilisateur.
    assert FakeSpotlightWatcher.instances[0].started
    assert FakeSpotlightWatcher.instances[0].roots == ["/Users", "/Applications"]
    service._poll_once(conn)  # no-op pour Spotlight (le direct vient de FSEvents)
    service.stop()
    conn.close()


@requires_trigram
def test_file_index_dispatch_to_usn_keeps_helper_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """backend « usn » : sans accès direct aux volumes, le service attend le dump du
    helper élevé — Spotlight n'est jamais appelé, même sur une plateforme darwin."""
    from tool_fakes import make_config

    monkeypatch.setattr("mikata.files.index.usn.can_read_volumes", lambda: False)
    monkeypatch.setattr(
        spotlight,
        "spotlight_snapshot",
        _raise_assertion("Spotlight ne doit pas être appelé"),
    )
    _force_backend(monkeypatch, "usn")
    service = fi.FileIndex(make_config(tmp_path))
    assert service._backend == "usn"
    conn = sqlite3.connect(":memory:")
    service._ensure_built(conn)
    assert service._source == "helper"
    assert not service.ready()  # pas de dump produit par le helper en test
    conn.close()
