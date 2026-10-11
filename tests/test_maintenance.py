"""Phase 7 : production — purge quotidienne, collecte bornée, score du vérificateur dans le log,
tâche planifiée de démarrage."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import write_test_config
from loguru import logger
from test_app_reply import FeedAudio, MarkerVAD

from mikata.app import App, State
from mikata.config import PROJECT_ROOT, load_config
from mikata.maintenance import BACKUPS_DIR, WAKE_LOGS_DIR, run_maintenance, trash_entries_older_than
from mikata.wake.recordings import MISSES_DIR

# Phase 13 : les traces de l'agent v2 sont purgées comme les autres journaux.
TRACES_DIR = "data/agent2/traces"

NOW = 1_000_000_000.0
DAY = 86_400


def _age(path: Path, days: float) -> None:
    os.utime(path, (NOW - days * DAY, NOW - days * DAY))


def test_trash_entries_older_than(tmp_path: Path) -> None:
    old_dir = tmp_path / "20260901-120000" / "rapport.docx"
    old_dir.parent.mkdir()
    old_dir.write_text("x")
    recent = tmp_path / "20261006-120000"
    recent.mkdir()
    _age(old_dir.parent, 20)
    _age(recent, 2)
    trashed: list[str] = []
    assert trash_entries_older_than(tmp_path, 14, now=NOW, trash=trashed.append) == 1
    assert trashed == [str(old_dir.parent)]
    assert trash_entries_older_than(tmp_path / "absent", 14, trash=trashed.append) == 0


def test_run_maintenance_purges_wake_logs_and_backups(tmp_path: Path, monkeypatch: Any) -> None:
    config = load_config(write_test_config(tmp_path / "config.yaml"))
    wav = tmp_path / WAKE_LOGS_DIR / "old_score0.90.wav"
    backup = tmp_path / BACKUPS_DIR / "20260101-000000"
    old_miss = tmp_path / MISSES_DIR / "20260101-000000-000000_rate.wav"
    recent_miss = tmp_path / MISSES_DIR / "20260301-000000-000000_rate.wav"
    for path in (wav, old_miss, recent_miss):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
    backup.mkdir(parents=True)
    for path, age in ((wav, 30 * DAY), (backup, 30 * DAY), (old_miss, 31 * DAY),
                      (recent_miss, 29 * DAY)):  # fmt: skip
        os.utime(path, (path.stat().st_mtime - age, path.stat().st_mtime - age))
    trashed: list[str] = []
    counts = run_maintenance(config, trash=trashed.append)
    # ratés : 30 jours ; les traces de l'agent v2 (phase 13) sont absentes ici.
    assert counts == {WAKE_LOGS_DIR: 1, MISSES_DIR: 1, BACKUPS_DIR: 1, TRACES_DIR: 0}
    assert sorted(trashed) == sorted([str(wav), str(backup), str(old_miss)])


def test_maintenance_runs_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("mikata.app.run_maintenance", lambda cfg: calls.append("purge") or {})
    app = App(load_config(write_test_config(tmp_path / "config.yaml")))

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        while not calls:
            await asyncio.sleep(0.01)
        app.request_shutdown("test")
        await task

    asyncio.run(asyncio.wait_for(scenario(), 5))
    assert calls == ["purge"]


def test_negative_collection_stops_by_itself(tmp_path: Path) -> None:
    """wake.negatives_max_hours : la collecte s'arrête seule (~115 Mo par heure sinon)."""
    config_file = write_test_config(
        tmp_path / "config.yaml",
        wake={"collect_negatives": True, "negatives_max_hours": 15 / 3600},  # 15 s
    )
    audio = FeedAudio()
    app = App(load_config(config_file), audio=audio, vad=MarkerVAD())
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        while app.state is not State.IDLE:
            await asyncio.sleep(0.01)
        audio.say(0, frames=0, silence=260)  # 20,8 s de micro : 2 extraits de 10 s
        while app._collector is not None:
            await asyncio.sleep(0.01)
        app.request_shutdown("test")
        await task

    try:
        asyncio.run(asyncio.wait_for(scenario(), 5))
    finally:
        logger.remove(sink)
    assert app._negatives_saved == 2  # (fichiers : même seconde ici, 10 s d'écart en vrai)
    assert list((tmp_path / "data" / "negatives").rglob("*.wav"))
    assert any("Collecte de la vie courante arrêtée" in m for m in seen)


def test_negative_collection_already_full_at_startup(tmp_path: Path) -> None:
    from mikata.audio.wavfile import write_wav

    for i in range(2):  # 20 s déjà collectées, limite à 15 s
        write_wav(
            tmp_path / "data" / "negatives" / "d" / f"{i}.wav", np.zeros(160, np.int16), 16_000
        )
    config_file = write_test_config(
        tmp_path / "config.yaml",
        wake={"collect_negatives": True, "negatives_max_hours": 15 / 3600},
    )
    app = App(load_config(config_file), audio=FeedAudio(), vad=MarkerVAD())

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        while not app._tasks:  # run() a démarré (compte des extraits fait)
            await asyncio.sleep(0.01)
        app.request_shutdown("test")
        await task

    asyncio.run(asyncio.wait_for(scenario(), 5))
    assert app._collector is None and app._negatives_saved == 2


def test_verifier_score_in_the_detection_log(tmp_path: Path) -> None:
    class Detector:
        last_verifier_score: float | None = None

        def process(self, frame: np.ndarray) -> float:
            hit = int(frame[0]) == 7
            self.last_verifier_score = 0.87 if hit else None
            return 1.0 if hit else 0.0

        def reset(self) -> None:
            pass

    audio = FeedAudio()
    app = App(
        load_config(write_test_config(tmp_path / "config.yaml")), audio=audio, detector=Detector()
    )
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="INFO")

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        while app.state is not State.IDLE:
            await asyncio.sleep(0.01)
        audio.say(7, frames=1, silence=2)
        while app.detections == 0:
            await asyncio.sleep(0.01)
        app.request_shutdown("test")
        await task

    try:
        asyncio.run(asyncio.wait_for(scenario(), 5))
    finally:
        logger.remove(sink)
    assert any("Mot d'activation détecté (score 1.00, voix 0.87)" in m for m in seen)


# -- tâche planifiée -------------------------------------------------------------------------

AUTOSTART = PROJECT_ROOT / "scripts" / "install_autostart.ps1"


def test_autostart_script_is_utf8_with_bom() -> None:
    """Windows PowerShell 5.1 lit un .ps1 sans BOM en ANSI : accents et « » abîmés."""
    raw = AUTOSTART.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")


def test_autostart_script_registers_a_logon_task() -> None:
    script = AUTOSTART.read_text(encoding="utf-8-sig")
    for needed in (
        "Register-ScheduledTask",
        "-AtLogOn",
        "-RunLevel Limited",  # droits normaux, jamais administrateur (§7)
        "-LogonType Interactive",  # session de l'utilisateur, pas la session 0 d'un service
        r".venv\Scripts\pythonw.exe",  # pas de console (§10)
        "-m mikata --supervise",  # relance si MIKATA plante
        "-RestartCount",  # « redémarrer en cas d'échec » (§10)
        "--check-config",
        "-ExecutionTimeLimit ([TimeSpan]::Zero)",  # tourne h24
        "Unregister-ScheduledTask",
        "-MultipleInstances IgnoreNew",
    ):
        assert needed in script, needed
    assert "Highest" not in script and "New-Service" not in script
