"""Phase 9 : ratés du mot d'activation notés au raccourci clavier (Ctrl+Alt+M).

Si « MIKATA » n'a pas été entendu, le raccourci fait ce qu'il faisait (bip, écoute) et
garde en plus les 4 dernières secondes du tampon dans data/wakeword/misses/, avec un son
distinct et une ligne de log, pour le prochain entraînement.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import edit_config
from test_app import BEEP, RecordingPlayer, ScriptedAudio

from mikata.app import App
from mikata.audio.earcons import synthesize
from mikata.audio.wavfile import float_to_int16, read_wav
from mikata.config import Config, load_config
from mikata.events import WakeDetected
from mikata.wake.recordings import MISS_CLIP_S, MISSES_DIR

MISS = np.full(10, 0.5, dtype=np.float32)
FRAMES = 70  # 70 trames de 80 ms (5,6 s) : plus que les 4 s gardées


def _app(config_file: Path, **wake: object) -> App:
    if wake:
        edit_config(config_file, wake=wake)
    return App(
        load_config(config_file), earcons={"wake": BEEP, "miss": MISS}, player=RecordingPlayer()
    )


def _press_hotkey_after_audio(app: App, frames: int = FRAMES, source: str = "hotkey") -> None:
    """Le micro donne `frames` trames (valeur = index), puis on appuie sur le raccourci."""
    audio = ScriptedAudio(
        frames, on_exhausted=lambda: app.bus.publish(WakeDetected(score=1.0, source=source))
    )
    app._audio = audio

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        for _ in range(300):
            await asyncio.sleep(0.01)
            if app.detections:
                break
        app.request_shutdown("test")
        await asyncio.wait_for(task, 5)  # attend aussi les écritures sur le disque

    asyncio.run(scenario())


def _misses(config_file: Path) -> list[Path]:
    return sorted((config_file.parent / MISSES_DIR).glob("*.wav"))


def test_hotkey_keeps_the_last_four_seconds(config_file: Path, caplog_loguru: list[str]) -> None:
    app = _app(config_file)
    _press_hotkey_after_audio(app)

    assert app.detections == 1 and app.misses == 1  # comportement habituel + raté noté
    (path,) = _misses(config_file)
    pcm, rate = read_wav(path)
    assert rate == 16_000 and len(pcm) == round(MISS_CLIP_S * 16_000)  # 4 s, 16 kHz mono
    frame_ids = float_to_int16(pcm)[::1280]
    assert list(frame_ids) == list(range(FRAMES - 50, FRAMES))  # les 50 dernières trames
    assert any("Raté du mot d'activation noté" in m for m in caplog_loguru)
    assert any(str(path) in m for m in caplog_loguru)
    player = app._player
    assert isinstance(player, RecordingPlayer)
    (effect,) = player.played  # bip d'écoute puis le son distinct, en un seul effet
    assert np.array_equal(effect[: len(BEEP)], BEEP) and np.array_equal(effect[-len(MISS) :], MISS)
    assert np.count_nonzero(effect[len(BEEP) : -len(MISS)]) == 0  # un court silence entre


def test_short_buffer_keeps_what_there_is(config_file: Path) -> None:
    app = _app(config_file)
    _press_hotkey_after_audio(app, frames=10)
    (path,) = _misses(config_file)
    assert len(read_wav(path)[0]) == 10 * 1280


def test_nothing_kept_without_microphone_audio(config_file: Path) -> None:
    app = _app(config_file)
    _press_hotkey_after_audio(app, frames=0)
    assert app.detections == 1 and app.misses == 0 and _misses(config_file) == []
    assert app._player.played == [BEEP]  # type: ignore[union-attr]


def test_can_be_turned_off(config_file: Path) -> None:
    app = _app(config_file, save_misses=False)
    _press_hotkey_after_audio(app)
    assert app.misses == 0 and _misses(config_file) == []
    assert app._player.played == [BEEP]  # type: ignore[union-attr]


def test_a_real_detection_is_not_a_miss(config_file: Path) -> None:
    app = _app(config_file)
    _press_hotkey_after_audio(app, source="wake")
    assert app.detections == 1 and app.misses == 0 and _misses(config_file) == []


def test_old_misses_go_to_the_trash(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", trashed.append)
    folder = config_file.parent / MISSES_DIR
    folder.mkdir(parents=True)
    old, recent = folder / "20260101-000000-000000_rate.wav", folder / "recent_rate.wav"
    for path, days in ((old, 31), (recent, 29)):
        path.write_bytes(b"RIFF")
        then = time.time() - days * 86_400
        os.utime(path, (then, then))
    app = _app(config_file)
    _press_hotkey_after_audio(app)
    # 30 jours (wake.miss_days), corbeille uniquement ; la corbeille simulée ne déplace rien,
    # d'où un ou deux passages (purge au démarrage, puis après l'enregistrement du raté).
    assert set(trashed) == {str(old)}


def test_miss_sound_is_distinct() -> None:
    miss, wake = synthesize("miss"), synthesize("wake")
    assert len(miss) != len(wake) or not np.allclose(miss, wake)
    assert len(miss) / 48_000 < 0.2  # bref : il suit le bip d'écoute


def test_config_defaults() -> None:
    config = Config()
    assert config.wake.save_misses is True and config.wake.miss_days == 30
