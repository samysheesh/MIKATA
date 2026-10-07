from __future__ import annotations

import asyncio
import time
import types

import numpy as np
from conftest import FakeOutputStream

from mikata.audio.devices import DeviceInfo
from mikata.audio.player import AudioPlayer
from mikata.tts.base import Player

HEADSET = DeviceInfo(2, "Casque (HyperX Cloud)", "Windows WASAPI", 0, 2, 48_000)


def _wait_until(predicate: object, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():  # type: ignore[operator]
            return
        time.sleep(0.005)
    raise AssertionError("délai dépassé")


def test_is_a_player() -> None:
    assert isinstance(AudioPlayer(HEADSET), Player)


def test_play_nowait_outputs_sound_then_silence(fake_sd: types.ModuleType) -> None:
    async def scenario() -> AudioPlayer:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        assert player.output_latency_s == FakeOutputStream.latency
        player.play_nowait(np.full(1000, 0.5, dtype=np.float32))
        _wait_until(lambda: player._current is None)
        await asyncio.sleep(0.02)
        player.close()
        return player

    player = asyncio.run(scenario())
    stream = FakeOutputStream.instances[-1]
    assert stream.kwargs["latency"] == "low"
    assert stream.kwargs["samplerate"] == 48_000
    assert stream.closed
    out = stream.output()
    assert out.shape[1] == 2
    assert not np.isnan(out).any()  # chaque bloc entièrement écrit
    assert int((out[:, 0] == 0.5).sum()) == 1000
    assert np.array_equal(out[:, 0], out[:, 1])
    assert player.output_latency_s == 0.0


def test_play_waits_for_end_and_resamples(fake_sd: types.ModuleType) -> None:
    async def scenario() -> None:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        await asyncio.wait_for(player.play(np.full(1600, 0.25, np.float32), 16_000), 2)
        player.close()

    asyncio.run(scenario())
    out = FakeOutputStream.instances[-1].output()[:, 0]
    assert 4700 < int((np.abs(out - 0.25) < 0.05).sum()) <= 4800  # 1600 éch. à 16 k -> 48 k


def test_stop_interrupts_play(fake_sd: types.ModuleType) -> None:
    async def scenario() -> float:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        task = asyncio.create_task(player.play(np.full(48_000 * 10, 0.1, np.float32), 48_000))
        await asyncio.sleep(0.05)
        start = time.monotonic()
        player.stop()
        await asyncio.wait_for(task, 1)
        elapsed = time.monotonic() - start
        player.close()
        return elapsed

    assert asyncio.run(scenario()) < 0.2


def test_new_sound_replaces_current(fake_sd: types.ModuleType) -> None:
    async def scenario() -> None:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        first = asyncio.create_task(player.play(np.full(48_000 * 10, 0.1, np.float32), 48_000))
        await asyncio.sleep(0.02)
        player.play_nowait(np.full(10, 0.9, np.float32))
        await asyncio.wait_for(first, 1)  # l'ancien son est terminé (coupé)
        player.close()

    asyncio.run(scenario())
