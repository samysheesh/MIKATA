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
        _wait_until(lambda: player._effect is None)
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


def test_new_phrase_replaces_current(fake_sd: types.ModuleType) -> None:
    async def scenario() -> None:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        first = asyncio.create_task(player.play(np.full(48_000 * 10, 0.1, np.float32), 48_000))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(player.play(np.full(480, 0.3, np.float32), 48_000))
        await asyncio.wait_for(first, 1)  # la phrase précédente est coupée
        await asyncio.wait_for(second, 1)
        player.close()

    asyncio.run(scenario())


def test_effect_is_mixed_over_speech_without_cutting_it(fake_sd: types.ModuleType) -> None:
    """Le bip d'une nouvelle commande ne coupe pas la réponse de la précédente."""

    async def scenario() -> None:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        speech = asyncio.create_task(player.play(np.full(9600, 0.25, np.float32), 48_000))
        await asyncio.sleep(0.01)
        player.play_nowait(np.full(960, 0.5, np.float32))
        await asyncio.wait_for(speech, 2)
        player.close()

    asyncio.run(scenario())
    out = FakeOutputStream.instances[-1].output()[:, 0]
    assert int(np.isclose(out, 0.25).sum()) == 9600 - 960  # la parole entière a été jouée
    assert int(np.isclose(out, 0.75).sum()) == 960  # bip additionné par-dessus


def test_stop_cuts_speech_but_not_effects(fake_sd: types.ModuleType) -> None:
    async def scenario() -> AudioPlayer:
        player = AudioPlayer(HEADSET)
        player.start(asyncio.get_running_loop())
        speech = asyncio.create_task(player.play(np.full(48_000, 0.1, np.float32), 48_000))
        player.play_nowait(np.full(48_000, 0.2, np.float32))
        await asyncio.sleep(0.01)
        player.stop()
        await asyncio.wait_for(speech, 1)
        assert player._speech is None and player._effect is not None
        player.close()
        return player

    asyncio.run(scenario())


def test_switch_reopens_the_stream_on_the_new_output(fake_sd: types.ModuleType) -> None:
    """Phase 15b : la sortie change sans redémarrer, au taux du nouveau périphérique."""
    speakers = DeviceInfo(5, "Haut-parleurs (Realtek)", "Windows WASAPI", 0, 8, 44_100)

    async def scenario() -> AudioPlayer:
        player = AudioPlayer(HEADSET)
        loop = asyncio.get_running_loop()
        player.start(loop)
        player.switch(speakers, loop)
        player.play_nowait(np.full(100, 0.5, dtype=np.float32))
        _wait_until(lambda: player._effect is None)
        player.close()
        return player

    player = asyncio.run(scenario())
    old, new = FakeOutputStream.instances[-2:]
    assert old.closed and old.kwargs["device"] == 2
    assert new.kwargs["device"] == 5 and new.kwargs["samplerate"] == 44_100
    assert new.kwargs["channels"] == 2 and player.sample_rate == 44_100
    assert int((new.output()[:, 0] == 0.5).sum()) == 100


def test_switch_falls_back_to_the_previous_output(fake_sd: types.ModuleType) -> None:
    """Un périphérique qui refuse de s'ouvrir ne laisse jamais MIKATA muet."""
    import pytest

    broken = DeviceInfo(9, "Casque débranché", "Windows WASAPI", 0, 2, 48_000)
    real = fake_sd.OutputStream

    def output_stream(**kwargs: object) -> object:
        if kwargs["device"] == 9:
            raise RuntimeError("périphérique indisponible")
        return real(**kwargs)

    fake_sd.OutputStream = output_stream  # type: ignore[attr-defined]

    async def scenario() -> AudioPlayer:
        player = AudioPlayer(HEADSET)
        loop = asyncio.get_running_loop()
        player.start(loop)
        with pytest.raises(RuntimeError):
            player.switch(broken, loop)
        return player

    player = asyncio.run(scenario())
    assert player.device is HEADSET and player._stream is not None
    assert FakeOutputStream.instances[-1].kwargs["device"] == 2
    player.close()
