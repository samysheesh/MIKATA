"""File de phrases : synthèse de N+1 pendant la lecture de N, flux joués dans l'ordre."""

from __future__ import annotations

import asyncio
import itertools
import time

import numpy as np

from mikata.audio.player import Speaker


class SlowTTS:
    """Synthèse en `delay` s ; l'audio vaut l'index de la phrase. Enregistre les horaires."""

    sample_rate = 16_000

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.started: list[tuple[str, float]] = []
        self.loaded = self.closed = False
        self._lock = asyncio.Lock()

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    async def synthesize(self, text: str) -> np.ndarray:
        async with self._lock:  # un seul worker, comme l'executor TTS
            self.started.append((text, time.monotonic()))
            await asyncio.sleep(self.delay)
        return np.full(160, float(len(self.started)), dtype=np.float32)


class RecordingPlayer:
    output_latency_s = 0.01

    def __init__(self, duration: float = 0.03) -> None:
        self.duration = duration
        self.played: list[tuple[float, int, float, float]] = []  # (valeur, sr, début, fin)
        self.stops = 0
        self._current: asyncio.Future[None] | None = None

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        start = time.monotonic()
        loop = asyncio.get_running_loop()
        self._current = loop.create_future()
        loop.call_later(self.duration, lambda f=self._current: f.done() or f.set_result(None))
        await self._current
        self.played.append((float(pcm[0]), sample_rate, start, time.monotonic()))

    def stop(self) -> None:
        self.stops += 1
        if self._current is not None and not self._current.done():
            self._current.set_result(None)


class Cache:
    sample_rate = 22_050

    def __init__(self) -> None:
        self.warmed = False

    def warm(self) -> None:
        self.warmed = True

    def lookup(self, text: str) -> np.ndarray | None:
        return np.full(10, 99.0, dtype=np.float32) if text == "C'est fait." else None


def test_phrases_play_in_order_with_synthesis_ahead() -> None:
    tts, player = SlowTTS(delay=0.02), RecordingPlayer(duration=0.05)

    async def scenario() -> list[float]:
        speaker = Speaker(tts, player)
        stream = speaker.open("#1")
        sounds: list[float] = []
        stream.on_first_sound = lambda: sounds.append(time.monotonic())
        for text in ("Un.", "Deux.", "Trois."):
            stream.say(text)
        stream.close()
        await asyncio.wait_for(stream.wait(), 2)
        await speaker.aclose()
        assert stream.first_sound_at is not None and len(sounds) == 1
        return sounds

    asyncio.run(scenario())
    assert [p[0] for p in player.played] == [1.0, 2.0, 3.0]
    # « Deux » est synthétisé pendant que « Un » est joué
    assert tts.started[1][1] < player.played[0][3]
    assert tts.closed


def test_streams_never_overlap_and_keep_order() -> None:
    tts, player = SlowTTS(delay=0.01), RecordingPlayer(duration=0.03)

    async def scenario() -> None:
        speaker = Speaker(tts, player)
        first = speaker.open("#1")
        second = speaker.open("#2")
        second.say("B1.")  # la 2e commande répond plus vite...
        second.close()
        first.say("A1.")
        first.say("A2.")
        first.close()
        await asyncio.wait_for(asyncio.gather(first.wait(), second.wait()), 2)
        await speaker.aclose()

    asyncio.run(scenario())
    starts_ends = [(p[2], p[3]) for p in player.played]
    assert all(e1 <= s2 + 1e-3 for (_, e1), (s2, _) in itertools.pairwise(starts_ends))
    # ...mais parle après la 1re (ordre d'ouverture) ; la synthèse, elle, n'a pas attendu.
    assert [t for t, _ in tts.started] == ["B1.", "A1.", "A2."]
    assert [p[0] for p in player.played] == [2.0, 3.0, 1.0]


def test_cached_phrase_skips_synthesis() -> None:
    tts, player, cache = SlowTTS(), RecordingPlayer(duration=0.0), Cache()

    async def scenario() -> None:
        speaker = Speaker(tts, player, cache)
        speaker.load()
        stream = speaker.open()
        stream.say("C'est fait.")
        stream.say("   ")
        stream.close()
        await asyncio.wait_for(stream.wait(), 2)
        await speaker.aclose()

    asyncio.run(scenario())
    assert tts.loaded and cache.warmed
    assert tts.started == []
    assert player.played[0][:2] == (99.0, 22_050)


def test_cancel_stops_current_and_drops_rest() -> None:
    tts, player = SlowTTS(delay=0.0), RecordingPlayer(duration=5.0)

    async def scenario() -> None:
        speaker = Speaker(tts, player)
        stream = speaker.open()
        for text in ("Un.", "Deux.", "Trois."):
            stream.say(text)
        await asyncio.sleep(0.05)
        stream.cancel()
        stream.cancel()
        stream.say("Après.")  # ignoré
        await asyncio.wait_for(stream.wait(), 1)
        next_stream = speaker.open()
        next_stream.say("Suivant.")
        next_stream.close()
        player.duration = 0.0
        await asyncio.wait_for(next_stream.wait(), 1)
        await speaker.aclose()

    asyncio.run(scenario())
    assert player.stops >= 1
    assert [p[0] for p in player.played] == [1.0, 4.0]  # « Deux », « Trois » abandonnés


def test_failed_synthesis_skips_fragment() -> None:
    class FlakyTTS(SlowTTS):
        async def synthesize(self, text: str) -> np.ndarray:
            if text == "Casse.":
                raise RuntimeError("onnx")
            return await super().synthesize(text)

    player = RecordingPlayer(duration=0.0)

    async def scenario() -> None:
        speaker = Speaker(FlakyTTS(), player)
        stream = speaker.open()
        stream.say("Casse.")
        stream.say("Ok.")
        stream.close()
        await asyncio.wait_for(stream.wait(), 1)
        await speaker.aclose()

    asyncio.run(scenario())
    assert len(player.played) == 1


def test_drain_waits_until_everything_said_is_played() -> None:
    """Phase 5 : une question est entièrement dite avant que MIKATA écoute la réponse."""
    tts, player = SlowTTS(delay=0.01), RecordingPlayer(duration=0.03)

    async def scenario() -> tuple[int, int]:
        speaker = Speaker(tts, player)
        stream = speaker.open("#1")
        stream.say("Cette commande va arrêter un programme.")
        stream.say("Tu confirmes ?")
        await asyncio.wait_for(stream.drain(), 2)
        played_at_drain = len(player.played)
        stream.say("Fini.")
        stream.close()
        await asyncio.wait_for(stream.drain(), 2)  # flux fermé : attend la fin
        await speaker.aclose()
        return played_at_drain, len(player.played)

    assert asyncio.run(scenario()) == (2, 3)


def test_drain_of_a_cancelled_stream_returns() -> None:
    tts, player = SlowTTS(delay=0.01), RecordingPlayer(duration=0.03)

    async def scenario() -> None:
        speaker = Speaker(tts, player)
        stream = speaker.open("#1")
        stream.say("Une phrase.")
        stream.cancel()
        await asyncio.wait_for(stream.drain(), 2)
        await speaker.aclose()

    asyncio.run(scenario())
