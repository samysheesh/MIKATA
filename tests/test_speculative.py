from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
import pytest

from mikata.config import SttConfig, VadConfig
from mikata.stt.base import Transcript
from mikata.stt.speculative import CommandRecorder, EndReason, frames_for
from mikata.stt.text import wake_word_pattern

PATTERN = wake_word_pattern(SttConfig().wake_words)
CFG = VadConfig()  # 500 ms de fin (6 trames), spéculatif 250 ms (3 trames), 4 s, 15 s
PRE_ROLL = np.full(32_000, -1, dtype=np.int16)


class FakeSTT:
    def __init__(self, text: str | Callable[[int, np.ndarray], str], delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[np.ndarray] = []

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        index = len(self.calls)
        self.calls.append(audio.copy())
        await asyncio.sleep(self.delay)
        text = self.text if isinstance(self.text, str) else self.text(index, audio)
        return Transcript(text=text, duration_s=len(audio) / 16_000, latency_s=self.delay)


def _frame(i: int) -> np.ndarray:
    return np.full(1280, i, dtype=np.int16)


async def _feed(recorder: CommandRecorder, pattern: str, pause: float = 0.0) -> None:
    """`pattern` : une lettre par trame, « s » = parole, « . » = silence."""
    for i, char in enumerate(pattern):
        recorder.add(_frame(i), char == "s")
        await asyncio.sleep(pause)


def _recorder(stt: FakeSTT, ends: list[EndReason], **kwargs: object) -> CommandRecorder:
    return CommandRecorder(stt, PRE_ROLL, CFG, PATTERN, on_end=ends.append, **kwargs)  # type: ignore[arg-type]


def test_frames_for() -> None:
    assert (frames_for(250), frames_for(500), frames_for(4000), frames_for(10)) == (3, 6, 50, 1)


def test_speculative_result_is_reused() -> None:
    async def scenario() -> tuple[FakeSTT, CommandRecorder, list[EndReason], Transcript]:
        stt, ends = FakeSTT("Hey Jarvis, ouvre Spotify."), []
        recorder = _recorder(stt, ends)
        await _feed(recorder, "ssss" + "......", pause=0.001)
        return stt, recorder, ends, await recorder.transcribe()

    stt, recorder, ends, transcript = asyncio.run(scenario())
    assert ends == [EndReason.SILENCE]
    assert recorder.speculative_runs == 1
    assert recorder.used_speculative
    assert len(stt.calls) == 1  # pas de second passage Whisper
    audio = stt.calls[0]
    assert len(audio) == 32_000 + 7 * 1280  # pré-roll entier + trames jusqu'au spéculatif
    assert audio[0] == -1  # commence au début du pré-roll
    assert transcript.text == "Hey Jarvis, ouvre Spotify."


def test_frames_after_end_are_ignored() -> None:
    async def scenario() -> CommandRecorder:
        recorder = _recorder(FakeSTT("x"), [])
        await _feed(recorder, "ss......" + "ssss")
        return recorder

    recorder = asyncio.run(scenario())
    assert recorder.finished
    assert len(recorder.audio) == 32_000 + 8 * 1280


def test_speech_resuming_discards_speculation() -> None:
    async def scenario() -> tuple[FakeSTT, CommandRecorder, Transcript]:
        stt, ends = FakeSTT(lambda i, a: f"passage {i}"), []
        recorder = _recorder(stt, ends)
        await _feed(recorder, "ss...ss......", pause=0.001)
        return stt, recorder, await recorder.transcribe()

    stt, recorder, transcript = asyncio.run(scenario())
    assert recorder.speculative_runs == 2
    assert transcript.text == "passage 1"  # le 2e spéculatif couvre toute la parole
    assert len(stt.calls[1]) == 32_000 + 10 * 1280


def test_stale_speculation_forces_final_pass() -> None:
    async def scenario() -> tuple[FakeSTT, CommandRecorder, Transcript]:
        stt = FakeSTT(lambda i, a: f"passage {i}", delay=0.05)
        recorder = _recorder(stt, [])
        # Le 1er spéculatif tourne encore quand la 2e pause arrive : pas de 2e spéculatif.
        await _feed(recorder, "ss...ss......")
        return stt, recorder, await recorder.transcribe()

    stt, recorder, transcript = asyncio.run(scenario())
    assert recorder.speculative_runs == 1
    assert not recorder.used_speculative
    assert transcript.text == "passage 1"
    assert len(stt.calls[1]) == 32_000 + 13 * 1280  # passage final sur tout l'audio


def test_command_said_in_one_breath_is_found_in_pre_roll() -> None:
    """« Hey Jarvis, pause » fini avant le bip : rien après, la commande est dans le pré-roll."""

    async def scenario() -> tuple[FakeSTT, list[EndReason], Transcript]:
        stt, ends = FakeSTT("Hey Jarvis, pause."), []
        recorder = _recorder(stt, ends)
        for i in range(20):
            recorder.add(_frame(i), False)
            await asyncio.sleep(0.001)
            if ends:
                break
        return stt, ends, await recorder.transcribe()

    stt, ends, transcript = asyncio.run(scenario())
    assert ends == [EndReason.PRE_ROLL]
    assert len(stt.calls) == 1
    assert len(stt.calls[0]) == 32_000  # pré-roll seul
    assert transcript.text == "Hey Jarvis, pause."


def test_wake_word_alone_waits_for_command() -> None:
    """« Hey Jarvis… [bip] … ouvre Spotify » : le pré-roll ne contient que le mot."""

    async def scenario() -> tuple[FakeSTT, list[EndReason], Transcript]:
        stt, ends = (
            FakeSTT(lambda i, a: "Hey Jarvis." if i == 0 else "Hey Jarvis. Ouvre Spotify."),
            [],
        )
        recorder = _recorder(stt, ends)
        await _feed(recorder, "........" + "sssss" + "......", pause=0.002)
        return stt, ends, await recorder.transcribe()

    stt, ends, transcript = asyncio.run(scenario())
    assert ends == [EndReason.SILENCE]
    assert transcript.text == "Hey Jarvis. Ouvre Spotify."
    assert len(stt.calls[0]) == 32_000


def test_pre_roll_without_wake_word_is_not_a_command() -> None:
    async def scenario() -> list[EndReason]:
        stt, ends = FakeSTT("une phrase sans rapport"), []
        recorder = _recorder(stt, ends)
        await _feed(recorder, "." * 20, pause=0.001)
        return ends

    assert asyncio.run(scenario()) == []


def test_no_speech_timeout() -> None:
    async def scenario() -> tuple[list[EndReason], int]:
        stt, ends = FakeSTT("Hey Jarvis."), []
        recorder = _recorder(stt, ends)
        await _feed(recorder, "." * 60)
        return ends, len(recorder.audio)

    ends, samples = asyncio.run(scenario())
    assert ends == [EndReason.NO_SPEECH]
    assert samples == 32_000 + 50 * 1280


def test_max_length() -> None:
    async def scenario() -> list[EndReason]:
        ends: list[EndReason] = []
        recorder = _recorder(FakeSTT("x"), ends)
        await _feed(recorder, "s" * 200)
        return ends

    assert asyncio.run(scenario()) == [EndReason.MAX_LENGTH]


def test_hotkey_skips_pre_roll_check() -> None:
    async def scenario() -> tuple[FakeSTT, list[EndReason]]:
        stt, ends = FakeSTT("pause"), []
        recorder = _recorder(stt, ends, check_pre_roll=False)
        await _feed(recorder, "." * 10, pause=0.001)
        return stt, ends

    stt, ends = asyncio.run(scenario())
    assert stt.calls == [] and ends == []


def test_failed_speculation_falls_back_to_final_pass() -> None:
    class FlakySTT(FakeSTT):
        async def transcribe(self, audio: np.ndarray) -> Transcript:
            if not self.calls:
                self.calls.append(audio)
                raise RuntimeError("GPU occupé")
            return await super().transcribe(audio)

    async def scenario() -> Transcript:
        recorder = _recorder(FlakySTT("ouvre Discord"), [])
        await _feed(recorder, "ss......", pause=0.001)
        return await recorder.transcribe()

    assert asyncio.run(scenario()).text == "ouvre Discord"


@pytest.mark.parametrize("reason_pattern", ["ss......", "." * 50])
def test_cancel_is_safe(reason_pattern: str) -> None:
    async def scenario() -> None:
        recorder = _recorder(FakeSTT("x", delay=1.0), [])
        await _feed(recorder, reason_pattern)
        recorder.cancel()
        recorder.cancel()

    asyncio.run(scenario())


def test_slow_pre_roll_check_is_awaited_after_timeout() -> None:
    """GPU très chargé : le délai sans parole expire avant la fin de la vérification."""

    async def scenario(text: str) -> tuple[list[EndReason], CommandRecorder]:
        ends: list[EndReason] = []
        recorder = _recorder(FakeSTT(text, delay=0.05), ends)
        await _feed(recorder, "." * 60)  # 4 s d'audio d'un coup, avant la fin du calcul
        assert ends == []
        await asyncio.sleep(0.1)
        return ends, recorder

    ends, recorder = asyncio.run(scenario("Hey Jarvis, pause."))
    assert ends == [EndReason.PRE_ROLL]
    assert len(recorder.audio) == 32_000 + 50 * 1280  # trames ignorées après le délai
    ends, _ = asyncio.run(scenario("Hey Jarvis."))
    assert ends == [EndReason.NO_SPEECH]


def test_failed_pre_roll_check_after_timeout_ends_listening() -> None:
    class BrokenSTT(FakeSTT):
        async def transcribe(self, audio: np.ndarray) -> Transcript:
            await asyncio.sleep(0.05)
            raise RuntimeError("GPU perdu")

    async def scenario() -> list[EndReason]:
        ends: list[EndReason] = []
        recorder = _recorder(BrokenSTT("x"), ends)
        await _feed(recorder, "." * 60)
        await asyncio.sleep(0.1)
        return ends

    assert asyncio.run(scenario()) == [EndReason.NO_SPEECH]
