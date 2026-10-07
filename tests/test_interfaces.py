from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from pydantic import BaseModel

from mikata.audio.base import FRAME_SAMPLES, SAMPLE_RATE
from mikata.llm.base import Done, LLMClient, LLMEvent, TextDelta, ToolCallReq
from mikata.stt.base import STT, Transcript
from mikata.tools.base import Tool
from mikata.tts.base import TTS, Player
from mikata.vad.base import VAD
from mikata.wake.base import WakeWordDetector


def test_frame_constants() -> None:
    assert SAMPLE_RATE == 16_000
    assert FRAME_SAMPLES == 1280


class DummyWake:
    def process(self, frame: np.ndarray) -> float:
        return 0.0

    def reset(self) -> None:
        pass


class DummyVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return False


class DummyStt:
    async def transcribe(self, audio: np.ndarray) -> Transcript:
        return Transcript("", 0.0, 0.0)


class DummyLlm:
    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        yield TextDelta("ok")
        yield ToolCallReq("1", "get_time", {})
        yield Done("stop")


class NoArgs(BaseModel):
    pass


class DummyTool:
    name = "get_time"
    description = "Donne l'heure."
    params = NoArgs
    terminal = False

    async def run(self, args: BaseModel) -> str:
        return "12:00"


class DummyTts:
    sample_rate = 22_050

    async def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)


class DummyPlayer:
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        pass

    def stop(self) -> None:
        pass


def test_protocols_are_satisfied_structurally() -> None:
    assert isinstance(DummyWake(), WakeWordDetector)
    assert isinstance(DummyVad(), VAD)
    assert isinstance(DummyStt(), STT)
    assert isinstance(DummyLlm(), LLMClient)
    assert isinstance(DummyTool(), Tool)
    assert isinstance(DummyTts(), TTS)
    assert isinstance(DummyPlayer(), Player)


def test_protocols_reject_incomplete_objects() -> None:
    assert not isinstance(object(), WakeWordDetector)
    assert not isinstance(DummyVad(), STT)
    assert not isinstance(DummyPlayer(), TTS)


def test_dataclasses() -> None:
    t = Transcript(text="bonjour", duration_s=1.0, latency_s=0.2)
    assert (t.text, t.duration_s, t.latency_s) == ("bonjour", 1.0, 0.2)
    assert ToolCallReq("a", "b", {"x": 1}).arguments == {"x": 1}
    assert Done("tool_calls").finish_reason == "tool_calls"
