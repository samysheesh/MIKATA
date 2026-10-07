from __future__ import annotations

import asyncio
import dataclasses
import threading

import numpy as np
import pytest

from mikata.events import (
    Cancel,
    EventBus,
    LLMDelta,
    Shutdown,
    SpeakDone,
    SpeechEnded,
    ToolCall,
    Transcript,
    WakeDetected,
)


def test_events_are_frozen_and_timestamped() -> None:
    first = WakeDetected(score=0.9)
    second = WakeDetected(score=0.9, source="hotkey")
    assert first.source == "wake_word"
    assert second.ts >= first.ts
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.score = 0.1  # type: ignore[misc]


def test_event_fields() -> None:
    audio = np.zeros(1280, dtype=np.int16)
    assert SpeechEnded(audio).audio is audio
    assert LLMDelta("Bonjour").text == "Bonjour"
    call = ToolCall("c1", "open_app", {"name": "Spotify"})
    assert (call.id, call.name, call.arguments) == ("c1", "open_app", {"name": "Spotify"})
    assert Cancel().reason == ""
    assert isinstance(SpeakDone().ts, float)
    assert Shutdown(reason="x").reason == "x"


def test_publish_and_get_in_order() -> None:
    async def scenario() -> list[object]:
        bus = EventBus()
        events = [WakeDetected(0.8), Transcript("ouvre Spotify", 1.2, 0.1), Shutdown()]
        for event in events:
            bus.publish(event)
        return [await bus.get() for _ in events]

    received = asyncio.run(scenario())
    assert [type(e) for e in received] == [WakeDetected, Transcript, Shutdown]
    assert received[1].text == "ouvre Spotify"  # type: ignore[union-attr]


def test_publish_threadsafe_from_other_thread() -> None:
    async def scenario() -> object:
        bus = EventBus()
        bus.bind_loop(asyncio.get_running_loop())
        threading.Thread(target=bus.publish_threadsafe, args=(Shutdown(reason="tray"),)).start()
        return await asyncio.wait_for(bus.get(), timeout=2)

    event = asyncio.run(scenario())
    assert isinstance(event, Shutdown)
    assert event.reason == "tray"


def test_publish_threadsafe_requires_loop() -> None:
    with pytest.raises(RuntimeError):
        EventBus().publish_threadsafe(Shutdown())


def test_publish_threadsafe_after_loop_closed() -> None:
    bus = EventBus()
    loop = asyncio.new_event_loop()
    bus.bind_loop(loop)
    loop.close()
    assert bus.publish_threadsafe(Shutdown()) is False
