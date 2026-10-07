from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from pathlib import Path

from mikata.app import App, State
from mikata.config import load_config
from mikata.events import LLMDelta, Shutdown


class RecordingTray:
    def __init__(self, quit_from_thread: bool = True) -> None:
        self.quit_from_thread = quit_from_thread
        self.started = False
        self.stopped = 0
        self.states: list[State] = []

    def start(self, on_quit: Callable[[], None]) -> None:
        self.started = True
        if self.quit_from_thread:
            # Comme pystray : le clic sur « Quitter » arrive depuis le thread de l'icône.
            threading.Timer(0.05, on_quit).start()

    def set_state(self, state: State) -> None:
        self.states.append(state)

    def stop(self) -> None:
        self.stopped += 1


def test_quit_from_tray_stops_cleanly(config_file: Path) -> None:
    tray = RecordingTray()
    app = App(load_config(config_file), tray=tray)
    asyncio.run(asyncio.wait_for(app.run(), timeout=5))
    assert tray.started
    assert tray.stopped == 1
    assert tray.states == [State.IDLE]
    assert app.state is State.IDLE


def test_other_events_are_ignored_until_shutdown(config_file: Path) -> None:
    app = App(load_config(config_file))

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0)
        app.bus.publish(LLMDelta("ignoré"))
        app.bus.publish(Shutdown(reason="test"))
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())
    assert app.bus.empty()


def test_cancellation_still_stops_tray(config_file: Path) -> None:
    tray = RecordingTray(quit_from_thread=False)
    app = App(load_config(config_file), tray=tray)

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0.01)
        task.cancel()  # ce que fait asyncio.run sur Ctrl+C
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert tray.stopped == 1


def test_states_cover_section_3() -> None:
    assert [s.name for s in State] == [
        "IDLE",
        "LISTENING",
        "TRANSCRIBING",
        "THINKING",
        "ACTING",
        "SPEAKING",
        "FOLLOW_UP",
    ]
