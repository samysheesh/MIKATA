from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mikata.app import App, State
from mikata.audio.capture import CapturedFrame
from mikata.audio.wavfile import float_to_int16, read_wav
from mikata.config import load_config
from mikata.events import LLMDelta, Shutdown, WakeDetected


class RecordingTray:
    def __init__(self, quit_from_thread: bool = True) -> None:
        self.quit_from_thread = quit_from_thread
        self.started = False
        self.stopped = 0
        self.states: list[State] = []

    def start(self, on_quit: Callable[[], None], on_action: object = None) -> None:
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


# -- phase 1 : audio, mot d'activation, raccourci ----------------------------------------


class ScriptedAudio:
    """Source audio : `count` trames (valeur = index) espacées de 80 ms dans le passé,
    puis attend indéfiniment."""

    def __init__(self, count: int, on_exhausted: Callable[[], None] | None = None) -> None:
        self.count = count
        self.on_exhausted = on_exhausted
        self.started = False
        self.stopped = False
        self.dropped_frames = 0
        self.overflows = 0
        self._i = 0
        self._t0 = time.monotonic() - count * 0.08

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.started = True

    async def read(self) -> CapturedFrame:
        if self._i >= self.count:
            if self.on_exhausted is not None:
                self.on_exhausted()
                self.on_exhausted = None
            await asyncio.Event().wait()
        frame = CapturedFrame(np.full(1280, self._i % 1000, np.int16), self._t0 + self._i * 0.08)
        self._i += 1
        return frame

    def stop(self) -> None:
        self.stopped = True


class ScriptedDetector:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.calls = 0
        self.resets = 0
        self.threads: set[str] = set()

    def process(self, frame: np.ndarray) -> float:
        self.threads.add(threading.current_thread().name)
        score = self.scores.get(self.calls, 0.0)
        self.calls += 1
        return score

    def reset(self) -> None:
        self.resets += 1


class RecordingPlayer:
    output_latency_s = 0.01

    def __init__(self) -> None:
        self.played: list[np.ndarray] = []
        self.started = False
        self.closed = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.started = True

    def play_nowait(self, pcm: np.ndarray) -> None:
        self.played.append(pcm)

    def close(self) -> None:
        self.closed = True


class FakeHotkeys:
    def __init__(self, fire: bool = True, fail: bool = False) -> None:
        self.fire = fire
        self.fail = fail
        self.stopped = False

    def start(self, on_trigger: Callable[[], None]) -> None:
        if self.fail:
            raise ImportError("No module named 'pynput'")
        if self.fire:
            threading.Timer(0.01, on_trigger).start()

    def stop(self) -> None:
        self.stopped = True


BEEP = np.ones(10, dtype=np.float32)


def _run_app(app: App, timeout: float = 5) -> None:
    asyncio.run(asyncio.wait_for(app.run(), timeout=timeout))


def _app(config_file: Path, **overrides: object) -> App:
    config = load_config(config_file)
    if overrides:
        config = config.model_copy(
            update={"wake": config.wake.model_copy(update=overrides)}, deep=True
        )
        config._base_dir = config_file.parent
    return App(config, earcons={"wake": BEEP}, player=RecordingPlayer())


def test_wake_word_beeps_and_logs_audio(config_file: Path) -> None:
    app = _app(config_file, consecutive_frames=2)
    # 2 trames consécutives >= 0,5 : détection à la trame 31 ; la trame 40 est isolée.
    detector = ScriptedDetector({30: 0.6, 31: 0.9, 40: 0.95})
    audio = ScriptedAudio(60, on_exhausted=lambda: app.request_shutdown("test"))
    app._audio, app._detector = audio, detector
    _run_app(app)

    player = app._player
    assert isinstance(player, RecordingPlayer)
    assert player.started and player.closed and audio.started and audio.stopped
    assert app.detections == 1
    assert len(player.played) == 1 and player.played[0] is BEEP
    assert detector.calls == 60
    assert detector.resets == 1
    assert detector.threads == {"mikata-wake_0"}
    wavs = list((config_file.parent / "data" / "wake_logs").glob("*.wav"))
    assert len(wavs) == 1
    assert "_score0.90" in wavs[0].name
    pcm, _ = read_wav(wavs[0])
    assert len(pcm) == 32_000  # 2 s
    assert float_to_int16(pcm)[-1] == 31  # se termine sur la trame déclenchante


def test_refractory_blocks_second_detection(config_file: Path) -> None:
    app = _app(config_file, consecutive_frames=1)
    app._detector = ScriptedDetector({5: 0.9, 10: 0.9, 40: 0.9})  # 10 : < 2 s après 5
    app._audio = ScriptedAudio(50, on_exhausted=lambda: app.request_shutdown("test"))
    _run_app(app)
    assert app.detections == 2


def test_hotkey_beeps_without_audio_log(config_file: Path) -> None:
    app = _app(config_file)
    hotkeys = FakeHotkeys()
    app._hotkeys = hotkeys
    app._audio = ScriptedAudio(0)

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if app.detections:
                break
        app.request_shutdown("test")
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert app.detections == 1
    assert hotkeys.stopped
    assert not (config_file.parent / "data" / "wake_logs").exists() or not list(
        (config_file.parent / "data" / "wake_logs").glob("*.wav")
    )


def test_hotkey_failure_is_not_fatal(config_file: Path) -> None:
    app = _app(config_file)
    app._hotkeys = FakeHotkeys(fail=True)
    app._audio = ScriptedAudio(1, on_exhausted=lambda: app.request_shutdown("test"))
    _run_app(app)
    assert app._hotkeys is None


def test_collect_negatives(config_file: Path) -> None:
    app = _app(config_file, collect_negatives=True)
    app._audio = ScriptedAudio(130, on_exhausted=lambda: app.request_shutdown("test"))
    _run_app(app)
    clips = list((config_file.parent / "data" / "negatives").rglob("*.wav"))
    assert len(clips) == 1
    pcm, rate = read_wav(clips[0])
    assert (len(pcm), rate) == (160_000, 16_000)


def test_ring_buffer_holds_pre_roll(config_file: Path) -> None:
    app = _app(config_file)
    app._audio = ScriptedAudio(70, on_exhausted=lambda: app.request_shutdown("test"))
    _run_app(app)
    # max(pre_roll_s (2 s), raté du mot d'activation (4 s, phase 9)) + 1 s de marge, plein
    assert len(app.ring) == 80_000
    assert app.ring.read_last(1)[0] == 69


def test_stats_loop(config_file: Path) -> None:
    config = load_config(config_file)
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"stats_interval_s": 0.01})}
    )

    class Cpu:
        samples = 0

        def sample(self) -> tuple[float, float]:
            Cpu.samples += 1
            return 1.5, 0.06

    app = App(config, cpu=Cpu())

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0.1)
        app.request_shutdown("test")
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert Cpu.samples >= 2


def test_failing_event_does_not_stop_app(config_file: Path) -> None:
    class BrokenPlayer(RecordingPlayer):
        def play_nowait(self, pcm: np.ndarray) -> None:
            raise RuntimeError("périphérique perdu")

    app = App(load_config(config_file), earcons={"wake": BEEP}, player=BrokenPlayer())

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0)
        app.bus.publish(WakeDetected(score=1.0, source="hotkey"))
        app.bus.publish(WakeDetected(score=1.0, source="hotkey"))
        await asyncio.sleep(0.05)
        assert not task.done()
        app.request_shutdown("test")
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def test_background_tasks_and_close_task(config_file: Path, tmp_path: Path) -> None:
    """Phase 13f : l'onglet « En cours » liste le navigateur de travail et peut le fermer."""
    from types import SimpleNamespace

    from agent_fakes import make_registry
    from tool_fakes import FakeBrowser

    app = _app(config_file)
    browser = FakeBrowser()
    browser.running = True
    app._agent = SimpleNamespace(tools=make_registry(tmp_path, browser=browser))
    assert app._background_tasks() == [
        {
            "id": "browser",
            "label": "Navigateur de travail — lecture en arrière-plan",
            "running": True,
            "visible": False,
            "tabs": [],
        }
    ]

    async def scenario() -> None:
        app._on_ui_command({"type": "close_task", "id": "browser"})
        await asyncio.sleep(0.05)
        assert ("quit", ()) in browser.log
        assert app._background_tasks()[0]["running"] is False

    asyncio.run(scenario())


def test_the_interface_manages_the_work_browser(config_file: Path, tmp_path: Path) -> None:
    """Phase 15a : onglets listés et fermables un par un, pause, fenêtre montrée ou masquée,
    sans jamais fermer le navigateur lui-même."""
    from types import SimpleNamespace

    from agent_fakes import make_registry
    from tool_fakes import FakeBrowser

    app = _app(config_file)
    published: list[dict[str, object]] = []
    app._publish_ui = lambda kind, **fields: published.append({"type": kind, **fields})  # type: ignore[method-assign]
    browser = FakeBrowser()
    browser.running = True
    browser.open_tabs = [
        {"id": "a", "title": "One Piece OP 27", "url": "https://youtube.com/watch?v=1"},
        {"id": "b", "title": "Mix", "url": "https://youtube.com/watch?v=2"},
    ]
    app._agent = SimpleNamespace(tools=make_registry(tmp_path, browser=browser))

    async def scenario() -> None:
        for command in (
            {"type": "list_tasks"},
            {"type": "browser", "action": "pause"},
            {"type": "browser", "action": "show"},
            {"type": "browser", "action": "close_tab", "tab": "b"},
            {"type": "browser", "action": "hide"},
        ):
            app._on_ui_command(command)
            await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert ("pause", ()) in browser.log and ("close_tab", "b") in browser.log
    assert ("visible", True) in browser.log and browser.log[-1] == ("visible", False)
    assert ("quit", ()) not in browser.log  # le navigateur lui-même reste lancé
    first, last = published[0]["tasks"][0], published[-1]["tasks"][0]  # type: ignore[index]
    assert [tab["id"] for tab in first["tabs"]] == ["a", "b"]
    assert [tab["id"] for tab in last["tabs"]] == ["a"] and last["visible"] is False
