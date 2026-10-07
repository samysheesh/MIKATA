"""Phase 5 : écoute d'une réponse sans mot d'activation (`App.listen_once`), utilisée par
la confirmation orale et le « stop » avant extinction ; parcours complet dans l'orchestrateur."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from test_agent import ScriptedLLM
from test_app import RecordingTray
from test_app_conversation import ALL_EARCONS, AudioOut, TextTTS
from tool_fakes import FakeDesktop, make_ctx

from mikata.agent.loop import Agent
from mikata.app import App, State
from mikata.audio.capture import CapturedFrame
from mikata.audio.player import Speaker
from mikata.config import load_config
from mikata.events import WakeDetected
from mikata.llm.base import Done, ToolCallReq
from mikata.stt.base import Transcript
from mikata.tools.registry import ToolRegistry

COMMAND, REPLY = 1, 2  # valeur des trames : parole de la commande, parole de la réponse


class FeedAudio:
    """Micro piloté par le test : les trames n'arrivent que quand on les envoie."""

    dropped_frames = 0
    overflows = 0

    def __init__(self) -> None:
        self.queue: asyncio.Queue[CapturedFrame] = asyncio.Queue()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    def stop(self) -> None:
        pass

    async def read(self) -> CapturedFrame:
        return await self.queue.get()

    def say(self, value: int, frames: int = 5, silence: int = 12) -> None:
        for v in [value] * frames + [0] * silence:
            self.queue.put_nowait(CapturedFrame(np.full(1280, v, np.int16), time.monotonic()))


class MarkerVAD:
    def is_speech(self, frame: np.ndarray) -> bool:
        return int(frame[0]) != 0

    def reset(self) -> None:
        pass


class SilentDetector:
    def process(self, frame: np.ndarray) -> float:
        return 0.0

    def reset(self) -> None:
        pass


class MarkerSTT:
    """Texte selon la parole entendue : la commande, ou la réponse."""

    def __init__(self, command: str, reply: str) -> None:
        self.command, self.reply = command, reply

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        await asyncio.sleep(0.005)
        values = set(np.unique(audio).tolist())
        text = self.reply if REPLY in values else self.command if COMMAND in values else ""
        return Transcript(text=text, duration_s=len(audio) / 16_000, latency_s=0.005)


def _app(
    config_file: Path, agent: Agent | None, stt: MarkerSTT
) -> tuple[App, FeedAudio, AudioOut, TextTTS, RecordingTray]:
    config = load_config(config_file)
    audio, out, tts = FeedAudio(), AudioOut(), TextTTS()
    tray = RecordingTray(quit_from_thread=False)
    app = App(
        config,
        tray=tray,
        audio=audio,
        detector=SilentDetector(),
        player=out,
        earcons=ALL_EARCONS,
        vad=MarkerVAD(),
        stt=stt,
        agent=agent,
        speaker=Speaker(tts, out),
    )
    return app, audio, out, tts, tray


async def _until(done: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "délai dépassé"
        await asyncio.sleep(0.005)


def _replying(app: App) -> bool:
    request = app._listening
    return request is not None and request.source == "réponse"


def test_listen_once_returns_what_was_said(config_file: Path) -> None:
    app, audio, out, _, tray = _app(config_file, None, MarkerSTT("", "Oui, vas-y."))

    async def scenario() -> tuple[str, str]:
        task = asyncio.create_task(app.run())
        await _until(lambda: app.state is State.IDLE)
        heard = asyncio.create_task(app.listen_once(3.0))
        await _until(lambda: _replying(app))
        audio.say(REPLY)
        first = await heard
        silent = asyncio.create_task(app.listen_once(0.5))
        await _until(lambda: _replying(app))
        audio.say(0, frames=0, silence=20)  # rien dit pendant 0,5 s
        second = await silent
        app.request_shutdown("test")
        await task
        return first, second

    assert asyncio.run(scenario()) == ("Oui, vas-y.", "")
    assert out.effects == ["wake", "wake"]  # un bip avant chaque écoute
    assert State.LISTENING in tray.states and app.state is State.IDLE


def test_listen_once_without_microphone(config_file: Path) -> None:
    app = App(load_config(config_file))
    assert asyncio.run(app.listen_once(1.0)) == ""


def _shutdown_app(config_file: Path, tmp_path: Path, reply: str) -> tuple:
    desktop = FakeDesktop()
    llm = ScriptedLLM([ToolCallReq("c1", "shutdown", {}), Done("tool_calls")])
    config = load_config(config_file)
    tools = ToolRegistry(make_ctx(tmp_path, desktop=desktop))
    agent = Agent(llm, config, window_title=lambda: None, tools=tools)
    app, audio, out, tts, _ = _app(config_file, agent, MarkerSTT("éteins le PC", reply))
    return app, audio, out, tts, desktop, llm


def _run_shutdown(app: App, audio: FeedAudio, reply_frames: int, silence: int) -> None:
    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await _until(lambda: app.state is State.IDLE)
        app.bus.publish(WakeDetected(score=1.0, source="hotkey"))  # comme « MIKATA »
        await _until(lambda: app._listening is not None)
        audio.say(COMMAND)
        await _until(lambda: _replying(app))  # annonce dite, MIKATA écoute
        audio.say(REPLY, frames=reply_frames, silence=silence)
        await _until(lambda: bool(app.answers) and app.state is State.IDLE)
        app.request_shutdown("test")
        await task

    asyncio.run(scenario())


def _power(desktop: FakeDesktop) -> list[tuple[str, object]]:
    return [entry for entry in desktop.log if entry[0] == "power"]


def test_saying_stop_cancels_the_shutdown(config_file: Path, tmp_path: Path) -> None:
    app, audio, out, tts, desktop, llm = _shutdown_app(config_file, tmp_path, "Stop !")
    _run_shutdown(app, audio, reply_frames=4, silence=12)
    assert _power(desktop) == []  # Windows n'a rien reçu
    assert tts.texts == [
        "Extinction dans 5 secondes. Dis stop pour annuler.",
        "Extinction annulée.",
    ]
    assert list(app.answers) == [
        "Extinction dans 5 secondes. Dis stop pour annuler. Extinction annulée."
    ]  # pas de « C'est fait » en plus
    assert len(llm.calls) == 1 and out.effects.count("wake") == 2


def test_silence_lets_the_shutdown_happen(config_file: Path, tmp_path: Path) -> None:
    app, audio, _, tts, desktop, _ = _shutdown_app(config_file, tmp_path, "")
    _run_shutdown(app, audio, reply_frames=0, silence=80)  # 5 s sans parole
    assert _power(desktop) == [("power", "shutdown")]
    assert tts.texts == ["Extinction dans 5 secondes. Dis stop pour annuler."]
