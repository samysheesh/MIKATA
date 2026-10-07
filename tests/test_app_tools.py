"""Phase 4 : commandes avec outils dans l'orchestrateur (parallèles, état ACTING, annonces)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from test_app import RecordingTray, ScriptedAudio, ScriptedDetector
from test_app_conversation import ALL_EARCONS, AudioOut, TextTTS
from test_app_listening import FakeSTT, ScriptedVAD, _run_until
from tool_fakes import FakeDesktop

from mikata.agent.loop import Agent
from mikata.app import App, State
from mikata.audio.player import Speaker
from mikata.config import load_config
from mikata.llm.base import Done, LLMEvent, ToolCallReq, Usage
from mikata.tools.base import ToolContext
from mikata.tools.registry import ToolRegistry

EARCONS = ALL_EARCONS


class OpenAppLLM:
    """« ouvre X » -> appel de open_app(X), après `delay` (le temps de « réfléchir »)."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.commands: list[str] = []

    async def prewarm(self) -> None:
        pass

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        command = messages[-1]["content"].split("« ")[-1].rstrip(" »")
        self.commands.append(command)
        assert any(t["function"]["name"] == "open_app" for t in tools)
        await asyncio.sleep(self.delay)
        app = command.split()[-1].rstrip(".")
        yield ToolCallReq(f"call-{app}", "open_app", {"name": app})
        yield Done("tool_calls", usage=Usage(2000, 12, 1900))


class SlowDesktop(FakeDesktop):
    def launch(self, app: Any) -> None:
        import time

        time.sleep(0.05)  # le temps de voir l'état ACTING
        super().launch(app)


def _app(config_file: Path, llm: OpenAppLLM, desktop: FakeDesktop, **script: Any) -> tuple:
    config = load_config(config_file)
    ctx = ToolContext(config, desktop=desktop)
    tray, out, tts = RecordingTray(quit_from_thread=False), AudioOut(), TextTTS()
    agent = Agent(llm, config, window_title=lambda: None, tools=ToolRegistry(ctx))
    app = App(
        config,
        tray=tray,
        audio=ScriptedAudio(script.get("frames", 200)),
        detector=ScriptedDetector(script["detections"]),
        player=out,
        earcons=EARCONS,
        vad=ScriptedVAD(script["speech"]),
        stt=FakeSTT(script["stt"]),
        agent=agent,
        speaker=Speaker(tts, out),
    )
    ctx.announce = app.announce
    return app, tray, out, tts


def test_open_spotify_then_brave_in_parallel(config_file: Path, caplog_loguru: list[str]) -> None:
    """Critère : « MIKATA ouvre Spotify » puis « MIKATA ouvre Brave », sans refus."""

    def text(audio: np.ndarray) -> str:
        frame_ids = {int(v) for v in audio[::1280]}
        return "Hey Jarvis, ouvre Brave." if 81 in frame_ids else "Hey Jarvis, ouvre Spotify."

    desktop, llm = SlowDesktop(), OpenAppLLM(delay=0.3)
    app, tray, out, tts = _app(
        config_file,
        llm,
        desktop,
        detections={31: 0.9, 80: 0.9},
        speech=set(range(25, 38)) | set(range(81, 88)),
        stt=text,
    )
    _run_until(app, lambda: len(app.answers) == 2 and app.state is State.IDLE, timeout=10)

    assert app.detections == 2  # la 2e commande n'a pas été refusée
    assert llm.commands == ["ouvre Spotify.", "ouvre Brave."]
    launched = [value for name, value in desktop.log if name == "launch"]
    assert launched == ["Spotify", "Brave"]
    assert list(app.answers) == ["C'est fait.", "C'est fait."]
    assert tts.texts == ["C'est fait.", "C'est fait."] and len(out.spoken) == 2
    assert State.ACTING in tray.states
    logs = [line for line in caplog_loguru if line.startswith("Réponse #")]
    assert len(logs) == 4  # mesures + texte, pour chaque commande
    measures = [line for line in logs if "premier son" in line]
    assert all("réflexion none" in line for line in measures)
    assert all(
        "chemin : court-circuit (outil terminal, 1 appel(s) LLM)" in line for line in measures
    )
    assert sum(line.startswith("Tokens #") for line in caplog_loguru) == 0  # pas de suivi ici


def test_announce_plays_timer_sound_then_speaks(config_file: Path) -> None:
    app, _, out, tts = _app(
        config_file, OpenAppLLM(), FakeDesktop(), detections={}, speech=set(), stt="", frames=5
    )

    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0.05)
        await app.announce("Le minuteur de 5 minutes est terminé.")
        app.request_shutdown("test")
        await task

    asyncio.run(scenario())
    assert out.effects == ["timer"]
    assert tts.texts == ["Le minuteur de 5 minutes est terminé."] and len(out.spoken) == 1
