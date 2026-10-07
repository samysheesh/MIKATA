"""Phase 3 : commande -> LLM -> voix, et commandes en parallèle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
from test_app import RecordingTray, ScriptedAudio, ScriptedDetector
from test_app_listening import EARCONS, FakeSTT, ScriptedVAD, _run_until

from mikata.agent.loop import Agent
from mikata.app import App, State
from mikata.audio.player import Speaker
from mikata.config import load_config
from mikata.llm.base import Done, LLMEvent, TextDelta
from mikata.llm.client import LlmError

ALL_EARCONS = {**EARCONS, "thinking": np.full(10, 9, dtype=np.float32)}


class AudioOut:
    """Faux AudioPlayer : effets (play_nowait) et parole (play) enregistrés."""

    output_latency_s = 0.01

    def __init__(self) -> None:
        self.effects: list[str] = []
        self.spoken: list[float] = []

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    def play_nowait(self, pcm: np.ndarray) -> None:
        names = {float(v[0]): k for k, v in ALL_EARCONS.items()}
        self.effects.append(names[float(pcm[0])])

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        await asyncio.sleep(0.01)
        self.spoken.append(float(pcm[0]))

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class TextTTS:
    """« Synthèse » : l'audio vaut l'index du texte dans `texts`."""

    sample_rate = 16_000

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.full(16, float(len(self.texts)), dtype=np.float32)


class ContentLLM:
    """Répond selon la commande ; `delay` entre deux morceaux ; garde les horaires d'appel."""

    def __init__(
        self, answer: Callable[[str], str], delay: float = 0.0, fail: bool = False
    ) -> None:
        self.answer = answer
        self.delay = delay
        self.fail = fail
        self.calls: list[tuple[str, float, float]] = []  # (commande, début, fin)
        self.prewarmed = 0

    async def prewarm(self) -> None:
        self.prewarmed += 1

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        command = messages[-1]["content"].split("« ")[-1].rstrip(" »")
        start = time.monotonic()
        if self.fail:
            raise LlmError("serveur injoignable")
        for word in self.answer(command).split(" "):
            await asyncio.sleep(self.delay)
            yield TextDelta(word + " ")
        self.calls.append((command, start, time.monotonic()))
        yield Done("stop")


def _app(
    config_file: Path,
    *,
    detections: dict[int, float],
    speech: set[int],
    stt: FakeSTT,
    llm: ContentLLM,
    frames: int = 200,
    thinking_after: float | None = None,
) -> tuple[App, RecordingTray, AudioOut, TextTTS]:
    config = load_config(config_file)
    if thinking_after is not None:
        config = config.model_copy(
            update={
                "agent": config.agent.model_copy(update={"thinking_sound_after_s": thinking_after})
            }
        )
        config._base_dir = config_file.parent
    tray, out, tts = RecordingTray(quit_from_thread=False), AudioOut(), TextTTS()
    app = App(
        config,
        tray=tray,
        audio=ScriptedAudio(frames),
        detector=ScriptedDetector(detections),
        player=out,
        earcons=ALL_EARCONS,
        vad=ScriptedVAD(speech),
        stt=stt,
        agent=Agent(llm, config, window_title=lambda: None),
        speaker=Speaker(tts, out),
    )
    return app, tray, out, tts


def test_command_is_answered_out_loud(config_file: Path) -> None:
    llm = ContentLLM(lambda c: "Il est quinze heures quarante-deux.")
    app, tray, out, tts = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, quelle heure est-il ?"),
        llm=llm,
    )
    _run_until(app, lambda: bool(app.answers) and app.state is State.IDLE)

    assert list(app.commands) == ["quelle heure est-il ?"]
    assert list(app.answers) == ["Il est quinze heures quarante-deux."]
    assert tts.texts == ["Il est quinze heures quarante-deux."]
    assert out.spoken == [1.0]
    assert llm.prewarmed >= 1  # connexion préchauffée dès le mot d'activation
    states = [s for i, s in enumerate(tray.states) if i == 0 or s != tray.states[i - 1]]
    assert states == [
        State.IDLE,
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ]
    assert out.effects == ["wake"]  # réponse rapide : pas de son de réflexion


def test_thinking_sound_when_llm_is_slow(config_file: Path) -> None:
    llm = ContentLLM(lambda c: "Voilà.", delay=0.3)
    app, _, out, _ = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, résume la page."),
        llm=llm,
        thinking_after=0.05,
    )
    _run_until(app, lambda: bool(app.answers) and app.state is State.IDLE)
    assert out.effects == ["wake", "thinking"]


def test_server_unreachable_is_said(config_file: Path) -> None:
    app, _, _, tts = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, bonjour."),
        llm=ContentLLM(lambda c: "", fail=True),
    )
    _run_until(app, lambda: bool(tts.texts) and app.state is State.IDLE)
    assert tts.texts == ["Je n'arrive pas à joindre le serveur."]


def test_two_commands_run_in_parallel(config_file: Path) -> None:
    """« ouvre Spotify » puis « ouvre Brave » pendant que la 1re réfléchit encore."""

    def text(audio: np.ndarray) -> str:
        frame_ids = {int(v) for v in audio[::1280]}
        return "Hey Jarvis, ouvre Brave." if 81 in frame_ids else "Hey Jarvis, ouvre Spotify."

    llm = ContentLLM(lambda c: f"J'ouvre {c.split()[-1].rstrip('.')} tout de suite.", delay=0.15)
    app, tray, out, _ = _app(
        config_file,
        detections={31: 0.9, 80: 0.9},
        speech=set(range(25, 38)) | set(range(81, 88)),
        stt=FakeSTT(text),
        llm=llm,
    )
    _run_until(app, lambda: len(app.answers) == 2 and app.state is State.IDLE, timeout=10)

    assert app.detections == 2  # la 2e commande n'a pas été refusée
    assert list(app.commands) == ["ouvre Spotify.", "ouvre Brave."]
    assert sorted(app.answers) == ["J'ouvre Brave tout de suite.", "J'ouvre Spotify tout de suite."]
    (first, _, e1), (second, s2, _) = sorted(llm.calls, key=lambda c: c[1])
    assert (first, second) == ("ouvre Spotify.", "ouvre Brave.")
    assert s2 < e1  # la 2e réfléchit pendant que la 1re n'a pas fini : pas de file d'attente
    assert out.effects.count("wake") == 2
    assert len(out.spoken) == 2  # les deux réponses dites, l'une après l'autre
    # La 2e écoute commence pendant que la 1re commande est encore en cours (transcription ou
    # réflexion selon la vitesse de la machine), sans repasser par IDLE.
    after_first = tray.states[tray.states.index(State.TRANSCRIBING) :]
    assert State.LISTENING in after_first
    assert State.IDLE not in after_first[: after_first.index(State.LISTENING)]
