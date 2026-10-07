"""Phase 6b : arrêt vocal sans mot d'activation, de bout en bout dans l'orchestrateur (micro,
mot d'activation, Whisper, LLM et casque simulés)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import write_test_config
from loguru import logger
from test_app_conversation import ALL_EARCONS, TextTTS
from test_app_phase6 import (
    TEXTS,
    CommandLLM,
    CutPlayer,
    MarkerDetector,
    _run,
    _until,
    _wake_and_say,
)
from test_app_reply import FeedAudio, MarkerVAD

from mikata.agent.loop import Agent
from mikata.app import App, State
from mikata.audio.player import Speaker
from mikata.config import load_config
from mikata.llm.base import Done, LLMEvent, TextDelta
from mikata.stt.base import Transcript

STORY, SEARCH, TIME = 4, 6, 1  # commandes (valeurs de test_app_phase6.TEXTS)
STOP, TAIS_TOI, LONGER, NO_THANKS, MUSIC = 20, 21, 22, 23, 24
SPOKEN = {
    STOP: "Stop !",
    TAIS_TOI: "Tais-toi.",
    LONGER: "Arrête de dire stop.",
    NO_THANKS: "Non merci.",
    MUSIC: "Arrête la musique.",
}


class PhraseSTT:
    """Texte de la dernière parole entendue (commandes et phrases dites pendant une réponse)."""

    def __init__(self) -> None:
        self.texts = {**TEXTS, **SPOKEN}
        self.heard: list[str] = []

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        await asyncio.sleep(0.005)
        values = [int(v) for v in audio[::1280] if int(v) in self.texts]
        text = self.texts[values[-1]] if values else ""
        self.heard.append(text)
        return Transcript(text=text, duration_s=len(audio) / 16_000, latency_s=0.005)


class TalkThenThinkLLM:
    """Dit une phrase, puis travaille encore longtemps avant la suite (outil lent...)."""

    def __init__(self) -> None:
        self.cancelled = False

    async def prewarm(self) -> None:
        pass

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        yield TextDelta("Je regarde ça tout de suite pour toi. ")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield TextDelta("Voilà le résultat.")
        yield Done("stop")


def _app(
    tmp_path: Path, llm: Any, *, duration: float = 5.0, **agent: Any
) -> tuple[App, FeedAudio, CutPlayer, TextTTS, PhraseSTT]:
    config_file = write_test_config(
        tmp_path / "config.yaml",
        agent={"follow_up_s": 1.0, **agent},
        wake={"refractory_s": 0.0},
    )
    config = load_config(config_file)
    audio, out, tts, stt = FeedAudio(), CutPlayer(duration), TextTTS(), PhraseSTT()
    app = App(
        config,
        audio=audio,
        detector=MarkerDetector(),
        player=out,
        earcons=ALL_EARCONS,
        vad=MarkerVAD(),
        stt=stt,
        agent=Agent(llm, config, window_title=lambda: None),
        speaker=Speaker(tts, out),
    )
    return app, audio, out, tts, stt


@pytest.fixture
def records() -> Iterator[list[Any]]:
    """Messages du log (niveau et texte) pendant le test."""
    seen: list[Any] = []
    sink = logger.add(lambda m: seen.append(m.record), level="DEBUG", format="{message}")
    yield seen
    logger.remove(sink)


def _voice_stop_logs(records: list[Any]) -> list[Any]:
    return [r for r in records if r["message"].startswith("Arrêt vocal : « ")]


@pytest.mark.parametrize("phrase", [STOP, TAIS_TOI])
def test_stop_without_wake_word_cuts_the_answer(
    tmp_path: Path, records: list[Any], phrase: int
) -> None:
    llm = CommandLLM()
    app, audio, out, tts, _ = _app(tmp_path, llm)
    latency: list[float] = []

    async def scenario() -> None:
        await _wake_and_say(app, audio, STORY)  # « MIKATA, raconte une longue histoire »
        await _until(lambda: app.state is State.SPEAKING)
        said_at = time.monotonic()
        audio.say(phrase, frames=4, silence=3)  # « stop », sans « MIKATA »
        await _until(lambda: bool(out.stops))
        latency.append(out.stops[0] - said_at)
        await _until(lambda: not app._active)
        await asyncio.sleep(0.05)

    _run(app, scenario)
    assert latency[0] < 0.3
    assert out.played == []  # 1re phrase coupée, la 2e jamais dite
    assert llm.commands == ["raconte une longue histoire"]  # « stop » n'est pas parti au LLM
    assert out.effects == ["wake"]  # pas de bip : le silence suffit
    assert app._listening is None  # pas de fenêtre de suivi après une réponse coupée
    logs = [r for r in _voice_stop_logs(records) if "coupée" in r["message"]]
    assert len(logs) == 1 and logs[0]["level"].name == "INFO"  # sous l'objectif de 300 ms
    assert "#1" in logs[0]["message"] and "objectif < 300 ms" in logs[0]["message"]
    assert tts.texts[0].startswith("Il était une fois")


@pytest.mark.parametrize("phrase", [LONGER, NO_THANKS, MUSIC])
def test_other_speech_does_not_cut(tmp_path: Path, phrase: int) -> None:
    """« arrête de dire stop », « non merci », « arrête la musique » : entendus, transcrits,
    ignorés ; la réponse est dite jusqu'au bout."""
    llm = CommandLLM()
    app, audio, out, _, stt = _app(tmp_path, llm, duration=0.5)

    async def scenario() -> None:
        await _wake_and_say(app, audio, STORY)
        await _until(lambda: app.state is State.SPEAKING)
        audio.say(phrase, frames=6, silence=3)
        await _until(lambda: bool(app.answers))

    _run(app, scenario)
    assert out.stops == [] and len(out.played) == 2
    assert SPOKEN[phrase] in stt.heard  # bien entendu, puis ignoré
    assert llm.commands == ["raconte une longue histoire"]


def test_voice_stop_can_be_disabled(tmp_path: Path) -> None:
    llm = CommandLLM()
    app, audio, out, _, stt = _app(tmp_path, llm, duration=0.5, voice_stop=False)

    async def scenario() -> None:
        await _wake_and_say(app, audio, STORY)
        await _until(lambda: app.state is State.SPEAKING)
        audio.say(STOP, frames=4, silence=3)
        await _until(lambda: bool(app.answers))

    _run(app, scenario)
    assert app._voice_stop is None
    assert out.stops == [] and len(out.played) == 2
    assert "Stop !" not in stt.heard  # pas même transcrit


def test_stop_cuts_only_the_answer_being_played(tmp_path: Path) -> None:
    """Comme le barge-in : la commande lancée en parallèle va au bout et sa réponse est dite."""
    llm = CommandLLM(delays={"raconte une longue histoire": 0.2, "cherche un truc compliqué": 1.0})
    app, audio, out, tts, _ = _app(tmp_path, llm, follow_up_s=0)

    async def scenario() -> None:
        await _wake_and_say(app, audio, STORY)  # A : longue réponse
        await _until(lambda: app._listening is None)
        await _wake_and_say(app, audio, SEARCH)  # B : réfléchit longtemps, en parallèle
        await _until(lambda: app.state is State.SPEAKING)  # A parle
        out.duration = 0.02
        audio.say(STOP, frames=4, silence=3)
        await _until(lambda: len(app.answers) == 2 and not app._active)

    _run(app, scenario)
    assert len(out.stops) == 1 and llm.cancelled == []
    assert out.played == [tts.texts.index("Voilà ce que j'ai trouvé.") + 1.0]  # réponse de B


def test_stop_also_stops_a_command_still_working(tmp_path: Path) -> None:
    """La réponse coupée ne sera plus entendue : sa commande s'arrête aussi (plus d'outil ni
    d'appel LLM), comme après « MIKATA, stop »."""
    llm = TalkThenThinkLLM()
    app, audio, out, _, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, SEARCH)
        await _until(lambda: app.state is State.SPEAKING)
        audio.say(STOP, frames=4, silence=3)
        await _until(lambda: llm.cancelled and not app._active)

    _run(app, scenario)
    assert len(out.stops) == 1 and not app.answers


def test_stop_while_idle_is_not_even_transcribed(tmp_path: Path) -> None:
    llm = CommandLLM()
    app, audio, _, _, stt = _app(tmp_path, llm, duration=0.02, follow_up_s=0)

    async def scenario() -> None:
        await _wake_and_say(app, audio, TIME)
        await _until(lambda: bool(app.answers) and not app._active)
        audio.say(STOP, frames=4, silence=3)  # rien n'est joué : le micro n'écoute pas
        await asyncio.sleep(0.2)

    _run(app, scenario)
    assert app._voice_stop is not None and app._voice_stop.transcriptions == 0
    assert "Stop !" not in stt.heard and llm.commands == ["quelle heure est-il"]


def test_stop_cuts_an_announcement(tmp_path: Path, records: list[Any]) -> None:
    """Minuteur ou rappel annoncé dans le casque : « stop » le coupe aussi."""
    app, audio, out, _, _ = _app(tmp_path, CommandLLM())

    async def scenario() -> None:
        announce = asyncio.create_task(app.announce("Ton minuteur de dix minutes est terminé."))
        await _until(lambda: app._playing_stream() is not None)
        audio.say(STOP, frames=4, silence=3)
        await asyncio.wait_for(announce, 2.0)

    _run(app, scenario)
    assert len(out.stops) == 1
    assert any("réponse annonce coupée" in r["message"] for r in _voice_stop_logs(records))


def test_wake_word_during_the_answer_still_wins(tmp_path: Path) -> None:
    """« MIKATA, stop » pendant une réponse : barge-in et annulation de la phase 6, une seule
    coupure, « stop » ne part pas au LLM."""
    llm = CommandLLM()
    app, audio, out, _, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, STORY)
        await _until(lambda: app.state is State.SPEAKING)
        await _wake_and_say(app, audio, 3)  # « MIKATA, stop »
        await _until(lambda: out.effects[-1:] == ["end"] and not app._active)

    _run(app, scenario)
    assert len(out.stops) == 1 and llm.commands == ["raconte une longue histoire"]
