"""Phase 6 : fenêtre de suivi, barge-in et annulation, de bout en bout dans l'orchestrateur
(micro, mot d'activation, Whisper, LLM et casque simulés)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import write_test_config
from test_agent import ScriptedLLM
from test_app import RecordingTray
from test_app_conversation import ALL_EARCONS, TextTTS
from test_app_reply import FeedAudio, MarkerVAD
from tool_fakes import FakeDesktop, make_ctx

from mikata.agent.loop import Agent
from mikata.app import App, State
from mikata.audio.player import Speaker
from mikata.config import load_config
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq
from mikata.stt.base import Transcript
from mikata.tools.registry import ToolRegistry

WAKE = 7  # trame « MIKATA »
TEXTS = {
    1: "quelle heure est-il",
    2: "et à Tokyo",
    3: "stop",
    4: "raconte une longue histoire",
    5: "annule",
    6: "cherche un truc compliqué",
    8: "et demain",
}
ANSWERS = {
    "quelle heure est-il": "Il est quinze heures.",
    "et à Tokyo": "À Tokyo, il est vingt-trois heures.",
    "stop": "Musique en pause.",
    "raconte une longue histoire": (
        "Il était une fois un assistant très bavard. Il parlait sans jamais s'arrêter."
    ),
    "cherche un truc compliqué": "Voilà ce que j'ai trouvé.",
    "et demain": "Demain, il fera beau.",
}


class MarkerDetector:
    """« MIKATA » sur la trame qui vaut WAKE."""

    def process(self, frame: np.ndarray) -> float:
        return 1.0 if int(frame[0]) == WAKE else 0.0

    def reset(self) -> None:
        pass


class LastWordSTT:
    """Texte de la dernière parole entendue (le pré-roll peut contenir une commande
    précédente : seule la plus récente compte)."""

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __init__(self, unreliable: set[int] | None = None) -> None:
        self.unreliable = unreliable or set()  # paroles mal entendues (bruit, voix lointaine)

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        await asyncio.sleep(0.005)
        values = [int(v) for v in audio[::1280] if int(v) in TEXTS]
        text = TEXTS[values[-1]] if values else ""
        noisy = bool(values) and values[-1] in self.unreliable
        return Transcript(
            text=text,
            duration_s=len(audio) / 16_000,
            latency_s=0.005,
            no_speech_prob=0.7 if noisy else 0.02,
            avg_logprob=-1.3 if noisy else -0.2,
        )


class CommandLLM:
    """Répond selon la commande, après `delays[commande]` s ; note les annulations."""

    def __init__(
        self, delays: dict[str, float] | None = None, answers: dict[str, str] | None = None
    ) -> None:
        self.delays = delays or {}
        self.answers = {**ANSWERS, **(answers or {})}
        self.calls: list[list[dict[str, Any]]] = []
        self.commands: list[str] = []
        self.cancelled: list[str] = []

    async def prewarm(self) -> None:
        pass

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        command = messages[-1]["content"].split("« ")[-1].rstrip(" »")
        self.calls.append(messages)
        self.commands.append(command)
        try:
            await asyncio.sleep(self.delays.get(command, 0.0))
        except asyncio.CancelledError:
            self.cancelled.append(command)
            raise
        yield TextDelta(self.answers[command])
        yield Done("stop")


class CutPlayer:
    """Casque : chaque phrase dure `duration` s, sauf si `stop()` la coupe."""

    output_latency_s = 0.01

    def __init__(self, duration: float = 0.02) -> None:
        self.duration = duration
        self.effects: list[str] = []
        self.played: list[float] = []  # phrases jouées jusqu'au bout
        self.stops: list[float] = []  # horaires des coupures (monotonic)
        self._current: asyncio.Future[bool] | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    def close(self) -> None:
        pass

    def play_nowait(self, pcm: np.ndarray) -> None:
        names = {float(v[0]): k for k, v in ALL_EARCONS.items()}
        self.effects.append(names[float(pcm[0])])

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        loop = asyncio.get_running_loop()
        current: asyncio.Future[bool] = loop.create_future()
        self._current = current
        timer = loop.call_later(self.duration, lambda: current.done() or current.set_result(False))
        cut = await current
        timer.cancel()
        if not cut:
            self.played.append(float(pcm[0]))

    def stop(self) -> None:
        current = self._current
        if current is not None and not current.done():  # une phrase est en train d'être jouée
            self.stops.append(time.monotonic())
            current.set_result(True)


def _app(
    tmp_path: Path,
    llm: Any,
    *,
    follow_up_s: float = 1.0,
    duration: float = 0.02,
    tools: ToolRegistry | None = None,
    stt: LastWordSTT | None = None,
) -> tuple[App, FeedAudio, CutPlayer, TextTTS, RecordingTray]:
    config_file = write_test_config(
        tmp_path / "config.yaml",
        agent={"follow_up_s": follow_up_s},
        wake={"refractory_s": 0.0},
    )
    config = load_config(config_file)
    audio, out, tts = FeedAudio(), CutPlayer(duration), TextTTS()
    tray = RecordingTray(quit_from_thread=False)
    app = App(
        config,
        tray=tray,
        audio=audio,
        detector=MarkerDetector(),
        player=out,
        earcons=ALL_EARCONS,
        vad=MarkerVAD(),
        stt=stt or LastWordSTT(),
        agent=Agent(llm, config, window_title=lambda: None, tools=tools),
        speaker=Speaker(tts, out),
    )
    return app, audio, out, tts, tray


async def _until(done: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "délai dépassé"
        await asyncio.sleep(0.005)


async def _wake_and_say(app: App, audio: FeedAudio, value: int) -> None:
    """« MIKATA » puis la commande, une fois l'écoute commencée."""
    audio.say(WAKE, frames=1, silence=0)
    await _until(lambda: app._listening is not None and app._listening.source == "wake_word")
    audio.say(value)


def _in_follow_up(app: App) -> bool:
    request = app._listening
    return request is not None and request.source == "suivi" and app.state is State.FOLLOW_UP


def _run(app: App, scenario: Callable[[], Any]) -> None:
    async def main() -> None:
        task = asyncio.create_task(app.run())
        await _until(lambda: app.state is State.IDLE)
        try:
            await scenario()
        finally:
            app.request_shutdown("test")
            await task

    asyncio.run(main())


# -- fenêtre de suivi ------------------------------------------------------------------------


def test_follow_up_question_without_wake_word(tmp_path: Path) -> None:
    """Critère de la phase 6 : une question de suivi sans redire « MIKATA »."""
    llm = CommandLLM()
    app, audio, out, tts, tray = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)  # « MIKATA, quelle heure est-il »
        await _until(lambda: _in_follow_up(app))
        audio.say(2)  # « et à Tokyo », sans mot d'activation
        await _until(lambda: len(app.answers) == 2)
        await _until(lambda: _in_follow_up(app))  # nouvelle fenêtre après la 2e réponse
        audio.say(0, frames=0, silence=15)  # 1 s sans rien dire
        await _until(lambda: app.state is State.IDLE and app._listening is None)

    _run(app, scenario)
    assert llm.commands == ["quelle heure est-il", "et à Tokyo"]
    history = llm.calls[1][1:-1]  # entre le prompt système et la commande
    assert {"role": "user", "content": "quelle heure est-il"} in history
    assert {"role": "assistant", "content": "Il est quinze heures."} in history
    assert tts.texts == ["Il est quinze heures.", "À Tokyo, il est vingt-trois heures."]
    assert out.effects == ["wake", "end"]  # pas de bip pour le suivi, bip de fin à la fin
    assert State.FOLLOW_UP in tray.states


def test_stop_in_follow_up_closes_it_without_llm(tmp_path: Path) -> None:
    llm = CommandLLM()
    app, audio, out, _, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: _in_follow_up(app))
        audio.say(3)  # « stop »
        await _until(lambda: out.effects[-1:] == ["end"] and not app._active)

    _run(app, scenario)
    assert llm.commands == ["quelle heure est-il"]  # « stop » n'est pas parti au LLM
    assert app._listening is None and app.state is State.IDLE  # pas de nouvelle fenêtre


def test_wake_word_inside_follow_up_is_acknowledged(tmp_path: Path) -> None:
    llm = CommandLLM()
    app, audio, out, _, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: _in_follow_up(app))
        audio.say(WAKE, frames=1, silence=0)  # « MIKATA » quand même
        audio.say(8)  # « et demain »
        await _until(lambda: len(app.answers) == 2)

    _run(app, scenario)
    assert llm.commands == ["quelle heure est-il", "et demain"]
    assert out.effects[:2] == ["wake", "wake"]


def test_no_follow_up_while_another_command_runs(tmp_path: Path) -> None:
    llm = CommandLLM(delays={"cherche un truc compliqué": 0.6})
    app, audio, _, _, _ = _app(tmp_path, llm)
    seen: list[bool] = []

    async def scenario() -> None:
        await _wake_and_say(app, audio, 6)  # commande lente
        await _until(lambda: app._listening is None)
        await _wake_and_say(app, audio, 1)  # commande rapide (sa réponse passe en second)
        await _until(lambda: len(app.answers) == 1)
        seen.append(app._listening is not None)  # pas de suivi : l'autre n'a pas fini
        await _until(lambda: len(app.answers) == 2)
        await _until(lambda: _in_follow_up(app))  # la dernière réponse ouvre le suivi
        audio.say(0, frames=0, silence=15)
        await _until(lambda: app._listening is None)

    _run(app, scenario)
    assert seen == [False]
    assert list(app.answers) == ["Voilà ce que j'ai trouvé.", "Il est quinze heures."]


def test_follow_up_disabled(tmp_path: Path) -> None:
    llm = CommandLLM()
    app, audio, out, _, _ = _app(tmp_path, llm, follow_up_s=0)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: bool(app.answers) and not app._active)
        await asyncio.sleep(0.05)

    _run(app, scenario)
    assert app._listening is None and out.effects == ["wake"]


# -- barge-in ------------------------------------------------------------------------------


def test_barge_in_cuts_the_answer_in_under_300_ms(tmp_path: Path) -> None:
    """Critère de la phase 6 : « MIKATA » pendant une réponse la coupe en < 300 ms."""
    llm = CommandLLM()
    app, audio, out, tts, _ = _app(tmp_path, llm, duration=5.0)  # phrases de 5 s
    latency: list[float] = []

    async def scenario() -> None:
        await _wake_and_say(app, audio, 4)  # « raconte une longue histoire »
        await _until(lambda: app.state is State.SPEAKING)
        said_at = time.monotonic()
        audio.say(WAKE, frames=1, silence=0)  # « MIKATA » pendant la réponse
        await _until(lambda: bool(out.stops))
        latency.append(out.stops[0] - said_at)
        out.duration = 0.02
        await _until(lambda: app._listening is not None)
        audio.say(1)  # nouvelle question
        await _until(lambda: len(app.answers) == 2)

    _run(app, scenario)
    assert latency[0] < 0.3
    # 1re phrase coupée, la 2e jamais dite ; la réponse suivante l'est
    assert out.played == [tts.texts.index("Il est quinze heures.") + 1.0]
    assert llm.commands == ["raconte une longue histoire", "quelle heure est-il"]


def test_barge_in_does_not_cancel_other_commands(tmp_path: Path) -> None:
    """Seul le son de la réponse en cours est coupé : la commande lancée en parallèle avant le
    barge-in va au bout, et sa réponse est dite."""
    llm = CommandLLM(delays={"raconte une longue histoire": 0.2, "cherche un truc compliqué": 1.0})
    app, audio, out, tts, _ = _app(tmp_path, llm, duration=5.0)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 4)  # A : longue réponse
        await _until(lambda: app._listening is None)
        await _wake_and_say(app, audio, 6)  # B : réfléchit longtemps, en parallèle
        await _until(lambda: app.state is State.SPEAKING)  # A parle
        out.duration = 0.02
        await _wake_and_say(app, audio, 1)  # barge-in : coupe A, puis C
        await _until(lambda: len(app.answers) == 3)

    _run(app, scenario)
    assert len(out.stops) == 1 and llm.cancelled == []
    found = tts.texts.index("Voilà ce que j'ai trouvé.") + 1.0
    assert found in out.played  # la réponse de B est dite en entier
    assert tts.texts.index("Il est quinze heures.") + 1.0 in out.played


def test_stop_after_barge_in_does_not_reach_the_llm(tmp_path: Path) -> None:
    """« MIKATA, stop » pendant une réponse : coupée, et « stop » ne met pas la musique en
    pause (pas d'appel LLM)."""
    llm = CommandLLM()
    app, audio, out, _, _ = _app(tmp_path, llm, duration=5.0)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 4)
        await _until(lambda: app.state is State.SPEAKING)
        await _wake_and_say(app, audio, 3)  # « MIKATA, stop »
        await _until(lambda: out.effects[-1:] == ["end"] and not app._active)
        await asyncio.sleep(0.05)

    _run(app, scenario)
    assert llm.commands == ["raconte une longue histoire"]
    assert app._listening is None  # pas de fenêtre de suivi après une réponse coupée
    assert out.played == []


# -- annulation ----------------------------------------------------------------------------


def test_cancel_command_cancels_the_running_command(tmp_path: Path) -> None:
    llm = CommandLLM(delays={"cherche un truc compliqué": 5.0})
    app, audio, out, tts, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 6)  # commande qui réfléchit longtemps
        await _until(lambda: app.state is State.THINKING)
        await _wake_and_say(app, audio, 5)  # « MIKATA, annule »
        await _until(lambda: llm.cancelled != [] and not app._active)

    _run(app, scenario)
    assert llm.cancelled == ["cherche un truc compliqué"]
    assert llm.commands == ["cherche un truc compliqué"]  # « annule » n'est pas parti au LLM
    assert tts.texts == [] and not app.answers
    assert out.effects[-1] == "end"


@pytest.mark.parametrize("follow_up_s", [0.0, 1.0])
def test_stop_with_nothing_to_cancel_is_a_normal_command(
    tmp_path: Path, follow_up_s: float
) -> None:
    """« MIKATA, stop » au repos : c'est une vraie commande (pause de la musique)."""
    llm = CommandLLM()
    app, audio, _, _, _ = _app(tmp_path, llm, follow_up_s=follow_up_s)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 3)
        await _until(lambda: bool(app.answers))

    _run(app, scenario)
    assert llm.commands == ["stop"] and list(app.answers) == ["Musique en pause."]


def test_barge_in_on_a_shutdown_announcement_cancels_it(tmp_path: Path) -> None:
    """Garde-fou : « MIKATA » pendant « Extinction dans 5 secondes... » ; l'annonce coupée
    n'a pas été entendue, l'extinction est donc annulée (jamais faite en silence)."""
    desktop = FakeDesktop()
    llm = ScriptedLLM(
        [ToolCallReq("c1", "shutdown", {}), Done("tool_calls")],
        [TextDelta("Il est quinze heures."), Done("stop")],
    )
    tools = ToolRegistry(make_ctx(tmp_path, desktop=desktop))
    app, audio, out, tts, _ = _app(tmp_path, llm, duration=5.0, tools=tools)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 4)  # « éteins le PC »
        await _until(lambda: app.state is State.SPEAKING)  # annonce en cours
        out.duration = 0.02
        await _wake_and_say(app, audio, 1)  # « MIKATA, quelle heure est-il »
        await _until(lambda: len(app.answers) == 2)

    _run(app, scenario)
    assert [entry for entry in desktop.log if entry[0] == "power"] == []
    assert any("Extinction annulée." in answer for answer in app.answers)
    assert "Extinction annulée." not in tts.texts  # réponse coupée : plus rien n'est dit


# -- fenêtre de suivi dans le bruit (retours de test de la phase 6) -------------------------


def test_unreliable_follow_up_is_ignored_without_a_word(tmp_path: Path) -> None:
    """Bruit ou voix lointaine dans la fenêtre de suivi : bip de fin, pas d'appel LLM, pas
    de « j'ai pas bien entendu »."""
    llm = CommandLLM()
    app, audio, out, tts, _ = _app(tmp_path, llm, stt=LastWordSTT(unreliable={2}))

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: _in_follow_up(app))
        audio.say(2)  # mal entendu
        await _until(lambda: out.effects[-1:] == ["end"] and not app._active)
        await asyncio.sleep(0.05)

    _run(app, scenario)
    assert llm.commands == ["quelle heure est-il"]
    assert tts.texts == ["Il est quinze heures."] and app._listening is None


def test_conversation_next_to_mikata_is_judged_by_the_llm(tmp_path: Path) -> None:
    """Une phrase bien entendue mais adressée à quelqu'un d'autre : le LLM répond le
    marqueur, rien n'est dit, la fenêtre se ferme."""
    llm = CommandLLM(answers={"et à Tokyo": "[IGNORER]"})
    app, audio, out, tts, _ = _app(tmp_path, llm)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: _in_follow_up(app))
        audio.say(2)
        await _until(lambda: len(llm.commands) == 2)
        await _until(lambda: out.effects[-1:] == ["end"] and not app._active)
        await asyncio.sleep(0.05)

    _run(app, scenario)
    assert tts.texts == ["Il est quinze heures."]  # le marqueur n'est jamais dit
    assert list(app.answers) == ["Il est quinze heures."]
    assert "Fenêtre de suivi" in llm.calls[1][-1]["content"]
    assert "Fenêtre de suivi" not in llm.calls[0][-1]["content"]  # pas après « MIKATA »
    assert app._listening is None  # pas de nouvelle fenêtre de suivi


def test_llm_gate_can_be_disabled(tmp_path: Path) -> None:
    llm = CommandLLM()
    config_file = write_test_config(
        tmp_path / "sans_juge.yaml",
        agent={"follow_up_s": 1.0, "follow_up_use_llm_gate": False},
        wake={"refractory_s": 0.0},
    )
    app, audio, _, _, _ = _app(tmp_path, llm)
    app.config = load_config(config_file)

    async def scenario() -> None:
        await _wake_and_say(app, audio, 1)
        await _until(lambda: _in_follow_up(app))
        audio.say(2)
        await _until(lambda: len(app.answers) == 2)

    _run(app, scenario)
    assert "Fenêtre de suivi" not in llm.calls[1][-1]["content"]
