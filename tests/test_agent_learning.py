# ruff: noqa: E501  (programmes et réponses de test sur une ligne)
"""Phase 14e (docs/AGENT_CORE.md §3.11) : la boucle agent apprend de ses tours.

Les leçons du carnet rejoignent le contexte quand elles se rapportent à la demande ; après un
tour, elles sont notées utiles ou nuisibles ; un échec corrigé ou une correction de
l'utilisateur donne une leçon, tirée en arrière-plan. Rien n'est appris d'un contenu lu."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from test_agent_scripts import PAGE, FakeLLM, Sink, script
from tool_fakes import FakeAudio, FakeBrowser, FakeDesktop, FakeUi, make_config, make_ctx

from mikata.agent.loop import Agent, TurnResult
from mikata.agent.memory import ShortTermMemory
from mikata.learn.playbook import Playbook
from mikata.learn.trace import TraceStore
from mikata.llm.base import LLMEvent, TextDelta, ToolCallReq
from mikata.tools.base import UiControlInfo
from mikata.tools.registry import ToolRegistry

MICRO = "Dans Discord, pour couper le micro, clique « Rendre muet » ; « Mettre en sourdine » coupe aussi le son."


class Learner:
    """Agent complet sur de faux services, avec un carnet et des traces dans tmp_path."""

    def __init__(self, tmp_path: Path, *turns: list[LLMEvent], **config: dict[str, Any]) -> None:
        self.audio, self.desktop = FakeAudio(), FakeDesktop()
        self.browser = FakeBrowser(page=PAGE)
        self.ui = FakeUi(
            controls=[
                UiControlInfo("Rendre muet", "bouton"),
                UiControlInfo("Mettre en sourdine", "bouton"),
            ]
        )
        self.ctx = make_ctx(
            tmp_path,
            config=make_config(tmp_path, **config),
            desktop=self.desktop,
            audio=self.audio,
            browser=self.browser,
            uia=self.ui,
        )
        self.llm = FakeLLM(*turns)
        self.playbook = Playbook(tmp_path / "playbook.json")
        self.traces = TraceStore(tmp_path / "traces")
        self.agent = Agent(
            self.llm,
            self.ctx.config,
            memory=ShortTermMemory(6000, 600.0),
            tools=ToolRegistry(self.ctx),
            playbook=self.playbook,
            traces=self.traces,
            window_title=lambda: None,  # jamais la vraie fenêtre active du PC de test
        )
        self.sink = Sink()

    def respond(self, command: str) -> TurnResult:
        async def scenario() -> TurnResult:
            result = await self.agent.respond(command, self.sink)
            await self.agent.learning_idle()  # la réflexion tourne en arrière-plan
            return result

        return asyncio.run(scenario())

    def system(self, call: int = 0) -> list[str]:
        return [m["content"] for m in self.llm.messages[call] if m["role"] == "system"]


def test_a_relevant_lesson_joins_the_context_and_is_scored(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path, [ToolCallReq("a", "uia_click", {"control": "Rendre muet", "window": "Discord"})]
    )
    lesson = learner.playbook.add("discord", MICRO, "correction")
    learner.respond("coupe mon micro sur Discord")
    block = next(text for text in learner.system() if text.startswith("Leçons tirées"))
    assert MICRO in block and "pas des règles" in block
    assert ("click", ("Discord", "Rendre muet", None, 1)) in learner.ui.log
    assert lesson.helpful == 1 and lesson.harmful == 0  # présente, et le tour a réussi

    # Une demande sans rapport ne reçoit aucune leçon : le chemin direct reste léger.
    other = Learner(tmp_path, [ToolCallReq("a", "set_volume", {"level": 30})])
    other.respond("mets le volume à trente")
    assert not any(text.startswith("Leçons tirées") for text in other.system())


def test_a_lesson_present_during_a_failure_is_scored_harmful(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path,
        [ToolCallReq("a", "uia_click", {"control": "Bouton absent", "window": "Discord"})],
        [TextDelta("Je ne trouve pas ce bouton.")],
    )
    learner.ui.fail = {"click"}
    lesson = learner.playbook.add("discord", MICRO, "correction")
    learner.respond("coupe mon micro sur Discord")
    assert lesson.harmful == 1 and lesson.helpful == 0


def test_a_recovered_failure_becomes_a_lesson_in_the_background(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path,
        [script("open_app('Photoshop')\nreturn 'Ouvert.'", "a")],
        [script("open_app('Steam')\nreturn 'Steam est ouvert.'", "b")],
        [
            TextDelta(
                '{"lecons": [{"domaine": "applications", "texte": "Si une application est introuvable, ouvre la plus proche parmi celles qui sont installées."}]}'
            )
        ],
    )
    result = learner.respond("ouvre mon logiciel")
    assert (
        result.answer == "Steam est ouvert." and result.llm_calls == 2
    )  # la réflexion ne compte pas dans le tour
    assert [lesson.scope for lesson in learner.playbook.lessons] == ["applications"]
    assert learner.playbook.lessons[0].source == "échec corrigé"
    reflection = learner.llm.messages[2]
    assert learner.llm.offered[2] == []  # la réflexion n'a aucun outil
    assert "Programme 1, qui a échoué" in reflection[1]["content"]
    assert "Programme 2, qui a réussi" in reflection[1]["content"]
    # La trace du tour est sur disque, lisible.
    saved = learner.traces.recent(1)[0]
    assert saved.request == "ouvre mon logiciel" and [s.ok for s in saved.scripts] == [False, True]


def test_nothing_is_learned_from_a_turn_that_read_untrusted_content(tmp_path: Path) -> None:
    """Un texte piégé ne doit pas pouvoir écrire dans le carnet, même par ricochet."""
    learner = Learner(
        tmp_path,
        [script("page = browser_read()\nopen_app('Photoshop')", "a")],
        [script("page = browser_read()\nopen_app('Steam')\nreturn 'Fait.'", "b")],
    )
    learner.respond("lis la page puis ouvre mon logiciel")
    assert learner.playbook.lessons == []
    assert len(learner.llm.messages) == 2  # aucun appel de réflexion
    assert learner.traces.recent(1)[0].tainted
    saved = json.loads(next((tmp_path / "traces").glob("*/*.json")).read_text(encoding="utf-8"))
    assert "ASSISTANT" not in json.dumps(saved, ensure_ascii=False)  # le contenu lu n'y est pas


def test_a_user_correction_becomes_a_lesson(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path,
        [ToolCallReq("a", "uia_click", {"control": "Mettre en sourdine", "window": "Discord"})],
        [ToolCallReq("b", "uia_click", {"control": "Rendre muet", "window": "Discord"})],
        [
            TextDelta(
                '{"lecons": [{"domaine": "discord", "texte": "Pour couper le micro dans Discord, clique Rendre muet, pas Mettre en sourdine."}]}'
            )
        ],
    )
    learner.respond("coupe mon micro sur Discord")
    assert learner.playbook.lessons == []  # réussi en apparence : rien à apprendre
    learner.respond("non, je voulais juste couper mon micro")
    assert [lesson.source for lesson in learner.playbook.lessons] == ["correction"]
    user = learner.llm.messages[2][1]["content"]
    assert "Ce que MIKATA avait fait juste avant" in user and "L'utilisateur l'a corrigé" in user
    # La fois suivante, la leçon est dans le contexte.
    again = Learner(
        tmp_path, [ToolCallReq("c", "uia_click", {"control": "Rendre muet", "window": "Discord"})]
    )
    again.respond("coupe mon micro sur Discord")
    assert any("Rendre muet, pas Mettre en sourdine" in text for text in again.system())


def test_a_correction_blames_the_lessons_that_were_present(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path,
        [ToolCallReq("a", "uia_click", {"control": "Mettre en sourdine", "window": "Discord"})],
        [ToolCallReq("b", "uia_click", {"control": "Rendre muet", "window": "Discord"})],
        [TextDelta('{"lecons": []}')],
    )
    wrong = learner.playbook.add(
        "discord", "Dans Discord, le bouton du micro s'appelle Mettre en sourdine.", "échec corrigé"
    )
    learner.respond("coupe mon micro sur Discord")
    assert wrong.helpful == 1  # elle était là et le tour a « réussi »
    learner.respond("non, pas ça, mon micro seulement")
    assert wrong.harmful == 1  # la correction la met en cause


def test_learning_can_be_turned_off(tmp_path: Path) -> None:
    learner = Learner(
        tmp_path,
        [script("open_app('Photoshop')", "a")],
        [script("open_app('Steam')\nreturn 'Ouvert.'", "b")],
        learning={"enabled": False},
    )
    learner.playbook.add(
        "applications",
        "Si une application est introuvable, ouvre la plus proche installée.",
        "échec corrigé",
    )
    learner.respond("ouvre mon application introuvable")
    assert not any(text.startswith("Leçons tirées") for text in learner.system())
    assert len(learner.llm.messages) == 2 and learner.traces.recent(5) == []


def test_a_plain_answer_leaves_no_trace(tmp_path: Path) -> None:
    learner = Learner(tmp_path, [TextDelta("Canberra.")])
    learner.respond("quelle est la capitale de l'Australie")
    assert learner.traces.recent(5) == [] and learner.playbook.lessons == []
