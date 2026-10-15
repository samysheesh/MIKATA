"""Phase 17f : une compétence apprise s'exécute sans le modèle — reconnue par l'aiguillage
dès l'essai — et sans redemander le contrôle qu'elle a déjà choisi (choix retenus)."""

from __future__ import annotations

import asyncio
import json
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from test_agent import NOW, Phrases, Sink
from tool_fakes import FakeUi, make_ctx

from mikata.action.runtime import SkillChoice, choice_key, find_choice
from mikata.agent.loop import Agent, TurnResult
from mikata.learn.jev import JevDecision
from mikata.learn.skills import CHOICES_FILE, Skill, SkillStore
from mikata.llm.base import Done, LLMEvent, TextDelta
from mikata.tools.base import UiControlInfo, UiWindowInfo
from mikata.tools.registry import ToolRegistry

COMMAND = "coupe mon micro sur Discord"
QUESTION = "Quel contrôle est le bouton du micro dans Discord ?"
CODE = (
    "window = uia_window(window=application)\n"
    "control = pick(window.controls, 'Quel contrôle est le bouton du micro dans ' + application"
    " + ' ?')\n"
    "uia_click(window=application, control=control)"
)
WINDOW = UiWindowInfo("Champions league | samania - Discord", "Discord")
# Comme le vrai Discord (relevé le 14/10) : deux « Rendre muet », l'appel et le profil.
CONTROLS = [
    UiControlInfo("Champions league | samania - Discord", "contrôle"),
    UiControlInfo("Options", "texte"),
    UiControlInfo("Rendre muet", "bouton", "désactivé"),
    UiControlInfo("Mettre en sourdine", "bouton"),
    UiControlInfo("Rendre muet", "bouton", "désactivé"),
]
KEY = "Rendre muet — bouton #1"


def _micro_skill() -> Skill:
    return Skill(
        name="couper-micro-application",
        description="Coupe le microphone dans une application de voix comme Discord.",
        patterns=["coupe mon micro sur {application}"],
        params=["application"],
        code=CODE,
    )


class FakeJev:
    """Jev de test : reconnaît la compétence demandée (ou rien) et choisit l'option `pick`."""

    def __init__(self, skill: str | None, pick: int | None) -> None:
        self.skill, self.pick_index = skill, pick
        self.decided: list[str] = []
        self.picked: list[str] = []

    async def prewarm(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def decide(self, sentence: str, skills: list[Skill]) -> JevDecision:
        self.decided.append(sentence)
        skill = next((s for s in skills if s.call_name == self.skill), None)
        if skill is None:
            return JevDecision("aucune", probability=1.0)
        return JevDecision("compétence", skill, {"application": "Discord"}, 1.0)

    async def pick(self, question: str, options: list[str]) -> tuple[int, float] | None:
        self.picked.append(question)
        return None if self.pick_index is None else (self.pick_index, 0.5)


class SlowLLM:
    """L'appel normal : répond au bout d'une seconde, sauf s'il est coupé avant."""

    def __init__(self, answer: str = "Je regarde.") -> None:
        self.answer = answer
        self.started = 0
        self.finished = 0

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[LLMEvent]:
        self.started += 1
        await asyncio.sleep(1.0)
        self.finished += 1
        yield TextDelta(self.answer)
        yield Done("stop")


def _turn(
    tmp_path: Path, jev: FakeJev, ui: FakeUi, llm: SlowLLM | None = None
) -> tuple[TurnResult, SkillStore]:
    store = SkillStore(tmp_path / "skills")
    ctx = make_ctx(tmp_path, uia=ui, skills=store)
    agent = Agent(
        llm or SlowLLM(),  # type: ignore[arg-type]
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        skills=store,
        jev=jev,
    )

    async def scenario() -> TurnResult:
        try:
            result = await agent.respond(COMMAND, Sink())
            await agent.learning_idle()
            return result
        finally:
            await agent.aclose()

    return asyncio.run(scenario()), store


def _adopted(tmp_path: Path) -> Skill:
    return SkillStore(tmp_path / "skills").adopt(_micro_skill())


def _clicks(ui: FakeUi) -> list[Any]:
    return [value for name, value in ui.log if name == "click"]


# -- de bout en bout ------------------------------------------------------------------------


def test_a_learned_skill_runs_without_the_model_then_remembers_its_button(
    tmp_path: Path,
) -> None:
    """La demande de l'utilisateur (15/10) : « coupe mon micro sur Discord » prenait 4 à 13 s
    une fois appris. À l'essai, l'aiguillage la reconnaît : pas d'appel au modèle. Le
    bouton choisi parmi les contrôles est retenu : la fois suivante, plus rien à choisir."""
    _adopted(tmp_path)
    jev, ui, llm = FakeJev("couper_micro_application", 2), FakeUi([WINDOW], CONTROLS), SlowLLM()
    result, store = _turn(tmp_path, jev, ui, llm)
    assert result.answer == "C'est fait." and result.skill_via == "aiguillage"
    assert llm.finished == 0  # l'appel normal a été coupé
    assert jev.picked == [QUESTION] and _clicks(ui) == [(WINDOW.title, "Rendre muet", None, 1)]
    skill = store.get("couper_micro_application")
    assert skill is not None and skill.successes == 1
    saved = json.loads((tmp_path / "skills" / skill.name / CHOICES_FILE).read_text("utf-8"))
    assert [(c["question"], c["valeurs"], c["choix"]) for c in saved] == [
        (QUESTION, {"application": "Discord"}, KEY)
    ]

    again, ui2 = FakeJev("couper_micro_application", None), FakeUi([WINDOW], CONTROLS)
    result, _ = _turn(tmp_path, again, ui2)  # relue sur disque
    assert result.answer == "C'est fait." and again.picked == []  # rien à choisir
    assert _clicks(ui2) == [(WINDOW.title, "Rendre muet", None, 1)]


def test_a_remembered_button_is_forgotten_when_the_skill_fails_with_it(
    tmp_path: Path,
) -> None:
    skill = _adopted(tmp_path)
    store = SkillStore(tmp_path / "skills")
    store.settle_choices(
        skill.call_name,
        [SkillChoice(skill.call_name, QUESTION, {"application": "Discord"}, KEY)],
        True,
    )
    ui = FakeUi([WINDOW], CONTROLS)
    ui.fail = {"click"}
    jev = FakeJev("couper_micro_application", None)
    result, store = _turn(tmp_path, jev, ui, SlowLLM("Je n'ai pas réussi à couper ton micro."))
    assert result.answer == "Je n'ai pas réussi à couper ton micro."  # le modèle reprend
    failed = store.get(skill.call_name)
    assert failed is not None and failed.failures == 1 and failed.choices == []
    assert jev.picked == []  # le choix retenu a servi... puis a été oublié


def test_a_remembered_button_that_disappeared_is_chosen_again(tmp_path: Path) -> None:
    skill = _adopted(tmp_path)
    SkillStore(tmp_path / "skills").settle_choices(
        skill.call_name,
        [SkillChoice(skill.call_name, QUESTION, {"application": "Discord"}, "Micro — bouton #1")],
        True,
    )
    jev = FakeJev("couper_micro_application", 2)
    result, store = _turn(tmp_path, jev, FakeUi([WINDOW], CONTROLS))
    assert result.answer == "C'est fait." and jev.picked == [QUESTION]
    kept = store.get(skill.call_name)
    assert kept is not None and [c["choix"] for c in kept.choices] == [KEY]


# -- identité d'une option, mémoire des choix -------------------------------------------------


def test_only_interface_controls_are_remembered() -> None:
    """Un bouton garde son rôle ; un contenu (lien, élément d'une liste) non : « le dernier
    opening » n'est pas le même d'une saison à l'autre."""
    items: list[Any] = [
        {"name": "Rendre muet", "kind": "bouton", "text": "désactivé"},
        {"name": "One Piece Opening 27", "kind": "lien", "text": ""},
        {"name": "Rendre muet", "kind": "bouton", "text": "activé"},
        {"name": "Jet Lag", "kind": "élément", "text": "Song • Tiakola"},
        "Accueil",
    ]
    assert choice_key(items, 0) == KEY and choice_key(items, 2) == "Rendre muet — bouton #2"
    assert [choice_key(items, i) for i in (1, 3, 4)] == ["", "", ""]
    assert find_choice(items, "Rendre muet — bouton #2") == 2
    assert find_choice(items[:2], "Rendre muet — bouton #2") is None


def test_choices_are_kept_per_question_and_values_and_dropped_on_correction(
    tmp_path: Path,
) -> None:
    store = SkillStore(tmp_path)
    skill = store.adopt(_micro_skill())
    name, discord = skill.call_name, {"application": "Discord"}
    fresh = SkillChoice(name, QUESTION, discord, KEY)
    store.settle_choices(name, [fresh], ok=False)
    assert skill.choices == []  # un échec ne retient rien
    store.settle_choices(name, [fresh], ok=True)
    reloaded = SkillStore(tmp_path).get(name)
    assert reloaded is not None and reloaded.recall(QUESTION, discord) == KEY
    assert skill.recall(QUESTION, {"application": "Teams"}) == ""  # autres valeurs
    store.settle_choices(name, [fresh], ok=False)  # pas repris : rien à oublier
    assert skill.recall(QUESTION, discord) == KEY
    store.correct([name])  # « non, c'est pas ça » juste après
    assert skill.recall(QUESTION, discord) == "" and skill.failures == 1


def test_choices_stay_personal(tmp_path: Path) -> None:
    """Ni exportés, ni crus quand ils viennent d'une compétence importée."""
    mine = SkillStore(tmp_path / "a")
    skill = mine.adopt(_micro_skill())
    mine.settle_choices(
        skill.call_name, [SkillChoice(skill.call_name, QUESTION, {"application": "X"}, KEY)], True
    )
    archive = mine.export(skill.call_name, tmp_path / "exports")
    with zipfile.ZipFile(archive) as packed:
        assert not any(CHOICES_FILE in name for name in packed.namelist())
    assert skill.path is not None and (skill.path / CHOICES_FILE).is_file()
    imported = SkillStore(tmp_path / "b").import_package(skill.path)  # dossier, avec choix
    assert imported.choices == []
