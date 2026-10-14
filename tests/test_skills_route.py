"""Phase 15j : aiguillage — une compétence active reconnue dans toute formulation ou langue,
en parallèle de l'appel normal, sans attente ajoutée."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from test_agent import NOW, Phrases, Sink
from tool_fakes import FakeAudio, FakeDesktop, make_ctx

from mikata.agent.loop import Agent, TurnResult
from mikata.learn.skills import Skill, SkillStore, locate, pick_route, route_prompt
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq
from mikata.tools.registry import ToolRegistry

CODE = "set_volume(level=int(niveau))\nmedia_control(action='pause')\nreturn None"


def _skill(status: str = "active") -> Skill:
    return Skill(
        name="son-et-pause",
        description="Règle le son puis met la lecture en pause.",
        patterns=["mets le son à {niveau} et pause"],
        params=["niveau"],
        code=CODE,
        status=status,
    )


class RaceLLM:
    """Le petit appel d'aiguillage (reconnu à son prompt) et l'appel normal répondent
    chacun après leur propre délai."""

    def __init__(
        self,
        route_answer: dict[str, Any] | str,
        route_delay: float,
        main: list[Any],
        main_delay: float,
    ) -> None:
        self.route_answer = route_answer
        self.route_delay = route_delay
        self.main = main
        self.main_delay = main_delay
        self.routes = 0
        self.mains = 0
        self.main_finished = 0

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[LLMEvent]:
        if messages[0]["content"].startswith("Tu aiguilles"):
            self.routes += 1
            await asyncio.sleep(self.route_delay)
            answer = self.route_answer
            yield TextDelta(answer if isinstance(answer, str) else json.dumps(answer))
            yield Done("stop")
            return
        self.mains += 1
        await asyncio.sleep(self.main_delay)
        for event in self.main:
            if isinstance(event, Done):
                self.main_finished += 1  # la boucle s'arrête au Done
            yield event


def _run(
    tmp_path: Path, llm: RaceLLM, command: str, status: str = "active"
) -> tuple[TurnResult, Sink, FakeAudio, SkillStore]:
    audio = FakeAudio()
    store = SkillStore(tmp_path / "skills")
    skill = store.adopt(_skill())
    skill.status = status
    store.save(skill)
    ctx = make_ctx(tmp_path, desktop=FakeDesktop(), audio=audio, skills=store)
    agent = Agent(
        llm,  # type: ignore[arg-type]
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        skills=store,
    )
    sink = Sink()

    async def scenario() -> TurnResult:
        try:
            result = await agent.respond(command, sink)
            await agent.learning_idle()
            return result
        finally:
            await agent.aclose()

    return asyncio.run(scenario()), sink, audio, store


def test_any_phrasing_runs_the_active_skill_and_cuts_the_normal_call(tmp_path: Path) -> None:
    llm = RaceLLM(
        {"competence": 1, "valeurs": {"niveau": "35"}},
        0.05,
        [TextDelta("Je règle le son."), Done("stop")],
        0.5,
    )
    result, sink, audio, store = _run(tmp_path, llm, "baisse le volume à 35 puis stoppe la musique")
    assert audio.volume == 35 and result.answer == "C'est fait."
    assert llm.routes == 1 and llm.main_finished == 0  # l'appel normal a été coupé
    assert sink.said == [] and result.skill_via == "aiguillage"
    assert result.path.startswith("compétence « son-et-pause » reconnue par l'aiguillage")
    skill = store.get("son_et_pause")
    assert skill is not None and skill.uses == 1 and skill.status == "active"


def test_no_skill_keeps_the_normal_answer(tmp_path: Path) -> None:
    llm = RaceLLM(
        {"competence": 0, "valeurs": {}},
        0.05,
        [TextDelta("Il fait beau."), Done("stop")],
        0.2,
    )
    result, sink, audio, _ = _run(tmp_path, llm, "quel temps fait-il ?")
    assert result.answer == "Il fait beau." and sink.said == ["Il fait beau."]
    assert audio.volume == 50 and result.skill_via == "" and llm.main_finished == 1


def test_a_normal_call_that_already_spoke_is_never_cut(tmp_path: Path) -> None:
    class Slow(RaceLLM):
        async def stream(self, messages: list[dict[str, Any]], tools: Any, **kw: Any) -> Any:
            if messages[0]["content"].startswith("Tu aiguilles"):
                async for event in super().stream(messages, tools, **kw):
                    yield event
                return
            self.mains += 1
            yield TextDelta("Je regarde ça tout de suite pour toi, une petite seconde. ")  # dit...
            await asyncio.sleep(0.3)  # ... et l'aiguillage répond pendant ce temps
            yield TextDelta("c'est réglé.")
            self.main_finished += 1
            yield Done("stop")

    llm = Slow({"competence": 1, "valeurs": {"niveau": "35"}}, 0.05, [], 0)
    result, _, audio, _ = _run(tmp_path, llm, "baisse le volume à 35 puis stoppe la musique")
    assert llm.main_finished == 1 and result.answer.endswith("c'est réglé.")
    assert audio.volume == 50  # jamais deux réponses à une même demande


def test_a_router_within_the_grace_period_wins_over_unexecuted_tools(tmp_path: Path) -> None:
    """L'appel normal demande un outil sans rien dire ; l'aiguillage répond juste après :
    la compétence s'exécute, l'outil demandé ne s'exécute jamais."""
    llm = RaceLLM(
        {"competence": 1, "valeurs": {"niveau": "35"}},
        0.3,
        [ToolCallReq("c1", "set_volume", {"level": 20}), Done("tool_calls")],
        0.05,
    )
    result, _, audio, _ = _run(tmp_path, llm, "baisse le volume à 35 puis stoppe la musique")
    assert audio.volume == 35 and result.skill_via == "aiguillage"


def test_a_slow_router_changes_nothing(tmp_path: Path) -> None:
    llm = RaceLLM(
        {"competence": 1, "valeurs": {"niveau": "35"}},
        2.0,
        [ToolCallReq("c1", "set_volume", {"level": 20}), Done("tool_calls")],
        0.05,
    )
    result, _, audio, _ = _run(tmp_path, llm, "baisse le volume à 35 puis stoppe la musique")
    assert audio.volume == 20 and result.skill_via == ""


def test_trial_skills_never_go_through_the_router(tmp_path: Path) -> None:
    """Le cycle décidé : une compétence ne s'exécute sans le modèle qu'une fois active."""
    llm = RaceLLM(
        {"competence": 1, "valeurs": {"niveau": "35"}}, 0.0, [TextDelta("Ok."), Done("stop")], 0.05
    )
    result, _, audio, _ = _run(tmp_path, llm, "baisse le volume à 35", status="trial")
    assert llm.routes == 0 and audio.volume == 50 and result.answer == "Ok."


@pytest.mark.parametrize(
    ("answer", "command", "expected"),
    [
        (
            '{"competence": 1, "valeurs": {"niveau": "35"}}',
            "le son à 35 et pause",
            {"niveau": "35"},
        ),
        (
            'ok : {"compétence": 1, "valeurs": {"niveau": "35"}}',
            "son 35 puis pause",
            {"niveau": "35"},
        ),
        (
            '{"competence": 1, "valeurs": {"niveau": "80"}}',
            "le son à 35 et pause",
            None,
        ),  # inventée
        ('{"competence": 1, "valeurs": {}}', "le son à 35 et pause", None),  # valeur manquante
        ('{"competence": 2, "valeurs": {"niveau": "35"}}', "le son à 35 et pause", None),
        ('{"competence": 0, "valeurs": {}}', "le son à 35 et pause", None),
        ("je ne sais pas", "le son à 35 et pause", None),
    ],
)
def test_router_answers_are_checked_in_code(
    answer: str, command: str, expected: dict[str, str] | None
) -> None:
    picked = pick_route(answer, [_skill()], command)
    assert (picked[1] if picked else None) == expected


def test_values_keep_the_spoken_spelling_and_prompt_lists_skills() -> None:
    assert locate("pon el último opening de One Piece", "one piece") == "One Piece"
    assert locate("lance l'opening de Naruto", "Bleach") is None
    prompt = route_prompt([_skill()])
    assert "1. skills.son_et_pause(niveau) : Règle le son" in prompt
