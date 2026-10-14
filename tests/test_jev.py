"""Phase 15k : Jev (TypeSafe, via OpenRouter) en premier étage de l'aiguillage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from test_skills_route import RaceLLM, _run

from mikata.learn.jev import (
    JEV_MODEL,
    JEV_URL,
    NONE_LABEL,
    JevDecision,
    JevRouter,
    build_request,
    read_answers,
    spans,
)
from mikata.learn.skills import Skill
from mikata.llm.base import Done, TextDelta

SKILL = Skill(
    name="lancer-opening",
    description="Lance sur YouTube le dernier opening d'un anime.",
    patterns=["lance le dernier opening de {anime}"],
    params=["anime"],
    code="",
    status="active",
)


def _answers(probabilities: dict[str, float], span: str | None = None) -> dict[str, Any]:
    answers: dict[str, Any] = {
        "competence": {
            "type": "choice",
            "choice": max(probabilities, key=lambda k: probabilities[k], default=""),
            "probabilities": probabilities,
        }
    }
    if span is not None:
        answers["lancer_opening__anime"] = {"type": "choice", "choice": span}
    return {"answers": answers, "usage": {"cost": 0.00004}}


def test_spans_keep_the_spoken_spelling() -> None:
    found = spans("balance-moi le dernier opening de Dr. Stone !")
    assert "Dr. Stone" in found and "balance-moi" in found and "opening de Dr. Stone" in found
    assert all(len(span.split()) <= 5 for span in found)
    assert len(found) == len(set(found))


def test_request_asks_the_skill_and_each_value_in_one_call() -> None:
    body, candidates = build_request("play the latest Bleach opening", [SKILL])
    assert body["model"] == JEV_MODEL and body["state"] == {
        "demande": "play the latest Bleach opening"
    }
    skill_question = body["questions"]["competence"]
    assert skill_question["type"] == "choice"
    assert set(skill_question["criteria"]) == {"lancer_opening", NONE_LABEL}
    assert "« lance le dernier opening de {anime} »" in skill_question["criteria"]["lancer_opening"]
    options = body["questions"]["lancer_opening__anime"]["criteria"]
    assert options[f"m{candidates.index('Bleach')}"] == "« Bleach »"


@pytest.mark.parametrize(
    ("probabilities", "span", "kind", "values"),
    [
        ({"lancer_opening": 0.97, NONE_LABEL: 0.03}, "m3", "compétence", {"anime": "Bleach"}),
        ({"lancer_opening": 0.03, NONE_LABEL: 0.97}, None, "aucune", {}),
        ({"lancer_opening": 0.68, NONE_LABEL: 0.32}, "m3", "incertain", {}),  # doute
        ({"lancer_opening": 0.97, NONE_LABEL: 0.03}, "m99", "incertain", {}),  # option hors liste
        ({"autre": 0.97, NONE_LABEL: 0.03}, None, "incertain", {}),  # compétence inconnue
        ({}, None, "incertain", {}),
    ],
)
def test_answers_are_read_with_a_threshold(
    probabilities: dict[str, float], span: str | None, kind: str, values: dict[str, str]
) -> None:
    sentence = "play the latest Bleach opening"
    _, candidates = build_request(sentence, [SKILL])
    assert candidates[3] == "Bleach"
    decision = read_answers(_answers(probabilities, span), [SKILL], candidates, 0.9)
    assert decision.kind == kind and decision.values == values


def _router(handler: Any) -> JevRouter:
    return JevRouter("sk-or-test", transport=httpx2.MockTransport(handler))


def test_router_calls_openrouter_and_never_sends_the_key_elsewhere() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_answers({"lancer_opening": 1.0, NONE_LABEL: 0}, "m3"))

    async def scenario() -> JevDecision:
        router = _router(handler)
        try:
            return await router.decide("play the latest Bleach opening", [SKILL])
        finally:
            await router.aclose()

    decision = asyncio.run(scenario())
    assert decision.kind == "compétence" and decision.values == {"anime": "Bleach"}
    (request,) = seen
    assert str(request.url) == JEV_URL and request.headers["authorization"] == "Bearer sk-or-test"
    assert json.loads(request.content)["questions"]["competence"]["type"] == "choice"


def test_router_errors_mean_unsure() -> None:
    async def scenario() -> JevDecision:
        router = _router(lambda request: httpx2.Response(401, json={"error": "clé refusée"}))
        try:
            return await router.decide("play the latest Bleach opening", [SKILL])
        finally:
            await router.aclose()

    assert asyncio.run(scenario()).kind == "incertain"


# -- dans l'aiguillage ------------------------------------------------------------------------


class FakeJev:
    def __init__(self, kind: str, values: dict[str, str] | None = None, delay: float = 0.02):
        self.kind = kind
        self.values = values or {}
        self.delay = delay
        self.calls = 0

    async def decide(self, sentence: str, skills: list[Skill]) -> JevDecision:
        self.calls += 1
        await asyncio.sleep(self.delay)
        skill = skills[0] if self.kind == "compétence" else None
        return JevDecision(self.kind, skill, self.values, 0.97)

    async def prewarm(self) -> None: ...

    async def aclose(self) -> None: ...


def _with_jev(monkeypatch: pytest.MonkeyPatch, jev: FakeJev) -> None:
    from mikata.agent import loop

    original = loop.Agent.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.jev = jev

    monkeypatch.setattr(loop.Agent, "__init__", init)


COMMAND = "baisse le volume à 35 puis stoppe la musique"
ANSWER = [TextDelta("Très bien."), Done("stop")]


def test_a_confident_jev_decides_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jev = FakeJev("compétence", {"niveau": "35"})
    _with_jev(monkeypatch, jev)
    llm = RaceLLM({"competence": 0, "valeurs": {}}, 0.5, ANSWER, 0.6)
    result, _, audio, _ = _run(tmp_path, llm, COMMAND)
    assert jev.calls == 1 and audio.volume == 35 and result.skill_via == "aiguillage"


def test_a_confident_no_from_jev_keeps_the_normal_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_jev(monkeypatch, FakeJev("aucune"))
    llm = RaceLLM({"competence": 1, "valeurs": {"niveau": "35"}}, 0.3, ANSWER, 0.5)
    result, _, audio, _ = _run(tmp_path, llm, COMMAND)
    assert audio.volume == 50 and result.answer == "Très bien." and result.skill_via == ""


def test_an_unsure_jev_lets_the_llm_router_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_jev(monkeypatch, FakeJev("incertain"))
    llm = RaceLLM({"competence": 1, "valeurs": {"niveau": "35"}}, 0.1, ANSWER, 0.6)
    result, _, audio, _ = _run(tmp_path, llm, COMMAND)
    assert audio.volume == 35 and result.skill_via == "aiguillage"


def test_a_value_jev_did_not_take_from_the_sentence_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_jev(monkeypatch, FakeJev("compétence", {"niveau": "80"}))  # absent de la phrase
    llm = RaceLLM({"competence": 0, "valeurs": {}}, 0.1, ANSWER, 0.4)
    result, _, audio, _ = _run(tmp_path, llm, COMMAND)
    assert audio.volume == 50 and result.answer == "Très bien."
