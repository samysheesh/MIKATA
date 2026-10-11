"""Agent v2 (phase 13a) : ask_json — JSON extrait, retenté une fois, jamais d'exception."""

from __future__ import annotations

import asyncio

from agent2_fakes import ScriptedLLM
from pydantic import BaseModel

from mikata.agent2.llm_json import ask_json


class Answer(BaseModel):
    value: str


def _run(llm: ScriptedLLM) -> tuple[Answer | None, object, int]:
    async def scenario() -> tuple[Answer | None, object, int]:
        return await ask_json(llm, "systeme", "question", Answer, effort="none")

    return asyncio.run(scenario())


def test_fenced_json_is_extracted() -> None:
    llm = ScriptedLLM(['voici la réponse :\n```json\n{"value": "bonjour"}\n```\nvoilà'])
    answer, usage, calls = _run(llm)
    assert answer is not None and answer.value == "bonjour"
    assert calls == 1 and usage is not None


def test_invalid_then_valid_retries_with_the_error() -> None:
    llm = ScriptedLLM(['{"valeur": "x"}', '{"value": "ok"}'])
    answer, _usage, calls = _run(llm)
    assert answer is not None and answer.value == "ok"
    assert calls == 2
    second_user = llm.calls[1][-1]["content"]
    assert "Réponse invalide" in second_user and "valeur" in second_user


def test_two_invalid_answers_give_none() -> None:
    llm = ScriptedLLM(['{"valeur": "x"}', "pas du json"])
    answer, _, calls = _run(llm)
    assert answer is None and calls == 2


def test_llm_exception_gives_none() -> None:
    class BrokenLlm:
        async def stream(self, messages, tools, reasoning_effort=None):  # type: ignore[no-untyped-def]
            raise ConnectionError("réseau coupé")
            yield  # pragma: no cover

    async def scenario() -> None:
        answer, _usage, calls = await ask_json(
            BrokenLlm(),
            "s",
            "u",
            Answer,
            effort=None,  # type: ignore[arg-type]
        )
        assert answer is None and calls == 0

    asyncio.run(scenario())
