"""Agent v2 (phase 13b) : les inconnues se résolvent du moins cher au plus cher."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent2_fakes import ScriptedLLM, make_registry
from tool_fakes import FakeIndex

from mikata.agent2.resolve import resolve_unknowns
from mikata.agent2.schema import Intent, Unknown, Verb
from mikata.agent2.sdk import Sdk


def _intent(unknowns: list[Unknown]) -> Intent:
    return Intent(verb=Verb.OPEN, object="x", unknowns=unknowns)


def _sdk(registry: Any) -> Sdk:
    return Sdk(registry, max_actions=25, observe_lines=40)


async def _scripted_search(question: str) -> str:
    return f"Résultats pour {question} :\n1. One Piece Opening 27 — « titre exact »"


def _run(
    intent: Intent,
    llm: ScriptedLLM,
    *,
    memory_facts: list[str],
    sdk_patch: str | None = None,
    **ctx: Any,
) -> tuple:
    registry = make_registry(Path(ctx.pop("tmp_path")), **ctx)

    async def scenario() -> tuple:
        sdk = _sdk(registry)
        if sdk_patch == "search":
            sdk.web_search = _scripted_search  # type: ignore[method-assign]
        return await resolve_unknowns(intent, sdk, llm, effort="none", memory_facts=memory_facts)

    return asyncio.run(scenario())


def test_web_unknown_is_resolved(tmp_path: Path) -> None:
    intent = _intent([Unknown(name="titre", question="titre exact ?", kind="web")])
    llm = ScriptedLLM(['{"value": "One Piece Opening 27", "confident": true}'])
    resolved, missing = _run(intent, llm, memory_facts=[], sdk_patch="search", tmp_path=tmp_path)
    assert missing == []
    assert resolved[0].name == "titre" and resolved[0].source == "web"
    assert resolved[0].value == "One Piece Opening 27"


def test_file_unknown_uses_the_indexed_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "Facture EDF 2026-10.pdf").write_text("x", encoding="utf-8")
    intent = _intent([Unknown(name="fichier", question="Facture EDF", kind="file")])
    resolved, missing = _run(
        intent, ScriptedLLM([]), memory_facts=[], tmp_path=tmp_path, index=FakeIndex()
    )
    assert missing == []
    assert resolved[0].source == "files"
    assert "Facture" in resolved[0].value


def test_memory_wins_before_web(tmp_path: Path) -> None:
    intent = _intent([Unknown(name="contact", question="qui est Paul ?", kind="web")])
    llm = ScriptedLLM(['{"value": "Paul Martin", "confident": true}'])
    calls: list[str] = []

    async def fake_search(self: Any, question: str) -> str:
        calls.append(question)
        return "résultats"

    registry = make_registry(tmp_path)

    async def scenario() -> tuple:
        sdk = _sdk(registry)
        sdk.web_search = fake_search  # type: ignore[method-assign]
        return await resolve_unknowns(
            intent, sdk, llm, effort="none", memory_facts=["Paul = Paul Martin sur Discord"]
        )

    resolved, _missing = asyncio.run(scenario())
    assert resolved[0].source == "memory" and resolved[0].value.startswith("Paul")
    assert calls == []  # le web n'a pas été appelé


def test_empty_value_is_unresolved(tmp_path: Path) -> None:
    intent = _intent([Unknown(name="titre", question="titre exact ?", kind="web")])
    llm = ScriptedLLM(['{"value": "", "confident": false}'])
    resolved, missing = _run(intent, llm, memory_facts=[], sdk_patch="search", tmp_path=tmp_path)
    assert resolved == [] and missing == ["titre"]


def test_user_kind_stays_unresolved(tmp_path: Path) -> None:
    intent = _intent([Unknown(name="choix", question="que préfères-tu ?", kind="user")])
    resolved, missing = _run(intent, ScriptedLLM([]), memory_facts=[], tmp_path=tmp_path)
    assert resolved == [] and missing == ["choix"]
