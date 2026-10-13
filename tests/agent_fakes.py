"""Outils de test partagés : LLM scripté et registre d'outils simulé.

Le LLM est scripté, le registre est branché sur les faux services de tests/tool_fakes.py :
rien ne touche le vrai PC ni le réseau.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from mikata.llm.base import Done, LLMEvent, TextDelta, Usage
from mikata.tools.registry import ToolRegistry


class ScriptedLLM:
    """LLMClient de test : renvoie les réponses données, dans l'ordre, et garde les
    messages reçus (self.calls). Lève AssertionError s'il n'y a plus de réponse."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append(messages)
        assert self.replies, "ScriptedLLM : plus de réponse disponible"
        reply = self.replies.pop(0)
        yield TextDelta(reply)
        yield Done("stop", usage=Usage(10, 5))


def make_registry(tmp_path: Path, **ctx_overrides: Any) -> ToolRegistry:
    """Registre avec tous les outils intégrés et les services simulés de
    tests/tool_fakes.py (FakeDesktop, FakeBrowser, FakeUi, FakeIndex...)."""
    from tool_fakes import make_ctx

    return ToolRegistry(make_ctx(tmp_path, **ctx_overrides))
