"""Interface du client LLM et événements de stream (docs/ARCHITECTURE.md §6 et §9)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallReq:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Done:
    finish_reason: str


LLMEvent: TypeAlias = TextDelta | ToolCallReq | Done


@runtime_checkable
class LLMClient(Protocol):
    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]: ...
