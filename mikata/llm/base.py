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
class Usage:
    """Tokens d'un appel (ajout phase 4, pour le coût). `cached` ⊂ `prompt`,
    `reasoning` ⊂ `completion`."""

    prompt: int = 0
    completion: int = 0
    cached: int = 0
    reasoning: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt + other.prompt,
            self.completion + other.completion,
            self.cached + other.cached,
            self.reasoning + other.reasoning,
        )


@dataclass
class Done:
    finish_reason: str
    # Ajouts phase 4 (facultatifs) : tokens consommés ; réflexion du modèle (DeepSeek V4 :
    # `reasoning_content`) : durée avant la fin de la réflexion et texte, à renvoyer au
    # modèle avec ses appels d'outils.
    usage: Usage | None = None
    reasoning_s: float | None = None
    reasoning: str = ""


LLMEvent: TypeAlias = TextDelta | ToolCallReq | Done


@runtime_checkable
class LLMClient(Protocol):
    # Ajout phase 4 : `reasoning_effort` (none | low | medium | high), choisi par commande.
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = ...,
    ) -> AsyncIterator[LLMEvent]: ...
