"""Interface d'un outil appelable par le LLM (docs/ARCHITECTURE.md §7 et §9).

terminal=True : si l'outil réussit, pas de second appel LLM (« C'est fait » en cache).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str  # en français, 1 phrase, lue par le LLM
    params: type[BaseModel]  # schéma JSON généré via model_json_schema()
    terminal: bool

    async def run(self, args: BaseModel) -> str:
        """Texte court renvoyé au LLM."""
        ...
