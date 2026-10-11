"""Agent v2 (phase 13d) : registre de test à blanc, pour valider un savoir-faire sans rien
faire. Même interface que ToolRegistry (`schemas()`, `execute()`, `.tools`), n'exécute
rien, renvoie `(simulé)` et garde la liste des appels. Connaît les noms des vrais outils
via le registre des classes (`registered_tools()`), sans état global ajouté. Réutilisé par
la passe de nuit (13e).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mikata.llm.base import ToolCallReq
from mikata.tools.registry import ToolResult, load_builtin_tools, registered_tools


@dataclass
class _DryTool:
    """Sentinelle : l'outil existe, mais son exécution est simulée."""

    name: str
    terminal: bool = False


class DryRunRegistry:
    def __init__(self) -> None:
        load_builtin_tools()
        self.tools = {name: _DryTool(name) for name in registered_tools()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in self.tools
        ]

    async def execute(self, call: ToolCallReq) -> ToolResult:
        self.calls.append((call.name, dict(call.arguments)))
        return ToolResult(
            call.id, call.name, ok=True, content="(simulé)", terminal=False, duration_s=0.0
        )
