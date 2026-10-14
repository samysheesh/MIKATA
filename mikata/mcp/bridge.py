"""Pont MCP -> outils de MIKATA (phase 16f, docs/AGENT_CORE.md §3.11, niveau 3).

Chaque outil d'un serveur configuré (`mcp.servers`) devient un outil ordinaire du registre,
nommé `mcp_<serveur>_<outil>` (et `mcp.<serveur>_<outil>` dans un programme) :

- ses paramètres sont validés à partir de son schéma JSON (types simples ; le reste passe
  tel quel, le serveur vérifie) ;
- son effet vient des indications du serveur : lecture seule -> « lire » ; sinon
  « conséquent », donc un « oui » est demandé après une lecture non fiable (sauf accès
  complet), comme pour n'importe quel outil de MIKATA ;
- ce qu'il rend est toujours « non fiable » : un serveur externe n'est jamais une source
  d'instructions.

Un serveur qui ne démarre pas est signalé dans le journal ; les autres restent utilisables.
"""

from __future__ import annotations

import asyncio
import keyword
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, create_model

from mikata.action.effects import Effect, Trust
from mikata.mcp.client import McpError, McpServer, McpTool
from mikata.tools.base import BaseTool, ToolError

if TYPE_CHECKING:
    from mikata.config import McpConfig
    from mikata.tools.registry import ToolRegistry

_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}
_SAFE = re.compile(r"[^a-z0-9_]+")


def safe_name(text: str, limit: int = 38) -> str:
    """Nom d'outil sûr ; `mcp_` + serveur (20) + outil (38) tient sous les 64 caractères
    qu'acceptent les API des modèles."""
    return _SAFE.sub("_", text.casefold()).strip("_")[:limit].strip("_") or "outil"


def _field_name(name: str, taken: set[str]) -> str:
    """Nom Python du paramètre : le sien s'il est sûr, sinon `arg_...` (`from`, `class`,
    `model_config`, `user-id`...) ; le vrai nom reste l'alias, vu par le modèle et le
    serveur."""
    if (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        and not keyword.iskeyword(name)
        and not name.startswith("model_")
        and not hasattr(BaseModel, name)
    ):
        base = name
    else:
        base = "arg_" + (re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "x")
    field, index = base, 2
    while field in taken:
        field, index = f"{base}_{index}", index + 1
    return field


def params_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Modèle Pydantic des paramètres d'après le schéma JSON de l'outil. Un nom que les
    API des modèles refusent (espaces, accents) est laissé de côté."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
            continue
        kind = _TYPES.get(str(prop.get("type")), Any)
        description = str(prop.get("description") or "")
        field = _field_name(name, set(fields))
        alias = name if field != name else None
        if name in required:
            fields[field] = (kind, Field(description=description, alias=alias))
        else:
            fields[field] = (kind | None, Field(default=None, description=description, alias=alias))
    return create_model(  # type: ignore[call-overload,no-any-return]
        f"Mcp_{safe_name(tool_name)}",
        __config__=ConfigDict(extra="allow", populate_by_name=True),
        **fields,
    )


def tool_class(server: McpServer, tool: McpTool) -> type[BaseTool[Any]]:
    """La classe d'outil MIKATA qui appelle `tool` sur `server`."""
    local = f"{safe_name(server.name, 20)}_{safe_name(tool.name)}"
    read_only = bool(tool.annotations.get("readOnlyHint"))
    description = (tool.description or f"Outil « {tool.name} » du serveur {server.name}.")[:600]

    async def run(self: BaseTool[Any], args: BaseModel) -> str:
        arguments = args.model_dump(by_alias=True, exclude_none=True)
        try:
            ok, text = await server.call_tool(tool.name, arguments)
        except McpError as exc:
            raise ToolError(str(exc)) from exc
        if not ok:
            raise ToolError(text or f"« {tool.name} » a échoué")
        return text or "Fait."

    return type(
        f"Mcp_{local}",
        (BaseTool,),
        {
            "name": f"mcp_{local}",
            "sdk": f"mcp.{local}",
            "description": f"[{server.name}] {description}",
            "params": params_model(tool.name, tool.input_schema),
            "effect": Effect.READ if read_only else Effect.CONSEQUENTIAL,
            "output": Trust.UNTRUSTED,
            "expected_s": 2.0,
            "terminal": False,
            "run": run,
        },
    )


class McpBridge:
    """Les serveurs configurés, démarrés ensemble ; leurs outils ajoutés au registre."""

    def __init__(self, config: McpConfig, env_file: Path | None = None) -> None:
        self.config = config
        self.env_file = env_file
        self.servers: list[McpServer] = []

    def _env(self, values: dict[str, str]) -> dict[str, str]:
        """« ${NOM} » remplacé par la variable d'environnement ou la ligne de .env."""
        from dotenv import dotenv_values

        known = dict(dotenv_values(self.env_file)) if self.env_file is not None else {}
        resolved = {}
        for key, value in values.items():
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())
            if match is None:
                resolved[key] = value
                continue
            found = os.environ.get(match.group(1)) or known.get(match.group(1))
            if found is None:
                logger.warning("MCP : variable {} absente (.env).", match.group(1))
                continue
            resolved[key] = str(found)
        return resolved

    async def start(self, registry: ToolRegistry) -> list[str]:
        """Démarre les serveurs activés ; rend les noms des outils ajoutés."""
        added: list[str] = []
        for spec in self.config.servers:
            if not spec.enabled:
                continue
            server = McpServer(
                spec.name, spec.command, spec.args, self._env(spec.env), self.config.timeout_s
            )
            try:
                await server.start()
                tools = await server.list_tools()
            except (McpError, OSError) as exc:
                logger.warning("Serveur MCP « {} » indisponible : {}", spec.name, exc)
                await server.close()
                continue
            self.servers.append(server)
            for tool in tools[: self.config.max_tools_per_server]:
                try:
                    instance = tool_class(server, tool)(registry.ctx)
                except Exception as exc:  # schéma inutilisable : cet outil seulement
                    logger.warning("Outil MCP « {} » ignoré : {}", tool.name, exc)
                    continue
                if instance.name in registry.tools:
                    continue  # deux outils au même nom une fois nettoyé : le premier reste
                registry.tools[instance.name] = instance
                added.append(instance.name)
            logger.info("Serveur MCP « {} » : {} outil(s).", spec.name, len(tools))
        return added

    async def close(self) -> None:
        await asyncio.gather(*(server.close() for server in self.servers))
        self.servers.clear()
