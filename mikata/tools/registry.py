"""Registre des outils (§7, §9) : un outil = une classe enregistrée par `@register_tool`.

- `ToolRegistry.schemas()` : définitions JSON pour le LLM, triées par nom (identiques d'un
  appel à l'autre, pour le cache de prompt).
- `ToolRegistry.execute()` : arguments validés par le modèle Pydantic de l'outil, délai
  maximal, résultat tronqué (~2 000 caractères, §6) ; une erreur devient un message pour le
  LLM, jamais une exception qui casserait la boucle agent.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from mikata.llm.base import ToolCallReq
from mikata.tools.base import Tool, ToolContext, ToolError

FAMILIES = ("system", "apps", "web", "media", "ui", "memory_tools")  # phase 4 (§7, §11)

_REGISTRY: dict[str, type[Any]] = {}
T = TypeVar("T", bound=type)


def register_tool(cls: T) -> T:
    """Décorateur : vérifie la forme de l'outil et l'enregistre sous son nom."""
    name = getattr(cls, "name", None)
    params = getattr(cls, "params", None)
    if not isinstance(name, str) or not name.isidentifier():
        raise TypeError(f"{cls.__name__} : `name` doit être un identifiant")
    if not isinstance(getattr(cls, "description", None), str) or not cls.description:
        raise TypeError(f"{cls.__name__} : `description` obligatoire (une phrase, en français)")
    if not (isinstance(params, type) and issubclass(params, BaseModel)):
        raise TypeError(f"{cls.__name__} : `params` doit être un modèle Pydantic")
    if not isinstance(getattr(cls, "terminal", None), bool):
        raise TypeError(f"{cls.__name__} : `terminal` doit être un booléen")
    if not asyncio.iscoroutinefunction(getattr(cls, "run", None)):
        raise TypeError(f"{cls.__name__} : `run` doit être une coroutine")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"outil « {name} » enregistré deux fois")
    _REGISTRY[name] = cls
    return cls


def registered_tools() -> dict[str, type[Any]]:
    return dict(_REGISTRY)


def load_builtin_tools() -> None:
    """Importe les familles d'outils : leurs `@register_tool` remplissent le registre."""
    for family in FAMILIES:
        importlib.import_module(f"mikata.tools.{family}")


def _clean_schema(node: Any) -> Any:
    """Schéma Pydantic allégé : sans les annotations « title » (tokens inutiles pour le
    LLM). Un paramètre qui s'appelle « title » (show_text) est gardé."""
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("properties", "$defs") and isinstance(value, dict):
            cleaned[key] = {name: _clean_schema(sub) for name, sub in value.items()}
        elif key != "title":
            cleaned[key] = _clean_schema(value)
    return cleaned


def function_schema(tool: Tool) -> dict[str, Any]:
    parameters = _clean_schema(tool.params.model_json_schema())
    parameters.pop("description", None)  # docstring du modèle : la description de l'outil suffit
    parameters.setdefault("properties", {})
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": parameters},
    }


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: str
    terminal: bool
    duration_s: float


def _validation_message(name: str, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'arguments'} : {err['msg']}"
        for err in exc.errors()
    )
    return f"arguments invalides pour {name} ({problems})"


class ToolRegistry:
    def __init__(self, ctx: ToolContext, include: Iterable[str] | None = None) -> None:
        self.ctx = ctx
        load_builtin_tools()
        wanted = set(include) if include is not None else None
        self.tools: dict[str, Tool] = {
            name: cls(ctx)
            for name, cls in sorted(_REGISTRY.items())
            if wanted is None or name in wanted
        }

    async def warm(self) -> None:
        """Index des applications construit d'avance (premier « ouvre ... » sans attente)."""
        desktop = self.ctx.desktop
        if desktop is None:
            return
        try:
            await self.ctx.run_blocking(desktop.list_apps)
        except Exception as exc:
            logger.warning("Index des applications indisponible : {}", exc)

    async def aclose(self) -> None:
        await self.ctx.aclose()

    def schemas(self) -> list[dict[str, Any]]:
        return [function_schema(tool) for tool in self.tools.values()]

    def is_terminal(self, name: str) -> bool:
        tool = self.tools.get(name)
        return bool(tool is not None and tool.terminal)

    async def execute(self, call: ToolCallReq) -> ToolResult:
        start = time.monotonic()
        tool = self.tools.get(call.name)
        terminal = tool is not None and tool.terminal
        if tool is None:
            ok, content = False, f"outil inconnu : {call.name}"
        else:
            try:
                args = tool.params.model_validate(call.arguments)
                async with asyncio.timeout(self.ctx.config.tools.timeout_s):
                    content, ok = await tool.run(args), True
            except ValidationError as exc:
                ok, content = False, _validation_message(call.name, exc)
            except ToolError as exc:
                ok, content = False, str(exc)
            except TimeoutError:
                ok = False
                content = f"{call.name} n'a pas répondu en {self.ctx.config.tools.timeout_s:.0f} s"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Outil {} : erreur inattendue.", call.name)
                ok, content = False, f"erreur inattendue dans {call.name} : {exc}"
        limit = self.ctx.config.tools.result_max_chars
        if len(content) > limit:
            content = content[:limit] + " […tronqué]"
        duration = time.monotonic() - start
        logger.info(
            "Outil {}({}) -> {} en {:.0f} ms : {}",
            call.name,
            ", ".join(f"{k}={v!r}" for k, v in call.arguments.items()),
            "ok" if ok else "ÉCHEC",
            duration * 1000,
            content[:160].replace("\n", " "),
        )
        return ToolResult(call.id, call.name, ok, content, terminal, duration)
