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
import re
import time
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from mikata.agent.guard import log_action
from mikata.llm.base import ToolCallReq
from mikata.tools.base import ActionCancelled, GuardDenied, Tool, ToolContext, ToolError

# Phase 4 : system, apps, web, media, ui, memory ; phase 5 : input, files, shell (§7, §11) ;
# phase 12a : uia (pilotage fin des applications) ; phase 12b : browser (navigateur CDP) ;
# phase 12c : skills (savoir-faire appris).
FAMILIES = (
    "system",
    "apps",
    "web",
    "media",
    "ui",
    "memory_tools",
    "input",
    "files",
    "shell",
    "uia",
    "browser",
    "skills",
)

_REGISTRY: dict[str, type[Any]] = {}
SDK_NAME_RE = re.compile(r"^[a-z][a-z_]*\.[a-z][a-z_]*$")
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
    sdk = getattr(cls, "sdk", "")
    if sdk and not SDK_NAME_RE.match(sdk):
        raise TypeError(f"{cls.__name__} : `sdk` doit avoir la forme « espace.fonction »")
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
    data: dict[str, Any] | None = None  # payload structuré (phase 12e), jamais envoyé au LLM
    status: str = "ok"  # ok | échec | refusé | annulé | délai (phase 14c : lu par les scripts)


# Payload structuré qu'un outil peut publier pendant son exécution (même tâche que
# `execute`, donc un ContextVar suffit) : le rejeu local des savoir-faire s'en sert pour
# lier les étapes entre elles (ex. le chemin trouvé par find_path -> open_in_explorer).
_TOOL_DATA: ContextVar[dict[str, Any] | None] = ContextVar("tool_data", default=None)


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
        """Index des applications construit d'avance (premier « ouvre ... » sans attente) ;
        sauvegardes trop vieilles envoyées à la corbeille (phase 5)."""
        if self.ctx.guard is not None:
            try:
                await asyncio.to_thread(self.ctx.guard.purge_backups)
            except Exception as exc:
                logger.warning("Purge des sauvegardes impossible : {}", exc)
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

    def time_limit(self, tool: Tool) -> float:
        limit = getattr(tool, "time_limit", None)
        own = limit() if callable(limit) else None
        return own or self.ctx.config.tools.timeout_s

    async def execute(self, call: ToolCallReq) -> ToolResult:
        start = time.monotonic()
        tool = self.tools.get(call.name)
        terminal = tool is not None and tool.terminal
        status = "échec"
        data: dict[str, Any] | None = None
        if tool is None:
            ok, content = False, f"outil inconnu : {call.name}"
        else:
            limit = self.time_limit(tool)
            # Phase 12a : le poller d'état ambiant (UIA) saute ses lectures pendant qu'un
            # outil agit, pour ne pas lire un écran en train de changer.
            uia = self.ctx.uia
            if uia is not None:
                uia.pause_state(True)
            try:
                args = tool.params.model_validate(call.arguments)
                terminal = tool.concludes(args)
                _TOOL_DATA.set(None)
                async with asyncio.timeout(limit):
                    content, ok = await tool.run(args), True
                data = _TOOL_DATA.get()
                status = "ok"
            except ValidationError as exc:
                ok, content = False, _validation_message(call.name, exc)
            except GuardDenied as exc:
                ok, content, status = False, f"refusé par le garde-fou : {exc}", "refusé"
            except ActionCancelled as exc:
                ok, content, status = False, str(exc), "annulé"
            except ToolError as exc:
                ok, content = False, str(exc)
            except TimeoutError:
                ok, status = False, "délai"
                content = f"{call.name} n'a pas répondu en {limit:.0f} s"
            except asyncio.CancelledError:
                log_action(call.name, call.arguments, "interrompu")
                raise
            except Exception as exc:
                logger.exception("Outil {} : erreur inattendue.", call.name)
                ok, content = False, f"erreur inattendue dans {call.name} : {exc}"
            finally:
                if uia is not None:
                    uia.pause_state(False)
        log_action(call.name, call.arguments, status, content)
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
        return ToolResult(call.id, call.name, ok, content, terminal, duration, data, status)
