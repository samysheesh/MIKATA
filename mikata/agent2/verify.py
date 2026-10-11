"""Agent v2 (phase 13c) : vérifier qu'un sous-but est atteint, sans jamais planter.

Chaque prédicat se vérifie sur l'observation et sur les actions déjà faites ; seul le
prédicat « judge » fait un appel LLM (données entre marqueurs, jamais d'instruction).
Une regex invalide vaut « non atteint, fiable » : c'est le plan, pas l'exécution, qui est
en cause — l'orchestrateur échouera proprement.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from mikata.agent2.llm_json import ask_json
from mikata.agent2.prompts import JUDGE_SYSTEM, judge_user
from mikata.agent2.schema import ActionRecord, Observation, Predicate, ToolSucceeded
from mikata.llm.base import LLMClient
from mikata.tools.registry import load_builtin_tools, registered_tools

load_builtin_tools()
_KNOWN_TOOLS = frozenset(registered_tools())


class _JudgeAnswer(BaseModel):
    reached: bool
    why: str = ""


def _search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.I) is not None
    except re.error as exc:
        logger.warning("Prédicat : regex invalide {!r} : {}", pattern, exc)
        return False


def _action_reached(pred: ToolSucceeded, actions: Sequence[ActionRecord]) -> bool:
    for action in reversed(actions):
        if action.tool == pred.tool:
            return action.ok
    return False


def _observation_text(obs: Observation) -> str:
    lines = [
        f"App active : {obs.active_app}",
        f"Titre : {obs.title}",
        "Résumé :",
    ]
    lines.extend(obs.summary.splitlines())
    return "\n".join(lines)


async def check(
    pred: Predicate,
    obs: Observation,
    actions: Sequence[ActionRecord],
    llm: LLMClient,
    *,
    effort: str | None,
) -> tuple[bool, bool]:
    """(atteint, fiable). fiable=False seulement pour le prédicat « judge »."""
    match pred.type:
        case "app_focused":
            return pred.name.casefold() in obs.active_app.casefold(), True
        case "window_title":
            return _search(pred.pattern, obs.title), True
        case "page_shows":
            return _search(pred.pattern, obs.page), True
        case "file_exists":
            expanded = Path(os.path.expanduser(pred.path))
            return expanded.exists(), True
        case "text_visible":
            text = "\n".join((obs.summary, obs.page, obs.title))
            return _search(pred.pattern, text), True
        case "tool_ok":
            if pred.tool not in _KNOWN_TOOLS:
                # Le modèle rapide écrit parfois un nom de canal ou de capacité
                # (« control.system », « schedule.timer ») : on vérifie alors que la
                # dernière action du sous-but a réussi.
                return bool(actions) and actions[-1].ok, True
            return _action_reached(pred, actions), True
        case "judge":
            answer, _, _ = await ask_json(
                llm,
                JUDGE_SYSTEM.format(question=pred.question),
                judge_user(_observation_text(obs)),
                _JudgeAnswer,
                effort=effort,
            )
            if answer is None:
                return False, False
            return answer.reached, False
    raise AssertionError(f"prédicat inconnu : {pred.type}")
