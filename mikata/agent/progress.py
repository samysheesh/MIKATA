"""Tâches en cours (phase 16c, docs/AGENT_CORE.md §3.8) : ce que fait chaque demande qui
tourne, pour répondre à « tu en es où ? » sans appel au modèle, et estimer la durée d'une
action pour décider d'un accusé de réception.

Les demandes tournent déjà en parallèle (une commande n'attend pas la précédente) et
« annule » / « stop » arrête la dernière en cours (`app.py`) : il manquait de pouvoir
demander où en est une tâche longue, et un accusé qui dépende de la durée prévue de
l'action plutôt que d'un délai fixe.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence

from mikata.action.sdk import SdkCatalog
from mikata.agent.script_tool import SCRIPT_TOOL
from mikata.learn.skills import fold
from mikata.llm.base import ToolCallReq

# Fonctions de l'interpréteur qui appellent un modèle : environ une seconde chacune.
_MODEL_FUNCTIONS = {"extract": 1.0, "pick": 0.5}
_CALL = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?)\s*\(")

# « tu en es où ? », « ça avance ? », « c'est bientôt fini ? », « where are you at? »
_STATUS = re.compile(
    r"^\s*(?:alors\s+)?(?:"
    r"(?:tu\s+)?(?:en\s+)?es\s+ou|t\s+en\s+es\s+ou|ou\s+(?:est[- ]ce\s+que\s+)?tu\s+en\s+es|"
    r"ca\s+(?:en\s+est\s+ou|avance)|c\s+est\s+bientot\s+(?:fini|pret|termine)|"
    r"tu\s+as\s+(?:bientot\s+)?fini|t\s+as\s+(?:bientot\s+)?fini|"
    r"qu\s+est[- ]ce\s+que\s+tu\s+fais|tu\s+fais\s+quoi|"
    r"how\s+is\s+it\s+going|where\s+are\s+you\s+at|are\s+you\s+done"
    r")\s*$"
)

ACTIVITY_LABELS = {
    "browser": "je suis sur la page",
    "search": "je cherche sur internet",
    "files": "je regarde dans tes fichiers",
    "app": "je regarde dans l'application",
    "work": "je m'en occupe",
    "thinking": "je réfléchis",
}


def is_status_question(text: str) -> bool:
    """La demande entière demande où en est MIKATA (aucune autre consigne dedans)."""
    return bool(_STATUS.match(" ".join(fold(text).split())))


def expected_duration(calls: Sequence[ToolCallReq], catalog: SdkCatalog | None) -> float:
    """Durée prévue de ces appels : les outils directs partent ensemble (le plus long
    compte), un programme enchaîne ses appels (ils s'additionnent)."""
    if catalog is None:
        return 0.0
    direct = 0.0
    script = 0.0
    for call in calls:
        if call.name == SCRIPT_TOOL:
            script += _program_duration(str(call.arguments.get("code", "")), catalog)
        else:
            entry = catalog.by_tool(call.name)
            if entry is not None and entry.announces:
                return 0.0  # l'outil s'annonce lui-même : pas d'accusé en plus
            direct = max(direct, entry.expected_s if entry is not None else 0.0)
    return direct + script


def _program_duration(code: str, catalog: SdkCatalog) -> float:
    total = 0.0
    for name in _CALL.findall(code):
        entry = catalog.get(name) or catalog.by_tool(name)
        if entry is not None:
            total += entry.expected_s
        else:
            total += _MODEL_FUNCTIONS.get(name, 0.0)
    return total


def describe(running: Iterable[tuple[str, str, float]], now: float | None = None) -> str:
    """Phrase dite pour « tu en es où ? » : (demande, activité, début) de chaque tâche."""
    now = time.monotonic() if now is None else now
    parts = []
    for command, activity, started in running:
        label = ACTIVITY_LABELS.get(activity, ACTIVITY_LABELS["work"])
        seconds = max(1, round(now - started))
        parts.append(f"pour « {command} », {label}, depuis {seconds} secondes")
    if not parts:
        return "Je n'ai rien en cours."
    sentence = " ; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."
