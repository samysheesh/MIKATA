"""Agent v2 (phase 13b) : résoudre les inconnues avant d'agir.

Ordre du moins cher au plus cher : mémoire (faits), index des fichiers, état de l'écran /
des applis, recherche web. L'utilisateur n'est jamais interrogé ici (c'est l'orchestrateur
qui décide, 13c). Le contenu lu (web, écran) n'est envoyé au LLM que dans le message
utilisateur, entre les marqueurs de données.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel

from mikata.agent2.llm_json import ask_json
from mikata.agent2.prompts import RESOLVE_SYSTEM, resolve_user
from mikata.agent2.schema import Intent, Observation, Resolved, Unknown
from mikata.agent2.sdk import Sdk
from mikata.llm.base import LLMClient


class _ResolveAnswer(BaseModel):
    value: str
    confident: bool


def _fold(text: str) -> str:
    """Comparaison insensible à la casse et aux accents (comme skills/library._fold)."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _first_result(files_text: str) -> str:
    """Le premier chemin d'une réponse find_files (« Trouvé :\n<chemin> (taille, ...) »)."""
    lines = files_text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Trouvé :" and index + 1 < len(lines):
            return lines[index + 1].strip().split(" (", 1)[0].strip()
        if stripped.startswith("Trouvé : ") and len(stripped) > 9:
            return stripped[9:].split(" (", 1)[0].strip()
    return ""


def _observation_text(obs: Observation) -> str:
    return f"App active : {obs.active_app}\nTitre : {obs.title}\nRésumé :\n{obs.summary}"


async def _extract(llm: LLMClient, question: str, text: str, *, effort: str | None) -> str:
    if not text.strip():
        return ""
    answer, _, _ = await ask_json(
        llm,
        RESOLVE_SYSTEM.format(question=question),
        resolve_user(text),
        _ResolveAnswer,
        effort=effort,
    )
    if answer is None or not answer.confident or not answer.value.strip():
        return ""
    return answer.value.strip()


async def resolve_unknowns(
    intent: Intent,
    sdk: Sdk,
    llm: LLMClient,
    *,
    effort: str | None,
    memory_facts: Sequence[str],
) -> tuple[list[Resolved], list[str]]:
    """(valeurs résolues, inconnues non résolues). L'intention reste inchangée."""
    resolved: list[Resolved] = []
    missing: list[str] = []
    for unknown in intent.unknowns:
        value = await _resolve_one(unknown, sdk, llm, effort=effort, memory_facts=memory_facts)
        if value:
            resolved.append(
                Resolved(name=unknown.name, value=value[0], source=value[1], confident=value[2])
            )
        else:
            missing.append(unknown.name)
    return resolved, missing


async def _resolve_one(
    unknown: Unknown,
    sdk: Sdk,
    llm: LLMClient,
    *,
    effort: str | None,
    memory_facts: Sequence[str],
) -> tuple[str, str, bool] | None:
    """(valeur, source, confiant) ou None."""
    question = unknown.question
    # 1. Mémoire : un fait contient un mot significatif de la question.
    question_words = {word for word in _fold(question).split() if len(word) >= 4}
    for fact in memory_facts:
        fact_words = {word for word in _fold(fact).split() if len(word) >= 4}
        if question_words & fact_words:
            return fact, "memory", True
    # 2. Selon le type d'inconnue.
    if unknown.kind == "file":
        try:
            found = await sdk.find_files(question)
        except Exception:
            found = ""
        first = _first_result(found)
        if first and "Aucun fichier" not in found:
            return first, "files", True
        return None
    if unknown.kind == "web":
        try:
            results = await sdk.web_search(question)
        except Exception:
            results = ""
        value = await _extract(llm, question, results, effort=effort)
        return (value, "web", True) if value else None
    if unknown.kind in ("screen", "app"):
        try:
            observation = await sdk.observe()
        except Exception:
            observation = None
        text = _observation_text(observation) if observation is not None else ""
        value = await _extract(llm, question, text, effort=effort)
        return (value, "screen", True) if value else None
    # contact : mémoire seulement (étape 1) ; user : l'orchestrateur décide.
    return None
