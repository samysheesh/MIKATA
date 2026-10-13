"""Réflexion après un tour (docs/AGENT_CORE.md §3.11) : d'un échec corrigé à une leçon.

Un appel LLM en arrière-plan, jamais pendant la réponse. Il ne voit que ce que MIKATA a
tenté et ce qui a fini par marcher ; il rend au plus deux leçons courtes et générales, que
le code ajoute une à une au carnet (`mikata.learn.playbook`).

On n'apprend que d'un signal fiable :

- un échec suivi d'une réussite dans le même tour (retour d'exécution) ;
- ou une correction de l'utilisateur (« non, pas ça... ») suivie d'une réussite.

Jamais d'un tour qui a lu du contenu non fiable : ce contenu ne doit pas pouvoir écrire
dans le carnet.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mikata.learn.trace import TurnTrace
from mikata.llm.base import Done, LLMClient, TextDelta, Usage

MAX_LESSONS = 2
CORRECTION_RE = re.compile(
    r"^\W*(non\b|nan\b|pas ça|pas ca|c'est pas|ce n'est pas|t'es tromp|tu t'es tromp|"
    r"mauvais|pas celui|pas celle|pas le bon|pas la bonne|je voulais|j'ai dit|j'avais dit)",
    re.IGNORECASE,
)

REFLECT_SYSTEM = """\
Tu aides MIKATA, un assistant vocal qui agit sur le PC de l'utilisateur, à apprendre de ses \
essais. On te montre ce qu'il a tenté, ce qui a échoué et ce qui a fini par marcher. \
Écris au plus deux leçons qui lui éviteront la même erreur la prochaine fois.

Une bonne leçon :
- est générale et réutilisable (une façon de faire avec une appli, un outil, un type de \
demande), pas un récit de cette demande précise ;
- tient en une phrase courte, en français, à l'impératif ou au présent ;
- cite les noms exacts qui comptent (nom d'un bouton, d'un outil, d'un argument) ;
- ne contient aucune donnée personnelle, aucun nom de fichier ou contenu propre à cette \
demande.

Réponds UNIQUEMENT par un objet JSON : {"lecons": [{"domaine": "...", "texte": "..."}]}. \
« domaine » : le nom de l'appli ou du sujet, en un ou deux mots minuscules (discord, \
fichiers, navigateur, programmes, général). S'il n'y a rien d'utile à retenir : \
{"lecons": []}."""


def is_correction(command: str) -> bool:
    """La commande corrige-t-elle ce que MIKATA vient de faire ?"""
    return CORRECTION_RE.search(command) is not None


def should_reflect(trace: TurnTrace, correction_of: TurnTrace | None = None) -> bool:
    """Y a-t-il un signal fiable à exploiter ? Jamais après un contenu non fiable."""
    if trace.tainted or trace.outcome != "ok" or not trace.acted:
        return False
    if correction_of is not None and not correction_of.tainted:
        return True
    return trace.recovered


def describe(trace: TurnTrace) -> str:
    """Le tour, en texte : la demande, les essais dans l'ordre, la réponse dite."""
    lines = [f"Demande : « {trace.request} »"]
    for index, script in enumerate(trace.scripts, start=1):
        state = "a réussi" if script.ok else f"a échoué ({script.error or script.kind})"
        lines.append(f"Programme {index}, qui {state} :\n{script.code}")
        if script.calls:
            lines.append("  appels : " + " ; ".join(script.calls[:10]))
    if trace.direct:
        lines.append("Outils directs appelés : " + ", ".join(trace.direct))
    for failure in trace.failures:
        lines.append(f"Échec : {failure}")
    lines.append(f"Réponse dite : « {trace.answer} »")
    return "\n".join(lines)


def parse_lessons(text: str) -> list[tuple[str, str]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return []
    try:
        raw: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    lessons: list[tuple[str, str]] = []
    for item in (raw.get("lecons") or [])[:MAX_LESSONS] if isinstance(raw, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("texte"), str):
            lessons.append((str(item.get("domaine") or "général"), item["texte"]))
    return lessons


async def reflect(
    llm: LLMClient,
    trace: TurnTrace,
    *,
    correction_of: TurnTrace | None = None,
    effort: str | None = "low",
) -> tuple[list[tuple[str, str]], Usage | None]:
    """Leçons (domaine, texte) tirées du tour ; liste vide s'il n'y a rien à retenir."""
    parts = []
    if correction_of is not None:
        parts.append("Ce que MIKATA avait fait juste avant :\n" + describe(correction_of))
        parts.append("L'utilisateur l'a corrigé, puis ceci a marché :\n" + describe(trace))
    else:
        parts.append(describe(trace))
    messages = [
        {"role": "system", "content": REFLECT_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    chunks: list[str] = []
    usage: Usage | None = None
    async for event in llm.stream(messages, [], reasoning_effort=effort):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, Done):
            usage = event.usage
    return parse_lessons("".join(chunks)), usage
