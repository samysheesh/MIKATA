"""Distillation d'une commande réussie en savoir-faire (phase 12c).

Un appel LLM en arrière-plan (reasoning none) transforme la trace d'une commande
multi-étapes réussie en procédure réutilisable : nom, description, exemple, étapes avec
les morceaux variables marqués {param}. Sortie invalide -> jetée en silence : une
procédure bancale ne doit jamais entrer dans la bibliothèque.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from mikata.config import Config

SKILL_DISTILL_PROMPT = """\
Tu es le compilateur de savoir-faire de MIKATA, l'assistant vocal du PC de l'utilisateur.
Voici une commande qui vient de réussir, avec les appels d'outils qui l'ont accomplie.

Transforme-la en procédure réutilisable. Réponds UNIQUEMENT par un objet JSON, rien
d'autre, avec ces clés :
- "name" : identifiant court en anglais, minuscules et underscores (ex. "youtube_recent_video") ;
- "description" : une phrase en français qui décrit la procédure en général, sans le détail
  de cette commande (ex. « chercher et lancer une vidéo récente sur YouTube ») ;
- "trigger" : la commande utilisateur d'origine, telle quelle ;
- "steps" : la liste des étapes (outil + arguments) telles qu'elles ont réussi, mais avec
  {param} à la place de chaque morceau qui change d'une commande à l'autre (terme de
  recherche, nom de fichier...) — par exemple
  {"tool": "browser_open", "args": {"url": "https://youtube.com/results?search_query={query}"}} ;
- "params" : la liste des noms de ces morceaux variables (ex. ["query"]), [] si la
  procédure est toujours identique ;
- "verify" : null, ou une étape de vérification (outil + arguments) qui confirme que la
  tâche est finie (ex. lire la page et vérifier que la vidéo est lancée).

Garde seulement les étapes utiles à rejouer (pas les lectures d'exploration inutiles).
N'invente aucune étape qui n'a pas été appelée.
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class DistillResult(BaseModel):
    name: str = Field(min_length=2, max_length=60, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=5, max_length=300)
    trigger: str = Field(min_length=2, max_length=300)
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    params: list[str] = Field(default_factory=list, max_length=8)
    verify: dict[str, Any] | None = None


def calls_to_json(calls: list[Any]) -> str:
    """La trace des appels d'outils, lisible par le compilateur."""
    return json.dumps(
        [{"tool": call.name, "args": call.arguments} for call in calls],
        ensure_ascii=False,
        indent=2,
    )


def parse_distill(text: str) -> DistillResult | None:
    """L'objet JSON du compilateur, ou None s'il n'est pas exploitable."""
    cleaned = text.strip()
    match = _FENCE_RE.search(cleaned)
    if match is not None:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    try:
        result = DistillResult.model_validate(data)
    except ValidationError:
        return None
    if any(not str(step.get("tool", "")).strip() for step in result.steps):
        return None
    # Chaque {param} utilisé dans les étapes doit être déclaré, et réciproquement.
    used: set[str] = set()
    for step in result.steps:
        for value in step.get("args", {}).values():
            used.update(re.findall(r"\{([a-z_]+)\}", str(value)))
    if used != set(result.params):
        return None
    return result


async def distill(
    llm: Any,
    config: Config,
    command: str,
    calls: list[Any],
    hint: str = "",
) -> DistillResult | None:
    """Un appel LLM en fond : la trace devient une procédure, ou rien (jamais d'erreur)."""
    from mikata.llm.base import Done, TextDelta

    note = f"\nL'utilisateur a demandé explicitement de retenir : « {hint} »." if hint else ""
    messages = [
        {"role": "system", "content": SKILL_DISTILL_PROMPT},
        {
            "role": "user",
            "content": (
                f"Commande : « {command} »{note}\n"
                f"Appels d'outils qui ont réussi :\n{calls_to_json(calls)}"
            ),
        },
    ]
    parts: list[str] = []
    try:
        async with asyncio.timeout(config.llm.turn_timeout_s):
            async for event in llm.stream(messages, [], reasoning_effort="none"):
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                elif isinstance(event, Done):
                    break
    except Exception as exc:
        logger.debug("Distillation du savoir-faire impossible : {}", exc)
        return None
    result = parse_distill("".join(parts))
    if result is None:
        logger.warning("Distillation du savoir-faire : réponse inexploitable, ignorée.")
    return result
