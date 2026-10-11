"""Agent v2 (phase 13a) : un seul point d'appel du LLM qui renvoie du JSON validé.

`ask_json` ne lève jamais pour une réponse invalide : il retente une fois avec le message
d'erreur Pydantic, puis renvoie `None`. Les exceptions réseau deviennent `None` aussi —
l'appelant décide de l'échec propre.
"""

from __future__ import annotations

import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

from mikata.llm.base import Done, LLMClient, TextDelta, Usage

M = TypeVar("M", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


async def ask_json(
    llm: LLMClient,
    system: str,
    user: str,
    model: type[M],
    *,
    effort: str | None,
    retries: int = 1,
) -> tuple[M | None, Usage | None, int]:
    """Demande au LLM un objet JSON conforme à `model`. Renvoie (objet ou None, usage
    cumulé, nombre d'appels). Ne lève jamais d'exception pour une réponse invalide."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    total_usage: Usage | None = None
    calls = 0
    for attempt in range(retries + 1):
        parts: list[str] = []
        usage: Usage | None = None
        try:
            async for event in llm.stream(messages, [], reasoning_effort=effort):
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                elif isinstance(event, Done):
                    usage = event.usage
                    break
        except Exception as exc:
            logger.warning("Appel JSON de l'agent v2 impossible : {}", exc)
            return None, total_usage, calls
        calls += 1
        if usage is not None:
            total_usage = usage if total_usage is None else total_usage + usage
        text = "".join(parts)
        try:
            return model.model_validate_json(_extract_json(text)), total_usage, calls
        except Exception as exc:
            if attempt >= retries:
                return None, total_usage, calls
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Réponse invalide : {_one_line(exc)}. "
                    "Réponds uniquement par le JSON corrigé.",
                }
            )
    return None, total_usage, calls


def _extract_json(text: str) -> str:
    """Le JSON d'une réponse, même entouré de texte ou de balises ```json```."""
    match = _FENCE_RE.search(text)
    if match is not None:
        text = match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return text.strip()
    return text[start : end + 1]


def _one_line(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__
    # La première ligne Pydantic (« N validation error for ... ») n'explique rien :
    # on garde la ligne qui nomme le champ fautif, sinon la dernière.
    for line in lines:
        if "input_value" in line or "type=" in line:
            return line[:300]
    return lines[-1][:300]
