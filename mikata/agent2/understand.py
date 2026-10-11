"""Agent v2 (phase 13b) : une demande devient une Intention structurée.

Un seul appel LLM (modèle rapide) ; la sortie est validée par Pydantic puis normalisée en
Python — jamais par le LLM.
"""

from __future__ import annotations

import re

from mikata.agent2.llm_json import ask_json
from mikata.agent2.prompts import UNDERSTAND_SYSTEM, UNDERSTAND_USER
from mikata.agent2.schema import Intent, Verb
from mikata.llm.base import LLMClient, Usage

# La troncature des sous-intentions est faite par l'orchestrateur (13c), qui dispose de
# `agent2.max_subgoals` : la signature ci-dessous est celle du document, sans le config.
_SUB_INTENTS_MAX = 8  # filet local en attendant 13c


async def understand(
    llm: LLMClient,
    request: str,
    *,
    context: str,
    preferences: str,
    os_name: str,
    effort: str | None,
) -> tuple[Intent | None, Usage | None, int]:
    """La demande normalisée, ou None si le LLM n'a rien produit d'exploitable."""
    intent, usage, calls = await ask_json(
        llm,
        UNDERSTAND_SYSTEM.format(os=os_name, context=context, preferences=preferences),
        UNDERSTAND_USER.format(request=request),
        Intent,
        effort=effort,
    )
    if intent is None:
        return None, usage, calls
    intent = _normalize(intent)
    return intent, usage, calls


def _normalize(intent: Intent) -> Intent:
    """Post-traitement déterministe : jamais confié au LLM."""
    if intent.channel_hint is not None:
        hint = re.sub(r"\s+", "", intent.channel_hint.casefold())
        intent.channel_hint = hint or None
    if intent.channel_hint is not None:
        # Un canal imposé doit savoir faire la famille du verbe ; sinon le LLM a confondu
        # l'objet de la demande avec un canal (« ouvre Discord » -> verb open, le canal
        # discord ne fait que du send -> hint rejeté, le score choisit apps).
        from mikata.agent2.channels import BUILTIN_CHANNELS

        prefix = f"{intent.verb.value}."
        known = {channel.name: channel for channel in BUILTIN_CHANNELS}
        channel = known.get(intent.channel_hint)
        if channel is not None and not any(
            capability.startswith(prefix) for capability in channel.capabilities
        ):
            intent.channel_hint = None
    if len(intent.sub_intents) > _SUB_INTENTS_MAX:
        intent.sub_intents = intent.sub_intents[:_SUB_INTENTS_MAX]
    if intent.verb is Verb.CHAT:
        intent.unknowns = []  # répondre sans agir : rien à découvrir
    return intent
