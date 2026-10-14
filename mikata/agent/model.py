"""Appel au modèle, en flux : le texte part à la voix fragment par fragment, les appels
d'outils sont rassemblés. Un retry immédiat sur une erreur réseau avant le premier mot
(docs/ARCHITECTURE.md §6) ; rôle « fort » si configuré (`llm.strong_model`)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from mikata.agent.turn import FollowUpGate, Reply, SpeechSink, TurnResult
from mikata.config import Config
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq
from mikata.tts.normalize import FragmentSplitter

MAX_ATTEMPTS = 2  # un retry immédiat (§6)


def with_reasoning(messages: list[dict[str, Any]], effort: str | None) -> list[dict[str, Any]]:
    """Messages envoyés pour un appel : la réflexion d'un appel d'outil de la commande en
    cours n'est jointe que si elle existe, ou si la réflexion est demandée pour cet appel.

    DeepSeek en mode réflexion exige `reasoning_content` sur ces messages : après un passage
    de « none » à « low » (filet), elle est donc envoyée vide plutôt qu'absente.
    """
    thinking = effort in ("low", "medium", "high")
    out = []
    for message in messages:
        if "reasoning_content" in message and not (message["reasoning_content"] or thinking):
            message = {k: v for k, v in message.items() if k != "reasoning_content"}
        out.append(message)
    return out


class ModelCaller:
    def __init__(self, llm: LLMClient, config: Callable[[], Config]) -> None:
        self.llm = llm
        self._config = config

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        speech: SpeechSink,
        result: TurnResult,
        gate: FollowUpGate | None = None,
        strong: bool = False,
    ) -> Reply | None:
        """Un appel (avec son retry) ; None si le serveur reste injoignable."""
        config = self._config()
        gate = gate or FollowUpGate(active=False)

        def say(fragments: list[str]) -> None:
            for fragment in fragments:
                if gate.allow(fragment):
                    speech.say(fragment)

        splitter = FragmentSplitter(config.tts.first_chunk_min_words)
        effort = result.effort_final
        parts: list[str] = []
        calls: list[ToolCallReq] = []
        done: Done | None = None
        start = time.monotonic()
        first_at: float | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts += 1
            result.llm_calls += 1
            start = time.monotonic()
            try:
                request = with_reasoning(messages, effort)
                # Rôle « fort » : un autre modèle seulement s'il est configuré (sinon l'appel
                # reste identique, la réflexion relevée fait la différence).
                extra = (
                    {"model": config.llm.strong_model} if strong and config.llm.strong_model else {}
                )
                async for event in self.llm.stream(
                    request, tools, reasoning_effort=effort, **extra
                ):
                    if isinstance(event, Done):
                        done = event
                        break
                    now = time.monotonic()
                    first_at = first_at or now
                    if result.first_token_at is None:
                        result.first_token_at = now
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                        say(splitter.feed(gate.feed(event.text)))
                    elif isinstance(event, ToolCallReq):
                        calls.append(event)
                if attempt > 1:
                    logger.info("Appel LLM réussi à la tentative {}/{}.", attempt, MAX_ATTEMPTS)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if parts:  # la réponse a commencé : on garde ce qui est dit
                    logger.warning("Flux LLM interrompu : {}", exc)
                    break
                logger.warning("Appel LLM en échec ({}/{}) : {}", attempt, MAX_ATTEMPTS, exc)
                calls.clear()
                if attempt == MAX_ATTEMPTS:
                    return None
        say(splitter.feed(gate.finish()))
        say(splitter.flush())
        if done is not None and done.usage is not None:
            result.usage = done.usage if result.usage is None else result.usage + done.usage
        _log_call(result.llm_calls, effort, start, first_at, done, calls)
        return Reply(
            "".join(parts), calls, done.reasoning if done is not None else "", gate.ignored
        )


def _log_call(
    number: int,
    effort: str | None,
    start: float,
    first_at: float | None,
    done: Done | None,
    calls: list[ToolCallReq],
) -> None:
    def ms(t: float | None) -> str:
        return "—" if t is None else f"{t * 1000:.0f} ms"

    reasoning = done.reasoning_s if done is not None else None
    tokens = done.usage.reasoning if done is not None and done.usage is not None else 0
    logger.info(
        "Appel LLM {} (réflexion {}) : premier élément {}, réflexion du modèle {}{}, "
        "total {}, outils : {}.",
        number,
        effort or "défaut",
        ms(first_at - start if first_at is not None else None),
        ms(reasoning),
        f" ({tokens} tokens)" if tokens else "",
        ms(time.monotonic() - start),
        ", ".join(call.name for call in calls) or "aucun",
    )
