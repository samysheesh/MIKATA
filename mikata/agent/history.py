"""Historique de conversation (docs/ARCHITECTURE.md §6) : en mémoire ou sur disque, et
résumé en arrière-plan des échanges les plus anciens quand le plafond est atteint."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from mikata.agent.memory import ShortTermMemory
from mikata.config import Config
from mikata.llm.base import Done, LLMClient, TextDelta, Usage

if TYPE_CHECKING:
    from mikata.llm.usage import UsageTracker

HISTORY_FILE = "data/history.json"

SUMMARY_PROMPT = """\
Tu tiens la mémoire de MIKATA, l'assistant vocal de l'utilisateur. Mets à jour le résumé de \
leur conversation avec les échanges ci-dessous. Garde l'essentiel : ce que l'utilisateur a \
dit de lui, ses demandes et préférences, les décisions, les réponses importantes, ce qui \
reste en cours. Laisse de côté les actions banales déjà faites (« C'est fait »). Au plus \
120 mots, en phrases simples, sans markdown, en français. Réponds seulement par le résumé."""


def short_term_memory(config: Config) -> ShortTermMemory:
    """Historique de conversation : en mémoire, ou sur disque si `agent.history_persist`."""
    agent = config.agent
    if agent.history_persist:
        return ShortTermMemory(
            agent.history_max_tokens,
            agent.history_persist_h * 3600,
            path=config.resolve(HISTORY_FILE),
            summarize=agent.history_summarize,
        )
    return ShortTermMemory(
        agent.history_max_tokens,
        agent.session_reset_min * 60,
        summarize=agent.history_summarize,
    )


class Summarizer:
    """Résume en arrière-plan les échanges sortis de l'historique (plafond atteint)."""

    def __init__(
        self,
        memory: ShortTermMemory,
        llm: LLMClient,
        config: Callable[[], Config],
        usage: UsageTracker | None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self._config = config
        self.usage = usage
        self.task: asyncio.Task[None] | None = None

    def schedule(self) -> None:
        if self.memory.pending and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._summarize(), name="résumé")

    async def _summarize(self) -> None:
        while self.memory.pending:
            generation, previous, exchanges = self.memory.summary_request()
            lines = [f"Résumé actuel : {previous or '(aucun)'}", "", "Échanges à intégrer :"]
            for user, assistant in exchanges:
                lines += [f"Utilisateur : {user}", f"MIKATA : {assistant}"]
            messages = [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ]
            parts: list[str] = []
            usage: Usage | None = None
            try:
                async with asyncio.timeout(self._config().llm.turn_timeout_s):
                    async for event in self.llm.stream(messages, [], reasoning_effort="none"):
                        if isinstance(event, TextDelta):
                            parts.append(event.text)
                        elif isinstance(event, Done):
                            usage = event.usage
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # hors ligne : nouvel essai au prochain échange
                logger.warning("Résumé de l'historique impossible : {}", exc)
                return
            if self.usage is not None:
                self.usage.record("résumé", usage, 1)
            summary = "".join(parts).strip()
            if not self.memory.apply_summary(generation, len(exchanges), summary):
                return  # conversation remise à zéro entre-temps, ou résumé vide
            logger.info(
                "Historique : {} ancien(s) échange(s) résumé(s) ({} caractères).",
                len(exchanges),
                len(self.memory.summary),
            )
