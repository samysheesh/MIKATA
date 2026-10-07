"""Boucle agent (§6). Phase 3 : réponse parlée en streaming, sans outils (phase 4).

1. Messages : prompt système + historique + contexte dynamique + commande transcrite.
2. Appel LLM en streaming ; chaque fragment terminé part aussitôt au TTS (§2, technique 3).
3. Erreur réseau / API avant le premier mot : un retry immédiat, puis « Je n'arrive pas à
   joindre le serveur » (en cache, fonctionne hors ligne).
4. Limite de `llm.turn_timeout_s` (60 s) par requête, puis phrase d'excuse courte.
Plusieurs tours peuvent tourner en parallèle (une commande n'attend pas la précédente).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from loguru import logger

from mikata.agent.memory import ShortTermMemory, is_reset_command
from mikata.config import Config
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq
from mikata.llm.prompts import active_window_title, build_messages, dynamic_context
from mikata.tts.cache import PHRASES
from mikata.tts.normalize import FragmentSplitter

MAX_ATTEMPTS = 2  # un retry immédiat (§6)


class SpeechSink(Protocol):
    """Destination des phrases d'une commande (`audio.player.SpeechStream`)."""

    def say(self, text: str) -> None: ...
    def say_audio(self, pcm: Any, sample_rate: int) -> None: ...


class CachedPhrases(Protocol):
    sample_rate: int

    def get(self, key: str) -> Any: ...


class SilentSpeech:
    """Destination muette (option --ask --no-audio) : le texte est seulement renvoyé."""

    def say(self, text: str) -> None:
        pass

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        pass


@dataclass
class TurnResult:
    answer: str = ""
    outcome: str = "ok"  # ok | reset | server | timeout
    first_token_at: float | None = None
    attempts: int = 0
    tool_calls: list[ToolCallReq] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        config: Config,
        memory: ShortTermMemory | None = None,
        phrases: CachedPhrases | None = None,
        window_title: Callable[[], str | None] = active_window_title,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.llm = llm
        self.config = config
        self.memory = memory or ShortTermMemory(
            config.agent.history_max_tokens, config.agent.session_reset_min * 60
        )
        self.phrases = phrases
        self._window_title = window_title
        self._clock = clock

    async def prewarm(self) -> None:
        prewarm = getattr(self.llm, "prewarm", None)
        if prewarm is not None:
            await prewarm()

    def _say_phrase(self, speech: SpeechSink, key: str) -> None:
        if self.phrases is not None and (pcm := self.phrases.get(key)) is not None:
            speech.say_audio(pcm, self.phrases.sample_rate)
        else:
            speech.say(PHRASES[key])

    async def respond(self, command: str, speech: SpeechSink) -> TurnResult:
        """Répond à voix haute à `command` ; renvoie le texte et les mesures du tour."""
        if is_reset_command(command):
            self.memory.clear()
            self._say_phrase(speech, "reset")
            return TurnResult(answer=PHRASES["reset"], outcome="reset")
        result = TurnResult()
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                await self._stream_answer(command, speech, result)
        except TimeoutError:
            logger.warning("Requête LLM trop longue (> {} s).", self.config.llm.turn_timeout_s)
            result.outcome = "timeout"
            self._say_phrase(speech, "timeout")
        return result

    async def _stream_answer(self, command: str, speech: SpeechSink, result: TurnResult) -> None:
        context = dynamic_context(self._clock(), self._window_title())
        messages = build_messages(self.memory.messages(), command, context)
        splitter = FragmentSplitter(self.config.tts.first_chunk_min_words)
        parts: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts = attempt
            try:
                async for event in self.llm.stream(messages, tools=[]):
                    if isinstance(event, TextDelta):
                        if result.first_token_at is None:
                            result.first_token_at = time.monotonic()
                        parts.append(event.text)
                        for fragment in splitter.feed(event.text):
                            speech.say(fragment)
                    elif isinstance(event, ToolCallReq):  # pas d'outils avant la phase 4
                        result.tool_calls.append(event)
                    elif isinstance(event, Done):
                        break
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if parts:  # la réponse a commencé : on garde ce qui est dit
                    logger.warning("Flux LLM interrompu : {}", exc)
                    break
                logger.warning("Appel LLM en échec ({}/{}) : {}", attempt, MAX_ATTEMPTS, exc)
                if attempt == MAX_ATTEMPTS:
                    result.outcome = "server"
                    self._say_phrase(speech, "server")
                    return
        for fragment in splitter.flush():
            speech.say(fragment)
        result.answer = "".join(parts).strip()
        if result.tool_calls and not result.answer:
            logger.info("Le LLM a demandé un outil, indisponible avant la phase 4.")
        if result.answer:
            self.memory.add(command, result.answer)

    async def aclose(self) -> None:
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()
