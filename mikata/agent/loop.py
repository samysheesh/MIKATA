"""Boucle agent (§6).

1. Messages : prompt système + faits mémorisés + historique + contexte dynamique + commande.
   Les définitions d'outils, fixes et triées, partent avec chaque appel (cache de prompt).
2. Appel LLM en streaming ; chaque fragment de texte terminé part aussitôt au TTS.
3. Appels d'outils : exécutés **en parallèle** (`asyncio.gather`). Si tous sont terminaux
   et ont réussi : « C'est fait » en cache, sans second appel LLM (chemin court-circuité).
   Sinon les résultats (tronqués) repartent au LLM, au plus `llm.max_tool_rounds` étapes.
4. Erreur réseau / API avant le premier mot : un retry immédiat, puis « Je n'arrive pas à
   joindre le serveur » (en cache, fonctionne hors ligne). `llm.turn_timeout_s` par requête.
5. Niveau de réflexion (ajout phase 4) : `llm.reasoning_effort` par défaut, ou celui d'un
   mot-clé dit en début / fin de commande (`llm.reasoning_keywords`, détecté par le code).
   Filet : un outil en échec, ou plus de `llm.reasoning_fallback_steps` étapes -> au moins
   « low » pour la suite de la commande.
Plusieurs tours peuvent tourner en parallèle (une commande n'attend pas la précédente).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from mikata.agent.memory import LongTermMemory, ShortTermMemory, is_reset_command
from mikata.config import Config
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq, Usage
from mikata.llm.prompts import active_window_title, build_messages, dynamic_context
from mikata.llm.reasoning import at_least, detect_reasoning
from mikata.tools.base import TurnIO, turn_context
from mikata.tts.cache import PHRASES
from mikata.tts.normalize import FragmentSplitter

if TYPE_CHECKING:
    from mikata.llm.usage import UsageTracker
    from mikata.tools.registry import ToolRegistry, ToolResult

MAX_ATTEMPTS = 2  # un retry immédiat (§6)
HISTORY_TOOL_CHARS = 400  # résultat d'outil gardé dans l'historique de conversation
# Applis ouvertes (contexte dynamique) : relevées au mot d'activation, pendant que
# l'utilisateur parle ; relues au moment de la commande seulement si elles datent.
OPEN_APPS_FRESH_S = 10.0
OPEN_APPS_TIMEOUT_S = 0.3  # le contexte ne doit pas retarder l'appel LLM
OPEN_APPS_PREWARM_TIMEOUT_S = 2.0


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

    async def drain(self) -> None:
        pass


@dataclass
class TurnResult:
    answer: str = ""
    outcome: str = "ok"  # ok | reset | server | timeout | limit
    first_token_at: float | None = None
    attempts: int = 0
    tool_calls: list[ToolCallReq] = field(default_factory=list)
    # Ajouts phase 4
    command: str = ""  # commande envoyée (sans mot-clé de réflexion)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_calls: int = 0
    rounds: int = 0  # étapes d'outils
    shortcut: bool = False  # outil terminal réussi : pas de second appel LLM
    usage: Usage | None = None
    cost_usd: float = 0.0
    effort: str | None = None  # niveau de réflexion au départ
    effort_final: str | None = None  # après le filet automatique
    effort_reasons: list[str] = field(default_factory=list)

    @property
    def effort_label(self) -> str:
        """« none », « medium (« réfléchis bien ») », « none -> low (outil en échec) »."""
        label = self.effort or "défaut"
        if self.effort_final != self.effort:
            label += f" -> {self.effort_final or 'défaut'}"
        return f"{label} ({', '.join(self.effort_reasons)})" if self.effort_reasons else label

    @property
    def path(self) -> str:
        """Chemin de la requête, pour le log (§10)."""
        if self.shortcut:
            return f"court-circuit (outil terminal, {self.llm_calls} appel(s) LLM)"
        if self.rounds:
            return f"{self.rounds} étape(s) d'outils, {self.llm_calls} appel(s) LLM"
        return f"réponse directe, {self.llm_calls} appel(s) LLM"


def _with_reasoning(messages: list[dict[str, Any]], effort: str | None) -> list[dict[str, Any]]:
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


@dataclass
class _Reply:
    text: str
    calls: list[ToolCallReq]
    reasoning: str


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        config: Config,
        memory: ShortTermMemory | None = None,
        phrases: CachedPhrases | None = None,
        window_title: Callable[[], str | None] = active_window_title,
        clock: Callable[[], datetime] = datetime.now,
        *,
        tools: ToolRegistry | None = None,
        long_memory: LongTermMemory | None = None,
        usage: UsageTracker | None = None,
        open_apps: Callable[[], Awaitable[Sequence[str]]] | None = None,
        installed_apps: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.memory = memory or ShortTermMemory(
            config.agent.history_max_tokens, config.agent.session_reset_min * 60
        )
        self.phrases = phrases
        self.tools = tools
        self.long_memory = long_memory
        self.usage = usage
        self._window_title = window_title
        self._clock = clock
        self._open_apps = open_apps
        self._installed_apps = installed_apps  # liste déjà en mémoire, jamais bloquante
        self._apps: Sequence[str] = ()
        self._apps_at = float("-inf")
        self._tools_warmed = False

    async def prewarm(self) -> None:
        """À chaque mot d'activation : connexion LLM préchauffée, applis ouvertes relevées ;
        une fois : index des applications."""
        jobs = []
        if self.tools is not None and not self._tools_warmed:
            self._tools_warmed = True
            jobs.append(self.tools.warm())
        if self._open_apps is not None:
            jobs.append(self._refresh_apps(OPEN_APPS_PREWARM_TIMEOUT_S))
        prewarm = getattr(self.llm, "prewarm", None)
        if prewarm is not None:
            jobs.append(prewarm())
        await asyncio.gather(*jobs)

    def _say_phrase(self, speech: SpeechSink, key: str) -> None:
        if self.phrases is not None and (pcm := self.phrases.get(key)) is not None:
            speech.say_audio(pcm, self.phrases.sample_rate)
        else:
            speech.say(PHRASES[key])

    async def respond(
        self,
        command: str,
        speech: SpeechSink,
        *,
        label: str = "commande",
        on_phase: Callable[[str], None] | None = None,
        listen: Callable[[float], Awaitable[str]] | None = None,
    ) -> TurnResult:
        """Répond à voix haute à `command` ; renvoie le texte et les mesures du tour.

        `on_phase("acting")` / `on_phase("thinking")` : outils en cours / LLM relancé.
        `listen(délai)` (phase 5) : écoute une réponse sans mot d'activation, pour les
        confirmations orales et le « stop » avant extinction ; None sans micro.
        """
        llm_cfg = self.config.llm
        request = detect_reasoning(command, llm_cfg.reasoning_keywords)
        result = TurnResult(command=request.text, effort=llm_cfg.reasoning_effort)
        if request.level is not None:
            result.effort = request.level
            result.effort_reasons.append(f"« {request.keyword} »")
            logger.info("Réflexion demandée : {} (mot-clé « {} »).", request.level, request.keyword)
        result.effort_final = result.effort
        if is_reset_command(request.text):
            self.memory.clear()
            self._say_phrase(speech, "reset")
            result.answer, result.outcome = PHRASES["reset"], "reset"
            return result
        try:
            async with asyncio.timeout(llm_cfg.turn_timeout_s):
                await self._run_turn(request.text, speech, result, on_phase, listen)
        except TimeoutError:
            logger.warning("Requête LLM trop longue (> {} s).", llm_cfg.turn_timeout_s)
            result.outcome = "timeout"
            self._say_phrase(speech, "timeout")
        finally:
            if self.usage is not None and result.llm_calls:
                result.cost_usd = self.usage.record(label, result.usage, result.llm_calls)
        return result

    async def _refresh_apps(self, timeout: float) -> None:
        assert self._open_apps is not None
        try:
            async with asyncio.timeout(timeout):
                self._apps = await self._open_apps()
            self._apps_at = time.monotonic()
        except Exception as exc:  # contexte facultatif
            logger.debug("Applis ouvertes inconnues : {!r}", exc)

    async def _context(self) -> str:
        stale = time.monotonic() - self._apps_at > OPEN_APPS_FRESH_S
        if self._open_apps is not None and stale:
            await self._refresh_apps(OPEN_APPS_TIMEOUT_S)
        return dynamic_context(self._clock(), self._window_title(), self._apps)

    def _escalate(self, result: TurnResult, reason: str) -> None:
        raised = at_least(result.effort_final, "low")
        if raised != result.effort_final:
            logger.info("Réflexion relevée : {} -> {} ({}).", result.effort_final, raised, reason)
            result.effort_final = raised
            result.effort_reasons.append(reason)

    async def _run_turn(
        self,
        command: str,
        speech: SpeechSink,
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
        listen: Callable[[float], Awaitable[str]] | None = None,
    ) -> None:
        facts = self.long_memory.facts if self.long_memory is not None else ()
        drain = getattr(speech, "drain", None)
        turn = TurnIO(say=speech.say, listen=listen, **({"drain": drain} if drain else {}))
        installed = self._installed_apps() if self._installed_apps is not None else ()
        messages = build_messages(
            self.memory.messages(), command, await self._context(), facts, installed
        )
        schemas = self.tools.schemas() if self.tools is not None else []
        max_rounds = self.config.llm.max_tool_rounds
        fallback_steps = self.config.llm.reasoning_fallback_steps
        spoken: list[str] = []
        steps: list[dict[str, Any]] = []  # appels d'outils gardés dans l'historique
        for round_no in range(max_rounds + 1):
            last = round_no == max_rounds  # dernier appel : sans outils, réponse obligée
            reply = await self._call_llm(messages, [] if last else schemas, speech, result)
            if reply is None:
                result.outcome = "server"
                self._say_phrase(speech, "server")
                return
            if reply.text.strip():
                spoken.append(reply.text.strip())
            if not reply.calls:
                break
            if last or self.tools is None:
                result.tool_calls.extend(reply.calls)
                logger.warning("Le LLM demande encore des outils : abandon ({} étapes).", round_no)
                result.outcome = "limit"
                if not spoken:
                    self._say_phrase(speech, "error")
                break
            with turn_context(turn):  # les outils parlent et écoutent dans cette commande
                results = await self._run_tools(reply, messages, steps, result, on_phase)
            spoken.extend(turn.spoken)  # ex. « Extinction dans 5 secondes... »
            turn.spoken.clear()
            if all(r.ok and r.terminal for r in results):
                result.shortcut = True
                if not spoken:  # une action simple : « C'est fait » en cache
                    self._say_phrase(speech, "done")
                    spoken.append(PHRASES["done"])
                break
            if on_phase is not None:
                on_phase("thinking")
            if any(not r.ok for r in results):
                self._escalate(result, "outil en échec")
            elif result.rounds >= fallback_steps:
                self._escalate(result, f"plus de {fallback_steps} étape(s)")
        result.answer = " ".join(spoken).strip()
        if result.answer or steps:
            self.memory.add(command, result.answer or PHRASES["done"], steps)

    async def _run_tools(
        self,
        reply: _Reply,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
    ) -> list[ToolResult]:
        assert self.tools is not None
        result.rounds += 1
        calls = [
            call if call.id else ToolCallReq(f"call_{result.rounds}_{i}", call.name, call.arguments)
            for i, call in enumerate(reply.calls)
        ]
        result.tool_calls.extend(calls)
        if on_phase is not None:
            on_phase("acting")
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": reply.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ],
        }
        # La réflexion repart avec ses appels d'outils pendant la commande (exigé par
        # DeepSeek en mode réflexion, cf. `_with_reasoning`), mais pas dans l'historique des
        # commandes suivantes.
        messages.append({**assistant, "reasoning_content": reply.reasoning})
        steps.append(assistant)
        results = list(await asyncio.gather(*(self.tools.execute(call) for call in calls)))
        result.tool_results.extend(results)
        for res in results:
            messages.append({"role": "tool", "tool_call_id": res.call_id, "content": res.content})
            short = res.content
            if len(short) > HISTORY_TOOL_CHARS:
                short = short[:HISTORY_TOOL_CHARS] + " […]"
            steps.append({"role": "tool", "tool_call_id": res.call_id, "content": short})
        return results

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        speech: SpeechSink,
        result: TurnResult,
    ) -> _Reply | None:
        """Un appel LLM (avec son retry) ; le texte part au TTS fragment par fragment."""
        splitter = FragmentSplitter(self.config.tts.first_chunk_min_words)
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
                request = _with_reasoning(messages, effort)
                async for event in self.llm.stream(request, tools, reasoning_effort=effort):
                    if isinstance(event, Done):
                        done = event
                        break
                    now = time.monotonic()
                    first_at = first_at or now
                    if result.first_token_at is None:
                        result.first_token_at = now
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                        for fragment in splitter.feed(event.text):
                            speech.say(fragment)
                    elif isinstance(event, ToolCallReq):
                        calls.append(event)
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
        for fragment in splitter.flush():
            speech.say(fragment)
        if done is not None and done.usage is not None:
            result.usage = done.usage if result.usage is None else result.usage + done.usage
        self._log_call(result.llm_calls, effort, start, first_at, done, calls)
        return _Reply("".join(parts), calls, done.reasoning if done is not None else "")

    @staticmethod
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

    async def aclose(self) -> None:
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()
        if self.tools is not None:
            await self.tools.aclose()
