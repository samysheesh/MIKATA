"""Boucle agent (docs/AGENT_CORE.md §3.1, §3.2) : l'assemblage des composants et le déroulé
d'un tour. Le détail vit dans des modules à interface étroite :

- `turn` : ce qui circule pendant un tour (demande, phrases, messages, état, résultat) ;
- `model` : un appel au modèle, en flux, avec retry et rôle fort ;
- `actions` : outils directs et programmes, sous les politiques ;
- `skill_path` : compétences (rejeu sans modèle, aiguillage, apprentissage) ;
- `learning` : traces, leçons, réflexion ;
- `history` : historique de conversation et son résumé ;
- `narration` : phrases d'attente.

Déroulé d'un tour :
1. Messages : prompt système + notice des programmes + faits + historique + contexte
   dynamique + commande (préfixe stable en tête pour le cache de prompt).
2. Compétence active qui colle : rejouée sans le modèle. Sinon l'aiguillage part avec le
   premier appel normal.
3. Appels au modèle, au plus `llm.max_tool_rounds` étapes : texte dit en flux ; outils
   directs en parallèle ; programme dans l'interpréteur. Si tout ce qui a été fait est
   terminal et réussi : « C'est fait » en cache, sans second appel.
4. Après le tour : historique, trace et leçons, bilan et apprentissage des compétences.

Ce qui est gardé des phases précédentes : retry réseau et phrase « serveur injoignable »
(phase 4), fenêtre de suivi (`FOLLOW_UP_MARKER`, phase 6), niveaux de réflexion et filet
automatique (phase 4). Plusieurs tours peuvent tourner en parallèle.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from mikata.action.effects import Effect, Trust
from mikata.action.sdk import SdkCatalog
from mikata.agent.actions import ActionRunner
from mikata.agent.explore import EXPLORE_SCHEMA
from mikata.agent.history import SUMMARY_PROMPT, Summarizer, short_term_memory
from mikata.agent.learning import Learner
from mikata.agent.memory import LongTermMemory, ShortTermMemory, is_reset_command
from mikata.agent.model import ModelCaller
from mikata.agent.narration import Narrator, activity_of, listening
from mikata.agent.progress import describe, expected_duration, is_status_question
from mikata.agent.script_tool import SCRIPT_SCHEMA, script_guide
from mikata.agent.skill_path import SkillPath, inducible
from mikata.agent.turn import (
    FOLLOW_UP_NOTE,
    UNTRUSTED_HISTORY,
    CachedPhrases,
    FollowUpGate,
    PhraseSpeaker,
    SilentSpeech,
    SpeechSink,
    Turn,
    TurnResult,
)
from mikata.config import Config
from mikata.learn.episodes import episodes, refers_to_past
from mikata.learn.episodes import render as render_episodes
from mikata.learn.playbook import Playbook, render
from mikata.learn.preferences import PreferenceStore
from mikata.learn.reflect import is_correction
from mikata.learn.skills import SkillStore
from mikata.learn.trace import TraceStore
from mikata.llm.base import LLMClient
from mikata.llm.prompts import active_window_title, build_messages, dynamic_context
from mikata.llm.reasoning import detect_reasoning
from mikata.plat import platform
from mikata.tools.base import TurnIO, turn_context
from mikata.tts.cache import PHRASES

if TYPE_CHECKING:
    from mikata.llm.usage import UsageTracker
    from mikata.tools.registry import ToolRegistry

# Noms gardés ici pour les modules qui les importaient de la boucle.
__all__ = [
    "SUMMARY_PROMPT",
    "UNTRUSTED_HISTORY",
    "Agent",
    "Narrator",
    "SilentSpeech",
    "TurnResult",
    "_inducible",
    "activity_of",
    "short_term_memory",
]
_inducible = inducible

# Applis ouvertes (contexte dynamique) : relevées au mot d'activation, pendant que
# l'utilisateur parle ; relues au moment de la commande seulement si elles datent.
OPEN_APPS_FRESH_S = 10.0
OPEN_APPS_TIMEOUT_S = 0.3  # le contexte ne doit pas retarder l'appel LLM
OPEN_APPS_PREWARM_TIMEOUT_S = 2.0


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
        skills: SkillStore | None = None,
        usage: UsageTracker | None = None,
        open_apps: Callable[[], Awaitable[Sequence[str]]] | None = None,
        installed_apps: Callable[[], Sequence[str]] | None = None,
        playbook: Playbook | None = None,
        traces: TraceStore | None = None,
        jev: Any = None,
        preferences: PreferenceStore | None = None,
    ) -> None:
        self.config = config
        self.memory = memory if memory is not None else short_term_memory(config)
        self.phrases = phrases
        self.tools = tools
        self.long_memory = long_memory
        self.skills = skills
        self.usage = usage
        self.jev = jev  # phase 15k : premier étage de l'aiguillage (mikata.learn.jev)
        self._window_title = window_title
        self._clock = clock
        self._open_apps = open_apps
        self._installed_apps = installed_apps  # liste déjà en mémoire, jamais bloquante
        self._apps: Sequence[str] = ()
        self._apps_at = float("-inf")
        self._tools_warmed = False
        self._turns: dict[int, Turn] = {}  # phase 16c : tours en cours (« tu en es où ? »)
        settings = lambda: self.config  # noqa: E731 — relu à chaque usage (réglages à chaud)
        self._say_phrase = PhraseSpeaker(phrases)
        self._model = ModelCaller(llm, settings)
        self._summarizer = Summarizer(self.memory, llm, settings, usage)
        self._learner = Learner(llm, settings, usage, playbook, traces, preferences)
        self.preferences = preferences  # phase 16e
        # Programmes (phase 14d) : SDK tiré du registre, une fois.
        self._catalog = SdkCatalog(tools.tools, platform()) if tools is not None else None
        self._script_guide = (
            script_guide(self._catalog.structured_returns())
            if self._catalog is not None and config.agent.scripts
            else None
        )
        self._actions: ActionRunner | None = None
        self._skill_path: SkillPath | None = None
        if tools is not None and self._catalog is not None:
            self._actions = ActionRunner(
                tools,
                self._catalog,
                llm,
                settings,
                lambda: self.jev,
                lambda: self._skill_path.callable() if self._skill_path is not None else {},
            )
            if skills is not None:
                self._skill_path = SkillPath(
                    skills,
                    self._actions,
                    self._model,
                    llm,
                    settings,
                    lambda: self.jev,
                    usage,
                    self._say_phrase,
                    on_start=self._starting,
                )
                tools.ctx.skills_capture = self._skill_path.capture  # outil remember_skill
            if tools.ctx.history is None:  # outil set_history_duration
                tools.ctx.history = self.memory
            if preferences is not None:  # outil forget (« oublie que je préfère... »)
                tools.ctx.preferences = preferences
        self._llm = llm

    def refresh_tools(self) -> None:
        """Phase 16f : des outils sont arrivés après le démarrage (serveurs MCP) : le SDK
        des programmes et sa notice sont reconstruits."""
        if self.tools is None:
            return
        self._catalog = SdkCatalog(self.tools.tools, platform())
        self._script_guide = (
            script_guide(self._catalog.structured_returns()) if self.config.agent.scripts else None
        )
        if self._actions is not None:
            self._actions.catalog = self._catalog

    # -- attributs partagés avec les composants ---------------------------------------------

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @llm.setter
    def llm(self, value: LLMClient) -> None:
        self._llm = value
        parts = (self._model, self._summarizer, self._learner, self._actions, self._skill_path)
        for part in parts:
            if part is not None:
                part.llm = value

    @property
    def playbook(self) -> Playbook | None:
        return self._learner.playbook

    @playbook.setter
    def playbook(self, value: Playbook | None) -> None:
        self._learner.playbook = value

    @property
    def traces(self) -> TraceStore | None:
        return self._learner.traces

    @property
    def _summary_task(self) -> asyncio.Task[None] | None:
        return self._summarizer.task

    # -- cycle de vie ---------------------------------------------------------------------

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
        if self.jev is not None and self._skill_path is not None and self._skill_path.has_active():
            jobs.append(self.jev.prewarm())  # phase 15k : aiguillage prêt
        await asyncio.gather(*jobs)

    async def learning_idle(self) -> None:
        """Attend la fin de la réflexion et de l'écriture de compétence en cours (tests,
        arrêt propre)."""
        skill_task = self._skill_path.task if self._skill_path is not None else None
        tasks = [t for t in (self._learner.task, skill_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        skill_task = self._skill_path.task if self._skill_path is not None else None
        for task in (self._summarizer.task, skill_task, self._learner.task):
            if task is not None and not task.done():
                task.cancel()
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()
        if self.jev is not None:
            await self.jev.aclose()
        if self.tools is not None:
            await self.tools.aclose()

    def say_phrase(self, speech: SpeechSink, key: str) -> None:
        """Dit une phrase fixe (cache audio si possible)."""
        self._say_phrase(speech, key)

    # -- contexte -------------------------------------------------------------------------

    async def _refresh_apps(self, timeout: float) -> None:
        assert self._open_apps is not None
        try:
            async with asyncio.timeout(timeout):
                self._apps = await self._open_apps()
            self._apps_at = time.monotonic()
        except Exception as exc:  # contexte facultatif
            logger.debug("Applis ouvertes inconnues : {!r}", exc)

    async def context(self) -> str:
        """Contexte dynamique : date, heure, fenêtre active, applis ouvertes."""
        stale = time.monotonic() - self._apps_at > OPEN_APPS_FRESH_S
        if self._open_apps is not None and stale:
            await self._refresh_apps(OPEN_APPS_TIMEOUT_S)
        return dynamic_context(self._clock(), self._window_title(), self._apps)

    # -- un tour --------------------------------------------------------------------------

    async def respond(
        self,
        command: str,
        speech: SpeechSink,
        *,
        label: str = "commande",
        on_phase: Callable[[str], None] | None = None,
        listen: Callable[[float], Awaitable[str]] | None = None,
        follow_up: bool = False,
    ) -> TurnResult:
        """Répond à voix haute à `command` ; renvoie le texte et les mesures du tour.

        `on_phase("acting")` / `on_phase("thinking")` : outils en cours / LLM relancé.
        `listen(délai)` (phase 5) : écoute une réponse sans mot d'activation, pour les
        confirmations orales et le « stop » avant extinction ; None sans micro.
        `follow_up` (phase 6) : phrase de la fenêtre de suivi, que le LLM peut ignorer.
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
        if is_status_question(request.text):  # phase 16c : sans appel au modèle
            running = [(t.command, t.activity, t.started_at) for t in self._turns.values()]
            result.answer, result.outcome = describe(running), "status"
            speech.say(result.answer)
            logger.info("Où en est MIKATA : {}", result.answer)
            return result
        narrator: Narrator | None = None
        narrating: asyncio.Task[None] | None = None
        # Fenêtre de suivi : la phrase sera peut-être ignorée, MIKATA ne dit rien d'avance.
        if self.config.agent.narration == "bavard" and not follow_up:
            narrator = Narrator(speech, self._say_phrase, self.config.agent.narration_after_s)
            speech = narrator
            listen = listening(narrator, listen)
            narrating = asyncio.create_task(narrator.run())
        try:
            async with asyncio.timeout(llm_cfg.turn_timeout_s):
                await self._run_turn(request.text, speech, result, on_phase, listen, follow_up)
        except TimeoutError:
            logger.warning("Requête LLM trop longue (> {} s).", llm_cfg.turn_timeout_s)
            result.outcome = "timeout"
            self._say_phrase(speech, "timeout")
        finally:
            if narrating is not None:
                narrating.cancel()
            if narrator is not None:
                result.narration = [PHRASES[key] for key in narrator.lines]
            if self.usage is not None and result.llm_calls:
                result.cost_usd = self.usage.record(label, result.usage, result.llm_calls)
        return result

    async def _start(
        self,
        command: str,
        speech: SpeechSink,
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
        listen: Callable[[float], Awaitable[str]] | None,
        follow_up: bool,
    ) -> tuple[Turn, list[Any]]:
        """Prépare le tour : contexte, leçons, messages."""
        drain = getattr(speech, "drain", None)
        io = TurnIO(say=speech.say, listen=listen, **({"drain": drain} if drain else {}))
        context = await self.context()
        if follow_up:
            context = f"{context} {FOLLOW_UP_NOTE}"
        lessons = self._learner.lessons_for(command, context)
        messages = build_messages(
            self.memory.messages(),
            command,
            context,
            self.long_memory.facts if self.long_memory is not None else (),
            self._installed_apps() if self._installed_apps is not None else (),
            script_guide=self._script_guide,
            lessons=render(lessons),
            narration=self.config.agent.narration == "bavard",
            preferences=self.preferences.render() if self.preferences is not None else "",
        )
        traces = self._learner.traces
        if traces is not None and refers_to_past(command):  # phase 16e : épisodes
            recent = await asyncio.to_thread(traces.recent, 60)
            block = render_episodes(episodes(command, recent, self._clock()))
            if block:
                messages.insert(-1, {"role": "system", "content": block})
        turn = Turn(command, speech, result, io, messages, on_phase=on_phase, follow_up=follow_up)
        return turn, lessons

    async def _run_turn(
        self,
        command: str,
        speech: SpeechSink,
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
        listen: Callable[[float], Awaitable[str]] | None = None,
        follow_up: bool = False,
    ) -> None:
        started = time.monotonic()
        turn, lessons = await self._start(command, speech, result, on_phase, listen, follow_up)
        self._turns[id(turn)] = turn
        try:
            await self._rounds(turn, lessons, started, follow_up)
        finally:
            self._turns.pop(id(turn), None)

    async def _rounds(
        self, turn: Turn, lessons: list[Any], started: float, follow_up: bool
    ) -> None:
        """Les étapes d'un tour : compétence, appels au modèle, actions."""
        command, speech, result = turn.command, turn.speech, turn.result
        schemas = self.tools.schemas() if self.tools is not None else []
        if self._script_guide is not None:
            schemas = [*schemas, SCRIPT_SCHEMA]
        if self._actions is not None and self.config.agent.explore:
            schemas = [*schemas, EXPLORE_SCHEMA]  # phase 16d
        max_rounds = self.config.llm.max_tool_rounds
        fallback_steps = self.config.llm.reasoning_fallback_steps
        skill_path = self._skill_path
        # Une compétence active dont un motif colle est rejouée sans le modèle ; sinon
        # l'aiguillage part avec le premier appel normal.
        replayed = await skill_path.before(turn) if skill_path is not None else None
        if replayed is not None:
            turn.spoken.extend(replayed)
            if result.shortcut:
                max_rounds = -1  # tout est dit : pas d'appel au modèle
        router = None
        if replayed is None and skill_path is not None:
            router = skill_path.start_router(command, follow_up)
        for round_no in range(max_rounds + 1):
            last = round_no == max_rounds  # dernier appel : sans outils, réponse obligée
            gate = FollowUpGate(follow_up and round_no == 0)
            offered = [] if last or turn.state.no_tools else schemas
            if router is not None and skill_path is not None:
                reply, routed = await skill_path.race(router, turn, offered, gate)
                router = None
                if routed is not None:
                    turn.spoken.extend(await skill_path.replay(*routed, "aiguillage", turn))
                    if result.shortcut:
                        break
                    continue  # une donnée à formuler, ou un échec : le modèle reprend
            else:
                reply = await self._model.call(
                    turn.messages, offered, speech, result, gate, turn.state.strong
                )
            if reply is None:
                result.outcome = "server"
                self._say_phrase(speech, "server")
                return
            if reply.ignored:  # rien n'est dit ni fait, rien n'entre dans l'historique
                logger.info("Fenêtre de suivi : « {} » ignorée ({}).", command, gate.reason)
                result.outcome = "ignored"
                return
            if reply.text.strip():
                turn.spoken.append(reply.text.strip())
            if not reply.calls:
                break
            if last or self._actions is None or turn.state.no_tools:
                result.tool_calls.extend(reply.calls)
                logger.warning("Le LLM demande encore des outils : abandon ({} étapes).", round_no)
                result.outcome = "limit"
                if not turn.spoken:
                    self._say_phrase(speech, "error")
                break
            self._starting(turn, reply.calls)
            with turn_context(turn.io):  # les outils parlent et écoutent dans cette commande
                results = await self._actions.run(reply, turn)
            self._starting(turn, [])
            tool_spoken = turn.take_tool_speech()  # ex. « Extinction dans 5 secondes... »
            turn.spoken.extend(tool_spoken)
            earlier_failure = any(not r.ok for r in result.direct_results) or any(
                not outcome.ok for outcome in result.scripts
            )
            # Après un échec dans le tour, pas de « C'est fait » automatique : seul un outil
            # ou un programme qui a dit sa propre phrase conclut ; sinon le modèle explique
            # (une phrase du modèle avant ses outils annonce une étape, elle ne conclut pas).
            if (
                all(r.ok and r.terminal for r in results)
                and (tool_spoken or not earlier_failure)
                and not self._redo_unfinished(turn, results)
            ):
                result.shortcut = True
                if not turn.spoken:  # une action simple : « C'est fait » en cache
                    self._say_phrase(speech, "done")
                    turn.spoken.append(PHRASES["done"])
                break
            turn.phase("thinking")
            if any(not r.ok for r in results):
                result.raise_effort("low", "outil en échec")
            elif result.rounds >= fallback_steps:
                result.raise_effort("low", f"plus de {fallback_steps} étape(s)")
        self._finish(turn, lessons, time.monotonic() - started)

    def _redo_unfinished(self, turn: Turn, results: list[Any]) -> bool:
        """Phase 16g : une correction (« non, sur Spotify ») refait la demande précédente.
        Une étape ne conclut pas le tour si elle n'a fait que défaire (fermer YouTube), ni
        tant que la correction n'a pas fait autant d'actions constructives que la demande
        corrigée (lancer un morceau, c'était ouvrir puis cliquer : ouvrir Spotify n'en est
        que le début). Le modèle voit alors le résultat et va au bout, au lieu d'un « C'est
        fait » sur la moitié du travail. Une correction simple (« non, à 20 ») conclut
        toujours en un appel."""
        if self.tools is None or not is_correction(turn.command):
            return False
        if all(self._undoes(r.name) for r in results):
            return True
        done = sum(
            1 for r in turn.result.direct_results if r.ok and self._constructive(r.name)
        ) + sum(1 for outcome in turn.result.scripts if outcome.ok)
        return done < self._actions_before()

    def _undoes(self, name: str) -> bool:
        tool = self.tools.tools.get(name) if self.tools is not None else None
        return bool(tool is not None and getattr(tool, "undoes", False))

    def _constructive(self, name: str) -> bool:
        """Une action qui fait avancer la demande : ni lecture, ni « défaire »."""
        entry = self._catalog.by_tool(name) if self._catalog is not None else None
        read = entry is not None and entry.effect is Effect.READ
        return not read and not self._undoes(name)

    def _actions_before(self) -> int:
        """Actions constructives de l'échange précédent, celui que l'on corrige."""
        return sum(1 for name in self.memory.last_tools() if self._constructive(name))

    def _starting(self, turn: Turn, calls: list[Any]) -> None:
        """Une action part (ou le modèle reprend la main, `calls` vide) : l'activité du
        tour change, et une action longue est annoncée maintenant (phase 16c)."""
        turn.activity = activity_of(calls) if calls else "thinking"
        if isinstance(turn.speech, Narrator):
            if calls:
                expected = expected_duration(calls, self._catalog)
                turn.speech.starting(turn.activity, expected, self.config.agent.ack_after_s)
            else:
                turn.speech.activity = "thinking"

    def _finish(self, turn: Turn, lessons: list[Any], seconds: float) -> None:
        """Historique, trace et leçons, compétences."""
        result = turn.result
        result.answer = " ".join(turn.spoken).strip()
        if result.answer or turn.steps:
            self.memory.add(turn.command, result.answer or PHRASES["done"], turn.steps)
            self._summarizer.schedule()
        result.tainted = turn.state.trust is Trust.UNTRUSTED
        self._learner.after_turn(turn.command, result, lessons, seconds)
        if self._skill_path is not None:
            self._skill_path.after_turn(turn.command, result)
