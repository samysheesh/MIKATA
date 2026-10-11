"""Agent v2 (phase 13c) : l'orchestrateur, enveloppe de l'Agent actuel.

Agent2 a la même interface publique que Agent : `App` ne voit aucune différence. Tout ce
qui n'est pas une action (réponses parlées, mémoire courte, historique, suivi) reste à
l'ancien agent ; Agent2 comprend, résout, choisit un canal, planifie, exécute et vérifie.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any

from loguru import logger

from mikata.agent.loop import TurnResult
from mikata.agent.memory import is_reset_command
from mikata.agent2.channels import BUILTIN_CHANNELS, choose, display_name
from mikata.agent2.executor import run_subgoal
from mikata.agent2.planner import make_plan
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.resolve import resolve_unknowns
from mikata.agent2.schema import Intent, Judge, Resolved, Trace, Verb
from mikata.agent2.sdk import Recorder, Sdk
from mikata.agent2.trace import TraceStore
from mikata.agent2.understand import understand
from mikata.config import Agent2Config
from mikata.llm.base import Done, LLMClient, LLMEvent, Usage
from mikata.plat import platform
from mikata.tools.base import TurnIO, turn_context
from mikata.tts.cache import PHRASES

# Raison d'échec réécrite sans détail technique (table de motifs, pas de LLM).
_FAILURE_REASONS: list[tuple[str, str]] = [
    ("refusé par le garde-fou", "le garde-fou a refusé une action"),
    ("n'a pas confirmé", "tu n'as pas confirmé"),
    ("pas de code", "je n'ai pas trouvé comment faire"),
    ("ActionLimitReached", "le script a fait trop d'actions"),
    ("outil inconnu", "un outil nécessaire est absent"),
    ("TimeoutError", "ça a pris trop de temps"),
]


class _CountingLLM:
    """Enveloppe le vrai LLM pour compter appels et tokens de toute la demande v2."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls = 0
        self.usage: Usage | None = None

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        async for event in self.inner.stream(messages, tools, reasoning_effort=reasoning_effort):
            if isinstance(event, Done) and event.usage is not None:
                self.usage = event.usage if self.usage is None else self.usage + event.usage
            yield event


class Agent2:
    """L'agent v2 : comprend, exécute en code vérifié, vérifie, et laisse l'ancien agent
    parler. `skills`/`recipes` arrivent en 13d/13e (None ici)."""

    def __init__(
        self,
        legacy: Any,
        *,
        traces: TraceStore,
        prefs: PreferenceStore,
        skills: Any = None,
        recipes: Any = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.legacy = legacy
        self.traces = traces
        self.prefs = prefs
        self.skills = skills
        self.recipes = recipes
        self._clock = clock

    def __getattr__(self, name: str) -> Any:
        # config, memory, tools, phrases, long_memory, usage, llm, prewarm, aclose... -> legacy
        return getattr(self.legacy, name)

    async def respond(
        self,
        command: str,
        speech: Any,
        *,
        label: str = "commande",
        on_phase: Callable[[str], None] | None = None,
        listen: Callable[[float], Awaitable[str]] | None = None,
        follow_up: bool = False,
    ) -> TurnResult:
        """Répond à la demande : compréhension, résolution, canal, plan, exécution.

        Les commandes de remise à zéro, la fenêtre de suivi, les questions et le bavardage
        restent à l'ancien agent (streaming et latence inchangés).
        """
        if follow_up or is_reset_command(command):
            return await self.legacy.respond(
                command,
                speech,
                label=label,
                on_phase=on_phase,
                listen=listen,
                follow_up=follow_up,
            )
        cfg: Agent2Config = self.legacy.config.agent2
        counted = _CountingLLM(self.legacy.llm)
        trace = self.traces.new(command)
        result = TurnResult(command=command)
        answer = ""
        try:
            context = await self.legacy.context()
            intent, _, _ = await understand(
                counted,
                command,
                context=context,
                preferences=self.prefs.describe(),
                os_name=platform(),
                effort=cfg.fast_effort,
            )
            trace.intent = intent
            if intent is None or intent.verb in (Verb.CHAT, Verb.ASK):
                trace.outcome = "chat"
                trace.answer = answer
                self._finish(trace, result, counted, label)
                return await self.legacy.respond(
                    command,
                    speech,
                    label=label,
                    on_phase=on_phase,
                    listen=listen,
                    follow_up=False,
                )
            failure = await self._run_intent(intent, counted, trace, speech, cfg, on_phase, listen)
            if failure is None:
                verified = self._all_verified(trace)
                trace.verified = verified
                trace.outcome = "ok"
                self.legacy.say_phrase(speech, "done")
                answer = PHRASES["done"]
            else:
                trace.outcome = "failed"
                answer = f"Je n'y arrive pas : {failure}"
                speech.say(answer)
        except Exception as exc:
            logger.exception("Agent v2 : erreur inattendue pendant la demande.")
            trace.outcome = "error"
            answer = f"Je n'y arrive pas : {_reason_for(str(exc))}"
            with contextlib.suppress(Exception):
                speech.say(answer)
        trace.answer = answer
        self._finish(trace, result, counted, label)
        result.answer = answer
        result.outcome = "ok" if trace.outcome == "ok" else "limit"
        return result

    async def _run_intent(
        self,
        intent: Intent,
        counted: _CountingLLM,
        trace: Trace,
        speech: Any,
        cfg: Agent2Config,
        on_phase: Callable[[str], None] | None,
        listen: Callable[[float], Awaitable[str]] | None,
    ) -> str | None:
        """Exécute une intention : None = réussite, sinon la raison courte de l'échec."""
        drain = getattr(speech, "drain", None)
        turn = TurnIO(say=speech.say, listen=listen, **({"drain": drain} if drain else {}))
        os_name = platform()
        with turn_context(turn):  # les confirmations orales du garde-fou fonctionnent
            sdk = Sdk(
                self.legacy.tools,
                max_actions=cfg.max_actions_per_script,
                observe_lines=cfg.observe_max_lines,
            )
            memory_facts = (
                self.legacy.long_memory.facts if self.legacy.long_memory is not None else ()
            )
            resolved, missing = await resolve_unknowns(
                intent, sdk, counted, effort=cfg.fast_effort, memory_facts=memory_facts
            )
            user_question = next(
                (item for item in intent.unknowns if item.kind == "user" and item.name in missing),
                None,
            )
            if user_question is not None:
                answer = await sdk.ask_user(user_question.question)
                if not answer:
                    return "tu n'as pas répondu"
                resolved.append(
                    Resolved(name=user_question.name, value=answer, source="user", confident=True)
                )
            trace.resolved.extend(resolved)
            installed = (
                self.legacy._installed_apps() if self.legacy._installed_apps is not None else ()
            )
            channel, announce = choose(intent, self.prefs, installed, {})
            trace.channel = channel.name
            if announce:
                speech.say(f"Je passe par {display_name(channel.name)}.")
                trace.channel_announced = True
            plan, _, _ = await make_plan(
                counted,
                intent,
                resolved,
                channel,
                channels_text=self._channels_text(),
                skills_text=self._skills_text(),
                os_name=os_name,
                max_subgoals=cfg.max_subgoals,
                effort=cfg.strong_effort,
            )
            trace.plan.extend(plan)

            def sdk_factory(recorder: Recorder) -> Sdk:
                return Sdk(
                    self.legacy.tools,
                    max_actions=cfg.max_actions_per_script,
                    observe_lines=cfg.observe_max_lines,
                    recorder=recorder,
                )

            for offset, subgoal in enumerate(plan):
                index = len(trace.plan) - len(plan) + offset
                if on_phase is not None:
                    on_phase("acting")
                steps = await run_subgoal(
                    index,
                    subgoal,
                    sdk_factory=sdk_factory,
                    llm=counted,
                    resolved=resolved,
                    cfg=cfg,
                    os_name=os_name,
                    skills=self.skills,
                )
                trace.steps.extend(steps)
                if not steps[-1].goal_reached:
                    return _reason_for(steps[-1].error or "condition non atteinte")
            return None

    def _finish(
        self,
        trace: Trace,
        result: TurnResult,
        counted: _CountingLLM,
        label: str,
    ) -> None:
        """Sauvegarde la trace et remplit les mesures du tour, comme l'ancien agent."""
        trace.llm_calls = counted.calls
        result.llm_calls = counted.calls
        result.usage = counted.usage
        if self.legacy.usage is not None and counted.calls:
            result.cost_usd = self.legacy.usage.record(label, counted.usage, counted.calls)
            trace.cost_usd = result.cost_usd
        self.traces.save(trace)

    @staticmethod
    def _all_verified(trace: Trace) -> bool:
        """Tous les sous-buts atteints, et pas par un prédicat « judge » (moins fiable)."""
        for index, subgoal in enumerate(trace.plan):
            if isinstance(subgoal.done_when, Judge):
                return False
            steps = [step for step in trace.steps if step.subgoal_index == index]
            if not steps or not steps[-1].goal_reached:
                return False
        return True

    @staticmethod
    def _channels_text() -> str:
        return "\n".join(
            f"- {channel.name} : {', '.join(channel.capabilities)}" for channel in BUILTIN_CHANNELS
        )

    def _skills_text(self) -> str:
        return "(aucun)" if self.skills is None else self.skills.describe()


def _reason_for(error: str) -> str:
    for marker, readable in _FAILURE_REASONS:
        if marker in error:
            return readable
    return "quelque chose n'a pas fonctionné"
