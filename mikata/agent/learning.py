"""Apprentissage par leçons (phase 14e, docs/AGENT_CORE.md §3.11, niveau 1) : trace de
chaque tour qui a agi, bilan des leçons présentes, réflexion en arrière-plan quand il y a
un signal fiable (échec corrigé, correction de l'utilisateur). Jamais bloquant."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from mikata.agent.turn import TurnResult
from mikata.config import Config
from mikata.learn.playbook import Lesson, Playbook
from mikata.learn.preferences import PreferenceStore, extract
from mikata.learn.reflect import is_correction, reflect, should_reflect
from mikata.learn.trace import TraceStore, TurnTrace, trace_from_result
from mikata.llm.base import LLMClient

if TYPE_CHECKING:
    from mikata.llm.usage import UsageTracker


class Learner:
    def __init__(
        self,
        llm: LLMClient,
        config: Callable[[], Config],
        usage: UsageTracker | None,
        playbook: Playbook | None,
        traces: TraceStore | None,
        preferences: PreferenceStore | None = None,
    ) -> None:
        self.llm = llm
        self._config = config
        self.usage = usage
        self.playbook = playbook
        self.traces = traces
        self.preferences = preferences  # phase 16e
        self._last_trace: tuple[TurnTrace, float] | None = None
        self.task: asyncio.Task[None] | None = None
        self.preference_task: asyncio.Task[None] | None = None  # phase 16g

    def lessons_for(self, command: str, context: str) -> list[Lesson]:
        cfg = self._config().learning
        if self.playbook is None or not cfg.enabled or cfg.lessons_in_context == 0:
            return []
        return self.playbook.select(command, context, cfg.lessons_in_context)

    def after_turn(
        self, command: str, result: TurnResult, lessons: list[Lesson], seconds: float
    ) -> None:
        """Après un tour qui a agi : trace, bilan des leçons présentes, et réflexion en
        arrière-plan s'il y a un signal fiable. Jamais après un contenu non fiable."""
        cfg = self._config().learning
        if not cfg.enabled or (self.playbook is None and self.traces is None):
            return
        ids = [lesson.id for lesson in lessons]
        trace = trace_from_result(command, result, seconds, ids, tainted=result.tainted)
        previous = self._last_trace
        self._last_trace = (trace, time.monotonic()) if trace.acted else None
        corrected = (
            previous[0]
            if previous is not None
            and time.monotonic() - previous[1] <= cfg.correction_window_s
            and is_correction(command)
            else None
        )
        if corrected is not None and self.preferences is not None:
            # Phase 16g : des seuls mots de l'utilisateur, même si le tour corrigé a lu une
            # page web et même si la correction n'a pas agi.
            self._learn_preferences(corrected.request, command)
        if not trace.acted:
            return
        if self.traces is not None and cfg.traces:
            self.traces.save(trace)
        if self.playbook is None:
            return
        failed = (
            trace.outcome != "ok"
            or bool(trace.failures)
            or any(not script.ok for script in trace.scripts)
        )
        if ids:
            self.playbook.mark(ids, helpful=not failed)
        correction_of = corrected
        if correction_of is not None and correction_of.lessons:
            # elles étaient là quand MIKATA s'est trompé
            self.playbook.mark(correction_of.lessons, helpful=False)
        if not should_reflect(trace, correction_of):
            return
        if self.task is not None and not self.task.done():
            return
        self.task = asyncio.create_task(self._reflect(trace, correction_of), name="réflexion")

    async def _reflect(self, trace: TurnTrace, correction_of: TurnTrace | None) -> None:
        assert self.playbook is not None
        config = self._config()
        try:
            async with asyncio.timeout(config.llm.turn_timeout_s):
                lessons, usage = await reflect(
                    self.llm,
                    trace,
                    correction_of=correction_of,
                    effort=config.learning.reflect_effort,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # hors ligne, réponse illisible : on n'apprend rien
            logger.warning("Réflexion impossible : {}", exc)
            return
        if self.usage is not None:
            self.usage.record("réflexion", usage, 1)
        source = "correction" if correction_of is not None else "échec corrigé"
        for scope, text in lessons:
            self.playbook.add(scope, text, source)

    def _learn_preferences(self, request: str, correction: str) -> None:
        if self.preference_task is not None and not self.preference_task.done():
            return
        self.preference_task = asyncio.create_task(
            self._extract_preferences(request, correction), name="préférences"
        )

    async def _extract_preferences(self, request: str, correction: str) -> None:
        assert self.preferences is not None
        config = self._config()
        try:
            async with asyncio.timeout(config.llm.turn_timeout_s):
                found, usage = await extract(self.llm, request, correction)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # hors ligne, réponse illisible : rien d'appris
            logger.warning("Préférences : extraction impossible ({})", exc)
            return
        if self.usage is not None:
            self.usage.record("préférences", usage, 1)
        for subject, value in found:
            self.preferences.learn(subject, value, "correction")
