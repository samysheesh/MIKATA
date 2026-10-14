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
from mikata.learn.preferences import PreferenceStore
from mikata.learn.reflect import is_correction, reflect_all, should_reflect
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
        correction_of = None
        if (
            previous is not None
            and time.monotonic() - previous[1] <= cfg.correction_window_s
            and is_correction(command)
        ):
            correction_of = previous[0]
            if correction_of.lessons:  # elles étaient là quand MIKATA s'est trompé
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
                lessons, preferences, usage = await reflect_all(
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
        # Une préférence ne vient que des mots de l'utilisateur : d'une correction.
        if correction_of is not None and self.preferences is not None:
            for subject, value in preferences:
                self.preferences.learn(subject, value, "correction")
