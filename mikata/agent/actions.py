"""Exécution des actions d'un tour (docs/AGENT_CORE.md §3.3 à §3.5).

Un appel au modèle rend des outils directs et/ou un programme (outil `script`) :

- les outils directs partent ensemble, chacun après la politique (`action.policy`) : si le
  modèle a lu un contenu non fiable dans ce tour, une action conséquente demande un « oui »
  (sauf accès complet) ;
- un programme s'exécute seul, dans l'interpréteur, sous les mêmes politiques
  (`action.runtime`), avec les compétences appelables et `pick` (Jev) ;
- tout est noté dans le tour : messages pour le modèle, étapes pour l'historique, sources
  pour l'apprentissage, provenance pour la suite du tour.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from mikata.action.effects import Trust
from mikata.action.policy import Verdict, confirm_question, decide
from mikata.action.runtime import ScriptRuntime
from mikata.action.sdk import SdkCatalog
from mikata.agent.explore import EXPLORE_TOOL, Explorer
from mikata.agent.guard import log_action
from mikata.agent.script_tool import (
    SCRIPT_TOOL,
    data_message,
    failure_message,
    has_data,
    spoken_result,
)
from mikata.agent.turn import HISTORY_TOOL_CHARS, UNTRUSTED_HISTORY, Reply, Turn
from mikata.config import Config
from mikata.llm.base import LLMClient, ToolCallReq
from mikata.tools.base import current_turn

if TYPE_CHECKING:
    from mikata.tools.registry import ToolRegistry, ToolResult

SCRIPT_FAILURES_BEFORE_ESCALATION = 2
MAX_SCRIPT_FAILURES = 3  # programmes en échec dans un même tour


class ActionRunner:
    def __init__(
        self,
        tools: ToolRegistry,
        catalog: SdkCatalog,
        llm: LLMClient,
        config: Callable[[], Config],
        jev: Callable[[], Any],
        skills: Callable[[], dict[str, Any]],
    ) -> None:
        self.tools = tools
        self.catalog = catalog
        self.llm = llm
        self._config = config
        self._jev = jev
        self._skills = skills
        self._confirm_lock = asyncio.Lock()  # une question à la fois, jamais deux voix

    def output_trust(self, tool_name: str) -> Trust:
        entry = self.catalog.by_tool(tool_name)
        return entry.output if entry is not None else Trust.LOCAL

    def action_names(self) -> set[str]:
        """Noms du SDK et des outils (vérification des compétences apprises)."""
        entries = self.catalog.entries.values()
        return {e.name for e in entries} | {e.tool for e in entries}

    async def run(self, reply: Reply, turn: Turn) -> list[ToolResult]:
        """Exécute les appels d'une réponse du modèle et les note dans le tour."""
        result, state = turn.result, turn.state
        result.rounds += 1
        calls = [
            call if call.id else ToolCallReq(f"call_{result.rounds}_{i}", call.name, call.arguments)
            for i, call in enumerate(reply.calls)
        ]
        result.tool_calls.extend(calls)
        turn.phase("acting")
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
        # DeepSeek en mode réflexion, cf. `model.with_reasoning`), mais pas dans
        # l'historique des commandes suivantes.
        turn.messages.append({**assistant, "reasoning_content": reply.reasoning})
        turn.steps.append(assistant)
        # Les outils directs partent ensemble ; un programme ou une exploration s'exécute
        # seul, à son tour.
        trust_before = state.trust
        own = (SCRIPT_TOOL, EXPLORE_TOOL)
        direct = [call for call in calls if call.name not in own]
        done = dict(
            zip(
                (call.id for call in direct),
                await asyncio.gather(*(self._direct(call, trust_before, turn) for call in direct)),
                strict=True,
            )
        )
        for call in calls:
            if call.name == SCRIPT_TOOL:
                done[call.id] = await self._script(call, turn)
            elif call.name == EXPLORE_TOOL:
                done[call.id] = await self._explore(call, turn)
        results = [done[call.id] for call in calls]
        result.direct_sources.extend((call, trust_before is Trust.TRUSTED) for call in direct)
        result.tool_results.extend(r for r in results if r.name not in own)
        result.direct_results.extend(r for r in results if r.name not in own)
        for res in results:
            turn.messages.append(
                {"role": "tool", "tool_call_id": res.call_id, "content": res.content}
            )
            untrusted = res.ok and self.output_trust(res.name) is Trust.UNTRUSTED
            if untrusted:
                state.trust = Trust.UNTRUSTED  # le modèle va lire un contenu venu d'ailleurs
            if res.status in ("politique", "annulé"):
                state.no_tools = True  # un refus de politique ou un « non » ne se contourne pas
            short = UNTRUSTED_HISTORY if untrusted else res.content
            if len(short) > HISTORY_TOOL_CHARS:
                short = short[:HISTORY_TOOL_CHARS] + " […]"
            turn.steps.append({"role": "tool", "tool_call_id": res.call_id, "content": short})
        return results

    async def execute(self, call: ToolCallReq, turn: Turn) -> ToolResult:
        """Une action décidée hors du modèle principal (exploration) : même politique qu'un
        outil direct, notée dans le tour ; ses arguments viennent de ce qui a été lu."""
        result = await self._direct(call, turn.state.trust, turn)
        turn.result.tool_calls.append(call)
        turn.result.tool_results.append(result)
        turn.result.direct_results.append(result)
        turn.result.direct_sources.append((call, False))
        return result

    async def _explore(self, call: ToolCallReq, turn: Turn) -> ToolResult:
        """Outil explore : la boucle bornée d'observation et d'action (phase 16d)."""
        from mikata.tools.registry import ToolResult

        start = time.monotonic()
        goal = str(call.arguments.get("but", "")).strip()
        target = str(call.arguments.get("cible", "")).strip()
        if not goal or not target:
            content = "explore(but, cible) : il manque le but ou la cible"
            return ToolResult(call.id, EXPLORE_TOOL, False, content, False, 0.0, None, "échec")
        outcome = await Explorer(self, self.llm, self._config).run(goal, target, turn)
        turn.result.llm_calls += outcome.llm_calls
        duration = time.monotonic() - start
        log_action(EXPLORE_TOOL, call.arguments, "ok" if outcome.ok else "échec", outcome.answer)
        if outcome.ok:
            io = current_turn()
            if outcome.answer and io is not None:
                io.speak(outcome.answer)
            content = outcome.answer or "But atteint."
            return ToolResult(call.id, EXPLORE_TOOL, True, content, True, duration)
        done = "; ".join(f"{c.name} : {'ok' if r.ok else 'échec'}" for c, r in outcome.actions)
        content = f"Exploration arrêtée : {outcome.answer} Actions faites : {done or 'aucune'}."
        return ToolResult(call.id, EXPLORE_TOOL, False, content, False, duration, None, "échec")

    async def _direct(self, call: ToolCallReq, turn_trust: Trust, turn: Turn) -> ToolResult:
        """Un outil direct, après la politique."""
        from mikata.tools.registry import ToolResult

        entry = self.catalog.by_tool(call.name)
        if entry is not None:
            config = self._config()
            decision = decide(
                entry, Trust.TRUSTED, turn_trust, full_access=config.guard.full_access
            )
            refusal = ""
            status = "politique"
            if decision.verdict is Verdict.DENY:
                refusal = f"refusé : {decision.reason}"
            elif decision.verdict is Verdict.CONFIRM:
                turn.result.confirmations += 1
                guard = self.tools.ctx.guard
                async with self._confirm_lock:
                    accepted = guard is not None and await guard.confirm(
                        confirm_question(entry), current_turn()
                    )
                if not accepted:
                    refusal, status = "l'utilisateur n'a pas confirmé", "annulé"
            if refusal:
                log_action(call.name, call.arguments, status, refusal)
                return ToolResult(call.id, call.name, False, refusal, False, 0.0, None, status)
        return await self.tools.execute(call)

    async def _script(self, call: ToolCallReq, turn: Turn) -> ToolResult:
        """Exécute un programme (outil script) et prépare la suite du tour.

        Réussi avec une phrase courte en retour : elle est dite telle quelle et le tour
        s'arrête là (résultat « terminal »). Réussi avec une donnée : le modèle la formule.
        En échec : l'erreur revient au modèle, qui corrige ; après un refus ou un « non »,
        plus aucun outil n'est proposé."""
        from mikata.tools.registry import ToolResult

        result, state = turn.result, turn.state
        config = self._config()
        start = time.monotonic()
        code = str(call.arguments.get("code", ""))
        runtime = ScriptRuntime(
            self.tools,
            self.catalog,
            llm=self.llm,
            full_access=config.guard.full_access,
            turn_trust=state.trust,
            extract_effort=config.llm.reasoning_effort,
            skills=self._skills(),
            jev=self._jev(),
        )
        outcome = await runtime.run(code, timeout_s=config.agent.script_timeout_s)
        result.scripts.append(outcome)
        result.script_sources.append((code, state.trust is Trust.TRUSTED))
        result.skill_uses.extend(runtime.skill_uses)
        result.skill_choices.extend(runtime.skill_choices)
        result.tool_calls.extend(runtime.tool_calls)
        result.tool_results.extend(runtime.tool_results)
        result.llm_calls += runtime.llm_calls
        if runtime.usage is not None:
            result.usage = runtime.usage if result.usage is None else result.usage + runtime.usage
        result.confirmations += sum(
            r.verdict in ("à confirmer", "non confirmé") for r in outcome.records
        )
        duration = time.monotonic() - start
        logger.info(
            "Programme : {} en {:.0f} ms ({} appel(s), {} opérations){}",
            "ok" if outcome.ok else f"ÉCHEC ({outcome.kind})",
            duration * 1000,
            len(outcome.records),
            outcome.steps,
            "" if outcome.ok else f" : {outcome.error}",
        )
        log_action(SCRIPT_TOOL, {"code": code}, "ok" if outcome.ok else outcome.kind, outcome.error)
        if not outcome.ok:
            status = {"refus": "politique", "annulé": "annulé", "garde-fou": "refusé"}.get(
                outcome.kind, "échec"
            )
            if outcome.retryable:
                state.script_failures += 1
                if state.script_failures >= SCRIPT_FAILURES_BEFORE_ESCALATION:
                    state.strong = True
                    result.raise_effort(config.llm.strong_effort, "programme en échec")
                if state.script_failures >= MAX_SCRIPT_FAILURES:
                    state.no_tools = True  # on n'insiste pas : le modèle l'explique
            else:
                state.no_tools = True
            return ToolResult(
                call.id, SCRIPT_TOOL, False, failure_message(outcome), False, duration, None, status
            )
        sentence = spoken_result(outcome)
        if not has_data(outcome):
            io = current_turn()
            if sentence is not None and io is not None:
                io.speak(sentence)
            content = sentence or "Programme terminé."
            return ToolResult(call.id, SCRIPT_TOOL, True, content, True, duration)
        # Une donnée à formuler : le modèle va la lire, donc le tour en prend la confiance.
        state.trust = state.trust.join(outcome.trust).join(outcome.read_trust)
        return ToolResult(call.id, SCRIPT_TOOL, True, data_message(outcome), False, duration)
