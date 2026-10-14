"""Le chemin des compétences dans un tour (docs/AGENT_CORE.md §3.6, phases 15e à 15l).

Avant le premier appel au modèle :
- une compétence **active** dont un motif colle à la phrase est rejouée sans le modèle ;
- sinon, celles qui correspondent sont signalées au modèle (note dans le contexte), et
  l'**aiguillage** (Jev, puis un petit appel LLM s'il hésite) part en même temps que le
  premier appel normal : s'il reconnaît une compétence active avant que MIKATA ait parlé,
  l'appel normal est coupé et la compétence s'exécute.

Après le tour : bilan des compétences utilisées (réussite surveillée, correction de
l'utilisateur), et apprentissage en arrière-plan d'un programme réussi (induction puis
vérification dans le code).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from mikata.agent.model import ModelCaller
from mikata.agent.script_tool import SCRIPT_TOOL
from mikata.agent.turn import (
    FollowUpGate,
    PhraseSpeaker,
    Reply,
    SpeechProbe,
    Turn,
    TurnResult,
)
from mikata.config import Config
from mikata.learn.reflect import is_correction
from mikata.learn.skills import Skill, SkillStore, call_code, induce, locate, route, verify
from mikata.llm.base import LLMClient, ToolCallReq
from mikata.tools.base import ToolError, turn_context
from mikata.tts.cache import PHRASES

if TYPE_CHECKING:
    from mikata.agent.actions import ActionRunner
    from mikata.llm.usage import UsageTracker

# Phase 15j : l'appel normal a fini le premier sans rien dire (il demande un outil) : on
# laisse encore ce temps à l'aiguillage avant d'exécuter quoi que ce soit.
ROUTE_GRACE_S = 1.0

Routed = tuple[Skill, dict[str, str]]


def inducible(result: TurnResult) -> str:
    """Le programme de ce tour qui peut devenir une compétence, ou "" : le dernier qui a
    réussi, écrit avant toute lecture d'un contenu non fiable (un texte piégé ne doit pas
    pouvoir s'inscrire dans une compétence), avec au moins deux appels, sans compétence.
    Sans programme, l'enchaînement des appels directs du tour (`steps_as_program`)."""
    for (code, trusted), outcome in zip(
        reversed(result.script_sources), reversed(result.scripts), strict=False
    ):
        if outcome.ok and trusted and len(outcome.records) >= 2 and "skills." not in code:
            return code
    if result.scripts:
        return ""
    return steps_as_program(result)


def steps_as_program(result: TurnResult) -> str:
    """« ouvre YouTube, lis la page, clique la vidéo » fait en trois appels directs, écrit
    comme un programme pour l'apprentissage. Un argument choisi après la lecture d'un
    contenu non fiable n'est jamais recopié : il devient `...`, et la compétence devra le
    retrouver elle-même dans ce qu'elle lit."""
    steps = result.direct_sources
    if len(steps) < 2 or len(result.direct_results) != len(steps):
        return ""
    if not all(r.ok for r in result.direct_results):
        return ""
    lines = []
    for call, trusted in steps:
        if call.name.endswith("_skill") or call.name == "remember":
            return ""  # apprendre à apprendre ou à retenir : non
        args = ", ".join(
            f"{key}={value!r}" if trusted else f"{key}=..." for key, value in call.arguments.items()
        )
        lines.append(f"{call.name}({args})")
    return "\n".join(lines)


class SkillPath:
    def __init__(
        self,
        store: SkillStore,
        actions: ActionRunner,
        model: ModelCaller,
        llm: LLMClient,
        config: Callable[[], Config],
        jev: Callable[[], Any],
        usage: UsageTracker | None,
        say_phrase: PhraseSpeaker,
    ) -> None:
        self.store = store
        self.actions = actions
        self.model = model
        self.llm = llm
        self._config = config
        self._jev = jev
        self.usage = usage
        self._say_phrase = say_phrase
        self.task: asyncio.Task[None] | None = None  # écriture d'une compétence en cours
        self._last_inducible: tuple[str, str] | None = None
        self._last_uses: tuple[list[str], float] | None = None

    # -- état -------------------------------------------------------------------------------

    def enabled(self) -> bool:
        return self._config().skills.enabled

    def callable(self) -> dict[str, Skill]:
        """Compétences appelables depuis un programme (`skills.<nom>(...)`)."""
        if not self.enabled():
            return {}
        return {skill.call_name: skill for skill in self.store.usable()}

    def has_active(self) -> bool:
        return any(skill.status == "active" for skill in self.store.usable())

    # -- avant le modèle ----------------------------------------------------------------------

    async def before(self, turn: Turn) -> list[str] | None:
        """Compétence active qui colle : rejouée sans le modèle ; rend les phrases dites,
        ou None si rien n'a été rejoué (une note pour le modèle a pu être ajoutée)."""
        if not self.enabled():
            return None
        config = self._config()
        found = self.store.match(turn.command)
        if found is not None and found[0].status == "active" and config.skills.local_replay:
            skill, values = found
            return await self.replay(skill, values, "motif", turn)
        hint = self._hint(turn.command, found)
        if hint:
            turn.messages.insert(-1, {"role": "system", "content": hint})
        return None

    async def replay(self, skill: Skill, values: dict[str, str], via: str, turn: Turn) -> list[str]:
        """Exécute une compétence active sans le modèle (le programme d'une ligne
        `return skills.<nom>(...)`, par le même chemin qu'un programme du modèle). Réussie
        avec une phrase ou rien à dire : tour fini (`result.shortcut`) ; avec une donnée, le
        modèle la formule ; en échec, le modèle reprend la demande. Rend les phrases dites."""
        result = turn.result
        call = ToolCallReq("competence", SCRIPT_TOOL, {"code": call_code(skill, values)})
        result.local_skill, result.skill_via = skill.name, via
        with turn_context(turn.io):
            results = await self.actions.run(Reply("", [call], ""), turn)
        spoken = turn.take_tool_speech()
        (outcome,) = results
        if outcome.ok and outcome.terminal:
            result.shortcut = True
            if not spoken:
                self._say_phrase(turn.speech, "done")
                spoken.append(PHRASES["done"])
            logger.info("Compétence « {} » ({}) : {}.", skill.name, via, values)
            return spoken
        if not outcome.ok:
            logger.info("Compétence « {} » : échec ({}), le modèle reprend.", skill.name, via)
            turn.messages.insert(
                len(turn.messages) - 2 - len(results),  # avant la demande et le rejeu
                {
                    "role": "system",
                    "content": f"MIKATA a essayé la compétence {skill.signature}, qui a "
                    "échoué (voir le résultat). Fais la demande autrement, sans elle.",
                },
            )
        return spoken

    def _hint(self, command: str, found: Routed | None) -> str:
        """Note pour le modèle : la compétence qui colle (avec ses valeurs), sinon les
        compétences proches. Rien pour une demande sans rapport."""
        if found is not None:
            skill, values = found
            args = ", ".join(f"{k}={v!r}" for k, v in values.items())
            return (
                f"Compétence apprise qui correspond : {skill.signature} — {skill.description} "
                f"Si elle convient, appelle-la dans un programme : "
                f"`return skills.{skill.call_name}({args})`."
            )
        related = self.store.related(command)
        if not related:
            return ""
        lines = [f"- {s.signature} — {s.description}" for s in related]
        return (
            "Compétences apprises proches (appelables dans un programme, si l'une convient "
            "exactement) :\n" + "\n".join(lines)
        )

    # -- aiguillage ---------------------------------------------------------------------------

    def start_router(self, command: str, follow_up: bool) -> asyncio.Task[Any] | None:
        """Lance l'aiguillage s'il y a des compétences actives. Jamais dans la fenêtre de
        suivi (la phrase n'est peut-être pas pour MIKATA)."""
        if follow_up or not self.enabled() or not self._config().skills.route:
            return None
        active = [s for s in self.store.usable() if s.status == "active"]
        if not active:
            return None
        return asyncio.create_task(self._route(command, active), name="aiguillage")

    async def _route(self, command: str, active: list[Skill]) -> Routed | None:
        """Deux étages (phase 15k). Jev (~0,4 s) décide seul quand il est sûr (compétence ou
        « aucune ») ; s'il hésite, ne répond pas ou n'est pas configuré, le petit appel LLM
        décide. Les deux partent ensemble : un doute de Jev ne coûte rien."""
        llm = asyncio.create_task(self._route_llm(command, active))
        jev = self._jev()
        try:
            if jev is not None:
                try:
                    async with asyncio.timeout(self._config().skills.jev_timeout_s):
                        decision = await jev.decide(command, active)
                except TimeoutError:
                    logger.info("Jev : trop lent, l'aiguillage LLM décide.")
                else:
                    if decision.kind == "aucune":
                        return None
                    if decision.kind == "compétence" and decision.skill is not None:
                        values = {k: locate(command, v) for k, v in decision.values.items()}
                        if all(values.values()):
                            return decision.skill, values  # type: ignore[return-value]
            return await llm
        finally:
            if not llm.done():
                llm.cancel()
                await asyncio.gather(llm, return_exceptions=True)

    async def _route_llm(self, command: str, active: list[Skill]) -> Routed | None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(self._config().skills.route_timeout_s):
                routed, usage = await route(self.llm, command, active)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # trop lent, hors ligne : l'appel normal décide seul
            logger.info("Aiguillage sans réponse : {}", str(exc) or type(exc).__name__)
            return None
        if self.usage is not None:
            self.usage.record("aiguillage", usage, 1)
        ms = (time.monotonic() - start) * 1000
        if routed is None:
            logger.info("Aiguillage : aucune compétence ({:.0f} ms).", ms)
        else:
            logger.info("Aiguillage : « {} » {} ({:.0f} ms).", routed[0].name, routed[1], ms)
        return routed

    async def race(
        self,
        router: asyncio.Task[Any],
        turn: Turn,
        offered: list[dict[str, Any]],
        gate: FollowUpGate,
    ) -> tuple[Reply | None, Routed | None]:
        """Premier appel normal et aiguillage en même temps. L'aiguillage l'emporte s'il
        reconnaît une compétence avant que l'appel normal ait dit quoi que ce soit : l'appel
        normal est alors coupé, ou sa réponse (des outils pas encore exécutés) écartée.
        S'il a fini le premier sans parler, l'aiguillage a encore `ROUTE_GRACE_S`."""
        result = turn.result
        probe = SpeechProbe(turn.speech)
        main = asyncio.create_task(
            self.model.call(turn.messages, offered, probe, result, gate, turn.state.strong)
        )
        done, _ = await asyncio.wait({router, main}, return_when=asyncio.FIRST_COMPLETED)
        if main in done and not router.done():
            reply = main.result()
            if reply is not None and not reply.text.strip() and not probe.said:
                await asyncio.wait({router}, timeout=ROUTE_GRACE_S)
        if not router.done():
            router.cancel()
            await asyncio.gather(router, return_exceptions=True)
            return await main, None
        routed = router.result()
        if routed is not None:
            result.llm_calls += 1
        if routed is None or probe.said:
            return await main, None
        if main.done():
            reply = main.result()
            if reply is not None and reply.text.strip():
                return reply, None  # il a déjà répondu en mots : on garde sa réponse
            logger.info("Aiguillage : réponse de l'appel normal écartée (outils non exécutés).")
        else:
            main.cancel()
            await asyncio.gather(main, return_exceptions=True)
        return None, routed

    # -- après le tour ------------------------------------------------------------------------

    def after_turn(self, command: str, result: TurnResult) -> None:
        """Bilan des compétences utilisées, correction de l'utilisateur, et programme à
        apprendre. Jamais bloquant : l'écriture d'une compétence part en arrière-plan."""
        if not self.enabled():
            return
        config = self._config()
        now = time.monotonic()
        previous = self._last_uses
        if (
            previous is not None
            and now - previous[1] <= config.learning.correction_window_s
            and is_correction(command)
        ):
            self.store.correct(previous[0])
        turn_ok = result.outcome == "ok"
        for name, ok, error in result.skill_uses:
            self.store.record(
                name, ok and turn_ok, error or result.outcome, local=bool(result.local_skill)
            )
        names = [name for name, _, _ in result.skill_uses]
        self._last_uses = (names, now) if names else None
        code = inducible(result)
        self._last_inducible = (command, code) if code and turn_ok else None
        if (
            self._last_inducible is None
            or names
            or not config.skills.learn
            or self.store.match(command) is not None
            or self.store.related(command)  # une voisine existe : pas de doublon
            or (self.task is not None and not self.task.done())
        ):
            return
        self.task = asyncio.create_task(
            self._learn(command, self._last_inducible[1]), name="compétence"
        )

    async def _learn(self, command: str, code: str) -> Skill | None:
        """Écrit la compétence (un appel LLM), la vérifie dans le code, la range à l'essai."""
        config = self._config()
        try:
            async with asyncio.timeout(config.llm.turn_timeout_s):
                candidate, usage = await induce(self.llm, command, code, config.skills.learn_effort)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # hors ligne, réponse illisible : rien n'est appris
            logger.warning("Compétence non écrite : {}", str(exc) or type(exc).__name__)
            return None
        if self.usage is not None:
            self.usage.record("compétence", usage, 1)
        if not candidate:
            logger.info("Rien de réutilisable dans « {} ».", command)
            return None
        skill, reason = verify(candidate, command, code, self.actions.action_names())
        if skill is None:
            logger.info("Compétence refusée à la vérification ({}) : {}", command, reason)
            return None
        try:
            return self.store.adopt(skill)
        except (OSError, ValueError) as exc:
            logger.warning("Compétence non rangée : {}", exc)
            return None

    async def capture(self, description: str) -> str:
        """Outil remember_skill : « retiens comment faire ça » sur le dernier programme."""
        if self._last_inducible is None:
            raise ToolError("aucun programme récent à retenir")
        command, code = self._last_inducible
        if self.task is not None and not self.task.done():
            await asyncio.gather(self.task, return_exceptions=True)
            found = self.store.match(command)
            if found is not None:
                return f"C'est déjà retenu : {found[0].description}"
        skill = await self._learn(command, code)
        if skill is None:
            raise ToolError("je n'ai pas réussi à en faire une compétence réutilisable")
        return (
            f"Retenu : {skill.description} Je la referai sans réfléchir après "
            f"{self._config().skills.promote_after} réussites."
        )
