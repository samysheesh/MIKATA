"""Outils de mémoire long terme (§6, §7) : remember ★, forget ★, set_history_duration.

Les faits sont gardés dans `data/memory.json` (`agent.memory.LongTermMemory`, 30 au plus)
et injectés dans le prompt système à chaque commande.

Ajout phase 6 : `set_history_duration` change, à la voix, la durée pendant laquelle la
conversation est gardée sans interaction (« garde la conversation un mois »).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mikata.action.effects import Effect
from mikata.agent.memory import LongTermMemory, MemoryFullError, ShortTermMemory
from mikata.tools.base import BaseTool, ToolContext, ToolError
from mikata.tools.registry import register_tool


def _memory(ctx: ToolContext) -> LongTermMemory:
    if not isinstance(ctx.memory, LongTermMemory):
        raise ToolError("mémoire long terme indisponible")
    return ctx.memory


class FactParams(BaseModel):
    fact: str = Field(
        min_length=2,
        max_length=300,
        description="Le fait, en une phrase autonome (ex. « Le chien de l'utilisateur "
        "s'appelle Rex »)",
    )


@register_tool
class Remember(BaseTool[FactParams]):
    name = "remember"
    sdk = "memory.remember"
    effect = Effect.CONSEQUENTIAL
    expected_s = 0.1
    description = "Mémorise durablement un fait sur l'utilisateur quand il le demande."
    params = FactParams
    terminal = True

    async def run(self, args: FactParams) -> str:
        try:
            added = _memory(self.ctx).remember(args.fact)
        except MemoryFullError as exc:
            raise ToolError(f"{exc} : demande à l'utilisateur quoi oublier") from exc
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return "Mémorisé." if added else "Je le savais déjà."


class ForgetParams(BaseModel):
    fact: str = Field(min_length=2, description="Le fait à oublier (formulation approchée)")


@register_tool
class Forget(BaseTool[ForgetParams]):
    name = "forget"
    sdk = "memory.forget"
    effect = Effect.CONSEQUENTIAL
    expected_s = 0.1
    description = "Oublie un fait mémorisé, quand l'utilisateur le demande."
    params = ForgetParams
    terminal = True

    async def run(self, args: ForgetParams) -> str:
        removed = _memory(self.ctx).forget(args.fact)
        if removed is None:
            raise ToolError(f"aucun fait mémorisé ne correspond à « {args.fact} »")
        return f"Oublié : {removed}"


MAX_HISTORY_HOURS = 24 * 366  # un an


def format_keep_duration(hours: float) -> str:
    """« 1 heure », « 12 heures », « 3 jours », « 2 semaines », « 1 mois »."""
    for size, singular, plural in (
        (24 * 365, "an", "ans"),
        (24 * 30, "mois", "mois"),
        (24 * 7, "semaine", "semaines"),
        (24, "jour", "jours"),
        (1, "heure", "heures"),
    ):
        count = hours / size
        if count >= 1 and abs(count - round(count)) < 1e-6:
            n = round(count)
            return f"{n} {singular if n == 1 else plural}"
    return f"{hours:g} heures"


class HistoryDurationParams(BaseModel):
    hours: float = Field(
        gt=0,
        le=MAX_HISTORY_HOURS,
        description="Durée en heures (1 jour = 24, 1 semaine = 168, 1 mois = 720, 1 an = 8760)",
    )


@register_tool
class SetHistoryDuration(BaseTool[HistoryDurationParams]):
    name = "set_history_duration"
    sdk = "memory.keep_history"
    effect = Effect.ACT
    expected_s = 0.1
    description = (
        "Change combien de temps la conversation est gardée sans échange (« garde la "
        "conversation un mois »)."
    )
    params = HistoryDurationParams

    async def run(self, args: HistoryDurationParams) -> str:
        history = self.ctx.history
        if not isinstance(history, ShortTermMemory):
            raise ToolError("historique de conversation indisponible")
        history.set_keep(args.hours * 3600)
        duration = format_keep_duration(args.hours)
        if history.path is None:
            return (
                f"Conversation gardée {duration} sans échange, jusqu'au redémarrage "
                "(agent.history_persist est désactivé)."
            )
        return f"Conversation gardée {duration} sans échange, même après un redémarrage."
