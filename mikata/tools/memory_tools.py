"""Outils de mémoire long terme (§6, §7) : remember ★, forget ★.

Les faits sont gardés dans `data/memory.json` (`agent.memory.LongTermMemory`, 30 au plus)
et injectés dans le prompt système à chaque commande.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mikata.agent.memory import LongTermMemory, MemoryFullError
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
    description = "Oublie un fait mémorisé, quand l'utilisateur le demande."
    params = ForgetParams
    terminal = True

    async def run(self, args: ForgetParams) -> str:
        removed = _memory(self.ctx).forget(args.fact)
        if removed is None:
            raise ToolError(f"aucun fait mémorisé ne correspond à « {args.fact} »")
        return f"Oublié : {removed}"
