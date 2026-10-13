"""Banc d'essai : fabrique de l'agent évalué (le même assemblage que `__main__.build_agent`,
avec une mémoire de conversation neuve à chaque essai)."""

from __future__ import annotations

from typing import Any

from mikata.config import Config
from mikata.eval.bench import AgentFactory
from mikata.tools.base import ToolContext


def agent_factory(llm: Any, usage: Any = None) -> AgentFactory:
    """Fabrique branchée sur `llm` (vrai client ou LLM scripté) ; un agent par essai."""

    def make(config: Config, ctx: ToolContext) -> Any:
        from mikata.agent.loop import Agent
        from mikata.agent.memory import ShortTermMemory
        from mikata.tools.apps import installed_app_names, open_apps
        from mikata.tools.registry import ToolRegistry

        desktop = ctx.desktop

        async def running_apps() -> list[str]:
            assert desktop is not None
            return open_apps(await ctx.run_blocking(desktop.windows))

        return Agent(
            llm,
            config,
            memory=ShortTermMemory(config.agent.history_max_tokens, 600.0),
            tools=ToolRegistry(ctx),
            long_memory=ctx.memory,
            skills=ctx.skills,
            usage=usage,
            open_apps=running_apps if desktop is not None else None,
            installed_apps=(
                (lambda: installed_app_names(desktop.cached_apps()))
                if desktop is not None
                else None
            ),
        )

    return make
