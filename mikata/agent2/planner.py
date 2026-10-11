"""Agent v2 (phase 13c) : l'intention devient une liste de sous-buts vérifiables.

Un appel LLM (réflexion forte) ; le post-traitement est déterministe : troncature au
nombre de sous-buts autorisé, canal inconnu remplacé par le canal choisi, plan vide ou
invalide remplacé par un unique sous-but de repli.
"""

from __future__ import annotations

from pydantic import BaseModel

from mikata.agent2.channels import BUILTIN_CHANNELS, Channel
from mikata.agent2.llm_json import ask_json
from mikata.agent2.prompts import PLAN_SYSTEM, PLAN_USER
from mikata.agent2.schema import Intent, Judge, Resolved, SubGoal
from mikata.llm.base import LLMClient, Usage

_KNOWN_CHANNELS = {channel.name for channel in BUILTIN_CHANNELS}


class _Plan(BaseModel):
    subgoals: list[SubGoal]


async def make_plan(
    llm: LLMClient,
    intent: Intent,
    resolved: list[Resolved],
    channel: Channel,
    *,
    channels_text: str,
    skills_text: str,
    os_name: str,
    max_subgoals: int,
    effort: str | None,
) -> tuple[list[SubGoal], Usage | None, int]:
    """Le plan, jamais vide. Canal inconnu corrigé, longueur bornée en Python."""
    system = PLAN_SYSTEM.format(
        os=os_name,
        max_subgoals=max_subgoals,
        channels=channels_text,
        channel=channel.name,
        skills=skills_text,
    )
    user = PLAN_USER.format(
        intent=intent.model_dump_json(exclude_defaults=True),
        resolved="\n".join(f"- {item.name} = {item.value}" for item in resolved) or "(aucune)",
    )
    plan, usage, calls = await ask_json(llm, system, user, _Plan, effort=effort)
    if plan is None or not plan.subgoals:
        fallback = SubGoal(
            goal=intent.object,
            channel=channel.name,
            done_when=Judge(question=f"La demande « {intent.object} » est-elle accomplie ?"),
        )
        return [fallback], usage, calls
    subgoals = plan.subgoals[:max_subgoals]
    for subgoal in subgoals:
        if subgoal.channel not in _KNOWN_CHANNELS:
            subgoal.channel = channel.name
    return subgoals, usage, calls
