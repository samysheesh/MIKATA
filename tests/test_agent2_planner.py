"""Agent v2 (phase 13c) : le plan est validé, borné et a toujours un repli."""

from __future__ import annotations

import asyncio

from agent2_fakes import ScriptedLLM

from mikata.agent2.channels import Channel
from mikata.agent2.planner import make_plan
from mikata.agent2.schema import Intent, Verb

_CHANNEL = Channel(name="youtube", kind="site", capabilities=["play.video"])

_PLAN_JSON = (
    '{"subgoals": ['
    '{"goal": "YouTube affiche les résultats", "channel": "youtube",'
    ' "done_when": {"type": "page_shows", "pattern": "Opening"},'
    ' "skill": null, "skill_args": {}},'
    '{"goal": "la vidéo se lance", "channel": "youtube",'
    ' "done_when": {"type": "page_shows", "pattern": "Lecture"},'
    ' "skill": null, "skill_args": {}}]}'
)


def _run(llm: ScriptedLLM, max_subgoals: int = 8) -> tuple:
    async def scenario() -> tuple:
        return await make_plan(
            llm,
            Intent(verb=Verb.PLAY, object="le dernier opening de One Piece"),
            [],
            _CHANNEL,
            channels_text="- youtube : play.video",
            skills_text="(aucun)",
            os_name="win32",
            max_subgoals=max_subgoals,
            effort="high",
        )

    return asyncio.run(scenario())


def test_valid_plan_is_returned() -> None:
    llm = ScriptedLLM([_PLAN_JSON])
    subgoals, usage, calls = _run(llm)
    assert len(subgoals) == 2
    assert subgoals[0].channel == "youtube"
    assert subgoals[0].done_when.type == "page_shows"
    assert calls == 1 and usage is not None


def test_unknown_channel_is_replaced_by_the_chosen_one() -> None:
    llm = ScriptedLLM([_PLAN_JSON.replace('"channel": "youtube",', '"channel": "mystère",', 1)])
    subgoals, _, _ = _run(llm)
    assert subgoals[0].channel == "youtube"


def test_plan_is_truncated_to_max_subgoals() -> None:
    llm = ScriptedLLM([_PLAN_JSON])
    subgoals, _, _ = _run(llm, max_subgoals=1)
    assert len(subgoals) == 1


def test_invalid_plan_falls_back_to_a_judge_subgoal() -> None:
    llm = ScriptedLLM(["pas du json", "encore raté"])
    subgoals, _, calls = _run(llm)
    assert calls == 2
    assert len(subgoals) == 1
    assert subgoals[0].channel == "youtube"
    assert subgoals[0].done_when.type == "judge"
