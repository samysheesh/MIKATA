"""Agent v2 (phase 13c) : les essais s'enchaînent, l'erreur guide le suivant, un refus
du garde-fou arrête tout."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent2_fakes import ScriptedLLM, make_registry
from tool_fakes import FakeDesktop

from mikata.agent2.executor import run_subgoal
from mikata.agent2.schema import AppFocused, SubGoal
from mikata.agent2.sdk import Sdk
from mikata.config import Agent2Config

_DEFAULT_PREDICATE = AppFocused(name="spotify")


def _subgoal(done_when: Any = None) -> SubGoal:
    return SubGoal(goal="ouvrir Spotify", channel="apps", done_when=done_when or _DEFAULT_PREDICATE)


def _run(tmp_path: Path, sub: SubGoal, llm: ScriptedLLM, **ctx: Any) -> list:
    registry = make_registry(tmp_path, **ctx)
    cfg = Agent2Config()

    async def scenario() -> list:
        return await run_subgoal(
            0,
            sub,
            sdk_factory=lambda recorder: Sdk(
                registry,
                max_actions=cfg.max_actions_per_script,
                observe_lines=cfg.observe_max_lines,
                recorder=recorder,
            ),
            llm=llm,
            resolved=[],
            cfg=cfg,
            os_name="win32",
        )

    return asyncio.run(scenario())


def test_first_failure_then_success_shows_the_error_to_the_llm(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            "{\"code\": \"await sdk.open_app('Introuvable')\\nreturn 'ok'\"}",
            "{\"code\": \"await sdk.open_app('Spotify')\\nreturn 'ok'\"}",
        ]
    )
    steps = _run(tmp_path, _subgoal(), llm, desktop=FakeDesktop())
    assert len(steps) == 2
    assert steps[0].error and "SdkError" in steps[0].error
    assert steps[1].goal_reached and steps[1].error is None
    # Le 2e prompt contient l'erreur du 1er essai (format type: message).
    second_user = llm.calls[1][-1]["content"]
    assert "SdkError" in second_user and "Introuvable" in second_user


def test_rejected_code_retries_instead_of_crashing(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            '{"code": "import os\\nreturn \'x\'"}',
            "{\"code\": \"await sdk.open_app('Spotify')\\nreturn 'ok'\"}",
        ]
    )
    steps = _run(tmp_path, _subgoal(), llm, desktop=FakeDesktop())
    assert len(steps) == 2
    assert steps[0].error and "CodeRejected" in steps[0].error
    assert steps[1].goal_reached


def test_three_failures_give_three_steps_not_reached(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    desktop.fail.add("launch")
    llm = ScriptedLLM(["{\"code\": \"await sdk.open_app('Introuvable')\\nreturn 'x'\"}"] * 3)
    steps = _run(tmp_path, _subgoal(), llm, desktop=desktop)
    assert len(steps) == 3
    assert all(not step.goal_reached for step in steps)
    assert all(step.error is not None for step in steps)


def test_guard_refusal_stops_the_subgoal_immediately(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        ["{\"code\": \"await sdk.write_file('D:/interdit/x.txt', 'x')\\nreturn 'x'\"}"]
    )
    steps = _run(tmp_path, _subgoal(), llm, desktop=FakeDesktop())
    assert len(steps) == 1
    assert steps[0].error and "garde-fou" in steps[0].error
    assert not steps[0].goal_reached
    assert len(llm.calls) == 1  # aucun nouvel essai demandé
