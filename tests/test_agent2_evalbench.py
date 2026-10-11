"""Agent v2 (phase 13e) : la batterie juge sur le simulé, jamais sur le vrai PC."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent2_fakes import ScriptedLLM

from mikata.agent2.evalbench import (
    EvalCase,
    EvalExpect,
    EvalFixtures,
    EvalResult,
    FixtureRegistry,
    regressions,
    run_case,
)
from mikata.llm.base import ToolCallReq

_CASE = EvalCase(
    id="play_latest_opening",
    request="lance le dernier opening de One Piece",
    fixtures=EvalFixtures(
        installed=["Spotify", "Discord", "Google Chrome"],
        web={"One Piece opening": "Le dernier opening de One Piece est « Assu ! »"},
    ),
    expect=EvalExpect(
        verb="play", channel="youtube", tools_called=["open_url"], tools_forbidden=["delete_file"]
    ),
)

_UNDERSTAND = (
    '{"verb": "play", "object": "dernier opening de One Piece",'
    ' "object_kind": "opening d\'anime", "entities": {}, "constraints": [],'
    ' "target": null, "channel_hint": null,'
    ' "unknowns": [{"name": "titre", "question": "titre exact ?", "kind": "web"}],'
    ' "sub_intents": []}'
)
_RESOLVE = '{"value": "Assu !", "confident": true}'
_PLAN = (
    '{"subgoals": [{"goal": "YouTube affiche la vidéo", "channel": "youtube",'
    ' "done_when": {"type": "tool_ok", "tool": "open_url"},'
    ' "skill": null, "skill_args": {}}]}'
)
_STEP = (
    '{"code": "await sdk.open_url(\'https://www.youtube.com/results?search_query=Assu\')'
    "\\nreturn 'ok'\"}"
)


def test_a_case_passes_when_expectations_hold(tmp_path: Path) -> None:
    llm = ScriptedLLM([_UNDERSTAND, _RESOLVE, _PLAN, _STEP])
    result = asyncio.run(run_case(_CASE, llm, root=tmp_path / "traces"))
    assert result.passed, result.reasons
    assert result.llm_calls == 4


def test_a_forbidden_tool_fails_the_case(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            _UNDERSTAND,
            _RESOLVE,
            _PLAN.replace("open_url", "delete_file", 2),
            "{\"code\": \"await sdk.trash('C:/vieux.txt')\\nreturn 'ok'\"}",
        ]
    )
    result = asyncio.run(run_case(_CASE, llm, root=tmp_path / "traces"))
    assert not result.passed
    assert any("outil interdit" in reason for reason in result.reasons)


def test_regressions_detect_a_pass_to_fail(tmp_path: Path) -> None:
    before = [EvalResult(id="a", passed=True), EvalResult(id="b", passed=True)]
    after = [
        EvalResult(id="a", passed=True),
        EvalResult(id="b", passed=False, reasons=["canal attendu spotify, obtenu youtube"]),
    ]
    broken = regressions(before, after)
    assert broken == ["b : canal attendu spotify, obtenu youtube"]


def test_fixture_registry_executes_nothing_real(tmp_path: Path) -> None:
    registry = FixtureRegistry(_CASE.fixtures)
    result = asyncio.run(
        registry.execute(ToolCallReq(id="1", name="delete_file", arguments={"path": "/x"}))
    )
    assert result.ok and result.content == "(simulé)"
    assert registry.calls == [("delete_file", {"path": "/x"})]
    assert not (tmp_path / "x").exists()
