"""Agent v2 (phase 13c) : chaque prédicat se vérifie sans jamais planter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent2_fakes import ScriptedLLM

from mikata.agent2.schema import (
    ActionRecord,
    AppFocused,
    FileExists,
    Judge,
    Observation,
    PageShows,
    TextVisible,
    ToolSucceeded,
    WindowTitleMatches,
)
from mikata.agent2.verify import check

_OBS = Observation(
    active_app="Spotify",
    title="Spotify Premium",
    page="YouTube — One Piece Opening 27",
    summary="Boutons : Lire, Pause\nEnvoyé",
)


def _check(pred, obs: Observation = _OBS, actions: tuple[ActionRecord, ...] = ()) -> tuple:
    async def scenario() -> tuple:
        return await check(pred, obs, actions, ScriptedLLM([]), effort="none")

    return asyncio.run(scenario())


def test_app_focused() -> None:
    assert _check(AppFocused(name="Spotify")) == (True, True)
    assert _check(AppFocused(name="Brave")) == (False, True)


def test_window_title() -> None:
    assert _check(WindowTitleMatches(pattern="spotify")) == (True, True)
    assert _check(WindowTitleMatches(pattern="^Discord")) == (False, True)


def test_page_shows() -> None:
    assert _check(PageShows(pattern="opening 27")) == (True, True)
    assert _check(PageShows(pattern="Naruto")) == (False, True)


def test_file_exists(tmp_path: Path) -> None:
    present = tmp_path / "facture.pdf"
    present.write_text("x", encoding="utf-8")
    assert _check(FileExists(path=str(present))) == (True, True)
    assert _check(FileExists(path=str(tmp_path / "absent.pdf"))) == (False, True)


def test_text_visible() -> None:
    assert _check(TextVisible(pattern="Envoyé")) == (True, True)
    assert _check(TextVisible(pattern="Rien de tel")) == (False, True)


def test_tool_ok_uses_the_last_action_of_that_tool() -> None:
    actions = (
        ActionRecord(tool="open_app", args={}, ok=False, result="x", duration_s=0.1),
        ActionRecord(tool="open_app", args={}, ok=True, result="x", duration_s=0.1),
    )
    assert _check(ToolSucceeded(tool="open_app"), actions=actions) == (True, True)
    failed = actions[:1]
    assert _check(ToolSucceeded(tool="open_app"), actions=failed) == (False, True)
    assert _check(ToolSucceeded(tool="autre_outil"), actions=actions) == (False, True)


def test_invalid_regex_is_false_and_reliable() -> None:
    assert _check(WindowTitleMatches(pattern="(")) == (False, True)


def test_judge_reaches_with_a_scripted_llm() -> None:
    llm = ScriptedLLM(['{"reached": true, "why": "la fenêtre est là"}'])

    async def scenario() -> tuple:
        return await check(Judge(question="Est-ce ouvert ?"), _OBS, (), llm, effort="none")

    assert asyncio.run(scenario()) == (True, False)


def test_judge_doubt_is_false() -> None:
    llm = ScriptedLLM(['{"reached": false, "why": "pas vu"}'])

    async def scenario() -> tuple:
        return await check(Judge(question="Est-ce ouvert ?"), _OBS, (), llm, effort="none")

    assert asyncio.run(scenario()) == (False, False)


def test_judge_invalid_answer_is_false_and_unreliable() -> None:
    llm = ScriptedLLM(["pas du json", "encore raté"])

    async def scenario() -> tuple:
        return await check(Judge(question="Est-ce ouvert ?"), _OBS, (), llm, effort="none")

    assert asyncio.run(scenario()) == (False, False)
