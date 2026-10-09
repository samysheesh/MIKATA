"""Outils UIA (phase 12a) : fenêtre, lecture, clic, champ, état ambiant — sur pilote simulé."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeUi, make_config, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolContext, TurnIO, UiControlInfo, turn_context
from mikata.tools.registry import ToolRegistry, ToolResult


def _execute(
    ctx: ToolContext, tool: str, turn: TurnIO | None = None, **arguments: Any
) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(ctx)
        try:
            if turn is None:
                return await registry.execute(ToolCallReq("1", tool, arguments))
            with turn_context(turn):
                return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


def test_uia_window_lists_the_controls(tmp_path: Path) -> None:
    ui = FakeUi(
        controls=[
            UiControlInfo("Rechercher", "bouton"),
            UiControlInfo("Résultats", "liste"),
        ]
    )
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_window", process="Discord")
    assert result.ok
    assert "Fenêtre « Discord »" in result.content
    assert "- Rechercher [bouton]" in result.content
    assert "- Résultats [liste]" in result.content
    assert ui.log[0] == ("window", (None, "Discord"))
    assert ui.log[1] == ("controls", ("Discord", 200))


def test_uia_window_with_no_readable_control_fails(tmp_path: Path) -> None:
    result = _execute(make_ctx(tmp_path, uia=FakeUi(controls=[])), "uia_window")
    assert not result.ok and "aucun contrôle lisible" in result.content


def test_uia_read_returns_the_text(tmp_path: Path) -> None:
    ui = FakeUi(text="Bonjour à tous.")
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_read", process="notepad", control="Document")
    assert result.ok and result.content == "Bonjour à tous."
    assert ui.log[0] == ("window", (None, "notepad"))
    assert ui.log[1] == ("read", ("Discord", "Document", 2000))


def test_uia_click_clicks_by_name(tmp_path: Path) -> None:
    ui = FakeUi()
    result = _execute(
        make_ctx(tmp_path, uia=ui), "uia_click", control="Lecture", kind="bouton", nth=2
    )
    assert result.ok and result.content == "« Lecture » cliqué."
    assert ("click", ("Discord", "Lecture", "bouton", 2)) in ui.log


def test_uia_set_text_writes_into_the_field(tmp_path: Path) -> None:
    ui = FakeUi()
    result = _execute(
        make_ctx(tmp_path, uia=ui), "uia_set_text", control="Recherche", text="météo Paris"
    )
    assert result.ok and result.content == "« Recherche » rempli."
    assert ("set_text", ("Discord", "Recherche", "météo Paris")) in ui.log


def test_uia_set_text_refuses_a_deletion_command(tmp_path: Path, caplog_loguru) -> None:
    ui = FakeUi()
    result = _execute(
        make_ctx(tmp_path, uia=ui), "uia_set_text", control="Console", text="Remove-Item notes.txt"
    )
    assert not result.ok and "suppression définitive" in result.content
    assert not any(name == "set_text" for name, _ in ui.log)  # rien n'est écrit
    assert any("REFUSÉ | uia_set_text(" in line for line in caplog_loguru)


def test_uia_state_reads_the_ambient_cache(tmp_path: Path) -> None:
    ui = FakeUi(state="Discord : salon vocal « Général ». En vocal : Samy.")
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_state")
    assert result.ok and result.content == ui.state
    assert any(name == "state" for name, _ in ui.log)


def test_uia_state_with_nothing_tracked_fails(tmp_path: Path) -> None:
    result = _execute(make_ctx(tmp_path, uia=FakeUi(state="")), "uia_state")
    assert not result.ok and "aucun état ambiant" in result.content


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("uia_window", {}),
        ("uia_read", {}),
        ("uia_click", {"control": "Lecture"}),
        ("uia_set_text", {"control": "Recherche", "text": "bonjour"}),
        ("uia_state", {}),
    ],
)
def test_uia_tools_without_the_service_explain_themselves(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    result = _execute(make_ctx(tmp_path), tool, **arguments)
    assert not result.ok and "pilotage fin" in result.content


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("uia_window", {}),
        ("uia_read", {}),
        ("uia_click", {"control": "Lecture"}),
        ("uia_set_text", {"control": "Recherche", "text": "bonjour"}),
        ("uia_state", {}),
    ],
)
def test_uia_tools_disabled_in_config_say_so(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    config = make_config(tmp_path, uia={"enabled": False})
    result = _execute(make_ctx(tmp_path, config=config, uia=FakeUi()), tool, **arguments)
    assert not result.ok and "désactivé" in result.content


def test_state_poller_is_paused_while_a_tool_runs(tmp_path: Path) -> None:
    ui = FakeUi()
    _execute(make_ctx(tmp_path, uia=ui), "uia_click", control="Lecture")
    assert ui.paused == [True, False]  # le poller saute ses lectures pendant l'action


def test_uia_click_is_logged_ok(tmp_path: Path, caplog_loguru) -> None:
    _execute(make_ctx(tmp_path, uia=FakeUi()), "uia_click", control="Lecture")
    assert any("OK | uia_click(" in line for line in caplog_loguru)
