"""Outils du navigateur (phase 12b) : onglets, ouverture, lecture, clic, champ, retour —
sur pilote simulé. Le garde-fou est vérifié dans test_guard_browser.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeBrowser, make_config, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolContext, TurnIO, turn_context
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


def test_browser_tabs_lists_the_tabs(tmp_path: Path) -> None:
    browser = FakeBrowser(tabs="- YouTube — https://youtube.com\n- Exemple — https://example.com")
    result = _execute(make_ctx(tmp_path, browser=browser), "browser_tabs")
    assert result.ok and "youtube.com" in result.content
    assert any(name == "tabs" for name, _ in browser.log)


def test_browser_open_navigates(tmp_path: Path) -> None:
    browser = FakeBrowser()
    result = _execute(
        make_ctx(tmp_path, browser=browser), "browser_open", url="https://www.youtube.com"
    )
    assert result.ok and result.content == "Page ouverte : https://www.youtube.com"
    assert ("open", "https://www.youtube.com") in browser.log


def test_browser_read_returns_the_page_and_marks_web(tmp_path: Path) -> None:
    browser = FakeBrowser(page="Titre : Vidéo\nURL : https://youtube.com\nTexte :\nContenu.")

    async def drain() -> None:
        pass

    turn = TurnIO(say=lambda text: None, drain=drain)
    result = _execute(make_ctx(tmp_path, browser=browser), "browser_read", turn)
    assert result.ok and "Vidéo" in result.content
    assert turn.web_read is True  # confirm_after_web s'applique ensuite
    assert any(name == "read" and value == 4000 for name, value in browser.log)


def test_browser_click_finds_then_clicks(tmp_path: Path) -> None:
    browser = FakeBrowser()
    result = _execute(make_ctx(tmp_path, browser=browser), "browser_click", text="Lecture", nth=2)
    assert result.ok and result.content == "« Lecture » cliqué."
    assert ("find", ("Lecture", 2)) in browser.log
    assert ("click", ("Lecture", 2)) in browser.log


def test_browser_type_writes_into_the_field(tmp_path: Path) -> None:
    browser = FakeBrowser()
    result = _execute(
        make_ctx(tmp_path, browser=browser), "browser_type", field="Recherche", text="black clover"
    )
    assert result.ok and "« Recherche »" in result.content
    assert ("type", ("Recherche", "black clover")) in browser.log


def test_browser_type_refuses_a_deletion_command(tmp_path: Path, caplog_loguru) -> None:
    browser = FakeBrowser()
    result = _execute(
        make_ctx(tmp_path, browser=browser),
        "browser_type",
        field="Console",
        text="Remove-Item notes.txt",
    )
    assert not result.ok and "suppression définitive" in result.content
    assert not any(name == "type" for name, _ in browser.log)
    assert any("REFUSÉ | browser_type(" in line for line in caplog_loguru)


def test_browser_back_goes_back(tmp_path: Path) -> None:
    browser = FakeBrowser()
    result = _execute(make_ctx(tmp_path, browser=browser), "browser_back")
    assert result.ok and result.content == "Page précédente."
    assert any(name == "back" for name, _ in browser.log)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("browser_tabs", {}),
        ("browser_open", {"url": "https://example.com"}),
        ("browser_read", {}),
        ("browser_click", {"text": "Lecture"}),
        ("browser_type", {"field": "Recherche", "text": "bonjour"}),
        ("browser_back", {}),
    ],
)
def test_browser_tools_without_the_service_explain_themselves(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    result = _execute(make_ctx(tmp_path), tool, **arguments)
    assert not result.ok and "contrôle du navigateur" in result.content


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("browser_tabs", {}),
        ("browser_open", {"url": "https://example.com"}),
        ("browser_read", {}),
        ("browser_click", {"text": "Lecture"}),
        ("browser_type", {"field": "Recherche", "text": "bonjour"}),
        ("browser_back", {}),
    ],
)
def test_browser_tools_disabled_in_config_say_so(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    config = make_config(tmp_path, browser={"enabled": False})
    result = _execute(make_ctx(tmp_path, config=config, browser=FakeBrowser()), tool, **arguments)
    assert not result.ok and "désactivé" in result.content


def test_browser_open_is_logged_ok(tmp_path: Path, caplog_loguru) -> None:
    _execute(make_ctx(tmp_path, browser=FakeBrowser()), "browser_open", url="https://example.com")
    assert any("OK | browser_open(" in line for line in caplog_loguru)
