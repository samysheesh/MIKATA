"""Garde-fou du navigateur (phase 12b) : seuls http/https s'ouvrent, un lien de
téléchargement demande confirmation, un texte de suppression ne s'écrit pas dans un champ."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeBrowser, make_ctx

from mikata.agent.guard import Verdict
from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolContext, TurnIO, turn_context
from mikata.tools.registry import ToolRegistry, ToolResult


class Conversation:
    """Une commande en cours : ce que MIKATA dit, et ce que l'utilisateur répond."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.said: list[str] = []
        self.listened: list[float] = []
        self.drained = 0

    def turn(self, listen: bool = True) -> TurnIO:
        async def drain() -> None:
            self.drained += 1

        async def hear(timeout_s: float) -> str:
            self.listened.append(timeout_s)
            return self.answers.pop(0) if self.answers else ""

        return TurnIO(say=self.said.append, drain=drain, listen=hear if listen else None)


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


@pytest.mark.parametrize(
    ("url", "verdict"),
    [
        ("https://www.youtube.com", Verdict.ALLOW),
        ("http://example.com", Verdict.ALLOW),
        ("https://example.com/x?q=1#part", Verdict.ALLOW),
        ("HTTPS://EXAMPLE.COM", Verdict.ALLOW),  # la casse du schéma n'a pas d'importance
        ("file://C:/notes.txt", Verdict.DENY),
        ("javascript:alert(1)", Verdict.DENY),
        ("data:text/html,<b>x</b>", Verdict.DENY),
        ("about:blank", Verdict.DENY),
        ("ftp://serveur/fichier", Verdict.DENY),
        ("", Verdict.DENY),
    ],
)
def test_navigation_schemes_are_classified(tmp_path: Path, url: str, verdict: Verdict) -> None:
    decision = make_ctx(tmp_path).guard.check_navigation(url)  # type: ignore[union-attr]
    assert decision.verdict is verdict


@pytest.mark.parametrize(
    ("url", "verdict"),
    [
        ("https://example.com/setup.exe", Verdict.CONFIRM),
        ("https://example.com/archive.zip?x=1", Verdict.CONFIRM),
        ("https://example.com/photo.jpg", Verdict.ALLOW),
        ("https://www.youtube.com/watch?v=abc", Verdict.ALLOW),
    ],
)
def test_download_links_are_classified(tmp_path: Path, url: str, verdict: Verdict) -> None:
    decision = make_ctx(tmp_path).guard.check_download(url)  # type: ignore[union-attr]
    assert decision.verdict is verdict


@pytest.mark.parametrize("url", ["file://C:/notes.txt", "javascript:alert(1)", "about:blank"])
def test_forbidden_navigations_never_open(tmp_path: Path, url: str, caplog_loguru) -> None:
    browser = FakeBrowser()
    result = _execute(make_ctx(tmp_path, browser=browser), "browser_open", url=url)
    assert not result.ok and "refusé par le garde-fou" in result.content
    assert not any(name == "open" for name, _ in browser.log)
    assert any("REFUSÉ | browser_open(" in line for line in caplog_loguru)


def test_download_click_asks_first(tmp_path: Path) -> None:
    browser = FakeBrowser(href="https://example.com/setup.exe")
    talk = Conversation("oui")
    result = _execute(
        make_ctx(tmp_path, browser=browser), "browser_click", talk.turn(), text="Télécharger"
    )
    assert talk.said == ["Cliquer « Télécharger » va télécharger un fichier. Tu confirmes ?"]
    assert talk.drained == 1 and talk.listened == [6]
    assert result.ok and any(name == "click" for name, _ in browser.log)


def test_download_click_cancelled_on_no(tmp_path: Path) -> None:
    browser = FakeBrowser(href="https://example.com/setup.exe")
    talk = Conversation("non")
    result = _execute(
        make_ctx(tmp_path, browser=browser), "browser_click", talk.turn(), text="Télécharger"
    )
    assert not result.ok and not any(name == "click" for name, _ in browser.log)


def test_download_click_without_microphone_is_refused(tmp_path: Path) -> None:
    browser = FakeBrowser(href="https://example.com/setup.exe")
    talk = Conversation("oui")
    result = _execute(
        make_ctx(tmp_path, browser=browser),
        "browser_click",
        talk.turn(listen=False),
        text="Télécharger",
    )
    assert not result.ok and not any(name == "click" for name, _ in browser.log)


def test_plain_click_does_not_ask(tmp_path: Path) -> None:
    browser = FakeBrowser(href="https://www.youtube.com/watch?v=abc")
    talk = Conversation()
    result = _execute(
        make_ctx(tmp_path, browser=browser), "browser_click", talk.turn(), text="Lecture"
    )
    assert result.ok and talk.said == []  # aucun risque : aucune question
