"""Garde-fou des outils UIA (phase 12a) : contrôles destructifs refusés, contrôles
sensibles à confirmer à voix haute, texte de suppression jamais écrit dans un champ."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeUi, make_ctx

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
    ("control", "verdict"),
    [
        ("Supprimer définitivement", Verdict.DENY),
        ("Vider la corbeille", Verdict.DENY),
        ("Réinitialiser ce PC", Verdict.DENY),
        ("Supprimer", Verdict.CONFIRM),
        ("Désinstaller", Verdict.CONFIRM),
        ("Acheter", Verdict.CONFIRM),
        ("Lecture", Verdict.ALLOW),
        ("Envoyer le message", Verdict.ALLOW),
    ],
)
def test_ui_controls_are_classified(tmp_path: Path, control: str, verdict: Verdict) -> None:
    decision = make_ctx(tmp_path).guard.check_ui_action(control)  # type: ignore[union-attr]
    assert decision.verdict is verdict


@pytest.mark.parametrize(
    "control",
    ["Supprimer définitivement", "Vider la corbeille", "Réinitialiser ce PC", "Formater"],
)
def test_destructive_controls_are_never_clicked(
    tmp_path: Path, control: str, caplog_loguru
) -> None:
    ui = FakeUi()
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_click", control=control)
    assert not result.ok and "refusé par le garde-fou" in result.content
    assert not any(name == "click" for name, _ in ui.log)  # rien n'est cliqué
    assert any("REFUSÉ | uia_click(" in line for line in caplog_loguru)


def test_sensitive_controls_ask_first(tmp_path: Path) -> None:
    ui = FakeUi()
    talk = Conversation("oui")
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_click", talk.turn(), control="Supprimer")
    assert talk.said == ["Cliquer « Supprimer » va supprimer quelque chose. Tu confirmes ?"]
    assert talk.drained == 1 and talk.listened == [6]  # question dite, puis 6 s d'écoute
    assert result.ok and ("click", ("Discord", "Supprimer", None, 1)) in ui.log


def test_sensitive_controls_cancelled_on_no(tmp_path: Path) -> None:
    ui = FakeUi()
    talk = Conversation("non")
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_click", talk.turn(), control="Supprimer")
    assert not result.ok and result.content.startswith("non lancé :")
    assert not any(name == "click" for name, _ in ui.log)


def test_sensitive_controls_without_microphone_are_refused(tmp_path: Path) -> None:
    ui = FakeUi()
    talk = Conversation("oui")
    turn = talk.turn(listen=False)  # --ask : personne pour répondre
    result = _execute(make_ctx(tmp_path, uia=ui), "uia_click", turn, control="Supprimer")
    assert not result.ok and not any(name == "click" for name, _ in ui.log)
    assert talk.said == []  # pas de question posée sans micro


def test_confirm_risky_disabled_still_denies_destructive(tmp_path: Path, caplog_loguru) -> None:
    config = make_ctx(tmp_path, uia=FakeUi()).config
    guard = config.guard.model_copy(update={"confirm_risky": False})
    config = config.model_copy(update={"guard": guard})
    ui = FakeUi()
    result = _execute(make_ctx(tmp_path, config=config, uia=ui), "uia_click", control="Supprimer")
    assert result.ok and ("click", ("Discord", "Supprimer", None, 1)) in ui.log  # sans question
    result = _execute(make_ctx(tmp_path, config=config, uia=ui), "uia_click", control="Formater")
    assert not result.ok
    assert not any(name == "click" and value[1] == "Formater" for name, value in ui.log)
