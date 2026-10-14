"""Phase 16g : une correction se refait jusqu'au bout (« non, sur Spotify »)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import FakeDesktop, make_ctx

from mikata.agent.loop import Agent, TurnResult
from mikata.eval.bench import history_steps
from mikata.llm.base import Done, TextDelta, ToolCallReq
from mikata.tools.registry import ToolRegistry


def _respond(
    tmp_path: Path,
    llm: ScriptedLLM,
    command: str,
    before: tuple[str, list[dict[str, Any]]] | None = None,
) -> tuple[TurnResult, FakeDesktop]:
    """`before` : l'échange précédent (demande, appels), celui que la commande corrige."""
    desktop = FakeDesktop()
    ctx = make_ctx(tmp_path, desktop=desktop)
    agent = Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
    )
    if before is not None:
        agent.memory.add(before[0], "C'est fait.", history_steps(before[1]))

    async def scenario() -> TurnResult:
        try:
            return await agent.respond(command, Sink())
        finally:
            await agent.aclose()

    return asyncio.run(scenario()), desktop


def _call(tool: str, **args: Any) -> list[Any]:
    return [ToolCallReq("c", tool, args), Done("tool_calls")]


def test_a_correction_that_only_undoes_goes_on_to_redo(tmp_path: Path) -> None:
    """Le 14/10 : « non, sur Spotify » fermait YouTube, puis « C'est fait » sans rien
    lancer sur Spotify (fermer est terminal, le tour s'arrêtait là)."""
    llm = ScriptedLLM(
        _call("close_app", name="Brave"),
        _call("open_app", name="Spotify"),
    )
    result, desktop = _respond(tmp_path, llm, "Non, sur Spotify.")
    assert len(llm.calls) == 2  # le modèle a vu la fermeture et a continué
    assert ("launch", "Spotify") in desktop.log
    assert result.shortcut  # l'ouverture, elle, conclut


def test_closing_on_request_still_ends_in_one_call(tmp_path: Path) -> None:
    llm = ScriptedLLM(_call("close_app", name="Brave"), [TextDelta("inutile"), Done("stop")])
    result, _ = _respond(tmp_path, llm, "ferme Brave")
    assert len(llm.calls) == 1 and result.shortcut and result.answer == "C'est fait."


def test_a_correction_that_redoes_directly_still_ends_in_one_call(tmp_path: Path) -> None:
    llm = ScriptedLLM(_call("open_app", name="Spotify"), [TextDelta("inutile"), Done("stop")])
    result, _ = _respond(tmp_path, llm, "non, ouvre plutôt Spotify")
    assert len(llm.calls) == 1 and result.shortcut


PLAY_ON_YOUTUBE = (
    "lance la musique Jet Lag de Tiakola",
    [
        {"name": "browser_open", "args": {"url": "https://www.youtube.com/results?q=jet+lag"}},
        {"name": "browser_read"},
        {"name": "browser_click", "args": {"text": "Tiakola - Jet Lag"}},
    ],
)


def test_correcting_a_multi_step_task_does_not_end_on_its_first_step(tmp_path: Path) -> None:
    """La demande corrigée avait demandé deux actions (ouvrir, cliquer) : ouvrir Spotify
    n'en refait que le début, le modèle voit le résultat et continue."""
    llm = ScriptedLLM(
        _call("open_app", name="Spotify"),
        [TextDelta("Je lance Jet Lag sur Spotify."), Done("stop")],
    )
    result, _ = _respond(tmp_path, llm, "Non, sur Spotify.", PLAY_ON_YOUTUBE)
    assert len(llm.calls) == 2 and not result.shortcut


def test_correcting_a_one_step_task_still_ends_in_one_call(tmp_path: Path) -> None:
    llm = ScriptedLLM(_call("open_app", name="Spotify"), [TextDelta("inutile"), Done("stop")])
    before = ("ouvre Steam", [{"name": "open_app", "args": {"name": "Steam"}}])
    result, _ = _respond(tmp_path, llm, "non, Spotify", before)
    assert len(llm.calls) == 1 and result.shortcut


def test_reads_do_not_count_as_actions_of_the_corrected_task(tmp_path: Path) -> None:
    llm = ScriptedLLM(_call("open_app", name="Spotify"), [TextDelta("inutile"), Done("stop")])
    before = (
        "ouvre Steam",
        [{"name": "uia_window", "args": {"window": "Steam"}}, {"name": "open_app"}],
    )
    result, _ = _respond(tmp_path, llm, "non, Spotify", before)
    assert len(llm.calls) == 1 and result.shortcut
