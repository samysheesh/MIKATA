"""Phase 16c : tâches en cours (« tu en es où ? ») et accusé selon la durée prévue."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel
from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import make_ctx

from mikata.action.sdk import SdkCatalog
from mikata.agent.loop import Agent, TurnResult
from mikata.agent.narration import Narrator
from mikata.agent.progress import describe, expected_duration, is_status_question
from mikata.llm.base import Done, TextDelta, ToolCallReq
from mikata.tools import registry as registry_module
from mikata.tools.base import BaseTool
from mikata.tools.registry import ToolRegistry, load_builtin_tools, register_tool, registered_tools
from mikata.tts.cache import PHRASES


@pytest.mark.parametrize(
    "text",
    [
        "tu en es où ?",
        "T'en es où ?",
        "Alors, ça avance ?",
        "c'est bientôt fini ?",
        "où est-ce que tu en es",
        "tu fais quoi ?",
        "are you done?",
    ],
)
def test_status_questions_are_recognised(text: str) -> None:
    assert is_status_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "où en est la livraison de mon colis ?",
        "tu en es où dans la série que je t'ai demandé de regarder et mets la suite",
        "ça avance la mise à jour de Windows ?",
        "lance le dernier opening de One Piece",
    ],
)
def test_other_requests_are_not_status_questions(text: str) -> None:
    assert not is_status_question(text)


def test_describe_running_tasks() -> None:
    running = [("lance le dernier opening de Bleach", "browser", 100.0)]
    assert describe(running, now=108.4) == (
        "Pour « lance le dernier opening de Bleach », je suis sur la page, depuis 8 secondes."
    )
    assert describe([]) == "Je n'ai rien en cours."


def _catalog() -> SdkCatalog:
    load_builtin_tools()
    return SdkCatalog(registered_tools(), "win32")


def test_expected_duration_of_actions() -> None:
    catalog = _catalog()
    open_page = catalog.by_tool("browser_open")
    assert open_page is not None and open_page.expected_s >= 2.5
    direct = [ToolCallReq("a", "browser_open", {"url": "x"}), ToolCallReq("b", "set_volume", {})]
    assert expected_duration(direct, catalog) == open_page.expected_s  # en parallèle : le plus long
    code = (
        "browser_open('x')\npage = browser_read()\n"
        "choix = pick(page.links, 'q')\nbrowser_click(choix)"
    )
    program = expected_duration([ToolCallReq("c", "script", {"code": code})], catalog)
    assert program > open_page.expected_s  # en programme : les appels s'additionnent
    assert expected_duration([ToolCallReq("d", "shutdown", {})], catalog) == 0  # s'annonce seul
    assert expected_duration([ToolCallReq("e", "set_volume", {"level": 3})], catalog) < 1


def test_long_actions_are_announced_when_they_start_never_twice() -> None:
    said: list[str] = []
    narrator = Narrator(Sink(), lambda sink, key: said.append(key), after_s=5)
    narrator.starting("search", 0.5, threshold_s=2.5)  # courte : rien
    assert said == []
    narrator.starting("browser", 4.0, threshold_s=2.5)
    assert said == ["wait_browser"] and narrator.lines == ["wait_browser"]
    narrator.starting("files", 4.0, threshold_s=2.5)  # déjà annoncé dans ce tour
    assert said == ["wait_browser"]
    spoken = Narrator(Sink(), lambda sink, key: said.append(key), after_s=5)
    spoken.say("Je cherche sur YouTube.")  # le modèle a déjà parlé
    spoken.starting("browser", 4.0, threshold_s=2.5)
    assert said == ["wait_browser"]


class _SlowParams(BaseModel):
    query: str


class _SlowSearch(BaseTool[_SlowParams]):
    name = "slow_search_test"
    description = "Recherche lente (test)."
    params = _SlowParams
    expected_s = 4.0

    async def run(self, args: _SlowParams) -> str:
        await asyncio.sleep(0.6)
        return "Trouvé."


def test_a_status_question_answers_from_the_running_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_module.load_builtin_tools()
    monkeypatch.setattr(registry_module, "_REGISTRY", dict(registry_module._REGISTRY))
    register_tool(_SlowSearch)
    llm = ScriptedLLM(
        [ToolCallReq("c1", "slow_search_test", {"query": "x"}), Done("tool_calls")],
        [TextDelta("C'est trouvé."), Done("stop")],
    )
    ctx = make_ctx(tmp_path)
    agent = Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
    )
    first, second = Sink(), Sink()

    async def scenario() -> tuple[TurnResult, TurnResult, TurnResult]:
        try:
            long_task = asyncio.create_task(agent.respond("cherche les horaires du musée", first))
            await asyncio.sleep(0.3)  # la recherche lente tourne
            status = await agent.respond("tu en es où ?", second)
            done = await long_task
            idle = await agent.respond("tu en es où ?", Sink())
            return status, done, idle
        finally:
            await agent.aclose()

    status, done, idle = asyncio.run(scenario())
    assert status.outcome == "status" and status.llm_calls == 0  # aucun appel au modèle
    assert status.answer.startswith("Pour « cherche les horaires du musée », je m'en occupe")
    assert second.said == [status.answer]
    assert done.answer == "C'est trouvé." and first.audio  # accusé de l'action longue
    assert first.audio[0][0][0] == len("wait_work")
    assert idle.answer == "Je n'ai rien en cours."
    assert PHRASES["wait_work"]
