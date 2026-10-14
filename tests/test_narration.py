"""Phase 15c : MIKATA dit ce qu'il fait quand une demande dure (réglage agent.narration)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import make_ctx

from mikata.agent.loop import Agent, Narrator, TurnResult, activity_of
from mikata.config import Config
from mikata.llm.base import Done, TextDelta, ToolCallReq
from mikata.llm.prompts import NARRATION_NOTE, build_messages
from mikata.tools import registry as registry_module
from mikata.tools.base import BaseTool
from mikata.tools.registry import ToolRegistry, register_tool
from mikata.tts.cache import PHRASES


def _call(tool: str, **arguments: Any) -> ToolCallReq:
    return ToolCallReq("c1", tool, arguments)


@pytest.mark.parametrize(
    ("calls", "kind"),
    [
        ([_call("browser_open", url="https://youtube.com")], "browser"),
        ([_call("web_search", query="météo")], "search"),
        ([_call("find_path", name="cyberpunk")], "files"),
        ([_call("uia_window", title="Discord")], "app"),
        ([_call("open_app", name="Steam")], "work"),
        ([_call("script", code="r = browser_read()\nreturn r")], "browser"),
        ([_call("script", code="for f in list_folder('Bureau'): pass")], "files"),
        ([_call("script", code="return 1 + 1")], "work"),
    ],
)
def test_activity_of_names_the_step(calls: list[ToolCallReq], kind: str) -> None:
    assert activity_of(calls) == kind
    assert f"wait_{kind}" in PHRASES  # chaque étape a sa phrase en cache


def test_narrator_speaks_only_after_a_silence() -> None:
    said: list[str] = []
    narrator = Narrator(Sink(), lambda sink, key: said.append(key), after_s=5)
    start = narrator._started
    assert narrator.due(start + 4.9) is None
    narrator.activity = "browser"
    assert narrator.due(start + 5.0) == "wait_browser"
    narrator.say("J'ai trouvé la vidéo, je la lance.")  # le modèle a parlé : silence remis à 0
    assert narrator.due(narrator._last_said + 9.9) is None  # type: ignore[operator]
    assert narrator.due(narrator._last_said + 10.0) == "wait_long"  # type: ignore[operator]
    narrator.listening = True  # confirmation en cours : jamais de phrase d'attente
    assert narrator.due(narrator._last_said + 60) is None  # type: ignore[operator]


def test_narrator_says_at_most_two_lines_and_forwards_speech() -> None:
    sink = Sink()
    sink.drain = lambda: "vidée"  # type: ignore[attr-defined]

    async def scenario() -> Narrator:
        narrator = Narrator(sink, lambda s, key: s.say(PHRASES[key]), after_s=0.1)
        narrator.activity = "search"
        await asyncio.wait_for(narrator.run(), 2)
        return narrator

    narrator = asyncio.run(scenario())
    assert narrator.lines == ["wait_search", "wait_long"]
    assert sink.said == [PHRASES["wait_search"], PHRASES["wait_long"]]
    assert narrator.drain() == "vidée"  # le reste de la destination est transmis


class _LongParams(BaseModel):
    query: str


class _LongSearch(BaseTool[_LongParams]):
    name = "web_search_long_test"
    description = "Recherche lente (test)."
    params = _LongParams

    async def run(self, args: _LongParams) -> str:
        await asyncio.sleep(1.4)
        return "Résultat : il fera beau."


def _agent(tmp_path: Path, llm: ScriptedLLM, narration: str) -> Agent:
    ctx = make_ctx(tmp_path)
    config = ctx.config.model_copy(
        update={
            "agent": ctx.config.agent.model_copy(
                update={"narration": narration, "narration_after_s": 1}
            )
        }
    )
    return Agent(
        llm,
        config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
    )


def _long_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, narration: str
) -> tuple[TurnResult, Sink, ScriptedLLM]:
    registry_module.load_builtin_tools()
    monkeypatch.setattr(registry_module, "_REGISTRY", dict(registry_module._REGISTRY))
    register_tool(_LongSearch)
    llm = ScriptedLLM(
        [_call("web_search_long_test", query="météo demain"), Done("tool_calls")],
        [TextDelta("Demain, il fera beau."), Done("stop")],
    )
    agent = _agent(tmp_path, llm, narration)
    sink = Sink()

    async def scenario() -> TurnResult:
        try:
            return await agent.respond("quel temps demain", sink)
        finally:
            await agent.aclose()

    return asyncio.run(scenario()), sink, llm


def test_a_long_step_is_narrated_without_changing_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, sink, llm = _long_turn(tmp_path, monkeypatch, "bavard")
    assert result.narration == [PHRASES["wait_search"]]  # en cache : aucun appel LLM de plus
    assert len(sink.audio) == 1 and sink.audio[0][0][0] == len("wait_search")
    assert result.answer == "Demain, il fera beau." and sink.said == ["Demain, il fera beau."]
    assert result.llm_calls == 2
    assert llm.calls[0][0]["content"].endswith(NARRATION_NOTE)


def test_discreet_mode_says_only_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, sink, llm = _long_turn(tmp_path, monkeypatch, "discret")
    assert result.narration == [] and sink.audio == []
    assert sink.said == ["Demain, il fera beau."]
    assert NARRATION_NOTE not in llm.calls[0][0]["content"]


def test_narration_note_is_part_of_the_fixed_prefix() -> None:
    """Le prompt reste identique d'une commande à l'autre (cache de prompt)."""
    first = build_messages([], "a", "ctx 1", narration=True)
    second = build_messages([], "b", "ctx 2", narration=True)
    assert first[:-1] == second[:-1] and first[0]["content"].endswith(NARRATION_NOTE)
    assert NARRATION_NOTE not in str(build_messages([], "a", "ctx"))


def test_narration_is_a_live_setting() -> None:
    from mikata.app import is_live_setting
    from mikata.ui.settings_store import SPECS

    assert is_live_setting("agent.narration") and SPECS["agent.narration"].live
    assert Config().agent.narration == "bavard"
