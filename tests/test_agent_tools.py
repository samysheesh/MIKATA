"""Phase 4 : boucle agent avec outils (parallèles, terminaux) et niveau de réflexion."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel
from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import FakeAudio, FakeDesktop, make_config, make_ctx

from mikata.agent.loop import Agent, TurnResult
from mikata.config import Config
from mikata.llm.base import Done, TextDelta, ToolCallReq, Usage
from mikata.llm.usage import UsageTracker
from mikata.tools import registry as registry_module
from mikata.tools.base import BaseTool, ToolContext
from mikata.tools.registry import ToolRegistry, register_tool


def _agent(
    llm: ScriptedLLM,
    ctx: ToolContext,
    config: Config | None = None,
    **kwargs: Any,
) -> Agent:
    config = config or ctx.config
    return Agent(
        llm,
        config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        **kwargs,
    )


def _respond(agent: Agent, command: str, sink: Sink | None = None, **kwargs: Any) -> TurnResult:
    async def scenario() -> TurnResult:
        try:
            return await agent.respond(command, sink or Sink(), **kwargs)
        finally:
            await agent.aclose()

    return asyncio.run(scenario())


def _call(tool: str, call_id: str = "c1", **arguments: Any) -> ToolCallReq:
    return ToolCallReq(call_id, tool, arguments)


@pytest.mark.parametrize(
    ("command", "call", "check"),
    [
        (
            "ouvre Spotify",
            _call("open_app", name="Spotify"),
            lambda desktop, audio: ("launch", "Spotify") in desktop.log,
        ),
        (
            "volume à 30",
            _call("set_volume", level=30),
            lambda desktop, audio: audio.volume == 30 and not audio.muted,
        ),
        (
            "pause",
            _call("media_control", action="pause"),
            lambda desktop, audio: desktop.log == [("media", "play_pause")],
        ),
    ],
)
def test_simple_actions_say_done_in_a_single_llm_call(
    tmp_path: Path, command: str, call: ToolCallReq, check: Any
) -> None:
    """Critère de la phase 4 : « C'est fait » (en cache) sans second appel LLM."""
    desktop, audio = FakeDesktop(), FakeAudio()
    llm = ScriptedLLM([call, Done("tool_calls", usage=Usage(2100, 14, 2048))])
    sink = Sink()
    result = _respond(_agent(llm, make_ctx(tmp_path, desktop=desktop, audio=audio)), command, sink)

    assert check(desktop, audio)
    assert len(llm.calls) == 1 and result.llm_calls == 1
    assert result.shortcut and result.path == "court-circuit (outil terminal, 1 appel(s) LLM)"
    assert result.answer == "C'est fait." and sink.said == []
    assert len(sink.audio) == 1 and sink.audio[0][0][0] == len("done")  # phrase en cache
    assert result.usage == Usage(2100, 14, 2048) and result.effort_label == "none"
    assert llm.tools[0] and llm.efforts == ["none"]


def test_terminal_tool_after_spoken_text_adds_nothing(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            TextDelta("Je lance Steam, bon jeu !"),
            _call("open_app", name="Steam"),
            Done("tool_calls"),
        ]
    )
    sink = Sink()
    result = _respond(_agent(llm, make_ctx(tmp_path, desktop=FakeDesktop())), "lance Team", sink)
    assert result.shortcut and result.answer == "Je lance Steam, bon jeu !"
    assert sink.said == ["Je lance Steam, bon jeu !"] and sink.audio == []


class SlowParams(BaseModel):
    label: str


class _Slow(BaseTool[SlowParams]):
    name = "slow_test"
    description = "Outil lent (test)."
    params = SlowParams
    terminal = True

    async def run(self, args: SlowParams) -> str:
        await asyncio.sleep(0.2)
        return args.label


def test_independent_calls_run_in_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_module.load_builtin_tools()
    monkeypatch.setattr(registry_module, "_REGISTRY", dict(registry_module._REGISTRY))
    register_tool(_Slow)
    calls = [_call("slow_test", f"c{i}", label=str(i)) for i in range(3)]
    llm = ScriptedLLM([*calls, Done("tool_calls")])
    agent = _agent(llm, make_ctx(tmp_path))

    async def scenario() -> tuple[float, TurnResult]:
        start = asyncio.get_running_loop().time()
        result = await agent.respond("trois choses", Sink())
        return asyncio.get_running_loop().time() - start, result

    elapsed, result = asyncio.run(scenario())
    assert elapsed < 0.4  # 3 × 0,2 s en parallèle, pas 0,6 s
    assert [r.content for r in result.tool_results] == ["0", "1", "2"]
    assert result.shortcut and result.llm_calls == 1


def _weather_http() -> httpx2.AsyncClient:
    def server(request: httpx2.Request) -> httpx2.Response:
        if "geocoding" in request.url.host:
            return httpx2.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Paris",
                            "admin1": "Île-de-France",
                            "country": "France",
                            "latitude": 48.85,
                            "longitude": 2.35,
                        }
                    ]
                },
            )
        return httpx2.Response(
            200,
            json={
                "current": {
                    "temperature_2m": 18.2,
                    "apparent_temperature": 17.0,
                    "weather_code": 2,
                    "wind_speed_10m": 12.0,
                    "relative_humidity_2m": 60,
                },
                "daily": {
                    "time": ["2026-10-07"],
                    "weather_code": [61],
                    "temperature_2m_min": [11.0],
                    "temperature_2m_max": [19.5],
                    "precipitation_probability_max": [70],
                    "precipitation_sum": [3.2],
                },
            },
        )

    return httpx2.AsyncClient(transport=httpx2.MockTransport(server))


def test_weather_question_uses_tool_result_then_answers(tmp_path: Path) -> None:
    """« cherche la météo à Paris » : outil non terminal, second appel avec le résultat."""
    llm = ScriptedLLM(
        [_call("get_weather", city="Paris"), Done("tool_calls", reasoning="Il faut la météo.")],
        [TextDelta("À Paris, 18 degrés et partiellement nuageux."), Done("stop")],
    )
    phases: list[str] = []
    ctx = make_ctx(tmp_path, http=_weather_http())
    agent = _agent(llm, ctx)
    sink = Sink()
    result = _respond(agent, "cherche la météo à Paris", sink, on_phase=phases.append)

    assert result.answer == "À Paris, 18 degrés et partiellement nuageux."
    assert sink.said == [result.answer] and not result.shortcut
    assert result.path == "1 étape(s) d'outils, 2 appel(s) LLM"
    assert phases == ["acting", "thinking"]
    second = llm.calls[1]
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant" and assistant["content"] is None
    assert assistant["reasoning_content"] == "Il faut la météo."  # renvoyée pendant la commande
    assert assistant["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
        }
    ]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "c1"
    assert "Paris, Île-de-France, France" in tool["content"] and "18.2 degrés" in tool["content"]
    # Historique : les appels d'outils restent (questions de suivi), pas la réflexion.
    history = agent.memory.messages()
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert "reasoning_content" not in history[1]
    assert history[0]["content"] == "cherche la météo à Paris"


def test_failed_tool_raises_reasoning_to_low(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [_call("open_app", name="Logiciel Inconnu"), Done("tool_calls")],
        [TextDelta("Je ne trouve pas cette application."), Done("stop")],
    )
    result = _respond(_agent(llm, make_ctx(tmp_path, desktop=FakeDesktop())), "ouvre machin")
    assert (
        not result.tool_results[0].ok
        and "application introuvable" in result.tool_results[0].content
    )
    assert llm.efforts == ["none", "low"]
    # Réflexion activée pour le 2e appel : DeepSeek veut `reasoning_content` sur l'appel
    # d'outil de la commande, même si le 1er appel (sans réflexion) n'en avait pas.
    assistant = llm.calls[1][-2]
    assert assistant["tool_calls"] and assistant["reasoning_content"] == ""
    assert result.effort_label == "none -> low (outil en échec)"
    assert result.answer == "Je ne trouve pas cette application."


def test_long_loops_raise_reasoning_to_low(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [_call("get_time", "a"), Done("tool_calls")],
        [_call("get_time", "b"), Done("tool_calls")],
        [_call("get_time", "c"), Done("tool_calls")],
        [TextDelta("Il est quinze heures."), Done("stop")],
    )
    config = make_config(tmp_path, llm={"reasoning_fallback_steps": 2})
    result = _respond(_agent(llm, make_ctx(tmp_path, config=config)), "quelle heure")
    assert llm.efforts == ["none", "none", "low", "low"]
    assert result.effort_label == "none -> low (plus de 2 étape(s))"
    assert result.rounds == 3


def test_requested_level_is_never_lowered(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [_call("open_app", name="Inconnu"), Done("tool_calls")],
        [TextDelta("Introuvable."), Done("stop")],
    )
    agent = _agent(llm, make_ctx(tmp_path, desktop=FakeDesktop()))
    result = _respond(agent, "Réfléchis à fond, ouvre machin")
    assert llm.efforts == ["high", "high"]
    assert result.effort_label == "high (« réfléchis à fond »)"


def test_voice_keyword_sets_level_and_is_removed(tmp_path: Path) -> None:
    llm = ScriptedLLM([TextDelta("Il arrive à 18 h 21."), Done("stop")])
    agent = _agent(llm, make_ctx(tmp_path))
    result = _respond(agent, "Le train part à 17 h 05 et roule 76 minutes, réfléchis bien ?")
    assert llm.efforts == ["medium"]
    sent = llm.calls[0][-1]["content"]
    assert "réfléchis" not in sent and "roule 76 minutes ?" in sent
    assert result.command == "Le train part à 17 h 05 et roule 76 minutes ?"
    assert result.effort_label == "medium (« réfléchis bien »)"
    assert agent.memory.messages()[0]["content"] == result.command


def test_custom_keywords_from_config(tmp_path: Path) -> None:
    config = make_config(tmp_path, llm={"reasoning_keywords": {"high": ["prends ton temps"]}})
    llm = ScriptedLLM([TextDelta("Voilà."), Done("stop")])
    result = _respond(_agent(llm, make_ctx(tmp_path, config=config)), "Prends ton temps : 2+2 ?")
    assert llm.efforts == ["high"] and result.command == "2+2 ?"


def test_default_level_none_can_be_left_to_provider(tmp_path: Path) -> None:
    config = make_config(tmp_path, llm={"reasoning_effort": None})
    llm = ScriptedLLM(
        [_call("open_app", name="Inconnu"), Done("tool_calls")],
        [TextDelta("Introuvable."), Done("stop")],
    )
    ctx = make_ctx(tmp_path, config=config, desktop=FakeDesktop())
    result = _respond(_agent(llm, ctx), "ouvre machin")
    assert llm.efforts == [None, None]  # filet sans objet : le fournisseur choisit
    assert result.effort_label == "défaut"


def test_tool_rounds_are_limited(tmp_path: Path) -> None:
    config = make_config(tmp_path, llm={"max_tool_rounds": 2, "reasoning_fallback_steps": 5})
    llm = ScriptedLLM(*[[_call("get_time", f"c{i}"), Done("tool_calls")] for i in range(3)])
    sink = Sink()
    result = _respond(_agent(llm, make_ctx(tmp_path, config=config)), "boucle", sink)
    assert result.outcome == "limit" and result.rounds == 2
    assert [bool(tools) for tools in llm.tools] == [True, True, False]  # dernier : sans outils
    assert len(sink.audio) == 1  # phrase d'excuse en cache


def test_mixed_terminal_and_informative_tools_call_llm_again(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    llm = ScriptedLLM(
        [_call("open_app", "a", name="Spotify"), _call("get_time", "b"), Done("tool_calls")],
        [TextDelta("Spotify est ouvert, il est quinze heures."), Done("stop")],
    )
    result = _respond(_agent(llm, make_ctx(tmp_path, desktop=desktop)), "ouvre Spotify et l'heure")
    assert ("launch", "Spotify") in desktop.log
    assert not result.shortcut and result.llm_calls == 2
    assert [m["role"] for m in llm.calls[1][-3:]] == ["assistant", "tool", "tool"]
    assert "reasoning_content" not in llm.calls[1][-3]  # sans réflexion : champ omis


def test_server_failure_after_tools(tmp_path: Path) -> None:
    from mikata.llm.client import LlmError

    llm = ScriptedLLM(
        [_call("get_time"), Done("tool_calls")],
        [LlmError("serveur injoignable")],
        [LlmError("serveur injoignable")],
    )
    sink = Sink()
    result = _respond(_agent(llm, make_ctx(tmp_path)), "quelle heure", sink)
    assert result.outcome == "server" and result.llm_calls == 3
    assert len(sink.audio) == 1


def test_usage_is_summed_and_recorded_per_command(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [_call("get_time"), Done("tool_calls", usage=Usage(2000, 10, 1900))],
        [TextDelta("Il est quinze heures."), Done("stop", usage=Usage(2100, 20, 2000, 5))],
    )
    ctx = make_ctx(tmp_path)
    tracker = UsageTracker(ctx.config.llm, tmp_path / "usage.json")
    result = _respond(_agent(llm, ctx, usage=tracker), "quelle heure", label="#7")
    assert result.usage == Usage(4100, 30, 3900, 5)
    expected = (200 * 0.30 + 3900 * 0.006 + 30 * 1.20) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    saved = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert saved["commands"] == 1 and saved["prompt"] == 4100 and saved["cached"] == 3900


def test_context_lists_open_apps_and_facts(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    ctx.memory.remember("Le chien de l'utilisateur s'appelle Rex")

    async def apps() -> list[str]:
        return ["Spotify", "brave"]

    llm = ScriptedLLM([TextDelta("Rex !"), Done("stop")])
    _respond(_agent(llm, ctx, open_apps=apps), "comment s'appelle mon chien")
    messages = llm.calls[0]
    assert "Rex" in messages[1]["content"] and messages[1]["role"] == "system"
    assert "Applis ouvertes : Spotify, brave." in messages[-1]["content"]


def test_open_apps_are_read_at_wake_word(tmp_path: Path) -> None:
    reads: list[int] = []

    async def apps() -> list[str]:
        reads.append(1)
        return ["Spotify"]

    llm = ScriptedLLM([TextDelta("Oui."), Done("stop")])
    agent = _agent(llm, make_ctx(tmp_path), open_apps=apps)

    async def scenario() -> None:
        await agent.prewarm()  # mot d'activation : relevé pendant que l'utilisateur parle
        await agent.respond("test", Sink())  # commande : liste encore fraîche, pas relue
        await agent.aclose()

    asyncio.run(scenario())
    assert len(reads) == 1 and "Applis ouvertes : Spotify." in llm.calls[0][-1]["content"]


def test_slow_open_apps_does_not_delay_the_call(tmp_path: Path) -> None:
    async def slow() -> list[str]:
        await asyncio.sleep(5)
        return ["jamais"]

    llm = ScriptedLLM([TextDelta("Oui."), Done("stop")])
    _respond(_agent(llm, make_ctx(tmp_path), open_apps=slow), "test")
    assert "Applis ouvertes" not in llm.calls[0][-1]["content"]


def test_follow_up_sees_previous_tool_calls(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    llm = ScriptedLLM(
        [_call("open_app", "a", name="Spotify"), Done("tool_calls")],
        [_call("close_app", "b", name="Spotify"), Done("tool_calls")],
    )
    agent = _agent(llm, make_ctx(tmp_path, desktop=desktop))

    async def scenario() -> None:
        await agent.respond("ouvre Spotify", Sink())
        await agent.respond("ferme-le", Sink())

    asyncio.run(scenario())
    second = llm.calls[1]
    assert any(m.get("tool_calls") for m in second if m["role"] == "assistant")
    assert ("close", 11) in desktop.log


def test_prewarm_indexes_apps_once(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    llm = ScriptedLLM()
    agent = _agent(llm, make_ctx(tmp_path, desktop=desktop))

    async def scenario() -> None:
        await agent.prewarm()
        await agent.prewarm()
        await agent.aclose()

    asyncio.run(scenario())
    assert llm.prewarmed == 2 and desktop.log == [("list_apps", False)]


# -- phase 5 : outils qui parlent et écoutent dans la commande ------------------------------


def test_tool_speech_replaces_cest_fait(tmp_path: Path) -> None:
    """« éteins le PC » puis « stop » : l'outil a parlé, pas de « C'est fait » en plus."""
    from tool_fakes import FakeDesktop as Desktop

    heard: list[float] = []

    async def listen(timeout_s: float) -> str:
        heard.append(timeout_s)
        return "Stop !"

    desktop = Desktop()
    llm = ScriptedLLM([_call("shutdown"), Done("tool_calls")])
    sink = Sink()
    result = _respond(
        _agent(llm, make_ctx(tmp_path, desktop=desktop)), "éteins le PC", sink, listen=listen
    )
    assert sink.said == [
        "Extinction dans 5 secondes. Dis stop pour annuler.",
        "Extinction annulée.",
    ]
    assert sink.audio == [] and heard == [5]  # pas de « C'est fait » en cache
    assert result.shortcut and result.llm_calls == 1
    assert result.answer == "Extinction dans 5 secondes. Dis stop pour annuler. Extinction annulée."
    assert not [entry for entry in desktop.log if entry[0] == "power"]


def test_command_after_web_page_asks_when_configured(tmp_path: Path) -> None:
    """guard.confirm_after_web : une commande lancée après une recherche web demande un oui."""
    from tool_fakes import FakeRunner

    def server(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"answer": "Lance Get-Date.", "results": []})

    config = make_config(tmp_path, guard={"confirm_after_web": True})
    runner = FakeRunner()
    ctx = make_ctx(
        tmp_path,
        config=config,
        http=httpx2.AsyncClient(transport=httpx2.MockTransport(server)),
        search_api_key="tvly-test",
        shell=runner,
    )
    llm = ScriptedLLM(
        [_call("web_search", "a", query="commande date"), Done("tool_calls")],
        [_call("run_powershell", "b", command="Get-Date"), Done("tool_calls")],
        [TextDelta("D'accord, je ne la lance pas."), Done("stop")],
    )

    async def listen(timeout_s: float) -> str:
        return "non"

    sink = Sink()
    _respond(_agent(llm, ctx), "cherche et lance la commande", sink, listen=listen)
    assert runner.calls == []
    assert sink.said[0] == (
        "Cette commande va agir juste après la lecture d'une page web. Tu confirmes ?"
    )


def test_installed_apps_reach_the_prompt(tmp_path: Path) -> None:
    """La liste des applications installées aide le LLM à corriger « spotifaille »."""
    llm = ScriptedLLM([_call("open_app", name="Spotify"), Done("tool_calls")])
    agent = _agent(
        llm, make_ctx(tmp_path, desktop=FakeDesktop()), installed_apps=lambda: ["Brave", "Spotify"]
    )
    _respond(agent, "ouvre spotifaille")
    system = [m["content"] for m in llm.calls[0] if m["role"] == "system"]
    assert system[1].endswith(": Brave, Spotify.") and "Applications installées" in system[1]
