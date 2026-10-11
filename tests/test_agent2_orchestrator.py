"""Agent v2 (phase 13c) : l'orchestrateur mène une demande de bout en bout et délègue
le bavardage à l'ancien agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agent2_fakes import ScriptedLLM, make_registry
from conftest import write_test_config
from tool_fakes import FakeBrowser, FakeDesktop

from mikata.agent.loop import PHRASES, TurnResult
from mikata.agent2.orchestrator import Agent2
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.trace import TraceStore
from mikata.config import Agent2Config, load_config

_UNDERSTAND = (
    '{"verb": "play", "object": "dernier opening de One Piece",'
    ' "object_kind": "opening d\'anime", "entities": {}, "constraints": [],'
    ' "target": null, "channel_hint": null,'
    ' "unknowns": [{"name": "titre", "question": "titre exact ?", "kind": "web"}],'
    ' "sub_intents": []}'
)
_RESOLVE = '{"value": "One Piece Opening 27", "confident": true}'
_PLAN = (
    '{"subgoals": ['
    '{"goal": "YouTube affiche la vidéo", "channel": "youtube",'
    ' "done_when": {"type": "page_shows", "pattern": "Opening 27"},'
    ' "skill": null, "skill_args": {}},'
    '{"goal": "la lecture démarre", "channel": "youtube",'
    ' "done_when": {"type": "tool_ok", "tool": "browser_click"},'
    ' "skill": null, "skill_args": {}}]}'
)
_STEP1 = (
    '{"code": "await sdk.browser_open(\'https://www.youtube.com/results?search_query=test\')'
    "\\nreturn 'ok'\"}"
)
_STEP2 = "{\"code\": \"await sdk.browser_click('One Piece Opening 27')\\nreturn 'ok'\"}"


class FakeSearchTool:
    """web_search simulé : l'orchestrateur ne touche jamais le réseau."""

    name = "web_search"
    description = "test"
    terminal = False
    params: Any = Any

    def __init__(self) -> None:
        from pydantic import BaseModel

        class SearchParams(BaseModel):
            query: str = ""

        self.params = SearchParams

    async def run(self, args: Any) -> str:
        return f"Résultats pour {args.query} :\n1. One Piece Opening 27"


class FakeSpeech:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def say_audio(self, pcm: Any, rate: int) -> None:
        pass

    async def drain(self) -> None:
        pass


class FakeLegacy:
    """L'Agent actuel réduit à ce que l'Agent2 utilise ; respond enregistre ses appels."""

    def __init__(self, config: Any, tools: Any, llm: ScriptedLLM) -> None:
        self.config = config
        self.tools = tools
        self.llm = llm
        self.long_memory = SimpleNamespace(facts=("Paul = Paul Martin",))
        self.usage = None
        self.responded: list[str] = []
        self.phrase_keys: list[str] = []

    def _installed_apps(self) -> list[str]:
        return ["Brave", "Spotify"]

    async def context(self) -> str:
        return "fenêtre active : Brave"

    def say_phrase(self, speech: FakeSpeech, key: str) -> None:
        self.phrase_keys.append(key)
        speech.say(PHRASES[key])

    async def respond(
        self,
        command: str,
        speech: FakeSpeech,
        *,
        label: str = "commande",
        on_phase: Any = None,
        listen: Any = None,
        follow_up: bool = False,
    ) -> TurnResult:
        self.responded.append(command)
        speech.say("réponse de l'ancien agent")
        return TurnResult(command=command, answer="réponse de l'ancien agent", outcome="ok")


def _agent2(
    tmp_path: Path, llm: ScriptedLLM, config: Any = None, browser: Any = None, desktop: Any = None
) -> tuple[Agent2, FakeSpeech, Any]:
    registry = make_registry(
        tmp_path, browser=browser or FakeBrowser(), desktop=desktop or FakeDesktop()
    )
    registry.tools["web_search"] = FakeSearchTool()  # type: ignore[assignment]
    legacy = FakeLegacy(config or SimpleNamespace(agent2=Agent2Config()), registry, llm)
    speech = FakeSpeech()
    traces = TraceStore(tmp_path / "data" / "agent2" / "traces")
    prefs = PreferenceStore(tmp_path / "data" / "agent2" / "preferences.json")
    return Agent2(legacy, traces=traces, prefs=prefs), speech, legacy


def test_full_request_reaches_done_and_saves_a_verified_trace(tmp_path: Path) -> None:
    llm = ScriptedLLM([_UNDERSTAND, _RESOLVE, _PLAN, _STEP1, _STEP2])
    browser = FakeBrowser(page="Titre : YouTube\nTexte :\nOne Piece Opening 27 en lecture")
    agent2, speech, _legacy = _agent2(tmp_path, llm, browser=browser)

    result = asyncio.run(agent2.respond("lance le dernier opening de One Piece", speech))
    assert result.outcome == "ok"
    assert any("C'est fait" in phrase for phrase in speech.said)
    assert any("Je passe par YouTube" in phrase for phrase in speech.said)
    traces = list((tmp_path / "data" / "agent2" / "traces").rglob("*.json"))
    assert len(traces) == 1
    from mikata.agent2.schema import Trace

    trace = Trace.model_validate_json(traces[0].read_text(encoding="utf-8"))
    assert trace.verified and trace.outcome == "ok"
    assert trace.channel == "youtube" and trace.channel_announced
    assert len(trace.plan) == 2 and len(trace.steps) == 2


def test_chat_is_delegated_to_the_legacy_agent(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            '{"verb": "chat", "object": "une blague", "object_kind": "", "entities": {},'
            ' "constraints": [], "target": null, "channel_hint": null, "unknowns": [],'
            ' "sub_intents": []}'
        ]
    )
    agent2, speech, legacy = _agent2(tmp_path, llm)
    result = asyncio.run(agent2.respond("raconte-moi une blague", speech))
    assert legacy.responded == ["raconte-moi une blague"]
    assert result.outcome == "ok"
    assert any("ancien agent" in phrase for phrase in speech.said)


def test_follow_up_is_delegated_without_any_llm_call(tmp_path: Path) -> None:
    llm = ScriptedLLM([])  # rien ne doit être consommé
    agent2, speech, legacy = _agent2(tmp_path, llm)
    asyncio.run(agent2.respond("et la suite", speech, follow_up=True))
    assert legacy.responded == ["et la suite"]
    assert llm.calls == []


def test_failure_says_je_n_y_arrive_pas_and_traces_failed(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [_UNDERSTAND, _RESOLVE, _PLAN.replace("browser_click", "open_app", 1), _STEP1]
        + ["{\"code\": \"await sdk.open_app('Introuvable')\\nreturn 'x'\"}"] * 3
    )
    desktop = FakeDesktop()
    desktop.fail.add("launch")
    agent2, speech, _legacy = _agent2(tmp_path, llm, desktop=desktop)

    result = asyncio.run(agent2.respond("lance le dernier opening de One Piece", speech))
    assert result.outcome == "limit"
    assert any("Je n'y arrive pas" in phrase for phrase in speech.said)
    traces = list((tmp_path / "data" / "agent2" / "traces").rglob("*.json"))
    from mikata.agent2.schema import Trace

    trace = Trace.model_validate_json(traces[0].read_text(encoding="utf-8"))
    assert trace.outcome == "failed" and not trace.verified


def test_build_agent_wraps_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    from tool_fakes import make_ctx

    from mikata import __main__ as main
    from mikata.agent.loop import Agent

    monkeypatch.setattr(main, "build_tool_context", lambda config: make_ctx(tmp_path))
    llm_test = {"base_url": "https://example.com/v1", "model": "test"}

    config = load_config(write_test_config(tmp_path / "config.yaml", llm=llm_test))
    agent = main.build_agent(config)
    assert isinstance(agent, Agent) and not isinstance(agent, Agent2)

    config = load_config(
        write_test_config(tmp_path / "config.yaml", llm=llm_test, agent2={"enabled": True})
    )
    wrapped = main.build_agent(config)
    assert isinstance(wrapped, Agent2)
