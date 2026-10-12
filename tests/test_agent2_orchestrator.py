"""Agent v2 (phases 13c/13d) : l'orchestrateur mène une demande de bout en bout, délègue
le bavardage, rejoue les recettes et entend les corrections."""

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
from mikata.agent2.learn.store import SkillMeta
from mikata.agent2.orchestrator import Agent2
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.schema import Intent, PageShows, Unknown, Verb
from mikata.agent2.trace import TraceStore
from mikata.config import Agent2Config, load_config


def _intent_play() -> Intent:
    return Intent(
        verb=Verb.PLAY,
        object="x",
        object_kind="opening d'anime",
        unknowns=[Unknown(name="titre", question="titre ?", kind="web")],
    )


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


def _agent2_with_learning(
    tmp_path: Path, llm: ScriptedLLM, browser: Any = None
) -> tuple[Agent2, FakeSpeech]:
    """Agent2 branché sur une bibliothèque de savoir-faire et des recettes (learner
    neutralisé : les tests de la 13d appellent learn() directement)."""
    from mikata.agent2.learn.recipes import RecipeStore
    from mikata.agent2.learn.store import SkillStore

    cfg = Agent2Config()
    agent2, speech, _legacy = _agent2(
        tmp_path, llm, config=SimpleNamespace(agent2=cfg), browser=browser
    )
    skills = SkillStore(tmp_path / "data" / "agent2" / "skills", cfg)
    recipes = RecipeStore(tmp_path / "data" / "agent2" / "recipes.json", skills, "win32")
    agent2.skills = skills
    agent2.recipes = recipes
    agent2.learner = None  # le learner est testé dans test_agent2_compiler.py
    return agent2, speech


def _seed_recipe(tmp_path: Path) -> tuple[Any, Any]:
    from mikata.agent2.learn.recipes import Recipe, RecipeStore
    from mikata.agent2.learn.store import SkillMeta, SkillStore

    cfg = Agent2Config()
    skills = SkillStore(tmp_path / "data" / "agent2" / "skills", cfg)
    skills.add(
        "play_youtube_video",
        SkillMeta(
            description="Jouer une vidéo YouTube",
            capability="play.video",
            channel="youtube",
            params={"query": "titre"},
            done_when=PageShows(pattern="{query}"),
            status="trial",
            os=["win32"],
        ),
        "await sdk.browser_open('https://www.youtube.com')\nreturn 'ok'",
    )
    recipes = RecipeStore(tmp_path / "data" / "agent2" / "recipes.json", skills, "win32")
    recipes.recipes["play|opening d'anime|web"] = Recipe(
        verb="play",
        object_kind="opening d'anime",
        resolve=[{"name": "titre", "kind": "web"}],
        capability="play.video",
        skill="play_youtube_video",
        skill_args={"query": "{titre}"},
        successes=1,
    )
    recipes._save()
    return skills, recipes


def test_full_request_reaches_done_and_saves_a_verified_trace(tmp_path: Path) -> None:
    # Le sous-but 0 (page_shows) est déjà vrai à l'écran : le pré-contrôle le valide
    # sans appel LLM ni action ; seul le sous-but 1 (browser_click) demande du code.
    llm = ScriptedLLM([_UNDERSTAND, _RESOLVE, _PLAN, _STEP2])
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
    assert trace.steps[0].code == "" and trace.steps[0].goal_reached  # pré-vérifié, rien refait
    assert [action.tool for action in trace.steps[1].actions] == ["browser_click"]


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


def test_a_neighbor_request_uses_the_recipe_without_a_planner(tmp_path: Path) -> None:
    skills, recipes = _seed_recipe(tmp_path)
    llm = ScriptedLLM([_UNDERSTAND, _RESOLVE.replace("One Piece Opening 27", "Naruto Ending 15")])
    browser = FakeBrowser(page="Titre : YouTube\nTexte :\nNaruto Ending 15 en lecture")
    agent2, speech = _agent2_with_learning(tmp_path, llm, browser=browser)
    agent2.skills = skills
    agent2.recipes = recipes

    result = asyncio.run(agent2.respond("lance le dernier ending de Naruto", speech))
    assert result.outcome == "ok"
    # understand + résolution, et rien d'autre : pas de planificateur, pas d'étape LLM.
    assert len(llm.calls) == 2
    traces = list((tmp_path / "data" / "agent2" / "traces").rglob("*.json"))
    from mikata.agent2.schema import Trace

    trace = Trace.model_validate_json(traces[0].read_text(encoding="utf-8"))
    assert trace.recipe == "play|opening d'anime|web"
    assert trace.skills_used == ["play_youtube_video"]
    assert len(trace.plan) == 1 and trace.plan[0].skill == "play_youtube_video"


def test_a_failed_recipe_falls_back_to_the_normal_path(tmp_path: Path) -> None:
    skills, recipes = _seed_recipe(tmp_path)
    skills.add(
        "play_youtube_video",
        SkillMeta(
            description="Jouer une vidéo YouTube",
            capability="play.video",
            channel="youtube",
            params={"query": "titre"},
            done_when=PageShows(pattern="{query}"),
            status="trial",
            os=["win32"],
        ),
        "await sdk.open_app('Introuvable')\nreturn 'ok'",
    )
    llm = ScriptedLLM(
        [
            _UNDERSTAND,
            _RESOLVE.replace("One Piece Opening 27", "Naruto Ending 15"),
            '{"subgoals": [{"goal": "YouTube affiche la vidéo", "channel": "youtube",'
            ' "done_when": {"type": "tool_ok", "tool": "browser_open"},'
            ' "skill": null, "skill_args": {}}]}',
            _STEP1,
        ]
    )
    agent2, speech = _agent2_with_learning(tmp_path, llm)
    agent2.skills = skills
    agent2.recipes = recipes

    result = asyncio.run(agent2.respond("lance le dernier ending de Naruto", speech))
    assert result.outcome == "ok"  # le chemin normal a réussi
    assert recipes.match(_intent_play()) is None  # la recette est punie, plus proposée
    assert skills.skills["play_youtube_video"].consecutive_failures == 1


def test_a_channel_correction_updates_prefs_and_replays(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            _UNDERSTAND,
            _RESOLVE,
            '{"subgoals": [{"goal": "YouTube affiche la vidéo", "channel": "youtube",'
            ' "done_when": {"type": "tool_ok", "tool": "browser_open"},'
            ' "skill": null, "skill_args": {}}]}',
            _STEP1,
            # rejeu : résolution, plan, étape
            _RESOLVE,
            '{"subgoals": [{"goal": "Spotify affiche la musique", "channel": "spotify",'
            ' "done_when": {"type": "tool_ok", "tool": "browser_open"},'
            ' "skill": null, "skill_args": {}}]}',
            _STEP1,
        ]
    )
    agent2, speech, _legacy = _agent2(tmp_path, llm)
    asyncio.run(agent2.respond("lance le dernier opening de One Piece", speech))
    asyncio.run(agent2.respond("non, sur Spotify", speech))
    assert agent2.prefs.scores("play", "opening d'anime")["spotify"] == 0.95
    traces = sorted(
        (tmp_path / "data" / "agent2" / "traces").rglob("*.json"),
        key=lambda path: path.stat().st_mtime,  # noms UUID : l'ordre n'est pas chronologique
    )
    from mikata.agent2.schema import Trace

    last = Trace.model_validate_json(traces[-1].read_text(encoding="utf-8"))
    assert last.channel == "spotify"
    assert any("C'est fait" in phrase for phrase in speech.said)
