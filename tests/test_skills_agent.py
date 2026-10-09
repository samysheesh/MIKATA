"""Savoir-faire dans la boucle agent (phase 12c) : rejeu sans LLM, repli sur échec,
plan pour les procédures paramétrées, apprentissage en arrière-plan et capture forcée."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import FakeAudio, FakeBrowser, FakeDesktop, FakeIndex, make_ctx

from mikata.agent.loop import Agent, TurnResult
from mikata.llm.base import Done, TextDelta, ToolCallReq, Usage
from mikata.skills.distill import DistillResult
from mikata.skills.library import SkillLibrary
from mikata.tools.registry import ToolRegistry, load_builtin_tools


def _skill_library(tmp_path: Path) -> SkillLibrary:
    load_builtin_tools()
    return SkillLibrary(tmp_path / "skills.json", max_skills=30, max_failures=3)


def _learn_silencieux(library: SkillLibrary) -> None:
    assert library.learn(
        "pc_silencieux",
        "mettre le PC en silencieux",
        "mets le pc en silencieux",
        [
            {"tool": "set_volume", "args": {"level": 20}},
            {"tool": "mute", "args": {"mute": True}},
        ],
    )


def _agent(
    tmp_path: Path,
    llm: ScriptedLLM,
    skills: SkillLibrary,
    audio: FakeAudio | None = None,
    browser: Any = None,
    index: Any = None,
    desktop: Any = None,
    **kwargs: Any,
) -> Agent:
    ctx = make_ctx(
        tmp_path,
        audio=audio or FakeAudio(),
        skills=skills,
        browser=browser,
        index=index,
        desktop=desktop,
    )
    return Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        skills=skills,
        **kwargs,
    )


def _respond(agent: Agent, command: str, sink: Sink | None = None) -> TurnResult:
    async def scenario() -> TurnResult:
        try:
            return await agent.respond(command, sink or Sink())
        finally:
            await agent.aclose()

    return asyncio.run(scenario())


def test_no_param_skill_replays_without_llm(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    _learn_silencieux(skills)
    audio = FakeAudio()
    llm = ScriptedLLM([])  # jamais appelé
    sink = Sink()
    result = _respond(_agent(tmp_path, llm, skills, audio=audio), "passe le pc en silencieux", sink)

    assert llm.calls == [] and result.llm_calls == 0  # zéro appel LLM
    assert audio.volume == 20 and audio.muted  # la chaîne a été rejouée dans l'ordre
    assert result.shortcut and result.answer == "C'est fait."
    assert len(sink.audio) == 1  # « C'est fait » en cache
    assert result.rounds == 2 and len(result.tool_calls) == 2
    assert skills.get("pc_silencieux").failures == 0  # type: ignore[union-attr]


def test_replay_failure_falls_back_to_the_llm(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    assert skills.learn(
        "ouvrir_introuvable",
        "ouvrir une application introuvable",
        "ouvre l'application introuvable",
        [
            {"tool": "set_volume", "args": {"level": 20}},
            {"tool": "open_app", "args": {"name": "Introuvable"}},
        ],
    )
    audio = FakeAudio()
    llm = ScriptedLLM(
        [TextDelta("Je n'arrive pas à l'ouvrir."), Done("stop", usage=Usage(10, 5, 0))]
    )
    result = _respond(_agent(tmp_path, llm, skills, audio=audio), "ouvre l'application introuvable")

    assert result.llm_calls == 1  # le LLM a repris la main
    assert "Je n'arrive pas" in result.answer and not result.shortcut
    assert audio.volume == 20  # la première étape a bien été exécutée
    assert not result.tool_results[1].ok  # la seconde a échoué
    assert skills.get("ouvrir_introuvable").failures == 1  # type: ignore[union-attr]
    tool_messages = [m for m in llm.calls[0] if m.get("role") == "tool"]
    assert len(tool_messages) == 2  # les deux étapes exécutées sont dans le contexte du LLM


def test_param_skill_injects_the_plan_and_lets_the_llm_fill_it(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    assert skills.learn(
        "youtube_recent_video",
        "chercher et lancer une vidéo récente sur YouTube",
        "lance le dernier opening de black clover",
        [
            {
                "tool": "browser_open",
                "args": {"url": "https://youtube.com/results?search_query={query}"},
            }
        ],
        params=["query"],
    )
    call = ToolCallReq(
        "c1", "browser_open", {"url": "https://youtube.com/results?search_query=one piece"}
    )
    llm = ScriptedLLM(
        [call, Done("tool_calls", usage=Usage(100, 10, 0))],
        [TextDelta("C'est fait."), Done("stop", usage=Usage(10, 5, 0))],
    )
    result = _respond(
        _agent(tmp_path, llm, skills, browser=FakeBrowser()),
        "lance le dernier opening de one piece",
    )

    plans = [
        message["content"]
        for message in llm.calls[0]
        if message.get("role") == "system" and "Procédure connue" in message.get("content", "")
    ]
    assert plans and "{query}" in plans[0]  # le plan avec le trou est donné au LLM
    assert result.tool_calls == [call] and result.llm_calls == 2


# -- phase 12e : rejeu local, paramètres remplis sans LLM -------------------------------


def _learn_ouvrir_dossier(skills: SkillLibrary) -> None:
    assert skills.learn(
        "ouvrir_dossier",
        "trouver un dossier et l'ouvrir dans l'explorateur",
        "où est cyberpunk 2077",
        [
            {
                "tool": "find_path",
                "args": {"name": "{query}"},
                "outputs": {"folder_path": "best"},
            },
            {"tool": "open_in_explorer", "args": {"path": "{folder_path}"}},
        ],
        params=["query", "folder_path"],
        slots="où est {query}",
    )


def test_local_replay_fills_slots_and_outputs_without_llm(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    _learn_ouvrir_dossier(skills)
    index = FakeIndex()  # trouve toujours D:\Steam\...\Cyberpunk 2077.exe
    desktop = FakeDesktop()
    llm = ScriptedLLM([])  # jamais appelé
    result = _respond(
        _agent(tmp_path, llm, skills, index=index, desktop=desktop),
        "où est league of legends",
        Sink(),
    )

    assert llm.calls == [] and result.llm_calls == 0  # zéro appel LLM
    assert result.shortcut and result.answer == "C'est fait."
    assert index.queries[0][0] == "league of legends"  # les mots prononcés remplissent {query}
    cyberpunk = "D:\\Steam\\steamapps\\common\\Cyberpunk 2077.exe"
    assert ("reveal", (cyberpunk, True)) in desktop.log  # {folder_path} lié depuis « best »
    assert skills.get("ouvrir_dossier").failures == 0  # type: ignore[union-attr]


def test_local_replay_unbound_step_falls_back_without_penalty(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    _learn_ouvrir_dossier(skills)
    index = FakeIndex(results=[])  # find_path ne trouve rien : « best » absent
    llm = ScriptedLLM([TextDelta("Je n'ai rien trouvé."), Done("stop", usage=Usage(10, 5, 0))])
    result = _respond(
        _agent(tmp_path, llm, skills, index=index, desktop=FakeDesktop()),
        "où est league of legends",
    )

    assert result.llm_calls == 1 and not result.shortcut  # le LLM a repris
    assert skills.get("ouvrir_dossier").failures == 0  # pas un échec de la procédure


def test_phrase_outside_the_pattern_gets_the_plan(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    _learn_ouvrir_dossier(skills)
    llm = ScriptedLLM(
        [
            ToolCallReq("c1", "find_path", {"name": "league"}),
            Done("tool_calls", usage=Usage(100, 10, 0)),
        ],
        [TextDelta("C'est fait."), Done("stop", usage=Usage(10, 5, 0))],
    )
    result = _respond(
        _agent(tmp_path, llm, skills, index=FakeIndex(), desktop=FakeDesktop()),
        "montre-moi le dossier league of legends",
    )

    plans = [
        message["content"]
        for message in llm.calls[0]
        if message.get("role") == "system" and "Procédure connue" in message.get("content", "")
    ]
    assert plans and result.llm_calls >= 1  # plan LLM, pas de rejeu local


def test_local_replay_off_goes_straight_to_the_plan(tmp_path: Path) -> None:
    skills = _skill_library(tmp_path)
    _learn_ouvrir_dossier(skills)
    llm = ScriptedLLM(
        [
            ToolCallReq("c1", "find_path", {"name": "league"}),
            Done("tool_calls", usage=Usage(100, 10, 0)),
        ],
        [TextDelta("C'est fait."), Done("stop", usage=Usage(10, 5, 0))],
    )
    config = make_ctx(tmp_path).config
    config = config.model_copy(
        update={"skills": config.skills.model_copy(update={"local_replay": False})}
    )
    ctx = make_ctx(tmp_path, skills=skills, index=FakeIndex(), desktop=FakeDesktop())
    ctx.config = config
    agent = Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        skills=skills,
    )
    result = _respond(agent, "où est league of legends")
    assert result.llm_calls >= 1 and not result.shortcut


def test_auto_learn_stores_a_distilled_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills = _skill_library(tmp_path)
    agent = _agent(tmp_path, ScriptedLLM([]), skills)

    async def fake_distill(
        llm: Any, config: Any, command: str, calls: list[Any], hint: str = ""
    ) -> DistillResult:
        return DistillResult(
            name="appris_ici",
            description="la procédure apprise",
            trigger=command,
            steps=[{"tool": "open_url", "args": {"url": "https://example.com"}}],
            params=[],
            verify=None,
        )

    monkeypatch.setattr("mikata.agent.loop.distill", fake_distill)
    # Deux tours d'outils non terminaux, tous réussis : ça déclenche l'apprentissage.
    agent.llm = ScriptedLLM(
        [ToolCallReq("c1", "get_time", {}), Done("tool_calls", usage=Usage(50, 5, 0))],
        [ToolCallReq("c2", "get_time", {}), Done("tool_calls", usage=Usage(50, 5, 0))],
        [TextDelta("Voilà."), Done("stop", usage=Usage(10, 5, 0))],
    )

    async def scenario() -> None:
        try:
            await agent.respond("quelle heure est-il, deux fois", Sink())
            task = agent._skill_task
            assert task is not None
            await asyncio.wait_for(task, timeout=2)  # la distillation tourne en fond
            assert [s.name for s in skills.all()] == ["appris_ici"]
        finally:
            await agent.aclose()

    asyncio.run(scenario())


def test_remember_skill_captures_the_last_successful_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills = _skill_library(tmp_path)
    agent = _agent(tmp_path, ScriptedLLM([]), skills)

    async def fake_distill(
        llm: Any, config: Any, command: str, calls: list[Any], hint: str = ""
    ) -> DistillResult:
        assert hint  # la description donnée à la voix arrive bien
        return DistillResult(
            name="capture_forcee",
            description="la procédure capturée",
            trigger=command,
            steps=[{"tool": "open_url", "args": {"url": "https://example.com"}}],
            params=[],
            verify=None,
        )

    monkeypatch.setattr("mikata.agent.loop.distill", fake_distill)
    agent.llm = ScriptedLLM(
        [ToolCallReq("c1", "get_time", {}), Done("tool_calls", usage=Usage(50, 5, 0))],
        [ToolCallReq("c2", "get_time", {}), Done("tool_calls", usage=Usage(50, 5, 0))],
        [TextDelta("Voilà."), Done("stop", usage=Usage(10, 5, 0))],
    )

    async def scenario() -> None:
        try:
            await agent.respond("quelle heure est-il, deux fois", Sink())
            text = await agent._capture_skill("régler le son du PC")
            assert text == "Mémorisé : la procédure capturée"
            assert [s.name for s in skills.all()] == ["capture_forcee"]
        finally:
            await agent.aclose()

    asyncio.run(scenario())
