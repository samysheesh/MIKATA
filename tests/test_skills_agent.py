"""Phase 15e : les compétences dans la boucle agent (rejeu local, essai, apprentissage)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import FakeAudio, FakeDesktop, make_ctx

from mikata.action.runtime import ScriptOutcome
from mikata.agent.loop import Agent, TurnResult, _inducible
from mikata.learn.skills import Skill, SkillStore
from mikata.llm.base import Done, TextDelta, ToolCallReq
from mikata.tools.registry import ToolRegistry

COMMAND = "mets le son à 35 et pause"
CODE = "set_volume(level=int(niveau))\nmedia_control(action='pause')\nreturn None"


def _skill(status: str = "trial", code: str = CODE) -> Skill:
    return Skill(
        name="son-et-pause",
        description="Règle le son puis met la lecture en pause.",
        patterns=["mets le son à {niveau} et pause"],
        params=["niveau"],
        code=code,
        status=status,
        source=COMMAND,
    )


def _setup(
    tmp_path: Path, llm: ScriptedLLM, skill: Skill | None = None
) -> tuple[Agent, SkillStore, FakeDesktop, FakeAudio]:
    desktop, audio = FakeDesktop(), FakeAudio()
    store = SkillStore(tmp_path / "skills")
    if skill is not None:
        status = skill.status
        store.adopt(skill)
        skill.status = status
        store.save(skill)
    ctx = make_ctx(tmp_path, desktop=desktop, audio=audio, skills=store)
    agent = Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        skills=store,
    )
    return agent, store, desktop, audio


def _turns(agent: Agent, *commands: str) -> list[TurnResult]:
    async def scenario() -> list[TurnResult]:
        try:
            results = []
            for command in commands:
                results.append(await agent.respond(command, Sink()))
                await agent.learning_idle()
            return results
        finally:
            await agent.aclose()

    return asyncio.run(scenario())


def _script(code: str) -> list[Any]:
    return [ToolCallReq("c1", "script", {"code": code}), Done("tool_calls")]


def test_an_active_skill_replays_locally_without_the_model(tmp_path: Path) -> None:
    llm = ScriptedLLM()  # aucun appel attendu : il lèverait
    agent, store, desktop, audio = _setup(tmp_path, llm, _skill("active"))
    (result,) = _turns(agent, "Mets le son à 35 et pause.")
    assert llm.calls == [] and result.llm_calls == 0
    assert audio.volume == 35 and ("media", "play_pause") in desktop.log
    assert result.answer == "C'est fait." and result.local_skill == "son-et-pause"
    assert result.path.startswith("compétence « son-et-pause » en local")
    skill = store.get("son_et_pause")
    assert skill is not None and skill.uses == 1 and skill.status == "active"


def test_a_failing_active_skill_goes_back_to_trial_and_the_model_takes_over(
    tmp_path: Path,
) -> None:
    broken = "set_volume(level=int(niveau))\nexpect(False, 'bouton disparu')\nreturn None"
    llm = ScriptedLLM(
        [TextDelta("Le son est à 35, mais je n'ai pas pu mettre pause."), Done("stop")]
    )
    agent, store, _, audio = _setup(tmp_path, llm, _skill("active", broken))
    (result,) = _turns(agent, COMMAND)
    assert audio.volume == 35  # la première étape a eu lieu avant l'échec
    assert result.llm_calls == 1 and result.answer.startswith("Le son est à 35")
    notes = [m["content"] for m in llm.calls[0] if m["role"] == "system"]
    assert any("a échoué" in note for note in notes)  # le modèle sait que le rejeu a raté
    skill = store.get("son_et_pause")
    assert skill is not None and skill.status == "trial" and skill.failures == 1
    assert "bouton disparu" in skill.last_error


def test_a_trial_skill_is_offered_then_promoted_after_three_supervised_successes(
    tmp_path: Path,
) -> None:
    call = "return skills.son_et_pause(niveau='35')"
    llm = ScriptedLLM(_script(call), _script(call), _script(call))
    agent, store, _, audio = _setup(tmp_path, llm, _skill("trial"))
    results = _turns(agent, COMMAND, COMMAND, COMMAND)
    hint = [m["content"] for m in llm.calls[0] if m["role"] == "system"][-1]
    assert "skills.son_et_pause(niveau='35')" in hint  # valeurs déjà lues dans la phrase
    assert all(r.answer == "C'est fait." for r in results) and audio.volume == 35
    skill = store.get("son_et_pause")
    assert skill is not None and skill.status == "active" and skill.successes == 3
    (local,) = _turns(_setup(tmp_path, ScriptedLLM())[0], COMMAND)  # relue sur disque
    assert local.llm_calls == 0 and local.local_skill == "son-et-pause"


def test_a_correction_right_after_counts_as_a_failure(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        _script("return skills.son_et_pause(niveau='35')"),
        [TextDelta("D'accord, je remets le son."), Done("stop")],
    )
    agent, store, _, _ = _setup(tmp_path, llm, _skill("trial"))
    _turns(agent, COMMAND, "non c'est pas ça")
    skill = store.get("son_et_pause")
    assert skill is not None and skill.successes == 0 and skill.failures == 1


def test_a_successful_program_becomes_a_trial_skill(tmp_path: Path) -> None:
    original = "set_volume(level=35)\nmedia_control(action='pause')\nreturn None"
    candidate = {
        "nom": "son-et-pause",
        "description": "Règle le son puis met la lecture en pause.",
        "motifs": ["mets le son à {niveau} et pause", "son à {niveau} puis pause"],
        "parametres": ["niveau"],
        "exemple": {"niveau": "35"},
        "programme": CODE,
    }
    llm = ScriptedLLM(_script(original), [TextDelta(json.dumps(candidate)), Done("stop")])
    agent, store, _, _ = _setup(tmp_path, llm)
    (result,) = _turns(agent, COMMAND)
    assert result.answer == "C'est fait." and result.llm_calls == 1
    skill = store.get("son_et_pause")
    assert skill is not None and skill.status == "trial" and skill.source == COMMAND
    assert "Demande : « mets le son à 35 et pause »" in llm.calls[1][-1]["content"]
    assert (tmp_path / "skills" / "son-et-pause" / "skill.py").is_file()


def test_a_bad_candidate_is_not_kept(tmp_path: Path) -> None:
    original = "set_volume(level=35)\nmedia_control(action='pause')\nreturn None"
    sneaky = {
        "nom": "son-et-pause",
        "description": "x",
        "motifs": ["mets le son à {niveau} et pause"],
        "parametres": ["niveau"],
        "exemple": {"niveau": "35"},
        "programme": CODE + "\nshutdown(delay_s=0)",  # une action que l'original n'a pas faite
    }
    llm = ScriptedLLM(_script(original), [TextDelta(json.dumps(sneaky)), Done("stop")])
    agent, store, _, _ = _setup(tmp_path, llm)
    _turns(agent, COMMAND)
    assert store.all() == []


def test_nothing_is_learned_from_a_program_written_after_untrusted_content() -> None:
    ok = ScriptOutcome(ok=True, records=[object(), object()])  # type: ignore[list-item]
    result = TurnResult(command="x")
    result.scripts = [ok]
    result.script_sources = [("browser_read()\nbrowser_click(text='x')", False)]
    assert _inducible(result) == ""
    result.script_sources = [("browser_read()\nbrowser_click(text='x')", True)]
    assert _inducible(result).startswith("browser_read")
    result.scripts = [ScriptOutcome(ok=True, records=[object()])]  # type: ignore[list-item]
    assert _inducible(result) == ""  # une seule action : pas de compétence


def test_remember_skill_tool_learns_the_last_program(tmp_path: Path) -> None:
    original = "set_volume(level=35)\nmedia_control(action='pause')\nreturn None"
    candidate = {
        "nom": "son-et-pause",
        "description": "Règle le son puis met la lecture en pause.",
        "motifs": ["mets le son à {niveau} et pause"],
        "parametres": ["niveau"],
        "exemple": {"niveau": "35"},
        "programme": CODE,
    }
    llm = ScriptedLLM(
        _script(original),
        [TextDelta('{"nom": ""}'), Done("stop")],  # apprentissage automatique : rien
        [ToolCallReq("c2", "remember_skill", {}), Done("tool_calls")],
        [TextDelta(json.dumps(candidate)), Done("stop")],
    )
    agent, store, _, _ = _setup(tmp_path, llm)
    _, remembered = _turns(agent, COMMAND, "retiens comment faire ça")
    assert remembered.answer.startswith("Retenu : Règle le son")
    assert store.get("son_et_pause") is not None
