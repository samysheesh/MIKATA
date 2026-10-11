"""Agent v2 (phase 13d) : la compilation ne garde que ce qui passe toutes les validations."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent2_fakes import ScriptedLLM

from mikata.agent2.learn.compiler import Learner
from mikata.agent2.learn.recipes import RecipeStore
from mikata.agent2.learn.store import SkillStore
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.schema import Intent, Resolved, Step, Trace, Verb
from mikata.agent2.trace import TraceStore
from mikata.config import Agent2Config

_COMPILED = (
    '{"name": "play_youtube_video",'
    ' "description": "Jouer une vidéo YouTube à partir d\'un titre.",'
    ' "capability": "play.video", "channel": "youtube",'
    ' "params": {"query": "titre exact à chercher"},'
    ' "done_when": {"type": "page_shows", "pattern": "{query}"},'
    ' "code": "await sdk.browser_open(\'https://www.youtube.com/results?search_query=\''
    " + args['query'])\\nreturn 'ok'\"}"
)


def _trace(tmp_path: Path, **overrides: object) -> Trace:
    store = TraceStore(tmp_path / "traces")
    trace = store.new("lance le dernier opening de One Piece")
    trace.intent = Intent(verb=Verb.PLAY, object="x", object_kind="opening d'anime")
    trace.resolved = [Resolved(name="titre", value="One Piece Opening 27", source="web")]
    trace.channel = "youtube"
    trace.verified = True
    trace.outcome = "ok"
    trace.steps = [
        Step(
            subgoal_index=0,
            attempt=1,
            code="await sdk.browser_open('https://www.youtube.com')\nreturn 'ok'",
            actions=[],
            observation={},
            goal_reached=True,
        )
    ]
    for key, value in overrides.items():
        setattr(trace, key, value)
    return trace


def _learner(tmp_path: Path, llm: ScriptedLLM) -> Learner:
    cfg = Agent2Config()
    store = SkillStore(tmp_path / "skills", cfg)
    recipes = RecipeStore(tmp_path / "recipes.json", store, "win32")
    prefs = PreferenceStore(tmp_path / "prefs.json")
    return Learner(llm, store, recipes, prefs, TraceStore(tmp_path / "traces"), cfg, "win32")


def test_verified_trace_produces_a_pending_skill(tmp_path: Path) -> None:
    """13e : rien n'entre directement dans la bibliothèque — le savoir-faire est mis en
    attente (pending.json) et adopté (verrouillé) par la passe de nuit."""
    trace = _trace(tmp_path)
    learner = _learner(tmp_path, ScriptedLLM([_COMPILED]))
    asyncio.run(learner.learn(trace))
    assert learner.store.skills == {}
    pending = learner.pending()
    assert len(pending) == 1
    name, meta, _code, recipe = pending[0]
    assert name == "play_youtube_video"
    assert meta.status == "proposed" and meta.source_traces == [trace.id]
    assert meta.needs_approval is False
    assert recipe is not None and recipe.skill == "play_youtube_video"


def test_unverified_trace_learns_nothing(tmp_path: Path) -> None:
    trace = _trace(tmp_path, verified=False)
    learner = _learner(tmp_path, ScriptedLLM([_COMPILED]))
    asyncio.run(learner.learn(trace))
    assert learner.store.skills == {}


def test_hardcoded_value_is_refused_by_anti_recopy(tmp_path: Path) -> None:
    hardcoded = _COMPILED.replace("args['query']", "'One Piece Opening 27'")
    trace = _trace(tmp_path)
    learner = _learner(tmp_path, ScriptedLLM([hardcoded]))
    asyncio.run(learner.learn(trace))
    assert learner.store.skills == {}


def test_code_failing_the_dry_run_is_refused(tmp_path: Path) -> None:
    failing = _COMPILED.replace(
        "await sdk.browser_open('https://www.youtube.com/results?search_query=' + args['query'])",
        "await sdk.outil_inconnu('x')",
    )
    trace = _trace(tmp_path)
    learner = _learner(tmp_path, ScriptedLLM([failing]))
    asyncio.run(learner.learn(trace))
    assert learner.store.skills == {}


def test_duplicate_name_is_suffixed(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    learner = _learner(tmp_path, ScriptedLLM([_COMPILED, _COMPILED]))
    asyncio.run(learner.learn(trace))
    asyncio.run(learner.learn(trace))
    names = [entry[0] for entry in learner.pending()]
    assert names == ["play_youtube_video", "play_youtube_video_2"]
