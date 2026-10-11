"""Agent v2 (phase 13e) : la passe de nuit adopte, corrige, fusionne et respecte son
budget ; l'ordonnanceur exige un créneau calme."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent2_fakes import ScriptedLLM

from mikata.agent2.dream import DREAM_BUDGET_USD, dream, dream_loop
from mikata.agent2.learn.compiler import Learner
from mikata.agent2.learn.recipes import RecipeStore
from mikata.agent2.learn.store import SkillMeta, SkillStore
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.schema import PageShows
from mikata.agent2.trace import TraceStore
from mikata.config import Agent2Config

_CODE = "await sdk.browser_open('https://www.youtube.com')\nreturn 'ok'"
_FIXED = (
    '{"name": "play_youtube_video",'
    ' "description": "Jouer une vidéo YouTube.",'
    ' "capability": "play.video", "channel": "youtube",'
    ' "params": {"query": "titre"},'
    ' "done_when": {"type": "page_shows", "pattern": "{query}"},'
    ' "code": "await sdk.browser_open(\'https://www.youtube.com/results?search_query=\''
    " + args['query'])\\nreturn 'ok'\"}"
)


class FakeUsage:
    """UsageTracker de test : chaque enregistrement coûte le même montant."""

    def __init__(self, cost: float) -> None:
        self.cost = cost

    def record(self, label: str, usage: Any, calls: int) -> float:
        return self.cost


def _meta(status: str = "trial", **overrides: object) -> SkillMeta:
    fields = dict(
        description="Jouer une vidéo",
        capability="play.video",
        channel="youtube",
        params={"query": "titre"},
        done_when=PageShows(pattern="{query}"),
        status=status,
        os=["win32"],
    )
    fields.update(overrides)
    return SkillMeta(**fields)


def _agent2(tmp_path: Path, llm: ScriptedLLM, usage: Any = None) -> Any:
    cfg = Agent2Config()
    store = SkillStore(tmp_path / "skills", cfg)
    recipes = RecipeStore(tmp_path / "recipes.json", store, "win32")
    prefs = PreferenceStore(tmp_path / "prefs.json")
    traces = TraceStore(tmp_path / "traces")
    learner = Learner(llm, store, recipes, prefs, traces, cfg, "win32")
    legacy = SimpleNamespace(config=SimpleNamespace(agent2=cfg), llm=llm, usage=usage)
    from mikata.agent2.orchestrator import Agent2

    agent2 = Agent2(legacy, traces=traces, prefs=prefs, skills=store, recipes=recipes)
    agent2.learner = learner
    return agent2


def test_pending_is_adopted_overnight(tmp_path: Path) -> None:
    llm = ScriptedLLM([])
    agent2 = _agent2(tmp_path, llm)
    agent2.learner._pend("play_youtube_video", _meta("proposed"), _CODE, None)
    report = asyncio.run(dream(agent2, now=datetime(2026, 10, 12, 3, 0)))
    assert report.learned == ["play_youtube_video"]
    assert agent2.learner.store.skills["play_youtube_video"].status == "trial"
    assert agent2.learner.pending() == []


def test_a_demoted_skill_is_fixed_then_trial(tmp_path: Path) -> None:
    llm = ScriptedLLM([_FIXED])
    agent2 = _agent2(tmp_path, llm)
    store = agent2.learner.store
    store.add("play_youtube_video", _meta("demoted"), _CODE)
    # une trace d'échec qui nomme le savoir-faire
    traces = agent2.traces
    failed = traces.new("lance un truc")
    failed.outcome = "failed"
    failed.skills_used = ["play_youtube_video"]
    from mikata.agent2.schema import Step

    failed.steps = [
        Step(
            subgoal_index=0,
            attempt=1,
            code="",
            actions=[],
            error="SdkError: application introuvable",
            observation={},
            goal_reached=False,
        )
    ]
    traces.save(failed)
    report = asyncio.run(dream(agent2, now=datetime(2026, 10, 12, 3, 0)))
    assert report.fixed == ["play_youtube_video"]
    assert store.skills["play_youtube_video"].status == "trial"


def test_active_duplicates_are_merged(tmp_path: Path) -> None:
    llm = ScriptedLLM([])
    agent2 = _agent2(tmp_path, llm)
    store = agent2.learner.store
    store.add("video_a", _meta("active"), _CODE)
    store.add("video_b", _meta("active"), _CODE)
    store.skills["video_a"].successes = 7
    store.skills["video_b"].successes = 2
    store._save()
    report = asyncio.run(dream(agent2, now=datetime(2026, 10, 12, 3, 0)))
    assert report.merged == ["video_b"]
    assert "video_a" in store.skills and "video_b" not in store.skills


def test_budget_is_enforced(tmp_path: Path) -> None:
    llm = ScriptedLLM([_FIXED, _FIXED])
    agent2 = _agent2(tmp_path, llm, usage=FakeUsage(DREAM_BUDGET_USD))
    store = agent2.learner.store
    store.add("video_a", _meta("demoted"), _CODE)
    store.add("video_b", _meta("demoted"), _CODE)
    traces = agent2.traces
    from mikata.agent2.schema import Step

    for name in ("video_a", "video_b"):
        failed = traces.new("une demande")
        failed.outcome = "failed"
        failed.skills_used = [name]
        failed.steps = [
            Step(
                subgoal_index=0,
                attempt=1,
                code="",
                actions=[],
                error="SdkError: échec",
                observation={},
                goal_reached=False,
            )
        ]
        traces.save(failed)
    report = asyncio.run(dream(agent2, now=datetime(2026, 10, 12, 3, 0)))
    # Le premier correctif coûte tout le budget : le second est laissé en place.
    assert report.fixed == ["video_a"]
    assert store.skills["video_b"].status == "demoted"
    assert report.cost_usd >= DREAM_BUDGET_USD


def test_dream_loop_waits_for_a_calm_slot(tmp_path: Path, monkeypatch: Any) -> None:
    llm = ScriptedLLM([])
    agent2 = _agent2(tmp_path, llm)
    import mikata.agent2.dream as dream_module

    ran: list[datetime] = []

    async def fake_dream(a2: Any, *, now: datetime) -> None:
        ran.append(now)

    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(dream_module, "dream", fake_dream)
    monkeypatch.setattr(dream_module.asyncio, "sleep", no_sleep)
    now = datetime(2026, 10, 12, 2, 59, 50)  # 10 s avant l'heure

    asyncio.run(
        dream_loop(
            agent2,
            now_fn=lambda: now,
            busy=lambda: False,
            last_command_at=lambda: 0.0,
            retry_s=0.01,
            idle_s=1200,
        )
    )
    assert ran  # la passe a eu lieu au créneau calme
