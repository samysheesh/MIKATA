"""Banc d'essai (docs/AGENT_CORE.md §3.10) : tâches livrées valides, PC simulé cohérent,
correcteurs d'état final, exécution d'un essai de bout en bout avec un LLM scripté."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from mikata.config import Config
from mikata.eval.agents import agent_factory
from mikata.eval.bench import (
    CHECKS,
    Env,
    Task,
    Trial,
    check_names,
    format_report,
    grade,
    load_tasks,
    run_trial,
    summarize,
)
from mikata.eval.sim import Page, SimBrowser, SimWorld, WorldSpec, find_page
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq, Usage

TASKS = Path(__file__).parent / "eval"


class EventLLM:
    """LLM de test : chaque appel rejoue la liste d'événements suivante."""

    def __init__(self, *turns: list[LLMEvent]) -> None:
        self.turns = list(turns)
        self.calls = 0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        for event in self.turns.pop(0):
            yield event
        yield Done("stop", usage=Usage(100, 10))


def _world(tmp_path: Path, **spec: Any) -> SimWorld:
    return SimWorld(WorldSpec(**spec), tmp_path, Config())


def test_shipped_tasks_are_valid() -> None:
    tasks = load_tasks(TASKS)
    assert len(tasks) >= 30
    assert len({task.id for task in tasks}) == len(tasks)
    for task in tasks:
        WorldSpec(**task.world)  # champs connus
        for checks in (task.expect, task.forbid, task.live_expect or []):
            unknown = [name for name in check_names(checks) if name not in CHECKS]
            assert unknown == [], f"{task.id} : {unknown}"
        assert task.expect or task.forbid, task.id
    families = {tag for task in tasks for tag in task.tags}
    assert {"direct", "chat", "fichiers", "web", "securite", "enchaine"} <= families
    # Les tâches rejouées sur le vrai PC la nuit : jamais de son, d'extinction ni d'envoi.
    for task in tasks:
        if task.live:
            silent_web = ("recherche", "titre_dans_presse_papiers")  # lire, puis copier
            assert not ({"web", "ui"} & set(task.tags)) or task.id in silent_web, task.id


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    file = tmp_path / "t.yaml"
    file.write_text("- {id: a, request: x}\n- {id: a, request: y}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="en double"):
        load_tasks(file)


def test_sim_desktop_keeps_a_state(tmp_path: Path) -> None:
    world = _world(tmp_path, open_apps=["Spotify", "Discord"])
    desktop = world.desktop
    assert desktop.foreground_window().title == "Discord"  # la dernière ouverte est devant
    desktop.focus_window(desktop.windows()[1])
    assert desktop.foreground_window().title == "Spotify"
    desktop.launch(desktop.list_apps()[5])
    assert desktop.foreground_window().title == "Bloc-notes"
    desktop.close_window(desktop.windows()[0])
    assert "Bloc-notes" not in [w.title for w in desktop.windows()]


def test_sim_browser_follows_links_and_patterns() -> None:
    pages = {
        r"youtube\.com/results": Page("Résultats", "…", {"Clip officiel": "https://y/watch?v=1"}),
        "https://site.fr": Page("Site", "Bonjour"),
    }
    assert find_page(pages, "https://site.fr/").title == "Site"
    assert find_page(pages, "https://www.youtube.com/results?search_query=a").title == "Résultats"
    assert find_page(pages, "https://inconnu.fr") is None

    async def scenario() -> SimBrowser:
        browser = SimBrowser(pages)
        await browser.open_url("https://www.youtube.com/results?search_query=clip")
        assert "Clip officiel" in await browser.read(2000)
        assert (await browser.find("officiel", 1))["href"] == "https://y/watch?v=1"
        await browser.click("clip", 1)
        return browser

    browser = asyncio.run(scenario())
    assert browser.url == "https://y/watch?v=1"
    asyncio.run(browser.back())
    assert "results" in browser.url


def test_graders_read_the_final_state(tmp_path: Path) -> None:
    world = _world(tmp_path, volume=30, files={"Bureau/a.txt": "Bonjour"}, open_apps=["Spotify"])
    env = Env(world.ctx, "Il fait 14 degrés.", [("get_weather", {"city": "Paris"})], world)
    task = Task(
        id="t",
        request="x",
        expect=[
            {"volume": 30},
            {"window_open": "spotify"},
            {"file_contains": {"path": "Bureau/a.txt", "text": "bonjour"}},
            {"answer": "14|quatorze"},
            {"tool": {"name": "get_weather", "args": {"city": "paris"}}},
            {"any_of": [{"playing": True}, {"file_exists": "Bureau/a.txt"}]},
        ],
        forbid=[{"shell": "."}, {"window_open": "Discord"}],
    )
    assert grade(task, env) == []
    world.audio.set_volume(100)
    world.shell.commands.append("Remove-Item x")
    assert grade(task, env) == ["attendu : volume=30", "interdit : shell='.'"]
    with pytest.raises(ValueError, match="contrôle inconnu"):
        grade(Task(id="t", request="x", expect=[{"inconnu": 1}]), env)


def test_sim_only_checks_are_skipped_on_the_real_pc(tmp_path: Path) -> None:
    world = _world(tmp_path)
    live = Env(world.ctx, "ok", [], None)  # pas de monde simulé : comme sur le vrai PC
    task = Task(
        id="t",
        request="x",
        expect=[{"media": "next"}, {"volume": 50}],
        live_expect=[{"volume": 50}, {"browser_url": "x"}],
    )
    assert grade(task, live) == []
    assert grade(task, Env(world.ctx, "ok", [], world)) == ["attendu : media='next'"]


def test_a_trial_passes_only_when_the_state_is_reached(tmp_path: Path) -> None:
    task = next(t for t in load_tasks(TASKS) if t.id == "volume_30")

    def trial(*turns: list[LLMEvent]) -> Trial:
        llm = EventLLM(*turns)
        return asyncio.run(run_trial(task, agent_factory(llm), Config()))

    done = trial([ToolCallReq("1", "set_volume", {"level": 30})])
    assert done.passed and done.llm_calls == 1 and done.calls == ["set_volume(level=30)"]
    assert done.tokens_in == 100 and done.outcome == "ok"
    said = trial([TextDelta("C'est fait.")])  # l'agent le dit sans le faire : échec
    assert not said.passed and said.failures == ["attendu : volume=30"]


def test_report_counts_tasks_passed_on_every_trial() -> None:
    def make(task: str, passed: bool, seconds: float) -> Trial:
        return Trial(task, passed, [] if passed else ["attendu : x"], seconds, 0.5, 1, 10, 5,
                     0.001, "", "ok", "")  # fmt: skip

    trials = [make("a", True, 1.0), make("a", False, 3.0), make("b", True, 2.0)]
    summary = summarize(trials)
    assert (summary.tasks, summary.passed_all, summary.passed_any) == (2, 1, 2)
    assert summary.median_s == 2.0 and summary.cost_usd == 0.003
    tasks = [Task(id="a", request="x", tags=["direct"]), Task(id="b", request="y", tags=["chat"])]
    report = format_report(tasks, trials)
    assert "a " in report and "1/2" in report and "TOTAL" in report and "1/2 réussies" in report
