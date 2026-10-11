"""Agent v2 (phase 13e) : la batterie de référence — le verrou d'adoption.

Un cas est un fichier JSON (tests/agent2_eval/ et data/agent2/eval/) décrivant une demande
et ce qu'on en attend. La batterie exécute l'Agent2 complet sur un registre simulé nourri
par les textes du cas : rien ne touche le vrai PC. `regressions` compare deux passes ;
c'est lui qui refuse un savoir-faire qui casse des cas qui passaient.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from mikata.agent2.learn.dryrun import DryRunRegistry
from mikata.agent2.orchestrator import Agent2
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.trace import TraceStore
from mikata.config import Agent2Config
from mikata.llm.base import ToolCallReq
from mikata.llm.usage import UsageTracker
from mikata.tools.registry import ToolResult

EVAL_CASES_DIR = "tests/agent2_eval"
USER_EVAL_DIR = "data/agent2/eval"
MIN_LOCK_CASES = 5  # au moins 5 cas de la même capacité pour verrouiller (13e)


class EvalExpect(BaseModel):
    verb: str
    channel: str
    tools_called: list[str] = []
    tools_forbidden: list[str] = []


class EvalFixtures(BaseModel):
    installed: list[str] = []
    web: dict[str, str] = {}


class EvalCase(BaseModel):
    id: str
    request: str
    os: list[str] = ["win32", "darwin"]
    fixtures: EvalFixtures = Field(default_factory=EvalFixtures)
    expect: EvalExpect
    capability: str | None = None  # pour le verrou d'adoption (7.3)


class EvalResult(BaseModel):
    id: str
    passed: bool
    reasons: list[str] = []
    llm_calls: int = 0
    cost_usd: float = 0.0


class FixtureRegistry(DryRunRegistry):
    """Registre simulé nourri par les textes du cas : les lectures renvoient les fixtures,
    tout le reste renvoie « (simulé) » sans rien exécuter."""

    def __init__(self, fixtures: EvalFixtures) -> None:
        super().__init__()
        self.fixtures = fixtures

    async def execute(self, call: ToolCallReq) -> ToolResult:
        if call.name == "web_search":
            query = str(call.arguments.get("query", "")).casefold()
            for needle, text in self.fixtures.web.items():
                if needle.casefold() in query or query in needle.casefold():
                    self.calls.append((call.name, dict(call.arguments)))
                    return ToolResult(
                        call.id, call.name, ok=True, content=text, terminal=False, duration_s=0.0
                    )
        if call.name == "list_windows":
            lines = [f"- {name} : Fenêtre {name}" for name in self.fixtures.installed[:3]]
            text = "Fenêtres ouvertes :\n" + "\n".join(lines) if lines else "(simulé)"
            self.calls.append((call.name, dict(call.arguments)))
            return ToolResult(
                call.id, call.name, ok=True, content=text, terminal=False, duration_s=0.0
            )
        if call.name in ("browser_read", "uia_read"):
            # Les outils de lecture renvoient les textes des fixtures : les prédicats de
            # vérification (page_shows, text_visible...) restent significatifs.
            text = "\n".join(self.fixtures.web.values()) or "(simulé)"
            self.calls.append((call.name, dict(call.arguments)))
            return ToolResult(
                call.id, call.name, ok=True, content=text, terminal=False, duration_s=0.0
            )
        return await super().execute(call)


class _EvalLegacy:
    """L'Agent actuel réduit à ce que l'Agent2 utilise pendant la batterie."""

    def __init__(self, llm: Any, registry: FixtureRegistry, fixtures: EvalFixtures) -> None:
        self.config = SimpleNamespace(agent2=Agent2Config())
        self.tools = registry
        self.llm = llm
        self.long_memory = SimpleNamespace(facts=())
        self.usage: UsageTracker | None = None

    def _installed_apps(self) -> list[str]:
        return list(self.tools.fixtures.installed)

    async def context(self) -> str:
        return "fenêtre active : Brave"

    def say_phrase(self, speech: Any, key: str) -> None:
        pass

    async def respond(self, *args: Any, **kwargs: Any) -> Any:
        # Les demandes « chat »/« ask » sont déléguées à l'ancien agent ; la batterie
        # accepte la délégation (le cas vérifie que la trace dit « chat »).
        from mikata.agent.loop import TurnResult

        return TurnResult(command=str(kwargs.get("command", "")), answer="ok", outcome="ok")


class _SilentSpeech:
    def say(self, text: str) -> None:
        pass

    def say_audio(self, pcm: Any, rate: int) -> None:
        pass

    async def drain(self) -> None:
        pass


def load_cases(*extra_dirs: Path) -> list[EvalCase]:
    """Les cas du dépôt puis ceux de l'utilisateur, triés par id."""
    dirs = [Path(EVAL_CASES_DIR), Path(USER_EVAL_DIR), *extra_dirs]
    cases: dict[str, EvalCase] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                case = EvalCase.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            cases[case.id] = case
    return [cases[key] for key in sorted(cases)]


def _tools_called(trace: Any) -> set[str]:
    return {action.tool for step in trace.steps for action in step.actions}


def _reasons(case: EvalCase, trace: Any) -> list[str]:
    reasons: list[str] = []
    if case.expect.verb in ("chat", "ask"):
        # Ces demandes sont déléguées à l'ancien agent : la trace dit « chat ».
        if trace.outcome == "chat":
            return []
        reasons.append("la demande n'a pas été déléguée à l'ancien agent")
        return reasons
    if trace.outcome != "ok":
        reasons.append("l'agent n'a pas réussi la demande")
        return reasons
    if trace.intent is None or trace.intent.verb.value != case.expect.verb:
        reasons.append(f"verbe attendu {case.expect.verb}")
    if trace.channel != case.expect.channel:
        reasons.append(f"canal attendu {case.expect.channel}, obtenu {trace.channel}")
    called = _tools_called(trace)
    for tool in case.expect.tools_called:
        if tool not in called:
            reasons.append(f"outil attendu jamais appelé : {tool}")
    for tool in case.expect.tools_forbidden:
        if tool in called:
            reasons.append(f"outil interdit appelé : {tool}")
    return reasons


async def run_case(
    case: EvalCase,
    llm: Any,
    *,
    skills: Any = None,
    recipes: Any = None,
    usage: UsageTracker | None = None,
    root: Path | None = None,
) -> EvalResult:
    """Joue un cas de bout en bout sur le registre simulé ; rien ne touche le vrai PC."""
    from mikata.agent2.orchestrator import _CountingLLM

    registry = FixtureRegistry(case.fixtures)
    counted = _CountingLLM(llm)
    legacy = _EvalLegacy(counted, registry, case.fixtures)
    root = root or Path("data/agent2/eval/traces")
    agent2 = Agent2(
        legacy,
        traces=TraceStore(root),
        prefs=PreferenceStore(root / ".." / "preferences.json"),
        skills=skills,
        recipes=recipes,
    )
    agent2.learner = None  # la batterie mesure, elle n'apprend pas
    await agent2.respond(case.request, _SilentSpeech())
    traces = agent2.traces.recent(days=1)
    trace = traces[0] if traces else None
    reasons = _reasons(case, trace) if trace is not None else ["aucune trace produite"]
    cost = 0.0
    if usage is not None and counted.calls:
        cost = usage.record(f"eval:{case.id}", counted.usage, counted.calls)
    return EvalResult(
        id=case.id, passed=not reasons, reasons=reasons, llm_calls=counted.calls, cost_usd=cost
    )


async def run_bench(
    cases: list[EvalCase],
    llm: Any,
    *,
    skills: Any = None,
    recipes: Any = None,
    usage: UsageTracker | None = None,
    root: Path | None = None,
) -> list[EvalResult]:
    """Toute la batterie, dans l'ordre des cas."""
    results: list[EvalResult] = []
    for case in cases:
        results.append(
            await run_case(case, llm, skills=skills, recipes=recipes, usage=usage, root=root)
        )
    return results


def regressions(before: list[EvalResult], after: list[EvalResult]) -> list[str]:
    """Les cas qui passaient avant et échouent après, avec leur raison."""
    after_by_id = {result.id: result for result in after}
    broken: list[str] = []
    for previous in before:
        if not previous.passed:
            continue
        current = after_by_id.get(previous.id)
        if current is None or not current.passed:
            reason = "; ".join(current.reasons) if current else "cas disparu"
            broken.append(f"{previous.id} : {reason}")
    return broken


def write_report(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
