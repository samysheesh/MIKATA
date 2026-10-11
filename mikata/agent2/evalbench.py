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
    """Registre simulé nourri par les textes du cas, avec un état : les actions changent
    ce que les lectures renvoient (fenêtre active, page affichée), pour que les prédicats
    de vérification (app_focused, page_shows...) restent significatifs. Rien ne touche le
    vrai PC."""

    def __init__(self, fixtures: EvalFixtures) -> None:
        super().__init__()
        self.fixtures = fixtures
        self.windows = list(fixtures.installed)  # la première = fenêtre active
        self.titles: dict[str, str] = {}
        self.page = "\n".join(fixtures.web.values()) or "(simulé)"

    async def execute(self, call: ToolCallReq) -> ToolResult:
        name = call.name
        args = call.arguments
        if name == "open_app":
            wanted = str(args.get("name", "")).casefold()
            for window in self.windows:
                if wanted and (wanted in window.casefold() or window.casefold() in wanted):
                    self.windows.remove(window)
                    self.windows.insert(0, window)
                    return self._note(call, f"{window} lancé, prêt à l'emploi.")
            else:  # comme le vrai bureau : une appli inconnue ne s'ouvre pas
                self.calls.append((name, dict(args)))
                return ToolResult(
                    call.id,
                    name,
                    ok=False,
                    content=f"application introuvable : {wanted}",
                    terminal=False,
                    duration_s=0.0,
                )
        elif name == "focus_window":
            wanted = str(args.get("name", "")).casefold()
            for window in self.windows:
                if wanted and (wanted in window.casefold() or window.casefold() in wanted):
                    self.windows.remove(window)
                    self.windows.insert(0, window)
                    break
            else:
                self.calls.append((name, dict(args)))
                return ToolResult(
                    call.id,
                    name,
                    ok=False,
                    content=f"fenêtre introuvable : {wanted}",
                    terminal=False,
                    duration_s=0.0,
                )
        elif name in ("browser_open", "open_url"):
            url = str(args.get("url", ""))
            texts = "\n".join(self.fixtures.web.values())
            self.page = f"Titre : Navigateur simulé\nURL : {url}\nTexte :\n{texts}"
            # Une recherche dans l'URL affiche ses résultats : les prédicats de page
            # peuvent vérifier ce qui a été cherché.
            import urllib.parse as _up

            query = _up.parse_qs(_up.urlparse(url).query).get("q", [""])[0]
            if query:
                self.page += f"\nRésultats pour : {query}"
            self.windows.insert(0, "Browser")  # le navigateur devient la fenêtre active
            self.titles["Browser"] = "YouTube" if "youtube" in url.casefold() else "Navigateur"
        elif name == "fetch_page":
            url = str(args.get("url", ""))
            texts = "\n".join(self.fixtures.web.values())
            self.page = f"Titre : Navigateur simulé\nURL : {url}\nTexte :\n{texts}"
        elif name == "set_volume":
            self._note(call, f"Volume réglé à {args.get('level')} %.")
        elif name == "set_timer":
            self._note(call, f"Minuteur lancé ({args.get('seconds')} s).")
        elif name == "set_reminder":
            self._note(call, f"Rappel programmé : {args.get('text')}.")
        elif name == "media_control":
            action = args.get("action", "play")
            state = "en cours" if action == "play" else f"« {action} » appliqué"
            self._note(call, f"Lecture : {state}.")
        elif name == "press_keys":
            keys = str(args.get("keys", "")).casefold()
            note = "Touches pressées."
            if "enter" in keys:
                note = "Validation (Entrée). Lecture en cours."
            self._note(call, note)
        elif name == "browser_click":
            self._note(call, f"Cliqué : {args.get('text')}. Lecture en cours.")
        elif name == "browser_type":
            self._note(call, f"Champ « {args.get('field')} » rempli : {args.get('text')}.")
        elif name == "uia_set_text":
            text = str(args.get("text", ""))
            if text and len(text) <= 30 and self.windows:
                # une saisie courte (une recherche) se voit dans le titre de la fenêtre
                window = self.windows[0]
                self.titles[window] = f"Fenêtre {window} — {text}"
            self._note(call, f"Texte écrit : {text}.")
        elif name == "uia_click":
            self._note(call, f"Contrôle cliqué : {args.get('control')}.")
        elif name == "write_file":
            content = str(args.get("content", ""))
            self._note(call, f"Fichier écrit : {args.get('path')}\n{content[:200]}")
        elif name == "move_file":
            self._note(call, f"Déplacé : {args.get('source')} -> {args.get('destination')}.")
        elif name == "remember":
            self._note(call, f"Mémorisé : {args.get('fact')}.")
        elif name == "show_text":
            self._note(call, f"Affiché : {args.get('text')}.")
        elif name == "open_in_explorer":
            self._note(call, f"Explorateur ouvert : {args.get('path')}.")
        elif name == "type_text":
            self._note(call, f"Texte tapé : {args.get('text')}.")
        elif name == "find_path":
            query = str(args.get("name", ""))
            self._note(call, f"Trouvé :\n- C:\\Utilisateur\\Documents\\{query}")
        elif name in ("run_powershell", "run_shell"):
            command = str(args.get("command", ""))[:100]
            self._note(call, f"Commande exécutée (simulée) : {command}")
        elif name == "web_search":
            query = str(args.get("query", "")).casefold()
            for needle, text in self.fixtures.web.items():
                if needle.casefold() in query or query in needle.casefold():
                    self.calls.append((name, dict(args)))
                    return ToolResult(
                        call.id, name, ok=True, content=text, terminal=False, duration_s=0.0
                    )
        elif name == "list_windows":
            lines = [
                f"- {window} : {self.titles.get(window, 'Fenêtre ' + window)}"
                for window in self.windows[:5]
            ]
            text = "Fenêtres ouvertes :\n" + "\n".join(lines) if lines else "(simulé)"
            self.calls.append((name, dict(args)))
            return ToolResult(call.id, name, ok=True, content=text, terminal=False, duration_s=0.0)
        elif name in ("browser_read", "uia_read"):
            # Les outils de lecture renvoient l'état simulé : les prédicats de vérification
            # (page_shows, text_visible...) restent significatifs.
            self.calls.append((name, dict(args)))
            return ToolResult(
                call.id, name, ok=True, content=self.page, terminal=False, duration_s=0.0
            )
        return await super().execute(call)

    def _note(self, call: ToolCallReq, text: str) -> ToolResult:
        """Une action qui laisse une trace visible : la lecture suivante la verra."""
        self.page = f"{self.page}\n{text}"
        self.calls.append((call.name, dict(call.arguments)))
        return ToolResult(call.id, call.name, ok=True, content=text, terminal=False, duration_s=0.0)


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
