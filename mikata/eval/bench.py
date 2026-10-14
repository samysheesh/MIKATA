"""Banc d'essai : tâches, correcteurs d'état final, exécution et rapport (AGENT_CORE §3.10).

Une tâche = une demande, un état de départ et des contrôles sur l'**état final** (volume,
fenêtres, fichiers, page affichée, réponse dite). Deux modes :

- `sim` : PC simulé (`mikata.eval.sim`), rejouable partout, sans effet réel ;
- `live` : le vrai PC, pour les seules tâches marquées `live: true` (silencieuses et sans
  risque), avec remise en état après chaque essai.

Les mesures (durée, premier élément, appels LLM, tokens, coût) sont rapportées avec la
réussite : une phase qui ralentit le chemin court se voit ici.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mikata.config import Config
from mikata.eval.sim import SimWorld, WorldSpec
from mikata.tools.base import ToolContext

AgentFactory = Callable[[Config, ToolContext], Any]
LIVE_SETTLE_S = 4.0  # vrai PC : délai laissé à l'état pour se stabiliser


class Task(BaseModel):
    id: str
    request: str
    tags: list[str] = Field(default_factory=list)
    world: dict[str, Any] = Field(default_factory=dict)  # champs de `WorldSpec`
    expect: list[dict[str, Any]] = Field(default_factory=list)  # tous vrais
    forbid: list[dict[str, Any]] = Field(default_factory=list)  # aucun vrai
    max_llm_calls: int | None = None  # budget (rapporté, pas éliminatoire)
    live: bool = False  # rejouable sur le vrai PC, la nuit, sans bruit ni risque
    live_expect: list[dict[str, Any]] | None = None  # contrôles propres au vrai PC
    # Réponse de l'utilisateur simulé à « Tu confirmes ? ». Par défaut : « oui » pour une
    # demande normale, « non » pour une tâche de sécurité (il n'a rien demandé de tel).
    confirm: str | None = None
    # Leçons déjà dans le carnet au début de l'essai : [{scope, text}] (phase 14e).
    lessons: list[dict[str, str]] = Field(default_factory=list)
    restore: list[str] = Field(default_factory=list)  # live : volume, clipboard, close:<appli>
    # Phase 16g : échanges précédents de la conversation, pour les corrections (« non, sur
    # Spotify ») : [{user, assistant, calls: [{name, args, result}]}].
    history: list[dict[str, Any]] = Field(default_factory=list)


def load_tasks(path: Path) -> list[Task]:
    """Tâches d'un fichier YAML (une liste) ou de tous les `.yaml` d'un dossier."""
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    tasks: list[Task] = []
    for file in files:
        for raw in yaml.safe_load(file.read_text(encoding="utf-8")) or []:
            tasks.append(Task.model_validate(raw))
    ids = [task.id for task in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"tâches en double : {', '.join(duplicates)}")
    return tasks


# -- correcteurs --------------------------------------------------------------------------


class Unverifiable(Exception):
    """Ce contrôle ne peut pas être relu dans ce mode (ex. page du navigateur sur le vrai PC)."""


@dataclass
class Env:
    ctx: ToolContext
    answer: str
    calls: list[tuple[str, dict[str, Any]]]
    world: SimWorld | None = None

    def sim(self) -> SimWorld:
        if self.world is None:
            raise Unverifiable
        return self.world


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None


def _in(needle: Any, text: str) -> bool:
    return str(needle).casefold() in text.casefold()


def _titles(env: Env) -> list[str]:
    assert env.ctx.desktop is not None
    return [f"{w.title} {w.process}".casefold() for w in env.ctx.desktop.windows()]


def _path(env: Env, relative: str) -> Path:
    return env.ctx.require_guard().resolve_path(relative)


def _tool(env: Env, value: Any) -> bool:
    name = value if isinstance(value, str) else value["name"]
    wanted = {} if isinstance(value, str) else value.get("args", {})
    for called, args in env.calls:
        if called == name and all(
            str(args.get(k, "")).casefold() == str(v).casefold() for k, v in wanted.items()
        ):
            return True
    return False


CHECKS: dict[str, Callable[[Env, Any], bool]] = {
    # état relisible dans les deux modes
    "volume": lambda e, v: abs(e.ctx.require_audio().get_volume() - float(v)) <= 1,
    "volume_above": lambda e, v: e.ctx.require_audio().get_volume() > float(v),
    "volume_below": lambda e, v: e.ctx.require_audio().get_volume() < float(v),
    "window_open": lambda e, v: any(str(v).casefold() in t for t in _titles(e)),
    "window_closed": lambda e, v: not any(str(v).casefold() in t for t in _titles(e)),
    "foreground": lambda e, v: bool(_titles(e)) and str(v).casefold() in _titles(e)[0],
    "file_exists": lambda e, v: _path(e, v).exists(),
    "file_absent": lambda e, v: not _path(e, v).exists(),
    "file_contains": lambda e, v: _path(e, v["path"]).is_file()
    and v["text"].casefold() in _path(e, v["path"]).read_text(encoding="utf-8").casefold(),
    "clipboard": lambda e, v: _in(v, e.ctx.require_desktop().clipboard_read()),
    "answer": lambda e, v: _has(str(v), e.answer),
    "remembered": lambda e, v: any(str(v).casefold() in f.casefold() for f in e.ctx.memory.facts),
    "tool": _tool,  # contrôle du chemin : seulement quand l'état n'est pas observable
    "no_tools": lambda e, v: (not e.calls) is bool(v),
    # état du PC simulé
    "muted": lambda e, v: e.sim().audio.muted is bool(v),
    "browser_url": lambda e, v: _has(str(v), e.sim().browser.url),
    "playing": lambda e, v: e.sim().browser.playing is bool(v),
    "opened": lambda e, v: any(_has(str(v), t) for t in e.sim().desktop.opened),
    "revealed": lambda e, v: any(_has(str(v), t) for t in e.sim().desktop.revealed),
    "media": lambda e, v: str(v) in e.sim().desktop.media,
    "typed": lambda e, v: any(str(v).casefold() in t.casefold() for t in e.sim().input.typed),
    "keys": lambda e, v: [str(k).casefold() for k in v]
    in [[k.casefold() for k in keys] for keys in e.sim().input.keys],
    "shell": lambda e, v: any(_has(str(v), c) for c in e.sim().shell.commands),
    "notified": lambda e, v: any(_has(str(v), f"{t} {b}") for t, b in e.sim().notifier.sent),
    "ui_clicked": lambda e, v: any(_in(v, name) for _, name in e.sim().ui.clicked),
    "trashed": lambda e, v: any(_has(str(v), item) for item in e.sim().trash.items),
    "locked": lambda e, v: e.sim().desktop.locked is bool(v),
    "power": lambda e, v: str(v) in e.sim().desktop.power_actions,
}  # fmt: skip


def check_names(checks: list[dict[str, Any]]) -> list[str]:
    """Noms des contrôles d'une liste, `any_of` déplié (validation des fichiers de tâches)."""
    names: list[str] = []
    for check in checks:
        for name, value in check.items():
            names.extend(check_names(value) if name == "any_of" else [name])
    return names


def _holds(env: Env, name: str, value: Any) -> bool:
    if name == "any_of":  # au moins un des contrôles de la liste
        results = []
        for check in value:
            for sub_name, sub_value in check.items():
                try:
                    results.append(_holds(env, sub_name, sub_value))
                except Unverifiable:
                    continue
        if not results:
            raise Unverifiable
        return any(results)
    return CHECKS[name](env, value)


def grade(task: Task, env: Env) -> list[str]:
    """Liste des contrôles en échec (vide = tâche réussie)."""
    failures: list[str] = []
    expect = task.live_expect if env.world is None and task.live_expect is not None else task.expect
    for wanted, checks in ((True, expect), (False, task.forbid)):
        unknown = [name for name in check_names(checks) if name not in CHECKS]
        if unknown:
            raise ValueError(f"tâche {task.id} : contrôle inconnu « {unknown[0]} »")
        for check in checks:
            for name, value in check.items():
                try:
                    ok = _holds(env, name, value)
                except Unverifiable:
                    continue  # mode réel : contrôle réservé au PC simulé
                except Exception as exc:
                    failures.append(f"{name}={value!r} : {type(exc).__name__} {exc}")
                    continue
                if ok is not wanted:
                    label = "attendu" if wanted else "interdit"
                    failures.append(f"{label} : {name}={value!r}")
    return failures


# -- exécution ----------------------------------------------------------------------------


class _Speech:
    """Destination muette : garde ce qui aurait été dit."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        pass

    async def drain(self) -> None:
        pass


@dataclass
class Trial:
    task: str
    passed: bool
    failures: list[str]
    seconds: float
    first_s: float | None
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    path: str
    outcome: str
    answer: str
    calls: list[str] = field(default_factory=list)
    error: str = ""
    scripts: int = 0  # programmes exécutés (outil script)
    confirmations: int = 0  # « Tu confirmes ? » posés à l'utilisateur


def history_steps(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Les appels d'outils d'un échange passé, au format des messages mémorisés."""
    steps: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        ident = f"h{index}"
        steps.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ident,
                        "type": "function",
                        "function": {
                            "name": str(call["name"]),
                            "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        steps.append(
            {"role": "tool", "tool_call_id": ident, "content": str(call.get("result", "ok"))}
        )
    return steps


async def run_trial(
    task: Task,
    factory: AgentFactory,
    config: Config,
    *,
    live_ctx: ToolContext | None = None,
    answers: Callable[[float], Awaitable[str]] | None = None,
) -> Trial:
    """Un essai : état de départ neuf (sim) ou vrai PC (live), demande, contrôles."""
    with tempfile.TemporaryDirectory(prefix="mikata-eval-", ignore_cleanup_errors=True) as tmp:
        world = (
            None if live_ctx is not None else SimWorld(WorldSpec(**task.world), Path(tmp), config)
        )
        ctx = live_ctx if live_ctx is not None else world.ctx  # type: ignore[union-attr]
        agent = factory(ctx.config, ctx)
        if task.lessons and world is not None:
            from mikata.learn.playbook import Playbook

            agent.playbook = Playbook(Path(tmp) / "playbook.json")
            for lesson in task.lessons:
                agent.playbook.add(lesson["scope"], lesson["text"], "correction")
        for exchange in task.history:
            agent.memory.add(
                str(exchange["user"]),
                str(exchange.get("assistant", "C'est fait.")),
                history_steps(exchange.get("calls") or []),
            )
        speech = _Speech()
        start = time.monotonic()
        error = ""
        result: Any = None
        # Sur le vrai PC, personne n'est là pour dire oui : toute confirmation est refusée.
        reply = task.confirm or ("non" if "securite" in task.tags else "oui")
        if live_ctx is not None:
            reply = "non"

        async def confirm(timeout_s: float) -> str:
            return reply

        try:
            result = await agent.respond(
                task.request, speech, label=f"eval:{task.id}", listen=answers or confirm
            )
        except Exception as exc:  # une tâche qui plante compte comme un échec, pas comme un arrêt
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - start
        answer = (getattr(result, "answer", "") or " ".join(speech.said)).strip()
        raw_calls = getattr(result, "tool_calls", []) or []
        calls = [(call.name, dict(call.arguments)) for call in raw_calls]
        env = Env(ctx, answer, calls, world)
        failures = [f"erreur : {error}"] if error else grade(task, env)
        # Vrai PC : une fenêtre met un instant à apparaître, on relit l'état quelques fois.
        deadline = time.monotonic() + (LIVE_SETTLE_S if live_ctx is not None else 0.0)
        while failures and not error and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            failures = grade(task, env)
        usage = getattr(result, "usage", None)
        first = getattr(result, "first_token_at", None)
        trial = Trial(
            task=task.id,
            passed=not failures,
            failures=failures,
            seconds=round(seconds, 2),
            first_s=round(first - start, 2) if first else None,
            llm_calls=getattr(result, "llm_calls", 0),
            tokens_in=usage.prompt if usage else 0,
            tokens_out=usage.completion if usage else 0,
            cost_usd=round(getattr(result, "cost_usd", 0.0), 6),
            path=getattr(result, "path", ""),
            outcome=getattr(result, "outcome", "error"),
            answer=answer[:300],
            calls=[
                f"{name}({', '.join(f'{k}={repr(v)[:80]}' for k, v in args.items())})"
                for name, args in calls
            ],
            error=error,
            scripts=len(getattr(result, "scripts", []) or []),
            confirmations=sum("Tu confirmes" in said for said in speech.said),
        )
        if world is not None:
            await world.ctx.aclose()
        return trial


async def restore_live(task: Task, ctx: ToolContext, before: dict[str, Any]) -> None:
    """Remet le vrai PC dans l'état d'avant l'essai (volume, presse-papiers, fenêtres)."""
    if any(item.startswith("close:") for item in task.restore):
        await asyncio.sleep(1.5)  # la fenêtre à refermer doit avoir eu le temps de s'ouvrir
    for item in task.restore:
        try:
            if item == "volume" and ctx.audio is not None and before["volume"] is not None:
                await ctx.run_blocking(ctx.audio.set_volume, before["volume"])
            elif item == "clipboard" and ctx.desktop is not None:
                await ctx.run_blocking(ctx.desktop.clipboard_write, before["clipboard"])
            elif item.startswith("close:") and ctx.desktop is not None:
                name = item.split(":", 1)[1].casefold()
                known = before["windows"]
                for window in await ctx.run_blocking(ctx.desktop.windows):
                    title = f"{window.title} {window.process}".casefold()
                    if name in title and window.handle not in known:
                        await ctx.run_blocking(ctx.desktop.close_window, window)
            elif item.startswith("trash:"):
                path = ctx.require_guard().resolve_path(item.split(":", 1)[1])
                if path.exists():
                    ctx.require_guard().trash(path)
        except Exception as exc:  # la remise en état ne doit jamais arrêter le banc
            print(f"  (remise en état « {item} » impossible : {exc})")


async def snapshot_live(ctx: ToolContext) -> dict[str, Any]:
    state: dict[str, Any] = {"volume": None, "clipboard": "", "windows": set()}
    if ctx.audio is not None:
        try:
            state["volume"] = await ctx.run_blocking(ctx.audio.get_volume)
        except Exception as exc:  # aucune sortie audio (casque éteint) : on continue sans
            print(f"  (volume illisible, les tâches de son échoueront : {type(exc).__name__})")
    if ctx.desktop is not None:
        state["clipboard"] = await ctx.run_blocking(ctx.desktop.clipboard_read)
        state["windows"] = {w.handle for w in await ctx.run_blocking(ctx.desktop.windows)}
    return state


# -- rapport ------------------------------------------------------------------------------


@dataclass
class Summary:
    tasks: int
    trials: int
    passed_all: int  # tâches réussies à tous les essais (pass^k)
    passed_any: int  # tâches réussies au moins une fois (pass@k)
    median_s: float
    median_first_s: float | None
    median_llm_calls: float
    cost_usd: float

    @property
    def rate(self) -> float:
        return self.passed_all / self.tasks if self.tasks else 0.0


def summarize(trials: list[Trial]) -> Summary:
    by_task: dict[str, list[Trial]] = {}
    for trial in trials:
        by_task.setdefault(trial.task, []).append(trial)
    firsts = [t.first_s for t in trials if t.first_s is not None]
    return Summary(
        tasks=len(by_task),
        trials=len(trials),
        passed_all=sum(all(t.passed for t in ts) for ts in by_task.values()),
        passed_any=sum(any(t.passed for t in ts) for ts in by_task.values()),
        median_s=round(statistics.median(t.seconds for t in trials), 2) if trials else 0.0,
        median_first_s=round(statistics.median(firsts), 2) if firsts else None,
        median_llm_calls=statistics.median(t.llm_calls for t in trials) if trials else 0,
        cost_usd=round(sum(t.cost_usd for t in trials), 5),
    )


def format_report(tasks: list[Task], trials: list[Trial]) -> str:
    """Tableau lisible : une ligne par tâche, puis le résumé par étiquette et global."""
    by_task: dict[str, list[Trial]] = {}
    for trial in trials:
        by_task.setdefault(trial.task, []).append(trial)
    head = f"{'tâche':28} {'réussite':9} {'durée':>7} {'1er':>6} {'LLM':>4} {'prog':>4}"
    lines = [f"{head}  détail"]
    for task in tasks:
        runs = by_task.get(task.id)
        if not runs:
            continue
        ok = sum(t.passed for t in runs)
        first = [t.first_s for t in runs if t.first_s is not None]
        detail = "" if ok == len(runs) else next(t for t in runs if not t.passed).failures[0]
        lines.append(
            f"{task.id:28} {ok}/{len(runs):<7} "
            f"{statistics.median(t.seconds for t in runs):6.1f}s "
            f"{(f'{statistics.median(first):.1f}s' if first else '—'):>6} "
            f"{statistics.median(t.llm_calls for t in runs):4.0f} "
            f"{sum(t.scripts for t in runs):4d}  {detail[:70]}"
        )
    tags = sorted({tag for task in tasks for tag in task.tags})
    lines.append("")
    for tag in [*tags, "TOTAL"]:
        ids = {task.id for task in tasks if tag == "TOTAL" or tag in task.tags}
        subset = [t for t in trials if t.task in ids]
        if not subset:
            continue
        s = summarize(subset)
        first = f"{s.median_first_s:.1f} s" if s.median_first_s is not None else "—"
        lines.append(
            f"{tag:14} {s.passed_all}/{s.tasks} réussies à tous les essais "
            f"({s.rate:.0%}) ; médianes : {s.median_s:.1f} s, premier élément {first}, "
            f"{s.median_llm_calls:g} appel(s) LLM ; coût {s.cost_usd:.4f} $"
        )
    return "\n".join(lines)


def save_report(path: Path, mode: str, tasks: list[Task], trials: list[Trial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mode": mode,
        "summary": asdict(summarize(trials)),
        "trials": [asdict(trial) for trial in trials],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
