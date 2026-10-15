"""Traces des tours qui ont agi (docs/AGENT_CORE.md §3.11) : la matière de l'apprentissage.

Une trace = la demande, ce qui a été fait (programmes et outils directs), ce qui a échoué
puis réussi, et la réponse. Les arguments qui dépendent d'un contenu lu ailleurs ne sont pas
gardés : une trace ne doit pas devenir un canal par lequel un texte piégé revient plus tard.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

TRACES_DIR = "data/traces"
CODE_MAX_CHARS = 2000


class ScriptTrace(BaseModel):
    code: str
    ok: bool
    kind: str = ""
    error: str = ""
    calls: list[str] = Field(default_factory=list)  # « outil(arguments) », expurgés
    tainted: bool = False  # le programme a lu un contenu non fiable


class TurnTrace(BaseModel):
    id: str
    at: str
    request: str
    answer: str = ""
    outcome: str = "ok"
    llm_calls: int = 0
    seconds: float = 0.0
    confirmations: int = 0
    scripts: list[ScriptTrace] = Field(default_factory=list)
    direct: list[str] = Field(default_factory=list)  # outils directs appelés
    failures: list[str] = Field(default_factory=list)  # outils directs en échec
    tainted: bool = False  # le tour a lu un contenu non fiable
    lessons: list[str] = Field(default_factory=list)  # leçons présentes dans le contexte

    @property
    def acted(self) -> bool:
        return bool(self.scripts or self.direct)

    @property
    def recovered(self) -> bool:
        """Un échec, puis une réussite dans le même tour : il y a quelque chose à retenir."""
        failed = self.failures or [s for s in self.scripts if not s.ok]
        succeeded = self.outcome == "ok" and (
            any(s.ok for s in self.scripts) or len(self.direct) > len(self.failures)
        )
        return bool(failed) and succeeded


def _call_text(sdk: str, args: dict[str, Any], hidden: bool) -> str:
    if hidden:
        return f"{sdk}(…)"
    rendered = ", ".join(f"{key}={str(value)[:80]!r}" for key, value in args.items())
    return f"{sdk}({rendered})"


def trace_from_result(
    request: str, result: Any, seconds: float, lessons: list[str], tainted: bool = False
) -> TurnTrace:
    """Trace d'un tour, depuis le `TurnResult` de la boucle agent."""
    codes = [
        str(call.arguments.get("code", ""))[:CODE_MAX_CHARS]
        for call in getattr(result, "tool_calls", [])
        if call.name == "script"
    ]
    scripts: list[ScriptTrace] = []
    for index, outcome in enumerate(getattr(result, "scripts", []) or []):
        read_untrusted = outcome.read_trust.label == "non fiable"
        scripts.append(
            ScriptTrace(
                code=codes[index] if index < len(codes) else "",
                ok=outcome.ok,
                kind=outcome.kind,
                error="" if read_untrusted else outcome.error[:300],
                calls=[
                    _call_text(r.tool, r.args, r.arg_trust == "non fiable") for r in outcome.records
                ],
                tainted=read_untrusted,
            )
        )
    direct = list(getattr(result, "direct_results", []) or [])
    return TurnTrace(
        id=uuid.uuid4().hex[:12],
        at=datetime.now().isoformat(timespec="seconds"),
        request=request[:300],
        answer=str(getattr(result, "answer", ""))[:300],
        outcome=str(getattr(result, "outcome", "ok")),
        llm_calls=int(getattr(result, "llm_calls", 0)),
        seconds=round(seconds, 2),
        confirmations=int(getattr(result, "confirmations", 0)),
        scripts=scripts,
        direct=[r.name for r in direct],
        failures=[f"{r.name} : {r.content[:160]}" for r in direct if not r.ok]
        + [f"sans effet : {text[:160]}" for text in getattr(result, "no_effect", []) or []],
        tainted=tainted or any(script.tainted for script in scripts),
        lessons=lessons,
    )


class TraceStore:
    """Traces en fichiers JSON, un dossier par jour ; les vieux dossiers vont à la corbeille."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, trace: TurnTrace) -> Path | None:
        folder = self.root / trace.at[:10]
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{trace.at[11:19].replace(':', '')}-{trace.id}.json"
            path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
            return path
        except OSError as exc:
            logger.warning("Trace non écrite : {}", exc)
            return None

    def recent(self, limit: int = 20) -> list[TurnTrace]:
        files = sorted(self.root.glob("*/*.json"), reverse=True)[:limit]
        traces: list[TurnTrace] = []
        for file in files:
            try:
                traces.append(
                    TurnTrace.model_validate(json.loads(file.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError):
                continue
        return traces

    def purge(self, days: float, trash: Any = None, now: float | None = None) -> int:
        """Dossiers de jour plus vieux que `days` : à la corbeille (jamais supprimés)."""
        from mikata.maintenance import trash_entries_older_than

        return trash_entries_older_than(self.root, days, trash=trash, now=now or time.time())
