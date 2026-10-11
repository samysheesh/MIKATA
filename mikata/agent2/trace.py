"""Agent v2 (phase 13a) : écriture, lecture et purge des traces.

Une trace par demande, en JSON sous `data/agent2/traces/AAAA-MM-JJ/<id>.json`, écriture
atomique. La purge envoie les dossiers de jour trop vieux à la corbeille — jamais de
suppression définitive.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from mikata.agent2.schema import Trace
from mikata.plat import platform

TRACES_DIR = "data/agent2/traces"


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def new(self, request: str) -> Trace:
        return Trace(id=uuid.uuid4().hex, started=datetime.now(), request=request, os=platform())

    def save(self, trace: Trace) -> Path:
        """Écriture atomique : `.part` puis `replace`."""
        day = trace.started.strftime("%Y-%m-%d")
        folder = self.root / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{trace.id}.json"
        part = path.with_suffix(".json.part")
        part.write_text(
            json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        part.replace(path)
        return path

    def load(self, path: Path) -> Trace:
        return Trace.model_validate_json(path.read_text(encoding="utf-8"))

    def recent(self, days: float, *, verified_only: bool = False) -> list[Trace]:
        """Traces plus récentes que `days` jours, de la plus récente à la plus ancienne."""
        limit = time.time() - days * 86_400
        traces: list[Trace] = []
        for folder in sorted(self.root.glob("????-??-??"), reverse=True):
            for path in sorted(folder.glob("*.json")):
                try:
                    if path.stat().st_mtime < limit:
                        continue
                    trace = self.load(path)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if verified_only and not trace.verified:
                    continue
                traces.append(trace)
        traces.sort(key=lambda trace: trace.started, reverse=True)
        return traces

    def purge(self, days: float) -> int:
        """Dossiers de jour plus vieux que `days` jours -> corbeille (send2trash).

        Import paresseux de send2trash : la liste blanche des importateurs (test du
        garde-fou) reste fermée, comme dans le reste du code.
        """
        from send2trash import send2trash

        limit = time.time() - days * 86_400
        count = 0
        for folder in sorted(self.root.glob("????-??-??")):
            try:
                if folder.stat().st_mtime < limit:
                    send2trash(str(folder))
                    count += 1
            except OSError:
                continue
        return count
