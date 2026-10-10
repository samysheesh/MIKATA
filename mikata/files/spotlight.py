"""Backend Spotlight (macOS) de l'index des fichiers (phase 12d).

Spotlight remplit et rafraîchit le cache SQLite/FTS5 de `mikata.files.index` :

- `spotlight_query` interroge l'index Spotlight du système via `mdfind`
  (sous-processus, jamais au chargement) ;
- `spotlight_snapshot` fait le dump initial (chemins + mtime) ;
- `SpotlightWatcher` rejoue les mises à jour en direct via FSEvents/watchdog
  (observateur lazy, no-op si absent).

Le consommateur SQLite/FTS5 reste identique : Spotlight ne fait que remplir /
rafraîchir les lignes. Les modules propres à macOS (watchdog) sont importés
paresseusement dans les fonctions, jamais au chargement.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

_SPOTLIGHT_LIMIT = 200


def spotlight_query(query: str, max_results: int = _SPOTLIGHT_LIMIT) -> list[str]:
    """Chemins renvoyés par `mdfind` pour `query` (darwin uniquement, lazy)."""
    if sys.platform != "darwin":
        raise NotImplementedError("Spotlight (mdfind) n'existe que sous macOS.")
    wanted = query.strip()
    if not wanted:
        return []
    try:
        proc = subprocess.run(
            ["mdfind", "-interpret", wanted],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise NotImplementedError("mdfind introuvable (macOS requis).") from exc
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line][:max_results]


def spotlight_snapshot(query: str = "kind:any") -> list[tuple[str, float]]:
    """(chemin, mtime) de tous les fichiers vus par Spotlight (dump initial darwin)."""
    paths = spotlight_query(query)
    rows: list[tuple[str, float]] = []
    for path in paths:
        try:
            rows.append((path, Path(path).stat().st_mtime))
        except OSError:
            rows.append((path, 0.0))
    return rows


class SpotlightWatcher:
    """Mises à jour en direct via FSEvents/watchdog (lazy, no-op si absent).

    `on_change` reçoit des chemins modifiés ; l'appelant les réindexe
    (upsert SQLite). Sans `watchdog` : journalise une fois, ne fait rien.
    """

    def __init__(self, roots: list[str], on_change: Any = None) -> None:
        self._roots = roots
        self._on_change = on_change
        self._observer: Any = None

    def start(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore[import-not-found]
            from watchdog.observers import Observer  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "Surveillance FSEvents inactive : watchdog non installé "
                "(mdfind reste disponible à la demande)."
            )
            return

        handler = self._make_handler(FileSystemEventHandler)

        from loguru import logger as _logger

        observer = Observer()
        for root in self._roots:
            try:
                observer.schedule(handler, root, recursive=True)
            except Exception as exc:
                _logger.warning("Surveillance de {} impossible : {}", root, exc)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("Surveillance FSEvents active sur {} racine(s).", len(self._roots))

    def _make_handler(self, base: Any) -> Any:
        callback = self._on_change

        class _Handler(base):  # type: ignore[valid-type,misc]
            def on_any_event(self, event: Any) -> None:
                if callback is not None and not getattr(event, "is_directory", False):
                    with contextlib.suppress(Exception):
                        callback(str(event.src_path))

        return _Handler()

    def stop(self) -> None:
        observer, self._observer = self._observer, None
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=2)
            except Exception:
                pass
