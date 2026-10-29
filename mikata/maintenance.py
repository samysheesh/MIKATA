"""Purge des fichiers qui grossissent quand MIKATA tourne h24 (ajout phase 7, §11).

Au démarrage puis toutes les 24 h (thread disque de l'orchestrateur) :
- `data/wake_logs/` : audios de détection de plus de `wake.log_days` jours ;
- `data/wakeword/misses/` (phase 9) : ratés du mot d'activation de plus de `wake.miss_days`
  jours ;
- `data/backups/` : sauvegardes avant écrasement de plus de `guard.backup_days` jours (le
  garde-fou ne purgeait qu'au moment d'une nouvelle sauvegarde).
Déjà bornés ailleurs : `data/logs/` et `data/actions.log` (rotation 5 Mo × 5, archives à la
corbeille), l'historique (plafond de tokens), `data/negatives/` (`wake.negatives_max_hours`).
Tout part à la corbeille (send2trash), jamais de suppression définitive.
Sur macOS, send2trash utilise la corbeille native du Finder : la purge reste inchangée.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from mikata.config import Config
from mikata.wake.recordings import MISSES_DIR, purge_older_than

WAKE_LOGS_DIR = "data/wake_logs"
TRACES_DIR = "data/traces"  # même valeur que mikata.learn.trace.TRACES_DIR
BACKUPS_DIR = "data/backups"  # = agent.guard.BACKUP_DIR (import paresseux du garde-fou évité)
INTERVAL_S = 24 * 3600


def trash_entries_older_than(
    directory: Path,
    days: float,
    now: float | None = None,
    trash: Callable[[str], None] | None = None,
) -> int:
    """Envoie à la corbeille les entrées directes de `directory` (fichiers ou dossiers) plus
    vieilles que `days` jours."""
    if not directory.is_dir():
        return 0
    if trash is None:
        from send2trash import send2trash as trash
    limit = (time.time() if now is None else now) - days * 86_400
    count = 0
    for entry in sorted(directory.iterdir()):
        try:
            if entry.stat().st_mtime < limit:
                trash(str(entry))
                count += 1
        except OSError as exc:
            logger.warning("Purge impossible pour {} : {}", entry, exc)
    return count


def run_maintenance(config: Config, trash: Callable[[str], None] | None = None) -> dict[str, int]:
    """Purge tout ce qui a dépassé sa durée de conservation ; renvoie le nombre par dossier."""
    return {
        WAKE_LOGS_DIR: purge_older_than(
            config.resolve(WAKE_LOGS_DIR), config.wake.log_days, trash=trash
        ),
        MISSES_DIR: purge_older_than(
            config.resolve(MISSES_DIR), config.wake.miss_days, trash=trash
        ),
        BACKUPS_DIR: trash_entries_older_than(
            config.resolve(BACKUPS_DIR), config.guard.backup_days, trash=trash
        ),
        # Phase 14e : traces des tours (dossiers de jour -> corbeille).
        TRACES_DIR: trash_entries_older_than(
            config.resolve(TRACES_DIR), config.learning.trace_days, trash=trash
        ),
    }
