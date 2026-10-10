"""Logs avec loguru : console (si elle existe) + fichier tournant (docs/ARCHITECTURE.md §10)."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from send2trash import send2trash

from mikata.config import LoggingConfig

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {thread.name} | {name}:{line} - {message}"
)
ACTION_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {message}"


def _is_action(record: Any) -> bool:
    """Lignes du journal des actions (`agent.guard.log_action`), à part du log général."""
    return bool(record["extra"].get("action"))


def _not_action(record: Any) -> bool:
    return not _is_action(record)


class _InterceptHandler(logging.Handler):
    """Redirige le module `logging` standard (bibliothèques tierces) vers loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def trash_old_logs(keep: int) -> Callable[[list[str]], None]:
    """Rétention loguru : garde les `keep` archives les plus récentes, envoie les autres à la
    corbeille (jamais de suppression définitive, cf. CLAUDE.md)."""

    def retention(files: list[str]) -> None:
        for path in sorted(files, key=lambda f: (-os.stat(f).st_mtime, f))[keep:]:
            try:
                send2trash(path)
            except OSError as exc:
                print(f"Impossible d'envoyer {path} à la corbeille : {exc}", file=sys.stderr)

    return retention


def _absolute(path: Path, base_dir: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else base_dir / path


def _enqueue_context() -> str | None:
    """Contexte multiprocessing des files de loguru (`enqueue=True`).

    Sous macOS, le contexte par défaut (« spawn ») crée des sémaphores POSIX **nommés**
    par handler, détachés seulement à la sortie normale (atexit). Si le processus se
    termine autrement — l'icône Cocoa de pystray pouvait aborter en SIGTRAP à l'arrêt —,
    le `resource_tracker` signale alors « leaked semaphore objects ». Le contexte
    « fork » crée des sémaphores anonymes (jamais enregistrés auprès du tracker) : plus
    rien à détacher, sans perte (la file sert seulement à découpler la journalisation,
    jamais à traverser un fork). Windows garde son défaut (ses sémaphores ne sont pas
    non plus enregistrés) ; Linux est déjà en « fork ».
    """
    if sys.platform == "win32":
        return None
    import multiprocessing

    return "fork" if "fork" in multiprocessing.get_all_start_methods() else None


def setup_logging(cfg: LoggingConfig, base_dir: Path) -> Path:
    """(Re)configure loguru. Idempotent : peut être rappelée après le chargement de la config.

    Retourne le chemin absolu du fichier de log. Les actions des outils (phase 5) vont dans
    un fichier à part, `cfg.actions_file` (data/actions.log), même rotation.
    """
    log_file = _absolute(cfg.file, base_dir)
    actions_file = _absolute(cfg.actions_file, base_dir)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    actions_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    # Sous pythonw.exe (démarrage automatique Windows), il n'y a pas de console :
    # sys.stderr vaut None. Sous launchd (darwin), le processus est au premier plan
    # (compatible launchd, pas de démon détaché) : stderr existe mais peut être
    # redirigé vers les journaux launchd (StandardOutPath/StandardErrorPath) — on
    # l'ajoute quand il est présent, comme ailleurs.
    if sys.stderr is not None:
        logger.add(
            sys.stderr, level=cfg.level, format=LOG_FORMAT, backtrace=False, filter=_not_action
        )
    common: dict[str, Any] = {
        "rotation": int(cfg.rotation_mb * 1024 * 1024),
        "retention": trash_old_logs(cfg.retention_files),
        "encoding": "utf-8",
        "enqueue": True,  # écriture depuis plusieurs threads (tray, executors) sans contention
        "context": _enqueue_context(),  # sémaphores anonymes hors Windows (cf. _enqueue_context)
        "backtrace": False,
    }
    logger.add(log_file, level=cfg.level, format=LOG_FORMAT, filter=_not_action, **common)
    logger.add(actions_file, level="INFO", format=ACTION_FORMAT, filter=_is_action, **common)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    return log_file


def shutdown_logging() -> None:
    """Vide la file d'écriture et ferme les fichiers de log."""
    logger.complete()
    logger.remove()
