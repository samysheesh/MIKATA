"""Logs avec loguru : console (si elle existe) + fichier tournant (docs/ARCHITECTURE.md §10)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

from mikata.config import LoggingConfig

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {thread.name} | {name}:{line} - {message}"
)


class _InterceptHandler(logging.Handler):
    """Redirige le module `logging` standard (bibliothèques tierces) vers loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(cfg: LoggingConfig, base_dir: Path) -> Path:
    """(Re)configure loguru. Idempotent : peut être rappelée après le chargement de la config.

    Retourne le chemin absolu du fichier de log.
    """
    log_file = cfg.file.expanduser()
    if not log_file.is_absolute():
        log_file = base_dir / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    # Sous pythonw.exe (démarrage automatique), il n'y a pas de console : sys.stderr vaut None.
    if sys.stderr is not None:
        logger.add(sys.stderr, level=cfg.level, format=LOG_FORMAT, backtrace=False)
    logger.add(
        log_file,
        level=cfg.level,
        format=LOG_FORMAT,
        rotation=int(cfg.rotation_mb * 1024 * 1024),
        retention=cfg.retention_files,
        encoding="utf-8",
        enqueue=True,  # écriture depuis plusieurs threads (tray, executors) sans contention
        backtrace=False,
    )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    return log_file


def shutdown_logging() -> None:
    """Vide la file d'écriture et ferme les fichiers de log."""
    logger.complete()
    logger.remove()
