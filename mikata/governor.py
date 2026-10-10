"""Priorité du processus et mesure CPU (§10) : le jeu doit toujours passer devant.

Threads : openWakeWord crée ses sessions ONNX avec 1 thread ; le wake word a son executor
dédié à 1 worker (app.py).
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def lower_priority(process: Any = None) -> None:
    """Passe le processus en priorité « inférieure à la normale »."""
    import psutil

    proc = process or psutil.Process()
    try:
        if sys.platform == "win32":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif sys.platform == "darwin":
            proc.nice(10)  # macOS : pas de classe de priorité, simple « nice » UNIX
        else:
            proc.nice(10)
    except (OSError, psutil.Error) as exc:
        logger.warning("Impossible de baisser la priorité du processus : {}", exc)
        return
    logger.info("Priorité du processus : inférieure à la normale.")


class CpuMonitor:
    """CPU du processus depuis le dernier appel : % d'un cœur et % de la machine entière."""

    def __init__(self, process: Any = None, cpu_count: int | None = None) -> None:
        import psutil

        self._proc = process or psutil.Process()
        self._cpus = cpu_count or psutil.cpu_count() or 1
        self._proc.cpu_percent(None)  # amorce la mesure

    def sample(self) -> tuple[float, float]:
        one_core = float(self._proc.cpu_percent(None))
        return one_core, one_core / self._cpus
