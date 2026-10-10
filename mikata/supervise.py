"""Relance de MIKATA s'il plante (ajout phase 7, démarrage automatique §10).

- win32 : la tâche planifiée lance `pythonw.exe -m mikata --supervise`. Ce petit
  processus démarre MIKATA dans un processus enfant et le relance s'il s'arrête sur
  une erreur : l'option « redémarrer en cas d'échec » du Planificateur de tâches ne
  couvre qu'un échec au lancement, pas un plantage en cours de route.
- darwin : compatible launchd — le superviseur tourne au premier plan (pas de
  pythonw, pas de DETACHED_PROCESS) sous un agent `~/Library/LaunchAgents`
  (voir scripts/install_launchd.py : RunAtLoad + KeepAlive) ; launchd relance
  déjà l'agent, la boucle ci-dessous couvre les plantages en cours de route.
- « Quitter » (code 0) ou une config invalide (code 2, rien ne changerait) : pas de relance.
- Sinon (erreur, GPU pas encore prêt juste après l'ouverture de session...) : relance après
  `delay_s`, au plus `max_restarts` fois par heure, puis abandon (pas de boucle infinie).
Journal à part (data/logs/supervise.log) : deux processus ne doivent pas tourner le même
fichier de log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from loguru import logger

NO_RESTART_CODES = (0, 2)  # EXIT_OK, EXIT_CONFIG (cf. __main__)
RESTART_CODE = 75  # EXIT_RESTART : « Redémarrer » demandé dans l'interface (cf. __main__)
SUPERVISED_ENV = "MIKATA_SUPERVISED"  # posée dans l'enfant : il sait qu'on le relancera
LOG_NAME = "supervise.log"


def supervise(
    command: Sequence[str],
    *,
    run: Callable[[Sequence[str]], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    max_restarts: int = 5,
    window_s: float = 3600,
    delay_s: float = 10,
) -> int:
    """Lance `command` jusqu'à un arrêt voulu ; renvoie son dernier code de sortie."""
    if run is None:

        def run(cmd: Sequence[str]) -> int:
            return subprocess.run(list(cmd), env={**os.environ, SUPERVISED_ENV: "1"}).returncode

    restarts: deque[float] = deque()
    while True:
        started = clock()
        code = run(command)
        if code == RESTART_CODE:  # voulu par l'utilisateur : tout de suite, sans compter
            logger.info("Redémarrage demandé : MIKATA est relancé.")
            continue
        if code in NO_RESTART_CODES:
            logger.info("MIKATA arrêté (code {}) : pas de relance.", code)
            return code
        now = clock()
        while restarts and now - restarts[0] > window_s:
            restarts.popleft()
        if len(restarts) >= max_restarts:
            logger.error(
                "MIKATA s'est arrêté {} fois en moins de {:.0f} min (dernier code {}) : abandon. "
                "Voir data/logs/mikata.log.",
                max_restarts + 1,
                window_s / 60,
                code,
            )
            return code
        restarts.append(now)
        logger.warning(
            "MIKATA arrêté sur une erreur (code {}, après {:.0f} s) : relance dans {:.0f} s.",
            code,
            now - started,
            delay_s,
        )
        sleep(delay_s)


def child_command(argv: Sequence[str]) -> list[str]:
    """Même interpréteur (pythonw.exe : pas de console), mêmes options, sans --supervise."""
    return [sys.executable, "-m", "mikata", *(a for a in argv if a != "--supervise")]


def relaunch_self(argv: Sequence[str]) -> None:
    """Redémarrage sans supervision : nouveau processus détaché, puis on quitte.

    win32 : sans console (DETACHED_PROCESS + CREATE_NO_WINDOW). darwin : premier
    plan compatible launchd (pas de détachement : launchd gère le cycle de vie,
    un double détachement casserait KeepAlive) ; ailleurs : simple Popen.
    """
    flags = 0
    if sys.platform == "win32":
        flags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined,unused-ignore]
            | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined,unused-ignore]
            | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined,unused-ignore]
        )
    subprocess.Popen(child_command(argv), creationflags=flags, close_fds=True)


def setup_supervisor_log(log_dir: Path) -> None:
    from mikata.logging_setup import trash_old_logs

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    if sys.stderr is not None:
        logger.add(sys.stderr, level="INFO")
    logger.add(
        log_dir / LOG_NAME,
        level="INFO",
        rotation=1024 * 1024,
        retention=trash_old_logs(2),
        encoding="utf-8",
    )
