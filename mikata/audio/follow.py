"""La sortie suit celle de Windows (ajout phase 15f).

Avec `audio.output_device: default`, MIKATA ouvrait au démarrage la sortie par défaut du
moment, puis n'en bougeait plus : un casque Bluetooth allumé après coup devenait la sortie de
Windows, mais MIKATA continuait de parler dans l'écran. Désormais la boucle de l'application
relève toutes les `POLL_S` secondes le nom de la sortie par défaut (Core Audio, via pycaw) et
bascule quand il change (voir `App._follow_output`).

`DefaultOutputWatch` décide seule, sans rien ouvrir : testable sous Linux.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

POLL_S = 2.0
STABLE_POLLS = 2  # un casque Bluetooth qui se connecte peut hésiter quelques secondes


def same_device(a: str, b: str) -> bool:
    """Même périphérique ? MME tronque les noms à 31 caractères : préfixe accepté."""
    a, b = a.casefold().strip(), b.casefold().strip()
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 25 and (a.startswith(b) or b.startswith(a)))


class DefaultOutputWatch:
    """Dit quand changer de sortie : celle de Windows diffère de la sortie en cours, et
    reste la même sur `stable` relevés de suite."""

    def __init__(self, stable: int = STABLE_POLLS) -> None:
        self.stable = stable
        self._seen: str | None = None
        self._count = 0

    def update(self, default: str | None, current: str) -> str | None:
        if not default or same_device(default, current):
            self._seen, self._count = None, 0
            return None
        if self._seen is not None and same_device(default, self._seen):
            self._count += 1
        else:
            self._seen, self._count = default, 1
        if self._count >= self.stable:
            self._seen, self._count = None, 0
            return default
        return None


def _com_init() -> None:
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception as exc:  # pycaw dira la vraie erreur au premier relevé
        logger.debug("Initialisation COM (suivi de sortie) : {}", exc)


def _default_name() -> str | None:
    from pycaw.utils import AudioUtilities

    speakers = AudioUtilities.GetSpeakers()
    name = getattr(speakers, "FriendlyName", None) if speakers is not None else None
    return str(name) if name else None


class WindowsDefaultOutput:
    """Nom de la sortie par défaut de Windows, relevé sur un thread COM dédié."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mikata-sortie", initializer=_com_init
        )

    async def name(self) -> str | None:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, _default_name)
        except Exception as exc:  # aucune sortie active, Core Audio indisponible...
            logger.debug("Sortie par défaut illisible : {}", exc)
            return None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def default_output_source() -> WindowsDefaultOutput | None:
    """Le relevé de la sortie par défaut, sous Windows seulement (ailleurs : None)."""
    return WindowsDefaultOutput() if sys.platform == "win32" else None
