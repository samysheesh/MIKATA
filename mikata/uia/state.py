"""Cache d'état ambiant (phase 12a) : ce qui se passe sur le PC, en RAM, jamais loggé.

Un thread dédié (pas celui des outils, pour ne jamais bloquer une action) relit l'état
toutes les `uia.state_poll_s` secondes ; pendant qu'un outil agit, le registre met le
poller en pause (`pause_state`) pour ne pas lire un écran en train de changer. Le contenu
n'est jamais écrit sur disque ni dans les journaux : ce qui est à l'écran reste privé.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable


def _com_init() -> None:
    """COM initialisé dans le thread du poller (Windows) : uiautomation l'exige, sinon
    « CoInitialize n'a pas été appelé » à chaque lecture. Sur darwin : rien à faire
    (AXUIElement n'a pas d'appartement COM). Ailleurs : rien à faire."""
    if sys.platform != "win32":
        return
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception:
        try:
            import ctypes

            ctypes.windll.ole32.CoInitializeEx(None, 0)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Pas de CoUninitialize à la sortie : avec des proxies COM encore vivants, comtypes
    # crashe (0x80010108, RPC déconnecté). Le thread daemon meurt et Windows détruit
    # l'appartement COM lui-même : rien ne fuit.


class AmbientState:
    def __init__(self, poll_s: float, snapshot: Callable[[], dict[str, str]]) -> None:
        self._poll_s = poll_s
        self._snapshot = snapshot
        self._data: dict[str, str] = {}
        self._pauses = 0  # compteur : deux outils en parallèle se relaient sans se gêner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._poll_s <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mikata-uia-state", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def pause(self, paused: bool) -> None:
        self._pauses += 1 if paused else -1
        if self._pauses < 0:  # défensif : un False sans True ne doit pas casser le compteur
            self._pauses = 0

    def _run(self) -> None:
        _com_init()
        while not self._stop.is_set():
            if self._pauses == 0:
                try:
                    data = self._snapshot() or {}
                except Exception:  # l'état ambiant est un plus : jamais une erreur fatale
                    data = {}
                if isinstance(data, dict):
                    self._data = data
            self._stop.wait(self._poll_s)

    def summary(self) -> str:
        """Texte court pour le LLM : chaque élément non vide, séparé par un point."""
        return " ".join(text.strip() for text in self._data.values() if text.strip())
