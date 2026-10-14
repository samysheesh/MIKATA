"""État ambiant (phase 12a) : ce qui se passe sur le PC, en RAM, jamais loggé.

Par défaut (`uia.state_poll_s` = 0), l'état est lu **à la demande**, quand l'outil
`uia_state` le demande, dans le fil COM des outils, et gardé `max_age_s` secondes. Rien ne
tourne au repos : une relève de Discord coûte 140 à 200 ms de CPU à MIKATA (724 nœuds,
mesuré le 14/10), soit 7 à 10 % d'un cœur quand elle était faite toutes les 2 s.

Avec `uia.state_poll_s` > 0, un thread dédié la refait en fond toutes les N secondes ;
pendant qu'un outil agit, le registre le met en pause (`pause_state`) pour ne pas lire un
écran en train de changer. Le contenu n'est jamais écrit sur disque ni dans les journaux :
ce qui est à l'écran reste privé.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any

# Le premier usage de `uiautomation` crée son client COM unique, et celui-ci génère des
# modules Python dans comtypes/gen (enveloppes de UIAutomationCore et de stdole). Cette
# création n'est pas protégée : le fil de l'état ambiant et celui des outils peuvent la
# lancer ensemble au premier démarrage ; l'un lit alors un module encore à moitié écrit
# (« cannot import name 'IEnumVARIANT' ») et le cache reste cassé jusqu'à ce qu'on le vide.
# Elle se fait donc une seule fois, sous verrou, avant tout autre usage.
_UIA_READY = threading.Lock()
_uia_ready = False


def load_uiautomation() -> Any:
    """Le module `uiautomation`, son client COM créé (et le cache comtypes généré) une
    seule fois pour tout le processus. COM doit être initialisé dans le fil appelant."""
    global _uia_ready
    with _UIA_READY:
        import uiautomation

        if not _uia_ready:
            uiautomation.GetRootControl()  # crée le client COM : génère le cache, en entier
            _uia_ready = True
    return uiautomation


def load_uia_interfaces() -> Any:
    """Les interfaces COM d'UI Automation (`comtypes.gen.UIAutomationClient`), générées au
    besoin par `load_uiautomation`."""
    load_uiautomation()
    from comtypes.gen import UIAutomationClient

    return UIAutomationClient


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
    def __init__(
        self,
        poll_s: float,
        snapshot: Callable[[], dict[str, str]],
        max_age_s: float = 5.0,
    ) -> None:
        self._poll_s = poll_s
        self._snapshot = snapshot
        self._max_age_s = max_age_s
        self._read_at = float("-inf")  # dernière lecture à la demande (time.monotonic)
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

    def _read(self) -> None:
        try:
            data = self._snapshot() or {}
        except Exception:  # l'état ambiant est un plus : jamais une erreur fatale
            data = {}
        if isinstance(data, dict):
            self._data = data

    def _run(self) -> None:
        _com_init()
        while not self._stop.is_set():
            if self._pauses == 0:
                self._read()
            self._stop.wait(self._poll_s)

    def summary(self) -> str:
        """Texte court pour le LLM : chaque élément non vide, séparé par un point. Sans
        thread de fond, l'état est relu ici s'il a plus de `max_age_s` secondes : appel
        bloquant, à faire depuis un fil où COM est initialisé (`ToolContext.run_blocking`)."""
        if self._thread is None and self._poll_s <= 0:
            now = time.monotonic()
            if now - self._read_at > self._max_age_s:
                self._read()
                self._read_at = now
        return " ".join(text.strip() for text in self._data.values() if text.strip())
