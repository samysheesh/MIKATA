"""Raccourci clavier de secours (§4), par défaut Ctrl+Alt+M : même effet que dire MIKATA.

pynput tourne dans son propre thread (hook clavier) ; il est importé paresseusement
et mocké dans les tests.

macOS (darwin) — autorisation « Surveillance de l'entrée » (Input Monitoring) requise :
    Réglages Système → Confidentialité et sécurité → Surveillance de l'entrée →
    cocher le terminal / l'application qui lance MIKATA, puis relancer MIKATA.
Sans cette autorisation, `GlobalHotKeys.start()` lève (ou le raccourci reste muet
selon la version de pynput) : accordez l'autorisation et redémarrez. Si l'autorisation
est révoquée après coup, même flux : ré-accorder dans Réglages puis relancer le
processus (re-grant flow). Le combo par défaut Ctrl+Alt+M est conservé, mais notez
qu'il peut entrer en collision avec des raccourcis d'applications macOS (les apps
au premier plan ont priorité sur le hook global).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

#: Combo par défaut (config `wake.hotkey`). Conservé sur macOS malgré le risque de
#: collision avec des raccourcis applicatifs (voir docstring du module).
DEFAULT_COMBO = "<ctrl>+<alt>+m"


class HotkeyListener:
    def __init__(self, combo: str) -> None:
        self.combo = combo
        self._listener: Any = None

    def start(self, on_trigger: Callable[[], None]) -> None:
        # pynput : déclaré pour win32 et darwin dans pyproject.toml (absent de la CI Linux).
        from pynput import keyboard

        try:
            self._listener = keyboard.GlobalHotKeys({self.combo: on_trigger})
        except ValueError as exc:
            raise ValueError(f"wake.hotkey = « {self.combo} » invalide : {exc}") from exc
        self._listener.start()
        logger.info("Raccourci clavier actif : {}", self.combo)

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
