"""Raccourci clavier de secours (§4), par défaut Ctrl+Alt+M : même effet que dire MIKATA.

pynput tourne dans son propre thread (hook clavier Windows) ; il est importé paresseusement
(installé seulement sous Windows) et mocké dans les tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger


class HotkeyListener:
    def __init__(self, combo: str) -> None:
        self.combo = combo
        self._listener: Any = None

    def start(self, on_trigger: Callable[[], None]) -> None:
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
