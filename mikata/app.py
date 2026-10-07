"""Orchestrateur et machine à états (docs/ARCHITECTURE.md §3).

Phase 0 : démarrage, icône, attente des événements et arrêt propre. Les transitions
IDLE → LISTENING → ... arrivent avec les phases 1 à 6.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from loguru import logger

from mikata.config import Config
from mikata.events import BusEvent, EventBus, Shutdown


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"


class StatusIcon(Protocol):
    """Icône de la zone de notification (implémentée par `ui.tray.TrayIcon`)."""

    def start(self, on_quit: Callable[[], None]) -> None: ...
    def set_state(self, state: State) -> None: ...
    def stop(self) -> None: ...


class App:
    def __init__(
        self, config: Config, tray: StatusIcon | None = None, bus: EventBus | None = None
    ) -> None:
        self.config = config
        self.bus = bus or EventBus()
        self._tray = tray
        self.state = State.IDLE

    def request_shutdown(self, reason: str) -> None:
        """Demande l'arrêt ; appelable depuis n'importe quel thread."""
        self.bus.publish_threadsafe(Shutdown(reason=reason))

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            logger.debug("État : {} -> {}", self.state, state)
        self.state = state
        if self._tray is not None:
            self._tray.set_state(state)

    async def run(self) -> None:
        """Boucle principale : rend la main après un `Shutdown` ou une annulation (Ctrl+C)."""
        self.bus.bind_loop(asyncio.get_running_loop())
        try:
            if self._tray is not None:
                self._tray.start(on_quit=lambda: self.request_shutdown("menu Quitter"))
            self._set_state(State.IDLE)
            logger.info("MIKATA démarré, en attente.")
            while True:
                event = await self.bus.get()
                if isinstance(event, Shutdown):
                    logger.info("Arrêt demandé ({}).", event.reason or "sans motif")
                    break
                await self._handle(event)
        finally:
            if self._tray is not None:
                self._tray.stop()
            logger.info("MIKATA arrêté proprement.")

    async def _handle(self, event: BusEvent) -> None:
        logger.debug("Événement non géré en phase 0 : {}", type(event).__name__)
