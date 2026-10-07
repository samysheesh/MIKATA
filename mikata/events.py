"""Événements de l'orchestrateur et bus minimal (docs/ARCHITECTURE.md §3).

Un seul consommateur : l'orchestrateur (`app.py`), seule pièce qui change d'état.
Les producteurs publient depuis la boucle asyncio (`publish`) ou depuis un autre thread
(`publish_threadsafe` : icône, raccourci clavier, thread audio).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from loguru import logger

from mikata.stt.base import Transcript


@dataclass(frozen=True)
class Event:
    # Horodatage monotone de création, pour les mesures de latence par étape.
    ts: float = field(default_factory=time.monotonic, kw_only=True)


@dataclass(frozen=True)
class WakeDetected(Event):
    score: float
    source: Literal["wake_word", "hotkey"] = "wake_word"


@dataclass(frozen=True, eq=False)
class SpeechEnded(Event):
    audio: np.ndarray  # int16 mono 16 kHz, pré-roll inclus


@dataclass(frozen=True)
class LLMDelta(Event):
    text: str


@dataclass(frozen=True)
class ToolCall(Event):
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class SpeakDone(Event):
    pass


@dataclass(frozen=True)
class Cancel(Event):
    reason: str = ""


@dataclass(frozen=True)
class Shutdown(Event):
    """Arrêt propre demandé (menu « Quitter », Ctrl+C...)."""

    reason: str = ""


# `Transcript` est le dataclass de l'interface STT (stt/base.py), publié tel quel.
BusEvent: TypeAlias = (
    WakeDetected | SpeechEnded | Transcript | LLMDelta | ToolCall | SpeakDone | Cancel | Shutdown
)


class EventBus:
    """File d'événements asyncio à consommateur unique."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[BusEvent] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Associe la boucle cible de `publish_threadsafe`."""
        self._loop = loop

    def publish(self, event: BusEvent) -> None:
        """Publie depuis la boucle asyncio (non bloquant)."""
        self._queue.put_nowait(event)

    def publish_threadsafe(self, event: BusEvent) -> bool:
        """Publie depuis n'importe quel thread. Retourne False si la boucle est arrêtée."""
        if self._loop is None:
            raise RuntimeError("EventBus.bind_loop() doit être appelé avant publish_threadsafe()")
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:  # boucle déjà fermée : l'application s'arrête
            logger.debug("Événement ignoré, boucle fermée : {}", event)
            return False
        return True

    async def get(self) -> BusEvent:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()
