"""Interface de détection d'activité vocale (docs/ARCHITECTURE.md §5 et §9)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mikata.audio.base import Frame


@runtime_checkable
class VAD(Protocol):
    def is_speech(self, frame: Frame) -> bool: ...
