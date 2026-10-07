"""Interface du détecteur de mot d'activation (docs/ARCHITECTURE.md §4 et §9)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mikata.audio.base import Frame


@runtime_checkable
class WakeWordDetector(Protocol):
    def process(self, frame: Frame) -> float:
        """Score 0..1 pour une trame de 80 ms."""
        ...

    def reset(self) -> None: ...
