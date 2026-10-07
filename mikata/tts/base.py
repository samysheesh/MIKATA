"""Interfaces de synthèse et de lecture audio (docs/ARCHITECTURE.md §8 et §9)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTS(Protocol):
    sample_rate: int

    async def synthesize(self, text: str) -> np.ndarray:
        """PCM float32 mono à `sample_rate`."""
        ...


@runtime_checkable
class Player(Protocol):
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None: ...

    def stop(self) -> None:
        """Arrêt immédiat (barge-in)."""
        ...
