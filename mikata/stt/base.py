"""Interface de transcription (docs/ARCHITECTURE.md §5 et §9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Transcript:
    text: str
    duration_s: float
    latency_s: float


@runtime_checkable
class STT(Protocol):
    async def transcribe(self, audio: np.ndarray) -> Transcript:
        """`audio` : int16 mono 16 kHz."""
        ...
