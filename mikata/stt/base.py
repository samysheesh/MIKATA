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
    # Ajouts phase 6 (fiabilité, pour la fenêtre de suivi) : None si le moteur ne les donne pas.
    no_speech_prob: float | None = None  # probabilité qu'il n'y ait pas de parole
    avg_logprob: float | None = None  # log-probabilité moyenne des tokens


@runtime_checkable
class STT(Protocol):
    async def transcribe(self, audio: np.ndarray) -> Transcript:
        """`audio` : int16 mono 16 kHz."""
        ...
