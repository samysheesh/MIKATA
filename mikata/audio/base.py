"""Types audio communs (docs/ARCHITECTURE.md §9).

Tableaux numpy int16 mono 16 kHz partout en entrée.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1 280, taille attendue par openWakeWord

Frame: TypeAlias = np.ndarray  # shape (1280,), dtype int16
