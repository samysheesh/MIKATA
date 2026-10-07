"""Tampon circulaire de pré-roll (§3) : les N dernières secondes d'audio 16 kHz int16.

Utilisé uniquement depuis la boucle asyncio : pas de verrou.
"""

from __future__ import annotations

import numpy as np

from mikata.audio.base import SAMPLE_RATE


class RingBuffer:
    def __init__(self, capacity: int, dtype: type[np.generic] = np.int16) -> None:
        if capacity <= 0:
            raise ValueError("capacity doit être > 0")
        self._data = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._end = 0  # position d'écriture
        self._filled = 0

    @classmethod
    def for_seconds(cls, seconds: float, sample_rate: int = SAMPLE_RATE) -> RingBuffer:
        return cls(round(seconds * sample_rate))

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._filled

    def write(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n >= self._capacity:
            self._data[:] = samples[-self._capacity :]
            self._end = 0
            self._filled = self._capacity
            return
        first = min(n, self._capacity - self._end)
        self._data[self._end : self._end + first] = samples[:first]
        self._data[: n - first] = samples[first:]
        self._end = (self._end + n) % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def read_last(self, n: int) -> np.ndarray:
        """Copie des `n` derniers échantillons (moins s'il n'y en a pas assez)."""
        n = max(0, min(n, self._filled))
        start = (self._end - n) % self._capacity
        if start + n <= self._capacity:
            return self._data[start : start + n].copy()
        return np.concatenate((self._data[start:], self._data[: self._end]))

    def clear(self) -> None:
        self._end = 0
        self._filled = 0
