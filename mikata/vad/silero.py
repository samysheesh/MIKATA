"""Silero VAD v5 en ONNX, sur CPU (§5) : < 1 ms par trame, actif seulement pendant l'écoute.

On appelle le modèle directement avec onnxruntime (le paquet `silero-vad` tire PyTorch,
2 Go inutiles ici). Le modèle attend des blocs de 512 échantillons à 16 kHz précédés de
64 échantillons de contexte, et garde un état récurrent entre deux appels : c'est ce que
fait le wrapper officiel. Une trame de 80 ms (1 280 échantillons) n'est pas un multiple
de 512 : le reste est gardé pour la trame suivante.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mikata.audio.base import SAMPLE_RATE, Frame
from mikata.config import ConfigError

CHUNK = 512  # 32 ms à 16 kHz
CONTEXT = 64


class VadModelMissingError(ConfigError):
    """Modèle Silero VAD introuvable."""


class SileroVAD:
    def __init__(self, model: Path, threshold: float = 0.5) -> None:
        if not model.is_file():
            raise VadModelMissingError(
                f"Modèle Silero VAD introuvable : {model}\n"
                "Lance : uv run python scripts/download_models.py"
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1  # §10 : 1 thread pour le VAD
        options.intra_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Oublie l'état récurrent : à appeler au début de chaque écoute."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self.last_probability = 0.0

    def probabilities(self, frame: Frame) -> list[float]:
        """Probabilité de voix pour chaque bloc complet de 32 ms disponible."""
        samples = np.concatenate((self._pending, frame.astype(np.float32) / 32768.0))
        count = len(samples) // CHUNK
        probs: list[float] = []
        for i in range(count):
            chunk = samples[i * CHUNK : (i + 1) * CHUNK]
            x = np.concatenate((self._context, chunk))[None, :]
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            self._context = chunk[-CONTEXT:]
            probs.append(float(out[0, 0]))
        self._pending = samples[count * CHUNK :]
        return probs

    def is_speech(self, frame: Frame) -> bool:
        """Vrai si l'un des blocs de 32 ms de la trame dépasse le seuil."""
        probs = self.probabilities(frame)
        if probs:
            self.last_probability = max(probs)
        return self.last_probability >= self.threshold
