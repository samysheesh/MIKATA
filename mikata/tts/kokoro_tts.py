"""Synthèse Kokoro-82M (§8) en ONNX via `kokoro-onnx`, sur CPU.

Plus naturel que Piper, plus lent. Le paquet `kokoro` officiel installe PyTorch : on
utilise le modèle ONNX (fp32 : la version int8 est ~5 fois plus lente sur CPU) avec une
session limitée à `tts.cpu_threads` threads (§10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mikata.tts.base import LocalTTS, cpu_session, require_files

KOKORO_RATE = 24_000


class KokoroTTS(LocalTTS):
    name = "kokoro"
    sample_rate = KOKORO_RATE

    def __init__(
        self,
        model: Path,
        voices: Path,
        voice: str = "ff_siwis",
        speed: float = 1.0,
        threads: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.voices = voices
        self.voice = voice
        self.speed = speed
        self.threads = threads
        self._kokoro: Any = None

    def load(self) -> None:
        require_files(self.model, self.voices)
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro.from_session(cpu_session(self.model, self.threads), str(self.voices))
        if self.voice not in self._kokoro.get_voices():
            from mikata.tts.base import TtsModelMissingError

            raise TtsModelMissingError(f"tts.kokoro_voice = « {self.voice} » inconnue de Kokoro.")

    def _synthesize(self, text: str) -> np.ndarray:
        if self._kokoro is None:
            raise RuntimeError("Kokoro n'est pas chargé")
        pcm, rate = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang="fr-fr")
        self.sample_rate = int(rate)
        return np.asarray(pcm, dtype=np.float32)
