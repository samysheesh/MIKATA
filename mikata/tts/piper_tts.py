"""Synthèse Piper (§8) : CPU, premier son quasi instantané, voix correcte.

La session ONNX est créée ici (et non par `PiperVoice.load`) pour la limiter à
`tts.cpu_threads` threads : le jeu doit toujours passer devant (§10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mikata.tts.base import LocalTTS, cpu_session, require_files


class PiperTTS(LocalTTS):
    name = "piper"

    def __init__(self, model: Path, speed: float = 1.0, threads: int = 4) -> None:
        super().__init__()
        self.model = model
        self.config_path = Path(f"{model}.json")
        self.speed = speed
        self.threads = threads
        self._voice: Any = None
        self._syn_config: Any = None

    def load(self) -> None:
        require_files(self.model, self.config_path)
        from piper import PiperVoice, SynthesisConfig
        from piper.config import PiperConfig

        config = PiperConfig.from_dict(json.loads(self.config_path.read_text(encoding="utf-8")))
        self._voice = PiperVoice(config=config, session=cpu_session(self.model, self.threads))
        self.sample_rate = int(config.sample_rate)
        # length_scale > 1 ralentit : vitesse 1,2 -> durée / 1,2
        self._syn_config = SynthesisConfig(length_scale=1.0 / self.speed)

    def _synthesize(self, text: str) -> np.ndarray:
        if self._voice is None:
            raise RuntimeError("Piper n'est pas chargé")
        chunks = [c.audio_float_array for c in self._voice.synthesize(text, self._syn_config)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
