"""Détecteur openWakeWord (ONNX, CPU) — §4.

Phase 1 : modèle pré-entraîné `hey_jarvis` ; phase 7 : `mikata.onnx` + vérificateur de voix,
sans changer ce code. Les modèles de features (`melspectrogram.onnx`, `embedding_model.onnx`)
sont lus dans le même dossier que le modèle (cf. scripts/download_models.py).
openWakeWord limite lui-même ses sessions ONNX à 1 thread (§10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mikata.audio.base import Frame
from mikata.config import ConfigError

FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")


class WakeModelMissingError(ConfigError):
    """Modèle de mot d'activation introuvable."""


def required_model_files(model: Path, verifier: Path | None = None) -> list[Path]:
    files = [model, *(model.parent / name for name in FEATURE_MODELS)]
    if verifier is not None:
        files.append(verifier)
    return files


class OpenWakeWordDetector:
    def __init__(self, model: Path, verifier: Path | None = None) -> None:
        missing = [p for p in required_model_files(model, verifier) if not p.is_file()]
        if missing:
            raise WakeModelMissingError(
                "Modèle(s) de mot d'activation introuvable(s) :\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nLance : uv run python scripts/download_models.py"
            )
        from openwakeword.model import Model

        kwargs: dict[str, Any] = {}
        if verifier is not None:
            kwargs["custom_verifier_models"] = {model.stem: str(verifier)}
        self.name = model.stem
        self._model = Model(
            wakeword_models=[str(model)],
            inference_framework="onnx",  # obligatoire sous Windows (§4)
            melspec_model_path=str(model.parent / FEATURE_MODELS[0]),
            embedding_model_path=str(model.parent / FEATURE_MODELS[1]),
            **kwargs,
        )

    def process(self, frame: Frame) -> float:
        scores = self._model.predict(np.asarray(frame, dtype=np.int16))
        return float(max(scores.values(), default=0.0))

    def reset(self) -> None:
        self._model.reset()
