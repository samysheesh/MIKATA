"""Détecteur openWakeWord (ONNX, CPU) — §4.

Phase 1 : modèle pré-entraîné `hey_jarvis` ; phase 7 : `mikata.onnx` + vérificateur de voix,
sans changer ce code. Les modèles de features (`melspectrogram.onnx`, `embedding_model.onnx`)
sont lus dans le même dossier que le modèle (cf. scripts/download_models.py).
openWakeWord limite lui-même ses sessions ONNX à 1 thread (§10).

Phase 7 : deux étages (§4).
1. Le modèle donne un score toutes les 80 ms.
2. Seulement quand ce score atteint `wake.threshold`, le vérificateur de ta voix
   (`wake.verifier`, entraîné par scripts/train_verifier.py sur tes prises) donne la
   probabilité que ce soit **toi** qui dises « MIKATA ». Sous `wake.verifier_threshold`, la
   trame est refusée (score 0) : un « MIKATA » d'une vidéo ou d'un ami ne déclenche pas.
Le vérificateur est un petit modèle scikit-learn appliqué aux features d'openWakeWord : coût
et délai quasi nuls. On l'appelle nous-mêmes (et non via `custom_verifier_models`) pour que
les deux seuils soient réglables séparément par eval_wakeword.py.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mikata.audio.base import Frame
from mikata.config import ConfigError

FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
REJECTION_LOG_S = 2.0  # un refus par période réfractaire dans le log, pas un par trame


class WakeModelMissingError(ConfigError):
    """Modèle de mot d'activation introuvable."""


def required_model_files(model: Path, verifier: Path | None = None) -> list[Path]:
    files = [model, *(model.parent / name for name in FEATURE_MODELS)]
    if verifier is not None:
        files.append(verifier)
    return files


def load_verifier(path: Path) -> Any:
    """Vérificateur de voix (pipeline scikit-learn sauvegardé par train_verifier.py)."""
    with path.open("rb") as file:
        return pickle.load(file)  # fichier créé sur ton PC par toi : de confiance


class OpenWakeWordDetector:
    def __init__(
        self,
        model: Path,
        verifier: Path | None = None,
        *,
        threshold: float = 0.5,
        verifier_threshold: float = 0.5,
    ) -> None:
        missing = [p for p in required_model_files(model, verifier) if not p.is_file()]
        if missing:
            raise WakeModelMissingError(
                "Modèle(s) de mot d'activation introuvable(s) :\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nLance : uv run python scripts/download_models.py"
                + (" (et pour mikata.onnx : README, phase 7)" if "mikata" in model.stem else "")
            )
        from openwakeword.model import Model

        self.name = model.stem
        self.threshold = threshold
        self.verifier_threshold = verifier_threshold
        self._model = Model(
            wakeword_models=[str(model)],
            inference_framework="onnx",  # obligatoire sous Windows (§4)
            melspec_model_path=str(model.parent / FEATURE_MODELS[0]),
            embedding_model_path=str(model.parent / FEATURE_MODELS[1]),
        )
        self._verifier = load_verifier(verifier) if verifier is not None else None
        self.last_verifier_score: float | None = None  # dernière trame vérifiée
        self.rejections = 0
        self._last_rejection_log = float("-inf")

    @property
    def has_verifier(self) -> bool:
        return self._verifier is not None

    def scores(self, frame: Frame, gate: float | None = None) -> tuple[float, float | None]:
        """(score du modèle, probabilité « ta voix » ou None). Le vérificateur ne tourne que si
        le score atteint `gate` (par défaut `threshold`)."""
        predictions = self._model.predict(np.asarray(frame, dtype=np.int16))
        score = float(max(predictions.values(), default=0.0))
        if self._verifier is None or score < (self.threshold if gate is None else gate):
            return score, None
        features = self._model.preprocessor.get_features(self._model.model_inputs[self.name])
        return score, float(self._verifier.predict_proba(features)[0][-1])

    def process(self, frame: Frame) -> float:
        score, voice = self.scores(frame)
        self.last_verifier_score = voice
        if voice is None or voice >= self.verifier_threshold:
            return score
        self.rejections += 1
        now = time.monotonic()
        if now - self._last_rejection_log >= REJECTION_LOG_S:
            self._last_rejection_log = now
            logger.info(
                "Mot d'activation refusé par le vérificateur : modèle {:.2f}, voix {:.2f} "
                "(seuil {:.2f}).",
                score,
                voice,
                self.verifier_threshold,
            )
        return 0.0

    def reset(self) -> None:
        self._model.reset()
