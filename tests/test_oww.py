from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from mikata.config import PROJECT_ROOT, ConfigError
from mikata.wake.base import WakeWordDetector
from mikata.wake.oww import (
    FEATURE_MODELS,
    OpenWakeWordDetector,
    WakeModelMissingError,
    resolve_wake_files,
)


class FakePreprocessor:
    def get_features(self, n: int) -> np.ndarray:
        return np.ones((1, n, 96), dtype=np.float32)


class FakeModel:
    instances: ClassVar[list[FakeModel]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.frames: list[np.ndarray] = []
        self.resets = 0
        self.preprocessor = FakePreprocessor()
        self.model_inputs = {"hey_jarvis": 16}
        FakeModel.instances.append(self)

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.frames.append(frame)
        return {"hey_jarvis": 0.25, "autre": 0.75}

    def reset(self) -> None:
        self.resets += 1


class FakeVerifier:
    """Vérificateur picklable : probabilité « ta voix » fixe."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, 16, 96)
        return np.array([[1 - self.probability, self.probability]])


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    for name in ("hey_jarvis.onnx", *FEATURE_MODELS):
        (tmp_path / name).write_bytes(b"onnx")
    return tmp_path


@pytest.fixture
def fake_oww(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("openwakeword")
    module = types.ModuleType("openwakeword.model")
    module.Model = FakeModel  # type: ignore[attr-defined]
    FakeModel.instances = []
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)


def test_missing_models_are_listed(tmp_path: Path) -> None:
    with pytest.raises(WakeModelMissingError) as info:
        OpenWakeWordDetector(tmp_path / "hey_jarvis.onnx")
    message = str(info.value)
    assert "hey_jarvis.onnx" in message
    assert "melspectrogram.onnx" in message
    assert "download_models.py" in message
    assert isinstance(info.value, ConfigError)


def test_resolve_wake_files_falls_back_to_dev_model(models_dir: Path) -> None:
    """Artefacts entraînés (phase 7) facultatifs : mikata.onnx absent → hey_jarvis.onnx."""
    model, verifier = resolve_wake_files(models_dir / "mikata.onnx", None)
    assert model == models_dir / "hey_jarvis.onnx"
    assert verifier is None


def test_resolve_wake_files_drops_missing_verifier(models_dir: Path) -> None:
    model, verifier = resolve_wake_files(models_dir / "hey_jarvis.onnx", models_dir / "absent.pkl")
    assert model == models_dir / "hey_jarvis.onnx"
    assert verifier is None  # vérificateur absent ignoré, pas d'erreur


def test_resolve_wake_files_keeps_present_artifacts(models_dir: Path) -> None:
    (models_dir / "mikata.onnx").write_bytes(b"onnx")
    verifier = _verifier(models_dir, 0.9)
    model, resolved = resolve_wake_files(models_dir / "mikata.onnx", verifier)
    assert model == models_dir / "mikata.onnx"
    assert resolved == verifier


def test_resolve_wake_files_leaves_error_when_nothing_present(tmp_path: Path) -> None:
    """Aucun modèle du tout : le résolveur ne masque pas l'erreur, le détecteur la lève."""
    model, verifier = resolve_wake_files(tmp_path / "mikata.onnx", None)
    assert model == tmp_path / "mikata.onnx" and verifier is None
    with pytest.raises(WakeModelMissingError):
        OpenWakeWordDetector(model)


def test_loads_onnx_with_local_feature_models(models_dir: Path, fake_oww: None) -> None:
    detector = OpenWakeWordDetector(models_dir / "hey_jarvis.onnx")
    assert isinstance(detector, WakeWordDetector)
    kwargs = FakeModel.instances[-1].kwargs
    assert kwargs["inference_framework"] == "onnx"
    assert kwargs["wakeword_models"] == [str(models_dir / "hey_jarvis.onnx")]
    assert kwargs["melspec_model_path"] == str(models_dir / "melspectrogram.onnx")
    assert kwargs["embedding_model_path"] == str(models_dir / "embedding_model.onnx")
    assert "custom_verifier_models" not in kwargs

    score = detector.process(np.zeros(1280, dtype=np.int16))
    assert score == 0.75
    assert isinstance(score, float)
    detector.reset()
    assert FakeModel.instances[-1].resets == 1


def _verifier(models_dir: Path, probability: float) -> Path:
    path = models_dir / "mikata_verifier.pkl"
    path.write_bytes(pickle.dumps(FakeVerifier(probability)))
    return path


def test_verifier_missing(models_dir: Path, fake_oww: None) -> None:
    with pytest.raises(WakeModelMissingError, match="mikata_verifier"):
        OpenWakeWordDetector(models_dir / "hey_jarvis.onnx", models_dir / "mikata_verifier.pkl")


def test_verifier_accepts_your_voice(models_dir: Path, fake_oww: None) -> None:
    """Phase 7, étage 2 : le modèle atteint le seuil, le vérificateur reconnaît ta voix."""
    detector = OpenWakeWordDetector(
        models_dir / "hey_jarvis.onnx", _verifier(models_dir, 0.9), verifier_threshold=0.5
    )
    assert "custom_verifier_models" not in FakeModel.instances[-1].kwargs  # appelé par nous
    assert detector.has_verifier
    assert detector.process(np.zeros(1280, dtype=np.int16)) == 0.75
    assert detector.last_verifier_score == pytest.approx(0.9)
    assert detector.rejections == 0


def test_verifier_rejects_another_voice(models_dir: Path, fake_oww: None) -> None:
    detector = OpenWakeWordDetector(
        models_dir / "hey_jarvis.onnx", _verifier(models_dir, 0.1), verifier_threshold=0.5
    )
    frame = np.zeros(1280, dtype=np.int16)
    assert detector.process(frame) == 0.0  # refusé : pas de déclenchement
    assert detector.process(frame) == 0.0
    assert detector.rejections == 2 and detector.last_verifier_score == pytest.approx(0.1)


def test_verifier_only_runs_above_the_threshold(models_dir: Path, fake_oww: None) -> None:
    detector = OpenWakeWordDetector(
        models_dir / "hey_jarvis.onnx", _verifier(models_dir, 0.1), threshold=0.8
    )
    frame = np.zeros(1280, dtype=np.int16)
    assert detector.process(frame) == 0.75  # sous le seuil : étage 2 pas consulté
    assert detector.last_verifier_score is None
    assert detector.scores(frame, gate=0.5) == (0.75, pytest.approx(0.1))  # eval_wakeword.py


REAL_MODEL = PROJECT_ROOT / "models" / "hey_jarvis.onnx"


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="modèles non téléchargés")
def test_real_model_stays_quiet_on_noise() -> None:
    detector = OpenWakeWordDetector(REAL_MODEL)
    rng = np.random.default_rng(0)
    scores = [detector.process(rng.normal(0, 300, 1280).astype(np.int16)) for _ in range(50)]
    assert max(scores) < 0.5
