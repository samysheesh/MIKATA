from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from mikata.config import PROJECT_ROOT, ConfigError
from mikata.wake.base import WakeWordDetector
from mikata.wake.oww import FEATURE_MODELS, OpenWakeWordDetector, WakeModelMissingError


class FakeModel:
    instances: ClassVar[list[FakeModel]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.frames: list[np.ndarray] = []
        self.resets = 0
        FakeModel.instances.append(self)

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.frames.append(frame)
        return {"hey_jarvis": 0.25, "autre": 0.75}

    def reset(self) -> None:
        self.resets += 1


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


def test_verifier(models_dir: Path, fake_oww: None) -> None:
    verifier = models_dir / "mikata_verifier.pkl"
    with pytest.raises(WakeModelMissingError, match="mikata_verifier"):
        OpenWakeWordDetector(models_dir / "hey_jarvis.onnx", verifier)
    verifier.write_bytes(b"pkl")
    OpenWakeWordDetector(models_dir / "hey_jarvis.onnx", verifier)
    assert FakeModel.instances[-1].kwargs["custom_verifier_models"] == {"hey_jarvis": str(verifier)}


REAL_MODEL = PROJECT_ROOT / "models" / "hey_jarvis.onnx"


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="modèles non téléchargés")
def test_real_model_stays_quiet_on_noise() -> None:
    detector = OpenWakeWordDetector(REAL_MODEL)
    rng = np.random.default_rng(0)
    scores = [detector.process(rng.normal(0, 300, 1280).astype(np.int16)) for _ in range(50)]
    assert max(scores) < 0.5
