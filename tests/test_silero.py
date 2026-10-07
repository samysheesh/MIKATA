from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mikata.audio.wavfile import float_to_int16, read_wav
from mikata.config import PROJECT_ROOT, ConfigError
from mikata.vad.base import VAD
from mikata.vad.silero import CHUNK, CONTEXT, SileroVAD, VadModelMissingError

REAL_MODEL = PROJECT_ROOT / "models" / "silero_vad.onnx"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "hey_jarvis_ouvre_spotify.wav"


class FakeSession:
    """Renvoie la moyenne absolue du bloc comme « probabilité » ; garde les entrées."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, outputs: None, feeds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(feeds)
        x = feeds["input"]
        return np.array([[float(np.abs(x[0, CONTEXT:]).mean())]]), feeds["state"] + 1


@pytest.fixture
def fake_vad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SileroVAD, FakeSession]:
    session = FakeSession()
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = types.SimpleNamespace  # type: ignore[attr-defined]
    ort.InferenceSession = lambda *a, **k: session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"onnx")
    return SileroVAD(model, threshold=0.25), session


def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(VadModelMissingError, match="download_models") as info:
        SileroVAD(tmp_path / "absent.onnx")
    assert isinstance(info.value, ConfigError)


def test_chunks_context_and_state(fake_vad: tuple[SileroVAD, FakeSession]) -> None:
    vad, session = fake_vad
    assert isinstance(vad, VAD)
    loud = np.full(1280, 16384, dtype=np.int16)  # 0,5
    assert vad.is_speech(loud)  # 2 blocs de 512, 256 en attente
    assert len(session.calls) == 2
    first, second = session.calls
    assert first["input"].shape == (1, CONTEXT + CHUNK)
    assert np.all(first["input"][0, :CONTEXT] == 0)  # pas de contexte au départ
    assert np.allclose(second["input"][0, :CONTEXT], 0.5)  # contexte = fin du bloc précédent
    assert second["state"][0, 0, 0] == 1  # état récurrent transmis
    assert int(second["sr"]) == 16_000

    vad.is_speech(np.zeros(1280, dtype=np.int16))  # 256 + 1280 = 3 blocs
    assert len(session.calls) == 5
    vad.reset()
    vad.is_speech(loud)
    assert np.all(session.calls[-2]["state"] == 0)
    assert np.all(session.calls[-2]["input"][0, :CONTEXT] == 0)


def test_threshold_uses_max_of_frame(fake_vad: tuple[SileroVAD, FakeSession]) -> None:
    vad, _ = fake_vad
    frame = np.zeros(1280, dtype=np.int16)
    frame[600:1000] = 16384  # voix seulement dans le 2e bloc
    assert vad.is_speech(frame)
    assert vad.last_probability > 0.25
    assert not vad.is_speech(np.zeros(1280, dtype=np.int16))


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="modèle Silero non téléchargé")
def test_real_model_finds_speech() -> None:
    vad = SileroVAD(REAL_MODEL)
    speech, _ = read_wav(FIXTURE)
    rng = np.random.default_rng(0)
    audio = np.concatenate([rng.normal(0, 0.003, 16_000), speech, rng.normal(0, 0.003, 16_000)])
    pcm = float_to_int16(audio.astype(np.float32))
    flags = [vad.is_speech(pcm[i : i + 1280]) for i in range(0, len(pcm) - 1279, 1280)]
    assert not any(flags[:10])  # bruit de fond seul
    assert sum(flags) >= 0.8 * len(speech) / 1280
    assert not any(flags[-8:])
