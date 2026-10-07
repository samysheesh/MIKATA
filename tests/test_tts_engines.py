from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from mikata.config import PROJECT_ROOT, Config, ConfigError
from mikata.tts.base import TTS, TtsModelMissingError, create_tts
from mikata.tts.kokoro_tts import KokoroTTS
from mikata.tts.piper_tts import PiperTTS

MODELS = PROJECT_ROOT / "models"


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def session(path: str, sess_options: Options, providers: list[str]) -> dict[str, Any]:
        options = sess_options
        sessions.append(
            {
                "path": path,
                "threads": options.intra_op_num_threads,
                "inter": options.inter_op_num_threads,
                "providers": providers,
            }
        )
        return sessions[-1]

    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = Options  # type: ignore[attr-defined]
    ort.InferenceSession = session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return sessions


class FakePiperVoice:
    instances: ClassVar[list[FakePiperVoice]] = []

    def __init__(self, config: Any, session: Any) -> None:
        self.config, self.session = config, session
        FakePiperVoice.instances.append(self)

    def synthesize(self, text: str, syn_config: Any) -> Any:
        self.syn_config = syn_config
        for part in (0.1, 0.2):
            yield types.SimpleNamespace(audio_float_array=np.full(100, part, dtype=np.float32))


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch, fake_ort: list) -> None:
    piper = types.ModuleType("piper")
    piper.PiperVoice = FakePiperVoice  # type: ignore[attr-defined]
    piper.SynthesisConfig = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    config = types.ModuleType("piper.config")
    config.PiperConfig = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dict=lambda d: types.SimpleNamespace(sample_rate=d["audio"]["sample_rate"])
    )
    monkeypatch.setitem(sys.modules, "piper", piper)
    monkeypatch.setitem(sys.modules, "piper.config", config)


def test_piper(tmp_path: Path, fake_piper: None, fake_ort: list) -> None:
    model = tmp_path / "voix.onnx"
    model.write_bytes(b"x")
    Path(f"{model}.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    tts = PiperTTS(model, speed=1.25, threads=3)
    assert isinstance(tts, TTS)
    tts.load()
    assert tts.sample_rate == 22050
    assert fake_ort[-1] == {"path": str(model), "threads": 3, "inter": 1,
                            "providers": ["CPUExecutionProvider"]}  # fmt: skip
    pcm = asyncio.run(tts.synthesize("Bonjour."))
    assert pcm.dtype == np.float32 and len(pcm) == 200
    assert FakePiperVoice.instances[-1].syn_config.length_scale == pytest.approx(0.8)
    tts.close()


def test_piper_missing_files(tmp_path: Path) -> None:
    with pytest.raises(TtsModelMissingError, match="download_models") as info:
        PiperTTS(tmp_path / "absent.onnx").load()
    assert "absent.onnx.json" in str(info.value)
    assert isinstance(info.value, ConfigError)


class FakeKokoro:
    @classmethod
    def from_session(cls, session: Any, voices: str) -> FakeKokoro:
        k = cls()
        k.session, k.voices = session, voices
        return k

    def get_voices(self) -> list[str]:
        return ["ff_siwis", "af_heart"]

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[np.ndarray, int]:
        self.last = (text, voice, speed, lang)
        return np.zeros(2400, dtype=np.float64), 24000


@pytest.fixture
def fake_kokoro(monkeypatch: pytest.MonkeyPatch, fake_ort: list) -> None:
    module = types.ModuleType("kokoro_onnx")
    module.Kokoro = FakeKokoro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)


def test_kokoro(tmp_path: Path, fake_kokoro: None, fake_ort: list) -> None:
    model, voices = tmp_path / "k.onnx", tmp_path / "v.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    tts = KokoroTTS(model, voices, "ff_siwis", speed=1.1, threads=4)
    tts.load()
    assert fake_ort[-1]["threads"] == 4
    pcm = tts.synthesize_sync("Salut.")
    assert pcm.dtype == np.float32 and tts.sample_rate == 24000
    assert tts._kokoro.last == ("Salut.", "ff_siwis", 1.1, "fr-fr")
    with pytest.raises(TtsModelMissingError, match="inconnue"):
        KokoroTTS(model, voices, "zz_nope").load()
    tts.close()


def test_create_tts_follows_config(tmp_path: Path) -> None:
    config = Config()
    config._base_dir = tmp_path
    piper, piper_id = create_tts(config)
    assert (
        isinstance(piper, PiperTTS)
        and piper.model == tmp_path / "models" / "fr_FR-siwis-medium.onnx"
    )
    kokoro, kokoro_id = create_tts(config, "kokoro")
    assert isinstance(kokoro, KokoroTTS) and kokoro.voice == "ff_siwis"
    assert piper_id != kokoro_id
    with pytest.raises(ValueError):
        create_tts(config, "espeak")


@pytest.mark.parametrize(
    ("backend", "files"),
    [
        ("piper", ["fr_FR-siwis-medium.onnx", "fr_FR-siwis-medium.onnx.json"]),
        ("kokoro", ["kokoro-v1.0.onnx", "voices-v1.0.bin"]),
    ],
)
def test_real_engine_speaks_french(backend: str, files: list[str]) -> None:
    if not all((MODELS / f).is_file() for f in files):
        pytest.skip("modèles de voix non téléchargés")
    config = Config()
    config._base_dir = PROJECT_ROOT
    tts, _ = create_tts(config, backend)
    tts.load()
    pcm = tts.synthesize_sync("C'est fait.")
    tts.close()
    assert 0.3 < len(pcm) / tts.sample_rate < 3.0
    assert np.abs(pcm).max() > 0.05


def test_trim_silence() -> None:
    from mikata.tts.base import KEEP_END_MS, KEEP_START_MS, trim_silence

    rate = 10_000
    pcm = np.concatenate([np.zeros(2000), np.full(500, 0.5), np.zeros(4000)]).astype(np.float32)
    out = trim_silence(pcm, rate)
    assert len(out) == KEEP_START_MS * 10 + 500 + KEEP_END_MS * 10
    assert out[KEEP_START_MS * 10] == 0.5
    assert len(trim_silence(np.zeros(100, np.float32), rate)) == 0
    assert len(trim_silence(np.zeros(0, np.float32), rate)) == 0
