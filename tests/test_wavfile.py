from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from mikata.audio.wavfile import float_to_int16, read_wav, write_wav


def test_roundtrip_int16(tmp_path: Path) -> None:
    data = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    path = tmp_path / "sub" / "a.wav"
    write_wav(path, data, 16_000)
    pcm, rate = read_wav(path)
    assert rate == 16_000
    assert float_to_int16(pcm).tolist() == data.tolist()
    assert not list(tmp_path.rglob("*.part"))


def test_float_is_clipped() -> None:
    assert float_to_int16(np.array([2.0, -2.0, 0.5], dtype=np.float32)).tolist() == [
        32767,
        -32768,
        16384,
    ]


def test_read_stereo_is_mixed(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(np.array([16384, 0, 16384, 0], dtype=np.int16).tobytes())
    pcm, rate = read_wav(path)
    assert rate == 48_000
    assert np.allclose(pcm, [0.25, 0.25])
