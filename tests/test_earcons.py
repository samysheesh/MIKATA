from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mikata.audio.earcons import DEFAULT_RATE, EARCON_NAMES, Earcons, synthesize
from mikata.audio.wavfile import read_wav
from mikata.config import PROJECT_ROOT

SOUNDS = PROJECT_ROOT / "assets" / "sounds"


@pytest.mark.parametrize("name", EARCON_NAMES)
def test_synthesized_sounds_are_short_and_safe(name: str) -> None:
    pcm = synthesize(name)
    assert pcm.dtype == np.float32
    max_s = 1.0 if name == "timer" else 0.4  # le minuteur doit s'entendre
    assert 0.05 < len(pcm) / DEFAULT_RATE < max_s
    assert np.abs(pcm).max() <= 1.0
    assert abs(pcm[0]) < 0.01 and abs(pcm[-1]) < 0.01  # pas de clic


def test_wake_beep_is_about_120ms() -> None:
    assert len(synthesize("wake")) == round(0.12 * DEFAULT_RATE)


@pytest.mark.parametrize("name", EARCON_NAMES)
def test_committed_wav_files(name: str) -> None:
    pcm, rate = read_wav(SOUNDS / f"{name}.wav")
    assert rate == DEFAULT_RATE
    assert len(pcm) > 0


def test_volume_and_resampling(tmp_path: Path) -> None:
    for name in EARCON_NAMES:
        shutil.copy(SOUNDS / f"{name}.wav", tmp_path)
    full = Earcons(tmp_path, 48_000, 1.0)
    half = Earcons(tmp_path, 48_000, 0.5)
    assert np.allclose(half["wake"], full["wake"] * 0.5)
    resampled = Earcons(tmp_path, 44_100, 1.0)
    assert abs(len(resampled["wake"]) - round(0.12 * 44_100)) <= 2


def test_missing_files_fall_back_to_synthesis(tmp_path: Path) -> None:
    earcons = Earcons(tmp_path, DEFAULT_RATE, 1.0)
    assert "wake" in earcons and "inconnu" not in earcons
    assert sorted(earcons) == sorted(EARCON_NAMES) and len(earcons) == len(EARCON_NAMES)
    assert np.allclose(earcons["end"], synthesize("end"), atol=1e-4)
    with pytest.raises(KeyError):
        earcons["inconnu"]
