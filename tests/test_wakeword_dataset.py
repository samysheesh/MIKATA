"""Phase 7 : données du mot d'activation (`mikata.wake.dataset`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mikata.audio.wavfile import write_wav
from mikata.wake.dataset import (
    HARD_NEGATIVE_TAKES,
    HARD_NEGATIVES,
    POSITIVE_PLAN,
    WakewordData,
    detector_positives,
    freeze_eval_negatives,
    list_wavs,
    next_take,
    noise_floor,
    pad_with_floor_noise,
    slug,
    total_hours,
    training_negatives,
    trim_to_voice,
    user_speech_clips,
    voice_bounds,
)

SR = 16_000


def _wav(path: Path, seconds: float = 0.1, value: int = 0) -> Path:
    write_wav(path, np.full(round(seconds * SR), value, dtype=np.int16), SR)
    return path


def _word(rng: np.random.Generator, before: float = 0.5, after: float = 0.3) -> np.ndarray:
    """Bruit faible, 0,4 s de « voix » forte, bruit faible."""
    quiet = lambda s: rng.normal(0, 50, round(s * SR))  # noqa: E731
    voice = rng.normal(0, 5000, round(0.4 * SR))
    return np.concatenate((quiet(before), voice, quiet(after))).astype(np.int16)


def test_recording_plan_is_100_train_and_30_eval() -> None:
    assert sum(c.train for c in POSITIVE_PLAN) == 100
    assert sum(c.eval for c in POSITIVE_PLAN) == 30
    assert len({c.key for c in POSITIVE_PLAN}) == len(POSITIVE_PLAN)
    assert [c.key for c in POSITIVE_PLAN if not c.detector] == ["phrase"]
    assert len(HARD_NEGATIVES) * HARD_NEGATIVE_TAKES == 36
    for word in ("mikado", "Nikita", "Akira", "il m'a quitté", "mi-carré"):  # §4
        assert word in HARD_NEGATIVES


def test_slug_and_paths(tmp_path: Path) -> None:
    assert slug("il m'a quitté") == "il_m_a_quitte"
    assert slug("Mi-carré") == "mi_carre"
    data = WakewordData(tmp_path)
    assert data.positives("eval", "loin") == tmp_path / "positives" / "eval" / "loin"
    assert data.hard_negatives("il m'a quitté") == tmp_path / "hard_negatives" / "il_m_a_quitte"
    assert data.eval_list == tmp_path / "eval_negatives.txt"


def test_next_take_fills_gaps(tmp_path: Path) -> None:
    assert next_take(tmp_path / "x") == tmp_path / "x" / "001.wav"
    _wav(tmp_path / "001.wav")
    _wav(tmp_path / "003.wav")
    assert next_take(tmp_path) == tmp_path / "002.wav"


def test_detector_positives_skip_phrases(tmp_path: Path) -> None:
    data = WakewordData(tmp_path)
    normal = _wav(data.positives("train", "normal") / "001.wav")
    _wav(data.positives("train", "phrase") / "001.wav")
    _wav(data.positives("eval", "normal") / "001.wav")
    assert detector_positives(data) == [normal]


def test_eval_negatives_are_frozen_and_spread(tmp_path: Path) -> None:
    negatives = tmp_path / "negatives"
    eval_list = tmp_path / "eval.txt"
    clips = [_wav(negatives / f"2026-10-0{d}" / f"{i:03d}.wav") for d in (1, 2) for i in range(5)]
    hours = 3 * 10 / 3600  # 3 extraits de 10 s
    assert freeze_eval_negatives(negatives, eval_list, hours=0.1) == []  # pas assez : rien figé
    assert not eval_list.exists()
    frozen = freeze_eval_negatives(negatives, eval_list, hours=hours)
    assert frozen == [clips[0], clips[4], clips[9]]  # réparti sur toute la collecte
    assert eval_list.read_text(encoding="utf-8").splitlines()[0] == "2026-10-01/000.wav"
    _wav(negatives / "2026-10-03" / "000.wav")  # nouvelle collecte : la liste ne bouge pas
    assert freeze_eval_negatives(negatives, eval_list, hours=hours) == frozen
    train = training_negatives(negatives, eval_list)
    assert len(train) == 8 and not set(train) & set(frozen)


def test_total_hours(tmp_path: Path) -> None:
    paths = [_wav(tmp_path / "a.wav", 1.8), _wav(tmp_path / "b.wav", 1.8)]
    (tmp_path / "bad.wav").write_bytes(b"pas un wav")
    assert total_hours([*paths, tmp_path / "bad.wav"]) == pytest.approx(0.001)


def test_user_speech_clips_exclude_the_wake_word(tmp_path: Path) -> None:
    for name, text in (
        ("01", "ouvre Spotify"),
        ("02", "merci MIKATA c'est tout"),
        ("03", "mi kata"),
    ):
        _wav(tmp_path / f"{name}.wav")
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    _wav(tmp_path / "04.wav")  # sans texte : ignoré
    assert user_speech_clips(tmp_path) == [tmp_path / "01.wav"]
    assert list_wavs(tmp_path / "absent") == []


def test_voice_bounds_and_trim() -> None:
    rng = np.random.default_rng(0)
    audio = _word(rng)
    start, end = voice_bounds(audio)  # type: ignore[misc]
    assert abs(start - 0.5 * SR) <= 160 and abs(end - 0.9 * SR) <= 160
    trimmed = trim_to_voice(audio, margin_s=0.1)
    assert len(trimmed) == pytest.approx(0.6 * SR, abs=400)
    silence = np.zeros(SR, dtype=np.int16)
    assert voice_bounds(silence) is None and trim_to_voice(silence) is silence


def test_pad_with_floor_noise() -> None:
    rng = np.random.default_rng(1)
    audio = _word(rng)
    padded = pad_with_floor_noise(audio, 1.0, 0.5, rng)
    assert len(padded) == len(audio) + 24_000 and padded.dtype == np.int16
    assert 20 < np.std(padded[:SR]) < 120  # au niveau du fond (~50), pas du silence numérique
    assert noise_floor(audio) < 100
