"""Phase 7 : enregistrement des « MIKATA » et archive pour Colab (record_wakeword_samples.py)."""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

import numpy as np
from scripts_loader import load_script

from mikata.audio.capture import CapturedFrame
from mikata.audio.wavfile import write_wav
from mikata.wake.dataset import WakewordData

rec = load_script("record_wakeword_samples")
SR = 16_000


def _wav(path: Path, seconds: float = 0.2) -> Path:
    write_wav(path, np.zeros(round(seconds * SR), dtype=np.int16), SR)
    return path


def test_takes_for_each_set(tmp_path: Path) -> None:
    data = WakewordData(tmp_path)
    train = rec.takes_for(data, "train")
    assert sum(t.target for t in train) == 100
    assert [t.word for t in train].count(False) == 1  # « phrase » : une commande après le mot
    assert sum(t.target for t in rec.takes_for(data, "eval")) == 30
    hard = rec.takes_for(data, "hard")
    assert sum(t.target for t in hard) == 36 and "mikado" in hard[0].say
    assert hard[3].directory == tmp_path / "hard_negatives" / "il_m_a_quitte"


def test_plan_report_tracks_progress(tmp_path: Path) -> None:
    data = WakewordData(tmp_path / "ww")
    for i in range(3):
        _wav(data.positives("train", "normal") / f"{i:03d}.wav")
    report = rec.plan_report(data, tmp_path / "negatives")
    assert "normal          3 / 20" in report
    assert "Mots proches : 0 / 36" in report
    assert "0.0 h sur 6 h" in report
    assert "record --set train : 97 prise(s)" in report
    assert "collect_negatives: true" in report


class _Capture:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    async def read(self) -> CapturedFrame:
        value = self.values.pop(0) if self.values else 0
        return CapturedFrame(np.full(1280, value, dtype=np.int16), 0.0)


class _Vad:
    def __init__(self) -> None:
        self.resets = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        return int(frame[0]) != 0

    def reset(self) -> None:
        self.resets += 1


def test_record_take_keeps_the_lead_in() -> None:
    vad = _Vad()
    capture = _Capture([0, 0, 0, 5, 5, 5] + [0] * 20)
    audio = asyncio.run(rec.record_take(capture, vad, end_silence_ms=480))
    assert vad.resets == 1
    assert audio is not None and len(audio) == (3 + 3 + 6) * 1280  # début gardé, 480 ms après
    assert asyncio.run(rec.record_take(_Capture([]), _Vad(), start_timeout_s=0.4)) is None


def test_check_take() -> None:
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 50, SR).astype(np.int16)
    word = quiet.copy()
    word[6000:12000] = rng.normal(0, 5000, 6000).astype(np.int16)
    assert rec.check_take(word, word=True) is None
    assert rec.check_take(quiet, word=True) == "voix trop faible ou absente"
    long = np.concatenate(
        [quiet[:4000], rng.normal(0, 5000, 3 * SR).astype(np.int16), quiet[:4000]]
    )
    assert "dis seulement" in rec.check_take(long, word=True)
    assert rec.check_take(long, word=False) is None  # « MIKATA, ouvre Spotify » : normal
    loud = word.copy()
    loud[8000] = 32767
    assert "saturé" in rec.check_take(loud, word=True)


def test_pack_excludes_the_evaluation_set(tmp_path: Path) -> None:
    data = WakewordData(tmp_path / "ww")
    _wav(data.positives("train", "normal") / "001.wav")
    _wav(data.positives("train", "phrase") / "001.wav")  # pas pour le détecteur
    _wav(data.positives("eval", "normal") / "001.wav")  # jamais dans l'archive
    _wav(data.hard_negatives("mikado") / "001.wav")
    negatives = tmp_path / "negatives"
    clips = [_wav(negatives / "2026-10-01" / f"{i:03d}.wav") for i in range(4)]
    data.root.mkdir(parents=True, exist_ok=True)
    data.eval_list.write_text("2026-10-01/000.wav\n", encoding="utf-8")  # figé pour le benchmark
    output = tmp_path / "out.zip"
    counts = rec.pack(data, negatives, output, negative_hours=1.0)
    assert counts == {"positives": 1, "hard_negatives": 1, "negatives": 3}
    with zipfile.ZipFile(output) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("data/manifest.json"))
    assert "train_wakeword.py" in names
    assert "data/positives/normal__001.wav" in names
    assert "data/hard_negatives/mikado__001.wav" in names
    assert "data/negatives/2026-10-01__000.wav" not in names  # extrait du benchmark
    assert {f"data/negatives/2026-10-01__{i:03d}.wav" for i in (1, 2, 3)} <= names
    assert not any("eval" in n or "phrase" in n for n in names)
    assert manifest["negatives"] == 3 and len(clips) == 4


def test_pack_limits_negative_hours(tmp_path: Path) -> None:
    data = WakewordData(tmp_path / "ww")
    _wav(data.positives("train", "normal") / "001.wav")
    negatives = tmp_path / "negatives"
    for i in range(6):
        _wav(negatives / "d" / f"{i:03d}.wav")
    counts = rec.pack(data, negatives, tmp_path / "o.zip", negative_hours=30 / 3600)
    assert counts["negatives"] == 3  # 3 extraits de 10 s


def test_main_pack_without_takes(tmp_path: Path) -> None:
    assert rec.main(["--root", str(tmp_path / "ww"), "pack"]) == 1


def test_parse_args() -> None:
    args = rec.parse_args(["record", "--set", "eval", "--only", "loin"])
    assert args.set == "eval" and args.only == "loin"
    assert rec.parse_args(["pack"]).negative_hours == rec.PACK_NEGATIVE_HOURS
