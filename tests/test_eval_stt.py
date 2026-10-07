from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from scripts_loader import load_script

from mikata.audio.capture import CapturedFrame
from mikata.config import PROJECT_ROOT
from mikata.stt.base import Transcript

ev = load_script("eval_stt")


@pytest.mark.parametrize(
    ("ref", "hyp", "errors", "words"),
    [
        ("ouvre Spotify", "Ouvre Spotify.", 0, 2),
        ("quelle heure est-il", "Quelle heure est-il ?", 0, 4),
        ("mets le volume à 30", "mets le volume à 13", 1, 5),
        ("lance Steam", "lance", 1, 2),
        ("pause", "pause pause", 1, 1),
        ("verrouille l'ordinateur", "verrouille l’ordinateur", 0, 3),
        ("", "bruit", 1, 0),
    ],
)
def test_word_errors(ref: str, hyp: str, errors: int, words: int) -> None:
    assert ev.word_errors(ref, hyp) == (errors, words)


def test_corpus_wer() -> None:
    pairs = [("ouvre Spotify", "ouvre Spotify"), ("mets le volume à 30", "mets le volume à 13")]
    assert ev.corpus_wer(pairs) == pytest.approx(1 / 7)
    assert ev.corpus_wer([]) == 0.0


def test_phrase_list_has_50_commands() -> None:
    phrases = ev.load_phrases(ev.DEFAULT_PHRASES)
    assert len(phrases) == 50
    assert "ouvre Spotify" in phrases
    assert not any(p.startswith("#") for p in phrases)


def test_load_dataset(tmp_path: Path) -> None:
    for name in ("01", "02", "03"):
        (tmp_path / f"{name}.wav").write_bytes(b"")
    (tmp_path / "01.txt").write_text("ouvre Spotify\n", encoding="utf-8")
    (tmp_path / "03.txt").write_text("pause", encoding="utf-8")
    assert ev.load_dataset(tmp_path) == [
        (tmp_path / "01.wav", "ouvre Spotify"),
        (tmp_path / "03.wav", "pause"),
    ]


class _Capture:
    def __init__(self, frames: int) -> None:
        self.i = 0
        self.frames = frames

    async def read(self) -> CapturedFrame:
        self.i += 1
        return CapturedFrame(np.full(1280, self.i, dtype=np.int16), 0.0)


class _Vad:
    def __init__(self, speech: set[int]) -> None:
        self.speech = speech
        self.resets = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        return int(frame[0]) in self.speech

    def reset(self) -> None:
        self.resets += 1


def test_record_one_stops_after_silence() -> None:
    vad = _Vad(set(range(3, 10)))
    audio = asyncio.run(ev.record_one(_Capture(100), vad))
    assert audio is not None
    assert vad.resets == 1
    assert len(audio) == (9 + 9) * 1280  # jusqu'à la trame 9, puis 700 ms (9 trames) de silence


def test_record_one_gives_up_without_speech() -> None:
    assert asyncio.run(ev.record_one(_Capture(100), _Vad(set()))) is None


def test_record_one_max_length() -> None:
    audio = asyncio.run(ev.record_one(_Capture(500), _Vad(set(range(1, 500))), max_s=2.0))
    assert audio is not None and len(audio) == 25 * 1280


class _FakeSTT:
    def __init__(self, answers: dict[int, str]) -> None:
        self.answers = answers
        self.loaded = self.closed = False

    def load(self) -> None:
        self.loaded = True

    def transcribe_sync(self, audio: np.ndarray) -> Transcript:
        return Transcript(self.answers[int(audio[0])], len(audio) / 16_000, 0.1 * int(audio[0]))

    def close(self) -> None:
        self.closed = True


def test_evaluate_and_report() -> None:
    dataset = [
        (np.full(1600, 1, dtype=np.int16), "ouvre Spotify"),
        (np.full(1600, 2, dtype=np.int16), "mets le volume à 30"),
        (np.full(1600, 3, dtype=np.int16), "pause"),
    ]
    stt = _FakeSTT({1: "Ouvre Spotify.", 2: "Mets le volume à 13.", 3: "Sous-titrage ST' 501"})
    result = ev.evaluate(stt, dataset, "float16", 1)
    assert stt.loaded and stt.closed
    assert result.wer == pytest.approx(2 / 8)  # 1 substitution + « pause » manquant
    assert result.latency_median_s == pytest.approx(0.2)
    assert result.worst[0] == (1.0, "pause", "")  # l'hallucination est retirée
    report = ev.format_results([result], len(dataset))
    assert "float16" in report and "25.0%" in report and "« pause » -> (rien)" in report


def test_run_without_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        ev.main(["--dir", str(tmp_path), "--config", str(PROJECT_ROOT / "config.yaml"), "run"]) == 1
    )
    assert "record" in capsys.readouterr().out
