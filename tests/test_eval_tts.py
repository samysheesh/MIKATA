from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scripts_loader import load_script

from mikata.audio.wavfile import read_wav
from mikata.config import PROJECT_ROOT

et = load_script("eval_tts")


class FakeTTS:
    sample_rate = 24_000

    def __init__(self, seconds_per_word: float = 0.3) -> None:
        self.loaded = self.closed = False
        self.texts: list[str] = []
        self.seconds_per_word = seconds_per_word

    def load(self) -> None:
        self.loaded = True

    def synthesize_sync(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.full(int(len(text.split()) * self.seconds_per_word * 24_000), 0.1, np.float32)

    def close(self) -> None:
        self.closed = True


def test_first_fragment_has_eight_words() -> None:
    assert len(et.FIRST_FRAGMENT.split()) == 8


def test_measure_writes_wavs(tmp_path: Path) -> None:
    tts = FakeTTS()
    result = et.measure(tts, "kokoro", runs=3, out_dir=tmp_path)
    assert tts.loaded and tts.closed
    assert tts.texts[0] == "Bonjour."  # préchauffage
    assert tts.texts.count(et.FIRST_FRAGMENT) == 4  # 3 mesures + la 1re phrase écrite
    assert "18 degrés" in tts.texts[-3]  # normalisé comme dans MIKATA
    assert [p.name for p in result.files] == [f"kokoro_{i:02d}.wav" for i in range(1, 6)]
    pcm, rate = read_wav(result.files[0])
    assert rate == 24_000 and len(pcm) > 0
    assert result.first_fragment_ms >= 0 and result.rtf > 0


def test_format_results_flags_target() -> None:
    fast = et.Result(
        "piper", 0.5, 45.0, 0.04, 22_050, [Path("a/piper_01.wav"), Path("a/piper_05.wav")]
    )
    slow = et.Result(
        "kokoro", 2.0, 350.0, 0.3, 24_000, [Path("a/kokoro_01.wav"), Path("a/kokoro_05.wav")]
    )
    table = et.format_results([fast, slow])
    lines = table.splitlines()
    assert "OK" in lines[1] and "> 200 ms" in lines[2]
    assert "piper_05.wav" in table


def test_main_reports_missing_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    assert et.main(["--config", str(config), "--out", str(tmp_path / "out")]) == 1
    assert "download_models" in capsys.readouterr().err
