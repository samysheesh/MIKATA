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
        self.prompts: list[str | None] = []

    def load(self) -> None:
        self.loaded = True

    def transcribe_sync(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        self.prompts.append(prompt)
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


def test_prompt_modes() -> None:
    """Phase 6 : sans prompt, prompt de la config, ou + noms des applis installées."""
    from mikata.config import SttConfig

    cfg = SttConfig(initial_prompt="MIKATA, Spotify")
    assert ev.prompt_for("none", cfg) == ""
    assert ev.prompt_for("config", cfg) is None
    apps = ev.prompt_for("apps", cfg, apps=lambda: ["Spotify", "Hades", "Balatro"])
    assert apps == "Applications : Balatro, Hades. MIKATA, Spotify"
    args = ev.parse_args(["run", "--prompt", "none", "apps"])
    assert args.prompt == ["none", "apps"]
    assert ev.parse_args(["run"]).prompt == ["config"]


def test_evaluate_passes_the_prompt() -> None:
    dataset = [(np.full(1600, 1, dtype=np.int16), "ouvre Spotify")]
    stt = _FakeSTT({1: "Ouvre Spotify."})
    result = ev.evaluate(stt, dataset, "float16", 1, "", "none")
    assert stt.prompts == [""] and result.prompt == "none"
    ev.evaluate(stt, dataset, "float16", 1)  # prompt de la config : rien d'imposé
    assert stt.prompts[-1] is None
    assert " none " in ev.format_results([result], 1)


# -- phase 6b : phrases courtes et prompt recopié sur du bruit -------------------------------


def test_short_phrase_list() -> None:
    phrases = ev.load_phrases(ev.PROJECT_ROOT / "scripts" / "stt_phrases_courtes_fr.txt")
    assert "et à Tokyo ?" in phrases and "et demain ?" in phrases
    assert len(phrases) >= 15 and all(len(ev.normalize(p)) <= 4 for p in phrases)


def test_prompt_leaks() -> None:
    prompt = "Applications : Hades. MIKATA, Bloc-notes, VS Code. Et à Tokyo ? Et demain ?"
    assert ev.prompt_leaks(prompt, "Et demain ?") == ["Et demain"]
    assert ev.prompt_leaks(prompt, "ouvre vs code") == ["VS Code"]
    assert ev.prompt_leaks(prompt, "Et à Paris ?") == []
    assert ev.prompt_leaks(prompt, "Hades") == []  # un seul mot : trop courant pour compter
    assert ev.prompt_leaks(prompt, "") == []


def test_evaluate_counts_prompt_copied_on_noise() -> None:
    dataset = [(np.full(1600, 1, dtype=np.int16), "et à Tokyo")]
    noise = [
        ("a.wav", np.full(1600, 2, dtype=np.int16)),
        ("b.wav", np.full(1600, 3, dtype=np.int16)),
        ("c.wav", np.full(1600, 4, dtype=np.int16)),
    ]
    stt = _FakeSTT({1: "Et à Tokyo ?", 2: "Et demain ?", 3: "", 4: "Bon, j'y vais."})
    prompt = "MIKATA, Spotify. Et à Tokyo ? Et demain ?"
    result = ev.evaluate(stt, dataset, "float16", 1, None, "config", noise, prompt)
    assert result.wer == 0.0 and result.noise_clips == 3
    assert result.noise_leaks == [("a.wav", "Et demain ?")]
    report = ev.format_results([result], 1)
    assert "1 extrait(s) sur 3 où le prompt est recopié" in report and "a.wav" in report


def test_load_noise(tmp_path: Path) -> None:
    from mikata.audio.wavfile import write_wav

    write_wav(tmp_path / "1.wav", np.zeros(1600, dtype=np.int16), 16_000)
    write_wav(tmp_path / "2.wav", np.zeros(4800, dtype=np.int16), 48_000)  # ignoré
    write_wav(tmp_path / "3.wav", np.zeros(800, dtype=np.int16), 16_000)
    clips = ev.load_noise(tmp_path)
    assert [name for name, _ in clips] == ["1.wav", "3.wav"]
    assert clips[0][1].dtype == np.int16 and len(clips[0][1]) == 1600
    assert len(ev.load_noise(tmp_path, max_files=1)) == 1


def test_run_options_for_short_phrases() -> None:
    args = ev.parse_args(
        ["--dir", "data/stt_eval_courtes", "run", "--prompt", "apps", "--initial-prompt", "MIKATA"]
    )
    assert args.dir == Path("data/stt_eval_courtes") and args.initial_prompt == "MIKATA"
    assert args.noise is None and args.noise_max == ev.NOISE_MAX_FILES
    args = ev.parse_args(["run", "--noise", "data/negatives", "--noise-max", "10"])
    assert args.noise == Path("data/negatives") and args.noise_max == 10
