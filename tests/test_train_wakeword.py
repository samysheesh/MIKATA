"""Phase 7 : entraînement sur Colab (train_wakeword.py) et vérificateur (train_verifier.py).

Le vrai entraînement (GPU, ~17 Go de données publiques) tourne sur Colab ; ici on vérifie la
préparation des données, la configuration donnée à openwakeword/train.py et les commandes
d'installation, avec Piper, openWakeWord et le réseau simulés."""

from __future__ import annotations

import random
import re
import sys
import types
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from scripts_loader import load_script

from mikata.audio.wavfile import write_wav
from mikata.wake.dataset import HARD_NEGATIVES, WakewordData

tw = load_script("train_wakeword")
tv = load_script("train_verifier")
SR = 16_000


def _settings(tmp_path: Path) -> Any:
    return tw.Settings(workdir=tmp_path / "work", data=tmp_path / "data", n_positive=40)


def _voice_clip(rng: np.random.Generator, rate: int = SR) -> np.ndarray:
    quiet = rng.normal(0, 40, rate // 2)
    return np.concatenate((quiet, rng.normal(0, 6000, rate // 2), quiet)).astype(np.int16)


def test_openwakeword_train_reads_only_keys_we_provide(tmp_path: Path) -> None:
    """Toutes les clés lues par openwakeword/train.py (0.6.0) sont dans notre configuration."""
    import openwakeword

    source = (Path(openwakeword.__file__).parent / "train.py").read_text(encoding="utf-8")
    read = set(re.findall(r"config\[[\"'](\w+)[\"']\]", source)) - {"total_length"}  # calculée
    s = _settings(tmp_path)
    config = tw.training_config(s, [tmp_path / "bg"], [], tmp_path / "f.npy", tmp_path / "stub")
    assert read <= set(config), read - set(config)
    assert config["target_false_positives_per_hour"] == 0.2  # cible du §4
    assert config["feature_data_files"]["mikata_vie_courante"] == str(tmp_path / "f.npy")
    assert config["batch_n_per_class"]["mikata_vie_courante"] > 0
    assert config["rir_paths"] == [] and config["background_paths"] == [str(tmp_path / "bg")]
    plain = tw.training_config(s, [], [], None, tmp_path / "stub")
    assert "mikata_vie_courante" not in plain["feature_data_files"]


def test_hard_negative_texts_cover_what_you_record() -> None:
    assert set(HARD_NEGATIVES) <= set(tw.NEGATIVE_TEXTS["fr"])
    assert {"mikata", "mi kata", "mee kah tah"} <= {t.lower() for t in tw.POSITIVE_TEXTS["en"]}


def test_resample_and_trim() -> None:
    rng = np.random.default_rng(0)
    clip = _voice_clip(rng, 22_050)
    resampled = tw.to_16k(clip, 22_050)
    assert resampled.dtype == np.int16 and abs(len(resampled) - round(len(clip) * SR / 22_050)) <= 1
    trimmed = tw.trim(tw.to_16k(_voice_clip(rng), SR), margin_s=0.1)
    assert len(trimmed) == pytest.approx(0.7 * SR, abs=400)  # 0,5 s de voix + 2 × 0,1 s


def test_copy_user_clips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    write_wav(tmp_path / "normal__001.wav", _voice_clip(rng), SR)
    n = tw.copy_user_clips([tmp_path / "normal__001.wav"], tmp_path / "out", 3, "user")
    assert n == 3 and sorted(p.name for p in (tmp_path / "out").glob("*.wav")) == [
        "user_normal__001_00.wav",
        "user_normal__001_01.wav",
        "user_normal__001_02.wav",
    ]
    with wave.open(str(tmp_path / "out" / "user_normal__001_00.wav")) as w:
        assert w.getframerate() == SR and w.getnframes() < SR * 1.5


def test_tts_jobs_mix_french_and_english_voices() -> None:
    voices = [("fr_FR-mls-medium.onnx", "fr", 125), ("en_US-libritts_r-medium.onnx", "en", 904)]
    jobs = tw.tts_jobs(voices, tw.POSITIVE_TEXTS, 100, random.Random(0))
    assert len(jobs) == 100
    french = [j for j in jobs if j.voice.startswith("fr")]
    assert len(french) == 50 and all(j.text in tw.POSITIVE_TEXTS["fr"] for j in french)
    assert all(0 <= j.speaker < 125 for j in french)  # type: ignore[operator]
    assert len({j.speaker for j in french}) > 20  # beaucoup de locuteurs différents
    assert all(0.75 <= j.length_scale <= 1.35 for j in jobs)
    single = tw.tts_jobs(
        [("fr_FR-siwis-medium.onnx", "fr", 1)], tw.POSITIVE_TEXTS, 3, random.Random(0)
    )
    assert {j.speaker for j in single} == {None}


def test_voice_inventory(tmp_path: Path) -> None:
    (tmp_path / "fr_FR-mls-medium.onnx").write_bytes(b"x")
    (tmp_path / "fr_FR-mls-medium.onnx.json").write_text('{"num_speakers": 125}')
    (tmp_path / "fr_FR-tom-medium.onnx").write_bytes(b"x")  # sans .json : ignorée
    assert tw.voice_inventory(tmp_path) == [("fr_FR-mls-medium.onnx", "fr", 125)]


def test_synthesize_with_piper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)

    class Chunk:
        sample_rate = 22_050

        def __init__(self) -> None:
            self.audio_int16_array = _voice_clip(rng, 22_050)

    class Voice:
        loaded: ClassVar[list[Path]] = []

        @classmethod
        def load(cls, path: Path) -> Voice:
            cls.loaded.append(path)
            return cls()

        def synthesize(self, text: str, config: Any) -> list[Chunk]:
            assert config.speaker_id == 7 and text == "Mikata"
            return [Chunk()]

    piper = types.ModuleType("piper")
    piper.PiperVoice = Voice  # type: ignore[attr-defined]
    piper.SynthesisConfig = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", piper)
    job = tw.TtsJob("fr_FR-mls-medium.onnx", 7, "Mikata", 1.0, 0.6, 0.8)
    tw.synthesize([job, job], tmp_path / "voices", tmp_path / "out", "tts")
    tw.synthesize([job, job], tmp_path / "voices", tmp_path / "out", "tts")  # reprise : rien
    assert sorted(p.name for p in (tmp_path / "out").glob("*.wav")) == [
        "tts_000000.wav",
        "tts_000001.wav",
    ]
    assert Voice.loaded == [tmp_path / "voices" / "fr_FR-mls-medium.onnx"]


def test_user_negative_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import openwakeword.utils

    class Features:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["device"] == "cpu"

        def _get_embeddings(self, audio: np.ndarray) -> np.ndarray:
            return np.ones((len(audio) // 1280 - 9, 96), dtype=np.float32)

    monkeypatch.setattr(openwakeword.utils, "AudioFeatures", Features)
    clips = []
    for i in range(2):
        clips.append(tmp_path / f"{i}.wav")
        write_wav(clips[-1], np.zeros(10 * SR, dtype=np.int16), SR)
    write_wav(tmp_path / "short.wav", np.zeros(SR, dtype=np.int16), SR)  # < 2 s : ignoré
    out = tmp_path / "f.npy"
    assert tw.user_negative_features([*clips, tmp_path / "short.wav"], out) == 2 * 7
    saved = np.load(out)
    assert saved.shape == (14, 16, 96) and saved.dtype == np.float16
    assert tw.user_negative_features(clips, out) == 14  # déjà calculées : reprise


def test_setup_installs_an_isolated_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    downloads: list[str] = []
    monkeypatch.setattr(tw, "run", lambda cmd, **kw: commands.append([str(c) for c in cmd]))
    monkeypatch.setattr(tw, "download", lambda url, dest: downloads.append(url) or dest)
    monkeypatch.setattr(
        tw.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=str(tmp_path / "oww") + "\n"),
    )
    tw.setup(_settings(tmp_path))
    flat = [" ".join(c) for c in commands]
    assert any("uv venv --python 3.11" in c for c in flat)
    assert any("torch==2.1.2" in c and tw.TORCH_INDEX in c for c in flat)
    assert any("--no-deps openwakeword==0.6.0 piper-tts==1.8.0" in c for c in flat)
    assert not any("tflite" in c or "onnxruntime-gpu" in c for c in flat)
    assert any(u.endswith("fr_FR-mls-medium.onnx") for u in downloads)
    assert any(u.endswith(tw.ACAV_FILE) for u in downloads)
    assert any(u.endswith("embedding_model.onnx") for u in downloads)


def test_parse_args() -> None:
    command, s = tw.parse_args(["train", "--workdir", "/content/m", "--steps", "1000"])
    assert command == "train" and s.steps == 1000 and s.workdir.name == "m"
    assert s.python == s.workdir / "venv" / "bin" / "python"  # Colab : Linux
    assert s.data == tw.HERE / "data"


# -- vérificateur ----------------------------------------------------------------------------


def test_verifier_uses_train_takes_and_your_voice_only(tmp_path: Path) -> None:
    data = WakewordData(tmp_path / "ww")
    rng = np.random.default_rng(3)
    for split, cond in (("train", "normal"), ("train", "phrase"), ("eval", "normal")):
        write_wav(data.positives(split, cond) / "001.wav", _voice_clip(rng), SR)
    write_wav(data.hard_negatives("mikado") / "001.wav", _voice_clip(rng), SR)
    stt = tmp_path / "stt"
    write_wav(stt / "01.wav", _voice_clip(rng), SR)
    (stt / "01.txt").write_text("ouvre Spotify", encoding="utf-8")
    write_wav(stt / "02.wav", _voice_clip(rng), SR)
    (stt / "02.txt").write_text("merci MIKATA c'est tout", encoding="utf-8")
    positives, negatives = tv.reference_clips(data, stt)
    assert [p.parent.name for p in positives] == ["normal", "phrase"]  # pas les prises « eval »
    assert negatives == [data.hard_negatives("mikado") / "001.wav", stt / "01.wav"]

    calls: list[dict[str, Any]] = []
    padded = tv.load_padded(positives, np.random.default_rng(0))
    assert len(padded[0]) == len(_voice_clip(rng)) + round((tv.LEAD_S + tv.TAIL_S) * SR)
    model = tmp_path / "models" / "mikata.onnx"
    tv.train(model, tmp_path / "v.pkl", padded, padded, trainer=lambda **kw: calls.append(kw))
    assert calls[0]["model_name"] == str(model) and calls[0]["inference_framework"] == "onnx"
    assert calls[0]["melspec_model_path"] == str(tmp_path / "models" / "melspectrogram.onnx")


def test_verifier_main_needs_the_model_and_takes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tv.main(["--model", str(tmp_path / "absent.onnx")]) == 1
    assert "Modèle introuvable" in capsys.readouterr().out
    model = tmp_path / "mikata.onnx"
    model.write_bytes(b"x")
    assert tv.main(["--model", str(model), "--root", str(tmp_path / "ww")]) == 1
    assert "Pas assez de prises" in capsys.readouterr().out


def _pcm_bytes(width: int, samples: int = 3200) -> bytes:
    t = np.sin(np.linspace(0, 60, samples)) * 0.4
    if width == 1:
        return ((t * 127) + 128).astype(np.uint8).tobytes()
    if width == 2:
        return (t * 32767).astype(np.int16).tobytes()
    if width == 3:
        v = (t * 8388607).astype(np.int32)
        return np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes()
    return (t * 2147483647).astype(np.int32).tobytes()


def test_extract_rirs_handles_any_sample_width_and_keeps_dest_flat(tmp_path: Path) -> None:
    """Réponses de pièces : 16, 24 et 32 bits sont lues, un fichier illisible est ignoré, et
    `dest` ne contient que des .wav (openwakeword plante sur un sous-dossier)."""
    import io
    import zipfile

    archive = tmp_path / "mit_rirs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for width in (2, 3, 4):
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(width)
                w.setframerate(32_000)
                w.writeframes(_pcm_bytes(width))
            z.writestr(f"rirs/w{width}.wav", buffer.getvalue())
        z.writestr("rirs/casse.wav", b"RIFFnot-a-wav")
    dest = tmp_path / "downloads" / "rirs16k"

    found = tw.extract_rirs(archive, dest)

    assert [p.name for p in found] == ["w2.wav", "w3.wav", "w4.wav"]
    assert all(p.suffix == ".wav" for p in dest.iterdir())  # que des .wav : ni dossier ni marqueur
    # (openwakeword essaie de lire tout ce que contient `dest` comme un son)
    assert (dest.parent / ".rirs16k_extrait").is_file()
    with wave.open(str(dest / "w3.wav"), "rb") as w:
        assert w.getframerate() == SR
    assert tw.extract_rirs(archive, dest) == found  # déjà fait : rien n'est refait


def test_extract_rirs_redoes_an_interrupted_extraction(tmp_path: Path) -> None:
    import io
    import zipfile

    archive = tmp_path / "mit_rirs.zip"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32_000)
        w.writeframes(_pcm_bytes(2))
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.wav", buffer.getvalue())
        z.writestr("b.wav", buffer.getvalue())
    dest = tmp_path / "rirs16k"
    dest.mkdir()
    tw.write_wav(
        dest / "a.wav", np.zeros(100, dtype=np.int16)
    )  # essai interrompu : un seul fichier

    assert [p.name for p in tw.extract_rirs(archive, dest)] == ["a.wav", "b.wav"]


def test_incomplete_features_from_an_interrupted_run_are_detected(tmp_path: Path) -> None:
    assert not tw.incomplete_features(tmp_path)  # rien : openwakeword calcule tout
    (tmp_path / "positive_features_train.npy").write_bytes(b"x")
    assert tw.incomplete_features(tmp_path)  # reste d'un essai interrompu : à recalculer
    for name in tw.FEATURE_FILES:
        (tmp_path / name).write_bytes(b"x")
    assert not tw.incomplete_features(tmp_path)  # tout est là : rien à refaire
