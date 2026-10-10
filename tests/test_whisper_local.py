from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from mikata.config import StartupError, SttConfig
from mikata.stt import whisper_local
from mikata.stt.base import STT
from mikata.stt.whisper_local import SttUnavailableError, WhisperSTT


class FakeSegment:
    def __init__(
        self, text: str, no_speech_prob: float | None = None, avg_logprob: float | None = None
    ) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class FakeWhisperModel:
    instances: ClassVar[list[FakeWhisperModel]] = []
    missing_locally: ClassVar[bool] = False
    error: ClassVar[Exception | None] = None

    def __init__(self, model: str, **kwargs: Any) -> None:
        if kwargs.get("local_files_only") and FakeWhisperModel.missing_locally:
            raise FileNotFoundError("Cannot find an appropriate cached snapshot folder")
        self.model = model
        self.kwargs = kwargs
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
        if FakeWhisperModel.error is not None:
            raise FakeWhisperModel.error
        self.calls.append((audio, kwargs))
        return iter([FakeSegment(" Hey Jarvis,"), FakeSegment(" ouvre Spotify.")]), None


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    state = types.SimpleNamespace(cuda_devices=1)
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    ct2 = types.ModuleType("ctranslate2")
    ct2.get_cuda_device_count = lambda: state.cuda_devices  # type: ignore[attr-defined]
    FakeWhisperModel.instances = []
    FakeWhisperModel.missing_locally = False
    FakeWhisperModel.error = None
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setattr(whisper_local, "gpu_memory_used_mb", lambda: 1000.0)
    return state


def _stt(tmp_path: Path, **overrides: Any) -> WhisperSTT:
    # Ces tests ciblent le backend faster-whisper/CUDA (modules simulés) : le moteur
    # est fixé car le défaut dépend de la plateforme depuis F1 (mlx sous macOS).
    return WhisperSTT(
        SttConfig().model_copy(update={"device": "cuda", **overrides}), download_root=tmp_path
    )


def test_load_and_transcribe(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    stt = _stt(tmp_path)
    assert isinstance(stt, STT)
    stt.load()
    model = FakeWhisperModel.instances[-1]
    assert model.model == "large-v3-turbo"
    assert model.kwargs == {
        "device": "cuda",
        "compute_type": "int8_float16",
        "download_root": str(tmp_path),
        "local_files_only": True,
    }
    warmup, _ = model.calls[0]
    assert len(warmup) == 16_000 and warmup.dtype == np.float32  # préchauffage 1 s

    audio = np.full(8_000, 16384, dtype=np.int16)
    transcript = asyncio.run(stt.transcribe(audio))
    assert transcript.text == "Hey Jarvis, ouvre Spotify."
    assert transcript.duration_s == 0.5
    pcm, kwargs = model.calls[-1]
    assert np.allclose(pcm, 0.5)
    assert kwargs["language"] == "fr"
    assert kwargs["beam_size"] == 1
    assert kwargs["without_timestamps"] is True
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False
    assert kwargs["initial_prompt"].startswith("MIKATA")
    stt.close()
    assert not stt.loaded


def test_downloads_when_not_cached(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    FakeWhisperModel.missing_locally = True
    _stt(tmp_path).load()
    assert "local_files_only" not in FakeWhisperModel.instances[-1].kwargs


def test_no_gpu_is_a_clear_error(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    fake_whisper.cuda_devices = 0
    with pytest.raises(SttUnavailableError, match="nvidia-smi") as info:
        _stt(tmp_path).load()
    assert isinstance(info.value, StartupError)
    assert "bascule CPU" in str(info.value)
    assert FakeWhisperModel.instances == []


@pytest.mark.parametrize(
    ("message", "hint"),
    [
        ("Library cublas64_12.dll is not found or cannot be loaded", "uv sync"),
        ("Could not load library cudnn_ops64_9.dll", "nvidia-cudnn-cu12"),
        ("CUDA failed with error out of memory", "int8_float16"),
        ("CUDA driver version is insufficient", "pilote NVIDIA"),
    ],
)
def test_cuda_errors_are_explained(
    fake_whisper: types.SimpleNamespace, tmp_path: Path, message: str, hint: str
) -> None:
    FakeWhisperModel.error = RuntimeError(message)  # cuDNN ne se charge qu'au 1er calcul
    with pytest.raises(SttUnavailableError, match=hint):
        _stt(tmp_path).load()


def test_cpu_device_skips_gpu_checks(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    fake_whisper.cuda_devices = 0
    _stt(tmp_path, device="cpu", compute_type="int8").load()
    assert FakeWhisperModel.instances[-1].kwargs["device"] == "cpu"


def test_transcribe_before_load(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="pas chargé"):
        _stt(tmp_path).transcribe_sync(np.zeros(10, dtype=np.int16))


def test_cuda_dll_directories_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for package in ("cublas", "cudnn"):
        (tmp_path / "nvidia" / package / "bin").mkdir(parents=True)
    spec = types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])
    monkeypatch.setattr(whisper_local.importlib.util, "find_spec", lambda name: spec)
    added: list[str] = []
    monkeypatch.setattr(whisper_local.sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows")

    dirs = whisper_local.add_cuda_dll_directories()
    expected = [tmp_path / "nvidia" / "cublas" / "bin", tmp_path / "nvidia" / "cudnn" / "bin"]
    assert dirs == expected
    assert added == [str(d) for d in expected]
    assert os.environ["PATH"].split(os.pathsep)[:2] == [str(d) for d in expected]


def test_no_dll_directories_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_local.sys, "platform", "linux")
    assert whisper_local.add_cuda_dll_directories() == []


@pytest.mark.skipif(sys.platform == "win32", reason="vérifie la machine de test sans GPU")
def test_real_ctranslate2_without_gpu(tmp_path: Path) -> None:
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        pytest.skip("cette machine a un GPU")
    with pytest.raises(SttUnavailableError, match="Aucun GPU NVIDIA"):
        _stt(tmp_path).load()


def test_build_prompt_adds_short_app_names_first() -> None:
    from mikata.stt.whisper_local import build_prompt

    base = "MIKATA, Bloc-notes, Spotify"
    names = [
        "Spotify",  # déjà dans le prompt
        "Counter-Strike 2",
        "Microsoft Flight Simulator 2024 Édition Premium",  # trop long pour être dit
        "Discord",
        "C:\\Program Files\\truc.exe",
        "Steam",
    ]
    prompt = build_prompt(base, names, max_chars=500)
    # Applis d'abord, vocabulaire et amorces à la fin (juste avant ta phrase)
    assert prompt == "Applications : Counter-Strike 2, Discord, Steam. MIKATA, Bloc-notes, Spotify"
    short = build_prompt(base, names, max_chars=len(base) + 30)
    assert short == "Applications : Steam. MIKATA, Bloc-notes, Spotify"  # les plus courts
    assert build_prompt(base, [], 500) == base
    assert build_prompt("", ["Steam"], 500) == "Applications : Steam."


def test_prompt_ends_with_short_follow_up_primers() -> None:
    """Phase 6b : « et à Tokyo ? » lu « Yeah, Tokyo » ; les amorces de suites courtes
    terminent le prompt, même avec des noms d'applications."""
    from mikata.stt.whisper_local import build_prompt

    base = SttConfig().initial_prompt
    assert base.endswith("Et à Tokyo ? Et demain ?") and base.startswith("MIKATA")
    prompt = build_prompt(base, ["Hades", "Balatro"], 500)
    assert prompt == f"Applications : Balatro, Hades. {base}"
    assert len(build_prompt(base, [f"Jeu {i}" for i in range(200)], 500)) <= 500


def test_prompt_follows_installed_apps(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    apps = ["Discord", "Hades"]  # Discord est déjà dans stt.initial_prompt
    cuda = SttConfig().model_copy(update={"device": "cuda"})
    stt = WhisperSTT(cuda, download_root=tmp_path, vocabulary=lambda: apps)
    assert stt.prompt == f"Applications : Hades. {SttConfig().initial_prompt}"
    apps.append("Balatro")
    assert stt.prompt.startswith("Applications : Balatro, Hades. MIKATA")  # recalculé
    stt.load()
    asyncio.run(stt.transcribe(np.zeros(1600, dtype=np.int16)))
    _, kwargs = FakeWhisperModel.instances[-1].calls[-1]
    assert kwargs["initial_prompt"] == stt.prompt
    off = WhisperSTT(
        cuda.model_copy(update={"prompt_apps": False}),
        download_root=tmp_path,
        vocabulary=lambda: apps,
    )
    assert off.prompt == SttConfig().initial_prompt

    def broken() -> list[str]:
        raise RuntimeError("index pas prêt")

    assert WhisperSTT(cuda, tmp_path, vocabulary=broken).prompt == stt.cfg.initial_prompt
    stt.close()


def test_explicit_prompt_overrides_config(
    fake_whisper: types.SimpleNamespace, tmp_path: Path
) -> None:
    stt = _stt(tmp_path)
    stt.load()
    stt.transcribe_sync(np.zeros(1600, dtype=np.int16), "")  # eval_stt --prompt none
    assert FakeWhisperModel.instances[-1].calls[-1][1]["initial_prompt"] is None
    stt.close()


def test_reliability_of_segments() -> None:
    """Phase 6 : confiance de la transcription (filtre de la fenêtre de suivi)."""
    from mikata.stt.whisper_local import _reliability

    segments = [FakeSegment(" Et à", 0.1, -0.2), FakeSegment(" Tokyo ?", 0.3, -0.6)]
    no_speech, logprob = _reliability(segments)
    assert no_speech == 0.3  # le pire segment
    assert logprob == pytest.approx((-0.2 * 4 + -0.6 * 7) / 11)  # pondéré par le texte
    assert _reliability([FakeSegment(" ok")]) == (None, None)
