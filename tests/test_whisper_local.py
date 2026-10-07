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
    def __init__(self, text: str) -> None:
        self.text = text


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
    return WhisperSTT(SttConfig().model_copy(update=overrides), download_root=tmp_path)


def test_load_and_transcribe(fake_whisper: types.SimpleNamespace, tmp_path: Path) -> None:
    stt = _stt(tmp_path)
    assert isinstance(stt, STT)
    stt.load()
    model = FakeWhisperModel.instances[-1]
    assert model.model == "large-v3-turbo"
    assert model.kwargs == {
        "device": "cuda",
        "compute_type": "float16",
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
