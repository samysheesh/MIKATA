from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from scripts_loader import load_script

dm = load_script("download_models")


def test_model_list() -> None:
    assert set(dm.MODELS) == {
        "hey_jarvis.onnx",
        "melspectrogram.onnx",
        "embedding_model.onnx",
        "silero_vad.onnx",
        "fr_FR-siwis-medium.onnx",
        "fr_FR-siwis-medium.onnx.json",
        "kokoro-v1.0.onnx",
        "voices-v1.0.bin",
    }
    assert "/resolve/v1.0.0/" in dm.MODELS["fr_FR-siwis-medium.onnx"]  # version figée
    assert "v5.1.2" in dm.MODELS["silero_vad.onnx"]  # version figée


def test_downloads_missing_files_and_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    (tmp_path / "hey_jarvis.onnx").write_bytes(b"deja la")
    fetched: list[str] = []

    def fake_download(url: str, dest: Path) -> int:
        fetched.append(dest.name)
        dest.write_bytes(b"x")
        return 1

    whisper: list[tuple[str, Path]] = []
    monkeypatch.setattr(dm, "download", fake_download)
    monkeypatch.setattr(dm, "download_whisper", lambda m, d: whisper.append((m, d)) or d)
    monkeypatch.setattr(dm, "download_parakeet_mlx", lambda: 0)  # pas de vrai snapshot_download
    assert dm.main(["--dir", str(tmp_path), "--config", str(config_file)]) == 0
    expected = set(dm.MODELS) - {"hey_jarvis.onnx"}
    if dm.sys.platform == "linux":  # assets Parakeet sherpa, Linux uniquement
        expected |= set(dm.SHERPA_FILES)
    assert sorted(fetched) == sorted(expected)
    assert (tmp_path / "hey_jarvis.onnx").read_bytes() == b"deja la"  # jamais écrasé
    if dm.sys.platform == "win32":  # Whisper = backend Windows uniquement
        assert whisper == [("large-v3-turbo", tmp_path / "whisper")]
    else:
        assert whisper == []


def test_parakeet_mlx_prefetch_darwin_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_hub = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or repo)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    monkeypatch.setattr(dm.sys, "platform", "win32")
    assert dm.download_parakeet_mlx() == 0
    assert calls == []  # rien téléchargé hors macOS

    monkeypatch.setattr(dm.sys, "platform", "darwin")
    assert dm.download_parakeet_mlx() == 0
    assert calls == [dm.PARAKEET_MLX_REPO]  # cache HuggingFace préchargé


def test_failures_give_exit_code_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> int:
        raise OSError("403")

    monkeypatch.setattr(dm, "download", broken)
    monkeypatch.setattr(dm, "download_whisper", broken)
    assert dm.main(["--dir", str(tmp_path)]) == 1
    assert dm.main(["--dir", str(tmp_path), "--no-whisper"]) == 1
