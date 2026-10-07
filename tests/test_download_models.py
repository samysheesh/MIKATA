from __future__ import annotations

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
    }
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
    assert dm.main(["--dir", str(tmp_path), "--config", str(config_file)]) == 0
    assert sorted(fetched) == ["embedding_model.onnx", "melspectrogram.onnx", "silero_vad.onnx"]
    assert (tmp_path / "hey_jarvis.onnx").read_bytes() == b"deja la"  # jamais écrasé
    assert whisper == [("large-v3-turbo", tmp_path / "whisper")]


def test_failures_give_exit_code_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> int:
        raise OSError("403")

    monkeypatch.setattr(dm, "download", broken)
    monkeypatch.setattr(dm, "download_whisper", broken)
    assert dm.main(["--dir", str(tmp_path)]) == 1
    assert dm.main(["--dir", str(tmp_path), "--no-whisper"]) == 1
