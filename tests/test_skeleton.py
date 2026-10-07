"""L'arborescence de la section 9 existe et chaque module s'importe sans Windows."""

from __future__ import annotations

import importlib

import pytest

from mikata.config import PROJECT_ROOT

MODULES = [
    "mikata.__main__",
    "mikata.app",
    "mikata.config",
    "mikata.events",
    "mikata.governor",
    "mikata.logging_setup",
    *(
        f"mikata.audio.{m}"
        for m in ("base", "capture", "player", "ringbuffer", "earcons", "devices", "wavfile")
    ),
    *(f"mikata.wake.{m}" for m in ("base", "oww", "trigger", "recordings")),
    *(f"mikata.vad.{m}" for m in ("base", "silero")),
    *(f"mikata.stt.{m}" for m in ("base", "whisper_local", "speculative")),
    *(f"mikata.llm.{m}" for m in ("base", "client", "prompts")),
    *(f"mikata.agent.{m}" for m in ("loop", "memory", "guard")),
    *(
        f"mikata.tools.{m}"
        for m in (
            "base",
            "registry",
            "system",
            "apps",
            "web",
            "media",
            "input",
            "files",
            "shell",
            "ui",
            "memory_tools",
        )
    ),
    *(f"mikata.tts.{m}" for m in ("base", "piper_tts", "kokoro_tts", "normalize", "cache")),
    *(f"mikata.ui.{m}" for m in ("tray", "notify", "hotkeys")),
]

PATHS = [
    "pyproject.toml",
    "uv.lock",
    "config.yaml",
    ".env.example",
    "assets/sounds",
    "models",
    "data",
    "scripts/install_autostart.ps1",
    "scripts/record_wakeword_samples.py",
    "scripts/eval_wakeword.py",
    "scripts/eval_stt.py",
    "scripts/list_audio_devices.py",
    "scripts/download_models.py",
    *(f"assets/sounds/{n}.wav" for n in ("wake", "end", "thinking", "error")),
    "scripts/user",
    "tests",
    "docs/ARCHITECTURE.md",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("path", PATHS)
def test_path_exists(path: str) -> None:
    assert (PROJECT_ROOT / path).exists()


def test_no_secret_file_committed() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
