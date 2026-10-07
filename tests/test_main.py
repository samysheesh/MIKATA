from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

import mikata.__main__ as entry
from mikata.app import State
from mikata.config import PROJECT_ROOT


def _log(config_file: Path) -> str:
    return (config_file.parent / "data" / "logs" / "mikata.log").read_text(encoding="utf-8")


def test_check_config_ok(config_file: Path) -> None:
    assert entry.main(["--config", str(config_file), "--check-config"]) == entry.EXIT_OK
    log = _log(config_file)
    assert "Configuration valide" in log
    assert "LLM_API_KEY : absente" in log


def test_check_config_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "config_invalide.yaml"
    bad.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    assert entry.main(["--config", str(bad), "--check-config"]) == entry.EXIT_CONFIG
    log = _log(bad)
    assert "wake.threshold" in log
    assert "wake.treshold" in log


def test_missing_config(tmp_path: Path) -> None:
    assert entry.main(["--config", str(tmp_path / "absent.yaml")]) == entry.EXIT_CONFIG


def test_secret_is_never_logged(config_file: Path) -> None:
    (config_file.parent / ".env").write_text("LLM_API_KEY=sk-ne-pas-loguer\n", encoding="utf-8")
    assert entry.main(["--config", str(config_file), "--check-config"]) == entry.EXIT_OK
    log = _log(config_file)
    assert "LLM_API_KEY : définie" in log
    assert "sk-ne-pas-loguer" not in log


class AutoQuitTray:
    instances: ClassVar[list[AutoQuitTray]] = []

    def __init__(self) -> None:
        self.stopped = False
        AutoQuitTray.instances.append(self)

    def start(self, on_quit: Callable[[], None]) -> None:
        threading.Timer(0.05, on_quit).start()

    def set_state(self, state: State) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def test_run_and_quit_from_tray(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entry, "TrayIcon", AutoQuitTray)
    assert entry.main(["--config", str(config_file)]) == entry.EXIT_OK
    assert AutoQuitTray.instances[-1].stopped
    log = _log(config_file)
    assert "Arrêt demandé (menu Quitter)" in log
    assert "MIKATA arrêté proprement." in log


def test_tray_failure_is_fatal(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenTray(AutoQuitTray):
        def start(self, on_quit: Callable[[], None]) -> None:
            raise ImportError("No module named 'pystray'")

    monkeypatch.setattr(entry, "TrayIcon", BrokenTray)
    assert entry.main(["--config", str(config_file)]) == entry.EXIT_ERROR
    assert "Erreur fatale" in _log(config_file)


def test_keyboard_interrupt_is_clean(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(coro: object) -> None:
        coro.close()  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(entry.asyncio, "run", interrupted)
    assert entry.main(["--config", str(config_file), "--no-tray"]) == entry.EXIT_OK
    assert "Ctrl+C" in _log(config_file)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        entry.main(["--version"])
    assert info.value.code == 0
    assert "MIKATA" in capsys.readouterr().out
