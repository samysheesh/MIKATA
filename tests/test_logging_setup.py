from __future__ import annotations

import logging
from pathlib import Path

import pytest
from loguru import logger

from mikata.config import LoggingConfig
from mikata.logging_setup import setup_logging, shutdown_logging, trash_old_logs


def test_file_sink_and_stdlib_interception(tmp_path: Path) -> None:
    log_file = setup_logging(LoggingConfig(level="DEBUG"), base_dir=tmp_path)
    assert log_file == tmp_path / "data" / "logs" / "mikata.log"
    logger.debug("message loguru é")
    logging.getLogger("tiers").warning("message stdlib")
    shutdown_logging()
    content = log_file.read_text(encoding="utf-8")
    assert "message loguru é" in content
    assert "message stdlib" in content


def test_level_filters(tmp_path: Path) -> None:
    log_file = setup_logging(LoggingConfig(level="WARNING"), base_dir=tmp_path)
    logger.info("trop bavard")
    logger.warning("important")
    shutdown_logging()
    content = log_file.read_text(encoding="utf-8")
    assert "trop bavard" not in content
    assert "important" in content


def test_without_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stderr", None)  # cas de pythonw.exe
    cfg = LoggingConfig(file=tmp_path / "abs.log", actions_file=tmp_path / "actions.log")
    log_file = setup_logging(cfg, base_dir=Path("/ailleurs"))  # chemins absolus : base ignorée
    logger.info("sans console")
    shutdown_logging()
    assert log_file == tmp_path / "abs.log"
    assert "sans console" in log_file.read_text(encoding="utf-8")


def test_setup_is_idempotent(tmp_path: Path) -> None:
    cfg = LoggingConfig()
    setup_logging(cfg, base_dir=tmp_path)
    log_file = setup_logging(cfg, base_dir=tmp_path)
    logger.info("une seule fois")
    shutdown_logging()
    assert log_file.read_text(encoding="utf-8").count("une seule fois") == 1


def test_retention_sends_old_logs_to_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    trashed: list[str] = []
    monkeypatch.setattr("mikata.logging_setup.send2trash", trashed.append)
    files = []
    for i in range(7):
        path = tmp_path / f"mikata.{i}.log"
        path.write_text("x", encoding="utf-8")
        os.utime(path, (1000 + i, 1000 + i))
        files.append(str(path))
    trash_old_logs(5)(files)
    assert sorted(trashed) == sorted(files[:2])  # les 2 plus anciennes


def test_actions_go_to_their_own_file(tmp_path: Path) -> None:
    """Phase 5 : chaque action d'outil dans data/actions.log, pas dans le log général."""
    from mikata.agent.guard import log_action

    log_file = setup_logging(LoggingConfig(), base_dir=tmp_path)
    log_action("delete_file", {"path": "notes.txt"}, "ok", "Mis à la corbeille : notes.txt.")
    log_action("run_powershell", {"command": "rm x"}, "refusé", "suppression définitive")
    logger.info("message ordinaire")
    shutdown_logging()
    actions = (tmp_path / "data" / "actions.log").read_text(encoding="utf-8").splitlines()
    assert len(actions) == 2
    assert actions[0].endswith(
        "| OK | delete_file(path='notes.txt') | Mis à la corbeille : notes.txt."
    )
    assert "| REFUSÉ | run_powershell(command='rm x') | suppression définitive" in actions[1]
    general = log_file.read_text(encoding="utf-8")
    assert "message ordinaire" in general and "delete_file" not in general
