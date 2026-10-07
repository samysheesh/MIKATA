from __future__ import annotations

import logging
from pathlib import Path

import pytest
from loguru import logger

from mikata.config import LoggingConfig
from mikata.logging_setup import setup_logging, shutdown_logging


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
    log_file = setup_logging(LoggingConfig(file=tmp_path / "abs.log"), base_dir=Path("/ailleurs"))
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
