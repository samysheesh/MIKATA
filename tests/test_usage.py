"""Tokens consommés et coût estimé par commande (ligne de log + cumul sur disque)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mikata.config import LlmConfig
from mikata.llm.base import Usage
from mikata.llm.usage import UsageTracker, cost_usd

CFG = LlmConfig()  # 0,30 $ entrée, 1,20 $ sortie, 0,006 $ cache, crédit 5 $


def test_cost_formula_for_deepseek_flash_on_opencode_zen() -> None:
    usage = Usage(prompt=1_000_000, completion=1_000_000, cached=0)
    assert cost_usd(usage, CFG) == pytest.approx(0.30 + 1.20)
    cached = Usage(prompt=1_000_000, completion=0, cached=1_000_000)
    assert cost_usd(cached, CFG) == pytest.approx(0.006)
    typical = Usage(prompt=2500, completion=15, cached=2304)  # « ouvre Spotify »
    assert cost_usd(typical, CFG) == pytest.approx((196 * 0.30 + 2304 * 0.006 + 15 * 1.20) / 1e6)


def test_record_logs_and_accumulates(tmp_path: Path, caplog_loguru: list[str]) -> None:
    path = tmp_path / "data" / "llm_usage.json"
    tracker = UsageTracker(CFG, path)
    first = tracker.record("#1", Usage(2500, 15, 2304, 0), calls=1)
    tracker.record("#2", Usage(4000, 60, 3000, 20), calls=2)
    line = caplog_loguru[0]
    assert line.startswith(
        "Tokens #1 : entrée 2500 (dont cache 2304), sortie 15 (dont réflexion 0)"
    )
    assert "1 appel(s)" in line and "coût ≈ 0.00009 $" in line
    assert "sur 5.00 $ de crédit" in line and "commandes restantes" in line
    assert "2 appel(s)" in caplog_loguru[1]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["commands"] == 2 and saved["prompt"] == 6500 and saved["cached"] == 5304
    assert saved["cost_usd"] == pytest.approx(first + cost_usd(Usage(4000, 60, 3000), CFG))
    reloaded = UsageTracker(CFG, path)  # le cumul survit au redémarrage
    assert reloaded.totals.commands == 2 and reloaded.totals.since == saved["since"]


def test_missing_usage_and_broken_file(tmp_path: Path, caplog_loguru: list[str]) -> None:
    path = tmp_path / "llm_usage.json"
    path.write_text("pas du json", encoding="utf-8")
    tracker = UsageTracker(CFG, path)
    assert tracker.totals.commands == 0
    assert tracker.record("#3", None, calls=1) == 0.0
    assert "décompte non fourni par le serveur" in caplog_loguru[-1]
    assert tracker.totals.commands == 0
