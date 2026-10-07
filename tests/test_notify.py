from __future__ import annotations

import sys
import types

import pytest

from mikata.ui.notify import ToastNotifier


def test_toast_uses_win11toast(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, str]] = []
    module = types.ModuleType("win11toast")
    module.notify = lambda title, body: sent.append((title, body))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "win11toast", module)
    ToastNotifier().notify("MIKATA — Minuteur", "Le minuteur est terminé.")
    assert sent == [("MIKATA — Minuteur", "Le minuteur est terminé.")]


def test_toast_failure_is_only_logged(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]
) -> None:
    module = types.ModuleType("win11toast")

    def broken(title: str, body: str) -> None:
        raise OSError("pas de centre de notifications")

    module.notify = broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "win11toast", module)
    ToastNotifier().notify("titre", "texte")
    assert "Notification Windows impossible" in caplog_loguru[-1]
