"""Chaines de repli des notifications par plateforme.

Sous macOS, le chemin est `pync` puis `osascript` ; `win11toast` (module Windows,
`sys_platform == 'win32'`) n'est jamais tenté. Sous Windows, `win11toast` uniquement.
Les modules sont simulés dans `sys.modules` (les vrais ne sont pas installés ici).
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from mikata.ui import notify as notify_module
from mikata.ui.notify import ToastNotifier


def test_darwin_prefers_pync(monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]) -> None:
    sent: list[tuple[str, str]] = []
    module = types.ModuleType("pync")

    def fake_notify(body: str, title: str) -> None:
        sent.append((title, body))

    module.notify = fake_notify  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pync", module)
    monkeypatch.setattr(notify_module, "platform", lambda: "darwin")
    ToastNotifier().notify("MIKATA — Minuteur", "Le minuteur est terminé.")
    assert sent == [("MIKATA — Minuteur", "Le minuteur est terminé.")]


def test_darwin_without_pync_falls_back_to_osascript(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]
) -> None:
    monkeypatch.delitem(sys.modules, "pync", raising=False)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    monkeypatch.setattr(notify_module, "platform", lambda: "darwin")
    ToastNotifier().notify("titre", "texte")
    assert len(calls) == 1
    assert calls[0][0] == "osascript"
    assert "display notification 'texte' with title 'titre'" in calls[0][2]


def test_darwin_never_tries_win11toast(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]
) -> None:
    """win11toast est un module Windows : il ne doit pas être importé sous macOS."""
    monkeypatch.delitem(sys.modules, "pync", raising=False)
    called: list[str] = []

    class ForbiddenModule(types.ModuleType):
        def __init__(self, name: str) -> None:
            super().__init__(name)

        def notify(self, title: str, body: str) -> None:
            called.append(title)

    monkeypatch.setitem(sys.modules, "win11toast", ForbiddenModule("win11toast"))

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    monkeypatch.setattr(notify_module, "platform", lambda: "darwin")
    ToastNotifier().notify("titre", "texte")
    assert called == []  # l'osascript a servi, jamais win11toast


def test_win32_uses_win11toast(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, str]] = []
    module = types.ModuleType("win11toast")
    module.notify = lambda title, body: sent.append((title, body))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "win11toast", module)
    monkeypatch.setattr(notify_module, "platform", lambda: "win32")
    ToastNotifier().notify("MIKATA — Minuteur", "Le minuteur est terminé.")
    assert sent == [("MIKATA — Minuteur", "Le minuteur est terminé.")]


def test_win32_never_tries_pync_or_osascript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le chemin Windows n'utilise que win11toast : ni pync ni osascript."""
    sent: list[tuple[str, str]] = []
    module = types.ModuleType("win11toast")
    module.notify = lambda title, body: sent.append((title, body))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "win11toast", module)

    def forbidden_run(argv: list[str], **kwargs: object) -> None:
        raise AssertionError(f"osascript ne doit pas être appelé sous Windows : {argv}")

    monkeypatch.setattr(notify_module.subprocess, "run", forbidden_run)
    monkeypatch.setattr(notify_module, "platform", lambda: "win32")
    ToastNotifier().notify("titre", "texte")
    assert sent == [("titre", "texte")]


def test_darwin_pync_failure_is_logged_and_returns(
    monkeypatch: pytest.MonkeyPatch, caplog_loguru: list[str]
) -> None:
    """Un échec de pync est journalisé et n'est pas suivi d'un osascript (le repli
    osascript ne sert que lorsque pync est absent ; comportement inchangé)."""
    module = types.ModuleType("pync")

    def broken(body: str, title: str) -> None:
        raise OSError("pync indisponible")

    module.notify = broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pync", module)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    monkeypatch.setattr(notify_module, "platform", lambda: "darwin")
    ToastNotifier().notify("titre", "texte")
    assert calls == []  # pas de repli osascript sur échec de pync (comportement d'origine)
    assert any("Notification macOS impossible" in line for line in caplog_loguru)
