from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from pathlib import Path

import pytest
from conftest import FakeIcon

from mikata.app import State
from mikata.ui import tray as tray_module
from mikata.ui.tray import (
    ICON_SIZE,
    STATE_COLORS,
    TrayIcon,
    make_icon_image,
    pump_cocoa,
    set_macos_app_icon,
)


def test_every_state_has_a_color() -> None:
    assert set(STATE_COLORS) == set(State)


def test_make_icon_image() -> None:
    image = make_icon_image(STATE_COLORS[State.IDLE])
    assert image.size == (ICON_SIZE, ICON_SIZE)
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0  # coin transparent
    center_left = image.getpixel((ICON_SIZE // 8, ICON_SIZE // 2))
    assert center_left[:3] == (0x8A, 0x8A, 0x8A)


@pytest.fixture
def win32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chemin Windows : icône dans son propre thread (inchangé)."""
    monkeypatch.setattr(tray_module.sys, "platform", "win32")


def test_tray_lifecycle_and_quit(fake_pystray: types.ModuleType, win32: None) -> None:
    quit_called = threading.Event()
    tray = TrayIcon()
    tray.start(on_quit=quit_called.set)

    icon = FakeIcon.instances[-1]
    assert icon.running.wait(timeout=2)
    assert not icon.detached  # win32 : boucle de messages dans son thread
    assert icon.title == "MIKATA — repos"
    labels = [item.text for item in icon.menu.items]
    assert labels == ["Quitter"]

    icon.menu.items[0].action(icon, icon.menu.items[0])  # clic sur « Quitter »
    assert quit_called.is_set()

    tray.set_state(State.LISTENING)
    assert icon.title == "MIKATA — écoute"

    tray.stop()
    tray.stop()  # idempotent
    assert icon.stop_calls == 1
    assert tray._thread is None
    tray.set_state(State.IDLE)  # sans effet une fois arrêtée


def test_set_error(fake_pystray: types.ModuleType, win32: None) -> None:
    tray = TrayIcon()
    tray.start(on_quit=lambda: None)
    tray.set_error()
    assert FakeIcon.instances[-1].title == "MIKATA — erreur"
    tray.stop()


# -- « Afficher la mascotte » : l'affordance de récupération --------------------------------


def test_menu_keeps_a_reveal_action_that_reveals_the_mascot(
    fake_pystray: types.ModuleType, win32: None
) -> None:
    """Mascotte masquée : l'icône de la barre garde son action « Afficher la mascotte »."""
    actions: list[str] = []
    tray = TrayIcon()
    tray.start(on_quit=lambda: None, on_action=actions.append)
    icon = FakeIcon.instances[-1]
    labels = [item.text for item in icon.menu.items]
    assert "Afficher la mascotte" in labels  # toujours présente
    show = next(item for item in icon.menu.items if item.text == "Afficher la mascotte")
    show.action(icon, show)
    assert actions == ["mascot"]
    tray.stop()


def test_hidden_mascot_is_told_by_the_tooltip(fake_pystray: types.ModuleType, win32: None) -> None:
    tray = TrayIcon()
    tray.start(on_quit=lambda: None, on_action=lambda action: None)
    icon = FakeIcon.instances[-1]
    tray.set_mascot_hidden(True)
    assert tray._mascot_hidden and "masquée" in icon.title
    tray.set_mascot_hidden(True)  # sans changement : rien à faire
    tray.set_mascot_hidden(False)
    assert icon.title == "MIKATA — repos"
    tray.stop()


def test_watch_reflects_the_mascot_state_file(
    fake_pystray: types.ModuleType, win32: None, tmp_path: Path
) -> None:
    path = tmp_path / "ui_state.json"
    tray = TrayIcon(state_path=path)
    tray.start(on_quit=lambda: None, on_action=lambda action: None)
    icon = FakeIcon.instances[-1]
    path.write_text(json.dumps({"hidden": True}), encoding="utf-8")
    tray._hidden_checked_at = 0.0  # force une relecture tout de suite
    tray._refresh_mascot_hidden()
    assert tray._mascot_hidden and "masquée" in icon.title
    path.write_text(json.dumps({"hidden": False}), encoding="utf-8")
    tray._hidden_checked_at = 0.0
    tray._refresh_mascot_hidden()
    assert not tray._mascot_hidden and icon.title == "MIKATA — repos"
    tray.stop()


def test_watch_mascot_state_sets_a_resolved_path(
    fake_pystray: types.ModuleType, win32: None, tmp_path: Path
) -> None:
    tray = TrayIcon()
    tray.watch_mascot_state(tmp_path / "data" / "ui_state.json")
    assert tray.state_path == tmp_path / "data" / "ui_state.json"


# -- darwin : thread principal + boucle Cocoa -----------------------------------------------


def test_darwin_runs_the_icon_detached_on_the_main_thread(
    fake_pystray: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """macOS : `run_detached` sur le thread principal (pas de thread « mikata-tray ») et
    l'icône d'application est posée ; la boucle Cocoa est pompée par la tâche de fond."""
    monkeypatch.setattr(tray_module.sys, "platform", "darwin")
    pumped: list[int] = []
    monkeypatch.setattr(tray_module, "pump_cocoa", lambda *a, **k: pumped.append(1) or 0)
    icons: list[object] = []

    def fake_app_icon(image: object, size: int = 256) -> bool:
        icons.append(image)
        return True

    monkeypatch.setattr(tray_module, "set_macos_app_icon", fake_app_icon)

    async def scenario() -> None:
        tray = TrayIcon(state_path=tmp_path / "ui_state.json")
        tray.start(on_quit=lambda: None, on_action=lambda action: None)
        icon = FakeIcon.instances[-1]
        assert icon.detached and icon.running.is_set()
        assert tray._thread is None  # rien dans un thread séparé
        assert threading.current_thread() is threading.main_thread()
        assert icons  # icône du Dock appliquée
        assert tray._watch_task is not None
        await asyncio.sleep(2 * tray_module.PUMP_INTERVAL_S)
        assert pumped  # la boucle Cocoa est bien pompée
        tray.stop()

    asyncio.run(scenario())


def test_darwin_off_the_main_thread_falls_back_to_a_thread(
    fake_pystray: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hors du thread principal (cas dégradé) : l'icône retombe sur son thread, sans pompe."""
    monkeypatch.setattr(tray_module.sys, "platform", "darwin")
    tray = TrayIcon()
    tray._icon = None

    def start_in_a_thread() -> None:
        import pystray

        tray._icon = pystray.Icon("mikata", icon=make_icon_image("#8a8a8a"), menu=pystray.Menu())
        tray._start_cocoa()

    worker = threading.Thread(target=start_in_a_thread)
    worker.start()
    worker.join(timeout=2)
    assert tray._thread is not None and not FakeIcon.instances[-1].detached
    tray.stop()


def test_pump_cocoa_and_app_icon_are_noop_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tray_module.sys, "platform", "win32")
    assert pump_cocoa() == 0
    assert set_macos_app_icon(make_icon_image("#8a8a8a")) is False


def test_pump_cocoa_and_app_icon_do_not_raise_without_appkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AppKit indisponible (import paresseux) : no-op journalisé, jamais d'exception."""
    monkeypatch.setattr(tray_module.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AppKit", None)
    monkeypatch.setitem(sys.modules, "Foundation", None)
    assert pump_cocoa() == 0
    assert set_macos_app_icon(make_icon_image("#8a8a8a")) is False
