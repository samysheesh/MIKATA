from __future__ import annotations

import threading
import types

from conftest import FakeIcon

from mikata.app import State
from mikata.ui.tray import ICON_SIZE, STATE_COLORS, TrayIcon, make_icon_image


def test_every_state_has_a_color() -> None:
    assert set(STATE_COLORS) == set(State)


def test_make_icon_image() -> None:
    image = make_icon_image(STATE_COLORS[State.IDLE])
    assert image.size == (ICON_SIZE, ICON_SIZE)
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0  # coin transparent
    center_left = image.getpixel((ICON_SIZE // 8, ICON_SIZE // 2))
    assert center_left[:3] == (0x8A, 0x8A, 0x8A)


def test_tray_lifecycle_and_quit(fake_pystray: types.ModuleType) -> None:
    quit_called = threading.Event()
    tray = TrayIcon()
    tray.start(on_quit=quit_called.set)

    icon = FakeIcon.instances[-1]
    assert icon.running.wait(timeout=2)
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


def test_set_error(fake_pystray: types.ModuleType) -> None:
    tray = TrayIcon()
    tray.start(on_quit=lambda: None)
    tray.set_error()
    assert FakeIcon.instances[-1].title == "MIKATA — erreur"
    tray.stop()
