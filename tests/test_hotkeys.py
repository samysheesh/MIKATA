from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import ClassVar

import pytest

from mikata.ui.hotkeys import HotkeyListener


class FakeGlobalHotKeys:
    instances: ClassVar[list[FakeGlobalHotKeys]] = []

    def __init__(self, hotkeys: dict[str, Callable[[], None]]) -> None:
        if "<inconnu>" in "".join(hotkeys):
            raise ValueError("touche inconnue")
        self.hotkeys = hotkeys
        self.started = False
        self.stopped = False
        FakeGlobalHotKeys.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def fake_pynput(monkeypatch: pytest.MonkeyPatch) -> None:
    keyboard = types.ModuleType("pynput.keyboard")
    keyboard.GlobalHotKeys = FakeGlobalHotKeys  # type: ignore[attr-defined]
    package = types.ModuleType("pynput")
    package.keyboard = keyboard  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynput", package)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)


def test_registers_combo_and_stops() -> None:
    calls: list[int] = []
    listener = HotkeyListener("<ctrl>+<alt>+m")
    listener.start(lambda: calls.append(1))
    fake = FakeGlobalHotKeys.instances[-1]
    assert fake.started
    fake.hotkeys["<ctrl>+<alt>+m"]()
    assert calls == [1]
    listener.stop()
    listener.stop()
    assert fake.stopped


def test_invalid_combo() -> None:
    with pytest.raises(ValueError, match=r"wake\.hotkey"):
        HotkeyListener("<inconnu>+m").start(lambda: None)
