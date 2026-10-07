from __future__ import annotations

import shutil
import sys
import threading
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mikata.config import PROJECT_ROOT


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les variables du poste de dev ne doivent pas influencer les tests."""
    for name in ("MIKATA_CONFIG", "LLM_API_KEY", "SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Copie du config.yaml du dépôt dans un dossier temporaire (logs isolés)."""
    dest = tmp_path / "config.yaml"
    shutil.copy(PROJECT_ROOT / "config.yaml", dest)
    return dest


class FakeIcon:
    """Substitut de pystray.Icon : run() bloque jusqu'à stop(), comme sous Windows."""

    instances: ClassVar[list[FakeIcon]] = []

    def __init__(self, name: str, icon: Any = None, title: str = "", menu: Any = None) -> None:
        self.name = name
        self.icon = icon
        self.title = title
        self.menu = menu
        self.running = threading.Event()
        self._stopped = threading.Event()
        self.stop_calls = 0
        FakeIcon.instances.append(self)

    def run(self) -> None:
        self.running.set()
        self._stopped.wait(timeout=10)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped.set()


class FakeMenuItem:
    def __init__(self, text: str, action: Any) -> None:
        self.text = text
        self.action = action


class FakeMenu:
    def __init__(self, *items: FakeMenuItem) -> None:
        self.items = items


@pytest.fixture
def fake_pystray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("pystray")
    module.Icon = FakeIcon  # type: ignore[attr-defined]
    module.Menu = FakeMenu  # type: ignore[attr-defined]
    module.MenuItem = FakeMenuItem  # type: ignore[attr-defined]
    FakeIcon.instances = []
    monkeypatch.setitem(sys.modules, "pystray", module)
    return module
