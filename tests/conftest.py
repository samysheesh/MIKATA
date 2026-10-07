from __future__ import annotations

import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
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


# -- faux sounddevice : callbacks appelés depuis un thread, comme PortAudio --------------

HOST_APIS = [
    {"name": "MME"},
    {"name": "Windows WASAPI", "default_input_device": 5, "default_output_device": 2},
]
DEVICES: list[dict[str, Any]] = [
    {"name": "Microphone (RØDE NT-USB+)", "hostapi": 0, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 48000.0},
    {"name": "Casque (HyperX Cloud)", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
    {"name": "Casque (HyperX Cloud) micro", "hostapi": 1, "max_input_channels": 1,
     "max_output_channels": 0, "default_samplerate": 48000.0},
    {"name": "Haut-parleurs (Realtek)", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
    {"name": "Microphone (RØDE NT-USB+)", "hostapi": 1, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 48000.0},
]  # fmt: skip


class FakeInputStream:
    """Envoie `signal` (float32, (n, canaux)) bloc par bloc au callback, puis du silence."""

    signal: ClassVar[np.ndarray | None] = None
    instances: ClassVar[list[FakeInputStream]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs.get("finished_callback")
        self.blocksize = kwargs["blocksize"]
        self.channels = kwargs["channels"]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        signal = self.signal
        pos = 0
        while not self._stop.is_set():
            block = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            if signal is not None and pos < len(signal):
                part = signal[pos : pos + self.blocksize]
                block[: len(part)] = part
                pos += self.blocksize
            self.callback(block, self.blocksize, None, False)
            time.sleep(0.001)
        if self.finished_callback:
            self.finished_callback()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def close(self) -> None:
        self.closed = True


class FakeOutputStream:
    """Appelle le callback en boucle et garde tout ce qui a été « joué »."""

    instances: ClassVar[list[FakeOutputStream]] = []
    latency = 0.012

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.channels = kwargs["channels"]
        self.played: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.closed = False
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = np.full((480, self.channels), np.nan, dtype=np.float32)
            self.callback(out, 480, None, False)
            self.played.append(out)
            time.sleep(0.001)

    def output(self) -> np.ndarray:
        return np.concatenate(self.played) if self.played else np.zeros((0, self.channels))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    sd = types.ModuleType("sounddevice")
    sd.query_hostapis = lambda: HOST_APIS  # type: ignore[attr-defined]
    sd.query_devices = lambda: DEVICES  # type: ignore[attr-defined]
    sd.InputStream = FakeInputStream  # type: ignore[attr-defined]
    sd.OutputStream = FakeOutputStream  # type: ignore[attr-defined]
    FakeInputStream.signal = None
    FakeInputStream.instances = []
    FakeOutputStream.instances = []
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd
