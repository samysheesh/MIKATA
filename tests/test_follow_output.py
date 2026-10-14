"""Phase 15f : la sortie de MIKATA suit celle de Windows (casque allumé après le démarrage)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mikata.app import App
from mikata.audio import follow
from mikata.audio.devices import DeviceInfo
from mikata.audio.follow import DefaultOutputWatch, same_device
from mikata.config import load_config

SCREEN = DeviceInfo(3, "PL2770H (NVIDIA High Definition Audio)", "Windows WASAPI", 0, 2, 48_000)
HEADSET = DeviceInfo(9, "Casque (AirPods Max de Samy)", "Windows WASAPI", 0, 2, 44_100)
MIC = DeviceInfo(4, "Microphone (RØDE NT-USB+)", "Windows WASAPI", 2, 0, 48_000)


def test_same_device_tolerates_case_and_mme_truncation() -> None:
    assert same_device("Casque (AirPods Max de Samy)", "casque (airpods max de samy)")
    assert same_device("Casque (AirPods Max de Samy)", "Casque (AirPods Max de Sa")
    assert not same_device("Casque (AirPods)", "Casque (HyperX Cloud)")
    assert not same_device("", "Casque")


def test_watch_switches_only_once_the_new_default_is_stable() -> None:
    watch = DefaultOutputWatch(stable=2)
    assert watch.update("PL2770H (NVIDIA High Definition Audio)", SCREEN.name) is None
    assert watch.update(HEADSET.name, SCREEN.name) is None  # premier relevé : on attend
    assert watch.update(HEADSET.name, SCREEN.name) == HEADSET.name
    assert watch.update(None, HEADSET.name) is None  # aucune sortie active : on ne bouge pas
    assert watch.update(SCREEN.name, HEADSET.name) is None
    assert watch.update(HEADSET.name, HEADSET.name) is None  # Bluetooth qui hésite : remis à 0
    assert watch.update(SCREEN.name, HEADSET.name) is None


class FakePlayer:
    def __init__(self) -> None:
        self.device = SCREEN
        self.sample_rate = 48_000
        self.switched: list[str] = []
        self.closed = 0

    def switch(self, device: DeviceInfo, loop: Any) -> None:
        self.device = device
        self.sample_rate = int(device.default_samplerate)
        self.switched.append(device.name)

    def close(self) -> None:
        self.closed += 1

    def play_nowait(self, pcm: Any) -> None: ...


class FakeSource:
    def __init__(self, names: list[str | None]) -> None:
        self.names = names
        self.closed = False

    async def name(self) -> str | None:
        return self.names.pop(0) if len(self.names) > 1 else self.names[0]

    def close(self) -> None:
        self.closed = True


def _app(config_file: Path) -> tuple[App, FakePlayer]:
    app = App(load_config(config_file))
    player = FakePlayer()
    app._player = player  # type: ignore[assignment]
    return app, player


def _follow(app: App, source: FakeSource, seconds: float) -> None:
    async def scenario() -> None:
        task = asyncio.create_task(app._follow_output(source))
        await asyncio.sleep(seconds)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_follows_the_windows_default_output(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(follow, "POLL_S", 0.01)
    app, player = _app(config_file)
    moved: list[str] = []

    async def move(name: str) -> None:
        moved.append(name)
        player.device = HEADSET

    app._move_output = move  # type: ignore[method-assign]
    source = FakeSource([SCREEN.name, HEADSET.name])
    _follow(app, source, 0.2)
    assert moved == [HEADSET.name] and source.closed


def test_never_moves_during_a_command_nor_with_a_chosen_device(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(follow, "POLL_S", 0.01)
    app, _ = _app(config_file)
    moved: list[str] = []

    async def move(name: str) -> None:
        moved.append(name)

    app._move_output = move  # type: ignore[method-assign]
    app._active = {1: object()}  # type: ignore[dict-item]
    _follow(app, FakeSource([HEADSET.name]), 0.1)
    assert moved == []
    app._active = {}
    audio = app.config.audio.model_copy(update={"output_device": "Haut-parleurs"})
    app.config = app.config.model_copy(update={"audio": audio})
    _follow(app, FakeSource([HEADSET.name]), 0.1)
    assert moved == []  # sortie choisie par son nom : MIKATA ne la change pas tout seul


def test_a_headset_unknown_to_portaudio_triggers_a_full_rescan(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mikata.__main__ as main_module
    from mikata.audio import devices

    app, player = _app(config_file)
    reopened: list[str] = []
    capture = SimpleNamespace(
        stop=lambda: reopened.append("stop"),
        reopen=lambda device, loop: reopened.append(device.name),
    )
    app._audio = capture  # type: ignore[assignment]
    lists = iter([[SCREEN, MIC], [SCREEN, HEADSET, MIC]])  # avant / après relecture
    refreshed: list[bool] = []
    monkeypatch.setattr(devices, "collect_devices", lambda sd: next(lists))
    monkeypatch.setattr(devices, "filter_preferred", lambda found: found)
    monkeypatch.setattr(main_module, "_refresh_portaudio", lambda sd: refreshed.append(True))
    monkeypatch.setattr(
        devices,
        "default_device_index",
        lambda sd, kind: HEADSET.index if kind == "output" else MIC.index,
    )
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace())
    asyncio.run(app._move_output(HEADSET.name))
    assert refreshed == [True] and player.closed == 1
    assert player.switched == [HEADSET.name] and player.sample_rate == 44_100
    assert reopened == ["stop", MIC.name] and app._audio_broken is False
    assert app._earcons.sample_rate == 44_100  # type: ignore[attr-defined]


def test_a_failed_rescan_is_retried(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mikata.__main__ as main_module
    from mikata.audio import devices

    app, _ = _app(config_file)

    def no_devices(sd: Any) -> list[DeviceInfo]:
        return []

    monkeypatch.setattr(devices, "collect_devices", no_devices)
    monkeypatch.setattr(main_module, "_refresh_portaudio", lambda sd: None)
    monkeypatch.setattr(devices, "default_device_index", lambda sd, kind: None)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", SimpleNamespace())
    asyncio.run(app._rescan_audio("test"))
    assert app._audio_broken is True  # la boucle de suivi réessaiera
