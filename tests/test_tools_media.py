"""Outils média / volume ; `PycawAudio` testé avec un faux pycaw."""

from __future__ import annotations

import asyncio
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeAudio, FakeDesktop, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools.media import PycawAudio, find_device
from mikata.tools.registry import ToolRegistry, ToolResult


def _run(
    tmp_path: Path, tool: str, desktop: Any = None, audio: Any = None, **arguments: Any
) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(make_ctx(tmp_path, desktop=desktop, audio=audio))
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    ("action", "key"),
    [
        ("pause", "play_pause"),
        ("play", "play_pause"),
        ("next", "next"),
        ("previous", "previous"),
        ("stop", "stop"),
    ],
)
def test_media_control_sends_media_keys(tmp_path: Path, action: str, key: str) -> None:
    desktop = FakeDesktop()
    result = _run(tmp_path, "media_control", desktop=desktop, action=action)
    assert result.ok and result.terminal and desktop.log == [("media", key)]


def test_media_control_rejects_unknown_action(tmp_path: Path) -> None:
    result = _run(tmp_path, "media_control", desktop=FakeDesktop(), action="rewind")
    assert not result.ok and "arguments invalides" in result.content


def test_set_volume_absolute_unmutes(tmp_path: Path) -> None:
    audio = FakeAudio()
    audio.muted = True
    result = _run(tmp_path, "set_volume", audio=audio, level=30)
    assert result.ok and result.content == "Volume réglé à 30 %."
    assert audio.volume == 30 and not audio.muted


@pytest.mark.parametrize(
    ("start", "change", "expected"), [(50, 10, 60), (95, 10, 100), (5, -20, 0)]
)
def test_set_volume_relative(tmp_path: Path, start: float, change: int, expected: float) -> None:
    audio = FakeAudio(volume=start)
    assert _run(tmp_path, "set_volume", audio=audio, change=change).ok
    assert audio.volume == expected


@pytest.mark.parametrize("arguments", [{}, {"level": 30, "change": 10}, {"level": 130}])
def test_set_volume_needs_one_valid_value(tmp_path: Path, arguments: dict[str, Any]) -> None:
    result = _run(tmp_path, "set_volume", audio=FakeAudio(), **arguments)
    assert not result.ok and "arguments invalides" in result.content


def test_mute_and_unmute(tmp_path: Path) -> None:
    audio = FakeAudio()
    assert _run(tmp_path, "mute", audio=audio).content == "Son coupé." and audio.muted
    assert _run(tmp_path, "mute", audio=audio, mute=False).content == "Son remis."
    assert not audio.muted


def test_set_output_device(tmp_path: Path) -> None:
    audio = FakeAudio()
    result = _run(tmp_path, "set_output_device", audio=audio, name="airpods")
    assert result.ok and audio.default == "{id-airpods}"
    assert result.content == "Sortie audio : Casque (AirPods de Samy)."
    missing = _run(tmp_path, "set_output_device", audio=audio, name="télé du salon")
    assert not missing.ok and "Haut-parleurs (Realtek(R) Audio)" in missing.content


def test_find_device() -> None:
    devices = [("a", "Haut-parleurs (Realtek(R) Audio)"), ("b", "Casque (AirPods de Samy)")]
    assert find_device("casque", devices) == devices[1]
    assert find_device("enceintes", devices) is None
    assert find_device("x", []) is None


class _Volume:
    def __init__(self) -> None:
        self.level = 0.5
        self.mute: int | None = None

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.level

    def SetMasterVolumeLevelScalar(self, level: float, context: Any) -> None:
        self.level = level

    def SetMute(self, mute: int, context: Any) -> None:
        self.mute = mute


@pytest.fixture
def fake_pycaw(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"volume": _Volume(), "default": None}

    class EDataFlow(Enum):
        eRender = 0

    class DEVICE_STATE(Enum):
        ACTIVE = 1

    class ERole(Enum):
        eConsole = 0
        eMultimedia = 1
        eCommunications = 2

    class Device:
        def __init__(self, dev_id: str, name: str | None) -> None:
            self.id, self.FriendlyName = dev_id, name
            self.EndpointVolume = state["volume"]

    class AudioUtilities:
        @staticmethod
        def GetSpeakers() -> Device:
            return Device("{spk}", "Haut-parleurs")

        @staticmethod
        def GetAllDevices(data_flow: int, device_state: int) -> list[Device | None]:
            state["filters"] = (data_flow, device_state)
            return [Device("{spk}", "Haut-parleurs"), Device("{x}", None), None]

        @staticmethod
        def SetDefaultDevice(dev_id: str, roles: list[ERole]) -> None:
            state["default"] = (dev_id, roles)

    pycaw = types.ModuleType("pycaw")
    utils = types.ModuleType("pycaw.utils")
    constants = types.ModuleType("pycaw.constants")
    utils.AudioUtilities = AudioUtilities  # type: ignore[attr-defined]
    constants.EDataFlow, constants.DEVICE_STATE, constants.ERole = EDataFlow, DEVICE_STATE, ERole  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycaw", pycaw)
    monkeypatch.setitem(sys.modules, "pycaw.utils", utils)
    monkeypatch.setitem(sys.modules, "pycaw.constants", constants)
    state["ERole"] = ERole
    return state


def test_pycaw_audio(fake_pycaw: dict[str, Any]) -> None:
    audio = PycawAudio()
    assert audio.get_volume() == 50
    audio.set_volume(130)
    assert fake_pycaw["volume"].level == 1.0
    audio.set_volume(30)
    assert fake_pycaw["volume"].level == pytest.approx(0.3)
    audio.set_mute(True)
    assert fake_pycaw["volume"].mute == 1
    assert audio.output_devices() == [("{spk}", "Haut-parleurs")]
    assert fake_pycaw["filters"] == (0, 1)  # sorties actives seulement
    audio.set_default_output("{spk}")
    dev_id, roles = fake_pycaw["default"]
    assert dev_id == "{spk}" and len(roles) == 3  # console, multimédia, communications
