from __future__ import annotations

import pytest
from conftest import DEVICES, HOST_APIS

from mikata.audio.devices import (
    DeviceInfo,
    DeviceNotFoundError,
    collect_devices,
    default_device_index,
    find_device,
)
from mikata.config import ConfigError


class _Sd:
    @staticmethod
    def query_hostapis() -> list:
        return HOST_APIS

    @staticmethod
    def query_devices() -> list:
        return DEVICES


@pytest.fixture
def devices() -> list[DeviceInfo]:
    return collect_devices(_Sd)


def test_prefers_wasapi(devices: list[DeviceInfo]) -> None:
    mic = find_device(devices, "nt-usb", "input")  # aussi présent en MME (index 0)
    assert mic.index == 5
    assert mic.host_api == "Windows WASAPI"
    assert mic.default_samplerate == 48_000


def test_kind_filters(devices: list[DeviceInfo]) -> None:
    assert find_device(devices, "Casque", "output").index == 2
    assert find_device(devices, "Casque", "input").index == 3


def test_exact_name_wins(devices: list[DeviceInfo]) -> None:
    extra = [*devices, DeviceInfo(9, "Casque", "Windows WASAPI", 0, 2, 48_000)]
    ambiguous = [*devices, DeviceInfo(9, "Casque 2", "Windows WASAPI", 0, 2, 48_000)]
    with pytest.raises(DeviceNotFoundError):
        find_device(ambiguous, "Casque", "output")
    assert find_device(extra, "casque", "output").index == 9


def test_not_found_lists_candidates(devices: list[DeviceInfo]) -> None:
    with pytest.raises(DeviceNotFoundError) as info:
        find_device(devices, "Rode", "input")
    message = str(info.value)
    assert "audio.input_device" in message
    assert "RØDE NT-USB+" in message
    assert "Haut-parleurs" not in message  # sortie seulement
    assert "list_audio_devices" in message
    assert isinstance(info.value, ConfigError)


def test_ambiguous(devices: list[DeviceInfo]) -> None:
    with pytest.raises(DeviceNotFoundError, match="ambigu"):
        find_device(devices, "(", "output")


def test_falls_back_to_other_host_api() -> None:
    mme_only = [DeviceInfo(0, "Micro USB", "MME", 1, 0, 44_100)]
    assert find_device(mme_only, "usb", "input").index == 0


def test_default_follows_windows_default(devices: list[DeviceInfo]) -> None:
    assert default_device_index(_Sd, "output") == 2
    assert find_device(devices, "default", "output", 2).name == "Casque (HyperX Cloud)"
    assert find_device(devices, "Default", "output", 4).name == "Haut-parleurs (Realtek)"
    with pytest.raises(DeviceNotFoundError):
        find_device(devices, "default", "output", None)
    with pytest.raises(DeviceNotFoundError):
        find_device(devices, "default", "output", 1)  # une entrée n'est pas une sortie
