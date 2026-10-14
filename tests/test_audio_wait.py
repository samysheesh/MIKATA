"""Phase 15d : casque éteint au démarrage -> MIKATA attend au lieu de s'arrêter."""

from __future__ import annotations

import types

from mikata.__main__ import AUDIO_RETRY_S, _refresh_portaudio, wait_for_audio_devices
from mikata.audio.devices import DeviceNotFoundError
from mikata.config import Config


def test_waits_until_the_headset_is_back(caplog_loguru: list[str]) -> None:
    tries: list[int] = []
    pauses: list[float] = []

    def probe() -> None:
        tries.append(1)
        if len(tries) < 4:  # casque éteint pendant trois essais
            raise DeviceNotFoundError("audio.output_device = « default » : aucun périphérique.")

    wait_for_audio_devices(Config(), probe=probe, sleep=pauses.append)
    assert len(tries) == 4 and pauses == [AUDIO_RETRY_S] * 3
    warnings = [line for line in caplog_loguru if "Nouvel essai" in line]
    assert len(warnings) == 1  # pas une ligne par essai
    assert any("après 3 essai(s)" in line for line in caplog_loguru)


def test_starts_at_once_when_devices_are_there() -> None:
    pauses: list[float] = []
    wait_for_audio_devices(Config(), probe=lambda: None, sleep=pauses.append)
    assert pauses == []


def test_device_list_is_read_again() -> None:
    calls: list[str] = []
    sd = types.SimpleNamespace(
        _terminate=lambda: calls.append("terminate"),
        _initialize=lambda: calls.append("initialize"),
    )
    _refresh_portaudio(sd)
    assert calls == ["terminate", "initialize"]
    _refresh_portaudio(types.SimpleNamespace())  # API absente : rien ne casse
