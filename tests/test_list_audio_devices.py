from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mikata.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "list_audio_devices", PROJECT_ROOT / "scripts" / "list_audio_devices.py"
)
assert _spec is not None and _spec.loader is not None
lad = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lad  # requis par @dataclass
_spec.loader.exec_module(lad)

HOST_APIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
DEVICES: list[dict[str, Any]] = [
    {"name": "Focusrite USB (Scarlett", "hostapi": 0, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Analogue 1 + 2 (Focusrite USB Audio)", "hostapi": 1, "max_input_channels": 2,
     "max_output_channels": 0, "default_samplerate": 48000.0},
    {"name": "Casque (HyperX Cloud)", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
    {"name": "Casque (HyperX Cloud) micro", "hostapi": 1, "max_input_channels": 1,
     "max_output_channels": 0, "default_samplerate": 48000.0},
    {"name": "Haut-parleurs (Realtek)", "hostapi": 1, "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
]  # fmt: skip


def _fake_sd() -> types.ModuleType:
    sd = types.ModuleType("sounddevice")
    sd.query_hostapis = lambda: HOST_APIS  # type: ignore[attr-defined]
    sd.query_devices = lambda: DEVICES  # type: ignore[attr-defined]
    return sd


def test_collect_and_filter() -> None:
    devices = lad.collect_devices(_fake_sd())
    assert len(devices) == 5
    assert devices[0].host_api == "MME"
    wasapi = lad.filter_host_api(devices, lad.PREFERRED_HOST_API)
    assert [d.index for d in wasapi] == [1, 2, 3, 4]
    assert lad.filter_host_api(devices, "ASIO") == devices  # API absente : tout garder


def test_match_devices_by_substring_and_kind() -> None:
    wasapi = lad.filter_host_api(lad.collect_devices(_fake_sd()), lad.PREFERRED_HOST_API)
    assert [d.index for d in lad.match_devices(wasapi, "focusrite", "input")] == [1]
    assert [d.index for d in lad.match_devices(wasapi, "Casque", "output")] == [2]
    assert [d.index for d in lad.match_devices(wasapi, "Casque", "input")] == [3]
    assert lad.match_devices(wasapi, "Focusrite", "output") == []


def test_describe_match() -> None:
    devices = lad.collect_devices(_fake_sd())
    ok, line = lad.describe_match("audio.output_device", "Casque", [devices[2]])
    assert ok and "[OK]" in line
    ok, line = lad.describe_match("audio.input_device", "Rode", [])
    assert not ok and "aucun" in line
    ok, line = lad.describe_match("audio.output_device", "(", devices[2:4])
    assert not ok and "ambigu" in line


def test_format_table() -> None:
    table = lad.format_table(lad.collect_devices(_fake_sd()))
    assert "Windows WASAPI" in table
    assert "Casque (HyperX Cloud)" in table
    assert "48000" in table


def test_main_with_repo_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config_file: Path
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    assert lad.main(["--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "MME" not in out  # WASAPI seulement par défaut
    assert "[OK] audio.input_device = 'Focusrite'" in out
    assert "[OK] audio.output_device = 'Casque'" in out

    assert lad.main(["--config", str(config_file), "--all"]) == 0
    assert "MME" in capsys.readouterr().out


def test_main_reports_missing_device(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    config = tmp_path / "config.yaml"
    config.write_text('audio:\n  input_device: "Rode"\n', encoding="utf-8")
    assert lad.main(["--config", str(config)]) == 1
    assert "aucun périphérique" in capsys.readouterr().out
