"""Choix des périphériques audio par sous-chaîne du nom, en privilégiant WASAPI (§3).

Les index PortAudio changent au redémarrage : on ne les stocke jamais dans la config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mikata.config import ConfigError

PREFERRED_HOST_API = "Windows WASAPI"

Kind = Literal["input", "output"]


class DeviceNotFoundError(ConfigError):
    """Aucun périphérique, ou plusieurs, ne correspond à la sous-chaîne de config.yaml."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    def supports(self, kind: Kind) -> bool:
        channels = self.max_input_channels if kind == "input" else self.max_output_channels
        return channels > 0


def collect_devices(sd: Any) -> list[DeviceInfo]:
    """Lit les périphériques via le module `sounddevice` (ou un substitut en test)."""
    host_apis = [api["name"] for api in sd.query_hostapis()]
    return [
        DeviceInfo(
            index=index,
            name=dev["name"],
            host_api=host_apis[dev["hostapi"]],
            max_input_channels=int(dev["max_input_channels"]),
            max_output_channels=int(dev["max_output_channels"]),
            default_samplerate=float(dev["default_samplerate"]),
        )
        for index, dev in enumerate(sd.query_devices())
    ]


def filter_host_api(devices: list[DeviceInfo], host_api: str | None) -> list[DeviceInfo]:
    """Garde l'API hôte demandée si elle existe, sinon tout."""
    if host_api is None or not any(d.host_api == host_api for d in devices):
        return devices
    return [d for d in devices if d.host_api == host_api]


def match_devices(devices: list[DeviceInfo], substring: str, kind: Kind) -> list[DeviceInfo]:
    needle = substring.casefold()
    return [d for d in devices if d.supports(kind) and needle in d.name.casefold()]


def default_device_index(sd: Any, kind: Kind) -> int | None:
    """Index du périphérique par défaut de Windows pour l'API WASAPI, s'il est connu."""
    key = "default_input_device" if kind == "input" else "default_output_device"
    for api in sd.query_hostapis():
        if api["name"] == PREFERRED_HOST_API:
            index = api.get(key, -1)
            return int(index) if index is not None and index >= 0 else None
    return None


def find_device(
    devices: list[DeviceInfo],
    substring: str,
    kind: Kind,
    default_index: int | None = None,
) -> DeviceInfo:
    """Le périphérique WASAPI (à défaut, de n'importe quelle API) dont le nom contient
    `substring`. Un nom exactement égal l'emporte sur les correspondances partielles.

    La valeur spéciale « default » désigne le périphérique par défaut de Windows
    (`default_index`), qui suit les changements de casque ou d'enceintes."""
    candidates = filter_host_api(devices, PREFERRED_HOST_API)
    if substring.casefold() == "default":
        for device in candidates:
            if device.index == default_index and device.supports(kind):
                return device
        key = "audio.input_device" if kind == "input" else "audio.output_device"
        raise DeviceNotFoundError(
            f"{key} = « default » : aucun périphérique par défaut trouvé dans Windows."
        )
    matches = match_devices(candidates, substring, kind)
    exact = [d for d in matches if d.name.casefold() == substring.casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]

    key = "audio.input_device" if kind == "input" else "audio.output_device"
    hint = "Voir : uv run python scripts/list_audio_devices.py"
    if not matches:
        available = ", ".join(f"« {d.name} »" for d in candidates if d.supports(kind))
        raise DeviceNotFoundError(
            f"{key} = « {substring} » : aucun périphérique ne correspond.\n"
            f"Disponibles : {available or 'aucun'}\n{hint}"
        )
    names = ", ".join(f"« {d.name} »" for d in matches)
    raise DeviceNotFoundError(
        f"{key} = « {substring} » est ambigu ({len(matches)} correspondances : {names}).\n"
        f"Précise la sous-chaîne. {hint}"
    )
