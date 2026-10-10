"""Choix des périphériques audio par sous-chaîne du nom, en privilégiant l'API native (§3).

API native selon la plateforme : WASAPI sous Windows, CoreAudio sous macOS.
Les index PortAudio changent au redémarrage : on ne les stocke jamais dans la config.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

from mikata.config import ConfigError

# Nom PortAudio de l'API native par plateforme (relevé via `query_hostapis`).
HOST_API_BY_PLATFORM: dict[str, str] = {
    "win32": "Windows WASAPI",
    "darwin": "Core Audio",
}

# Conservé pour la compatibilité (tests, script) : valeur Windows.
PREFERRED_HOST_API = HOST_API_BY_PLATFORM["win32"]


def preferred_host_api() -> str:
    """Nom de l'API hôte native de la plateforme courante.

    Plateforme inconnue (Linux de test, CI) : WASAPI, pour garder un tri
    déterministe ; `filter_preferred` retombe sur l'autre API native si absente.
    """
    return HOST_API_BY_PLATFORM.get(sys.platform, HOST_API_BY_PLATFORM["win32"])


def preferred_host_apis() -> list[str]:
    """Ordre de préférence des API natives : celle de la plateforme d'abord."""
    native = preferred_host_api()
    return [native, *[name for name in HOST_API_BY_PLATFORM.values() if name != native]]


def filter_preferred(devices: list[DeviceInfo]) -> list[DeviceInfo]:
    """Périphériques de l'API native préférée, présente ; sinon l'autre API native ;
    sinon tout (aucune des deux n'existe, ex. : MME seul)."""
    for api_name in preferred_host_apis():
        if any(d.host_api == api_name for d in devices):
            return [d for d in devices if d.host_api == api_name]
    return devices


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
    """Index du périphérique par défaut du système pour l'API native, s'il est connu."""
    wanted = preferred_host_apis()
    key = "default_input_device" if kind == "input" else "default_output_device"
    by_name = {api["name"]: api for api in sd.query_hostapis()}
    for api_name in wanted:
        if api_name in by_name:
            index = by_name[api_name].get(key, -1)
            return int(index) if index is not None and index >= 0 else None
    return None


def find_device(
    devices: list[DeviceInfo],
    substring: str,
    kind: Kind,
    default_index: int | None = None,
) -> DeviceInfo:
    """Le périphérique de l'API native (à défaut, de n'importe quelle API) dont le nom
    contient `substring`. Un nom exactement égal l'emporte sur les correspondances partielles.

    La valeur spéciale « default » désigne le périphérique par défaut du système
    (`default_index`), qui suit les changements de casque ou d'enceintes. La
    correspondance se fait sur le nom CoreAudio / WASAPI tel que PortAudio le rapporte.
    """
    api_name = preferred_host_api()
    candidates = filter_preferred(devices)
    if substring.casefold() == "default":
        for device in candidates:
            if device.index == default_index and device.supports(kind):
                return device
        key = "audio.input_device" if kind == "input" else "audio.output_device"
        raise DeviceNotFoundError(
            f"{key} = « default » : aucun périphérique par défaut trouvé ({api_name})."
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
