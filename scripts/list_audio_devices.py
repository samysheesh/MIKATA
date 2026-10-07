"""Liste les périphériques audio et vérifie ceux choisis dans config.yaml.

Usage : uv run python scripts/list_audio_devices.py [--all] [--config config.yaml]

Sous Windows, seuls les périphériques WASAPI sont affichés par défaut (c'est l'API hôte
utilisée par MIKATA, §3) ; `--all` affiche aussi MME, DirectSound, WDM-KS.
Les valeurs `audio.input_device` / `audio.output_device` sont des sous-chaînes du nom
(insensibles à la casse) : elles doivent désigner exactement un périphérique.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PREFERRED_HOST_API = "Windows WASAPI"

Kind = Literal["input", "output"]


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


def format_table(devices: list[DeviceInfo]) -> str:
    header = f"{'#':>3}  {'Entrées':>7}  {'Sorties':>7}  {'Hz':>6}  {'API':<16}  Nom"
    rows = [header, "-" * len(header)]
    for d in devices:
        rows.append(
            f"{d.index:>3}  {d.max_input_channels:>7}  {d.max_output_channels:>7}  "
            f"{d.default_samplerate:>6.0f}  {d.host_api:<16}  {d.name}"
        )
    return "\n".join(rows)


def describe_match(key: str, substring: str, matches: list[DeviceInfo]) -> tuple[bool, str]:
    if len(matches) == 1:
        return True, f"[OK] {key} = '{substring}' -> #{matches[0].index} {matches[0].name}"
    if not matches:
        return False, f"[!!] {key} = '{substring}' -> aucun périphérique ne correspond"
    names = ", ".join(f"#{d.index} {d.name}" for d in matches)
    return False, f"[!!] {key} = '{substring}' -> ambigu, {len(matches)} correspondances : {names}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="toutes les API hôtes")
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import sounddevice as sd
    except OSError as exc:  # bibliothèque PortAudio absente
        print(f"PortAudio introuvable : {exc}", file=sys.stderr)
        return 1

    devices = collect_devices(sd)
    shown = devices if args.all else filter_host_api(devices, PREFERRED_HOST_API)
    print(format_table(shown))

    try:
        from mikata.config import ConfigError, load_config, resolve_config_path

        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"\nConfiguration non vérifiée : {exc}", file=sys.stderr)
        return 1

    print(f"\nConfiguration : {config_path}")
    candidates = filter_host_api(devices, PREFERRED_HOST_API)
    all_ok = True
    for key, substring, kind in (
        ("audio.input_device", config.audio.input_device, "input"),
        ("audio.output_device", config.audio.output_device, "output"),
    ):
        if substring.casefold() == "default":
            print(f"[OK] {key} = 'default' -> périphérique par défaut de Windows")
            continue
        ok, line = describe_match(key, substring, match_devices(candidates, substring, kind))
        all_ok &= ok
        print(line)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
