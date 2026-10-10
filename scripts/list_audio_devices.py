"""Liste les périphériques audio et vérifie ceux choisis dans config.yaml.

Usage : uv run python scripts/list_audio_devices.py [--all] [--config config.yaml]

Seuls les périphériques de l'API native sont affichés par défaut (WASAPI sous Windows,
CoreAudio sous macOS : c'est l'API hôte utilisée par MIKATA, §3) ; `--all` affiche
aussi les autres API (MME, DirectSound, WDM-KS...).
Les valeurs `audio.input_device` / `audio.output_device` sont des sous-chaînes du nom
(insensibles à la casse) : elles doivent désigner exactement un périphérique.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mikata.audio.devices import (
    PREFERRED_HOST_API,
    DeviceInfo,
    collect_devices,
    default_device_index,
    filter_host_api,
    filter_preferred,
    match_devices,
    preferred_host_api,
)

__all__ = ["PREFERRED_HOST_API", "collect_devices", "filter_host_api", "main", "match_devices"]


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
    api_name = preferred_host_api()
    shown = devices if args.all else filter_preferred(devices)
    print(format_table(shown))

    try:
        from mikata.config import ConfigError, load_config, resolve_config_path

        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"\nConfiguration non vérifiée : {exc}", file=sys.stderr)
        return 1

    print(f"\nConfiguration : {config_path}")
    candidates = filter_preferred(devices)  # --all n'élargit que l'affichage, pas le choix
    all_ok = True
    for key, substring, kind in (
        ("audio.input_device", config.audio.input_device, "input"),
        ("audio.output_device", config.audio.output_device, "output"),
    ):
        if substring.casefold() == "default":
            index = default_device_index(sd, kind)
            default = next((d for d in candidates if d.index == index), None)
            if default is None:
                print(f"[!!] {key} = 'default' -> aucun périphérique par défaut ({api_name})")
                all_ok = False
            else:
                print(f"[OK] {key} = 'default' -> #{default.index} {default.name}")
            continue
        ok, line = describe_match(key, substring, match_devices(candidates, substring, kind))
        all_ok &= ok
        print(line)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
