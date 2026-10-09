"""Contrat d'échange entre le helper élevé (lecture MFT/journal USN, droits admin) et
MIKATA (utilisateur standard) : des lignes JSONL.

- `build.jsonl` : un dump complet (une ligne par chemin) que le helper régénère au
  démarrage et sur erreur.
- `events.log` : les deltas du journal USN, ajoutés en O_APPEND (une ligne par
  événement). MIKATA consomme ce qu'il lit et tronque les octets lus ; les appends sont
  atomiques (< 64 Ko), donc une troncature ne coupe jamais une ligne déjà écrite.

Ces fichiers vivent dans `data/index/` (dossier partagé : le script d'installation règle
les ACL pour que l'utilisateur puisse lire/écrire, le helper tourne en admin de toute
façon). Rien d'autre ne transite entre les deux processus.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mikata.files import usn

DUMP = "build.jsonl"  # dump complet (une ligne par chemin), régénéré par le helper
EVENTS = "events.log"  # deltas du journal USN, en append, consommés par MIKATA


def dump_line(path: str, entry: usn.UsnEntry) -> str:
    return json.dumps(
        {"frn": entry.frn, "path": path, "mtime": entry.mtime, "is_dir": entry.is_dir},
        ensure_ascii=False,
    )


def parse_dump_line(line: str) -> dict[str, Any] | None:
    """Ligne de dump, ou None si la ligne est tronquée/invalide."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not all(k in data for k in ("frn", "path", "mtime", "is_dir")):
        return None
    return data


def event_line(volume: str, event: usn.UsnEvent) -> str:
    return json.dumps(
        {
            "volume": volume,
            "frn": event.frn,
            "parent": event.parent_frn,
            "name": event.name,
            "reason": event.reason,
            "mtime": event.mtime,
            "is_dir": event.is_dir,
            "usn": event.usn,
        },
        ensure_ascii=False,
    )


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Ligne d'événement, ou None si la ligne est tronquée/invalide."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not all(
        k in data for k in ("volume", "frn", "parent", "name", "reason", "mtime", "is_dir")
    ):
        return None
    return data


def iter_lines(path: Path) -> Iterator[str]:
    """Les lignes du fichier, la dernière ligne tronquée est ignorée (append en cours)."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.endswith("\n"):
                yield line.rstrip("\n")
        # la dernière ligne sans saut de ligne peut être un append inachevé : ignorée


def consume(path: Path, limit_bytes: int) -> None:
    """Retire les `limit_bytes` premiers octets (ce qui a été appliqué).

    Règle de sécurité : si la taille a changé depuis la lecture (append arrivé entre),
    on ne touche qu'aux `limit_bytes` premiers octets, jamais aux lignes nouvelles.
    """
    try:
        with path.open("r+b") as handle:
            size = handle.seek(0, os.SEEK_END)
            if size < limit_bytes:
                return
            if size == limit_bytes:
                handle.truncate(0)
                return
            handle.seek(limit_bytes)
            rest = handle.read()
            handle.seek(0)
            handle.write(rest)
            handle.truncate(size - limit_bytes)
    except OSError:
        return
