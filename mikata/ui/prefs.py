"""Préférences de la mascotte (ajout phase 8) : data/ui_state.json, écrit par l'interface.

Position, écran, taille, masquée, bulle : ce sont des préférences de
fenêtre qui changent souvent (à chaque déplacement), pas des réglages de MIKATA ; elles ne
vont donc pas dans config.yaml.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

STATE_FILE = "data/ui_state.json"
SIZES = {"petite": 96, "moyenne": 128, "grande": 192}


@dataclass
class MascotPrefs:
    screen: str = ""  # nom de l'écran (QScreen.name)
    x: int | None = None  # coin haut gauche, coordonnées du bureau
    y: int | None = None
    size: str = "moyenne"
    hidden: bool = False
    bubble: bool = True

    @property
    def pixels(self) -> int:
        return SIZES.get(self.size, SIZES["moyenne"])


def load_prefs(path: Path) -> MascotPrefs:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return MascotPrefs()
    if not isinstance(data, dict):
        return MascotPrefs()
    known = {f.name for f in fields(MascotPrefs)}  # clés anciennes ou inconnues ignorées
    prefs = MascotPrefs(**{k: v for k, v in data.items() if k in known})
    if prefs.size not in SIZES:
        prefs.size = "moyenne"
    return prefs


def save_prefs(path: Path, prefs: MascotPrefs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(asdict(prefs), indent=2), encoding="utf-8")
    tmp.replace(path)
