"""Exporte Mika (la mascotte dessinée en code) en skin d'images (ajout phase 8).

Exemple du format « image par image » : une bande d'images PNG (spritesheet) par état et un
skin.json prêt à l'emploi. Pour un skin à toi à partir d'images générées, voir plutôt
scripts/make_skin.py et docs/MASCOT_SKIN_GUIDE.md.

    uv run python scripts/export_mascot_skin.py mika-images
    uv run python scripts/export_mascot_skin.py mika-images --size 256 --fps 12

Résultat : assets/mascot/mika-images/skin.json + idle.png, wake.png... Puis réglages > Skins
(ou config.yaml : ui.skin: mika-images). Fonctionne sans écran
(QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THEMES = PROJECT_ROOT / "assets" / "mascot"
# Durée exportée par état (s) : une boucle complète de l'animation dessinée.
DURATIONS = {
    "idle": 2.6,
    "wake": 0.9,
    "listening": 1.2,
    "thinking": 2.6,
    "acting": 3.0,
    "speaking": 1.4,
    "success": 1.6,
    "error": 1.6,
    "follow_up": 2.0,
    "mic_off": 1.0,
}


def export(name: str, size: int = 128, fps: float = 10.0, root: Path = THEMES) -> Path:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from mikata.ui.qt.painter import Frame, paint_mascot
    from mikata.ui.skins import ONE_SHOT

    app = QApplication.instance() or QApplication([])  # noqa: F841 (Qt a besoin d'une app)
    folder = root / name
    if folder.exists():
        raise SystemExit(f"{folder} existe déjà : choisis un autre nom.")
    folder.mkdir(parents=True)
    states = {}
    for state, seconds in DURATIONS.items():
        count = max(1, round(seconds * fps))
        sheet = QImage(size * count, size, QImage.Format.Format_ARGB32)
        sheet.fill(QColor(0, 0, 0, 0))
        painter = QPainter(sheet)
        for i in range(count):
            t = i / fps
            left = 1 - i / count if state == "follow_up" else None
            frame = Frame(state, t=t, age=t, follow_up_left=left)
            paint_mascot(painter, QRectF(i * size, 0, size, size), frame)
        painter.end()
        sheet.save(str(folder / f"{state}.png"))
        states[state] = {"file": f"{state}.png", "frames": count, "fps": fps}
        if state in ONE_SHOT:
            states[state]["loop"] = False
    manifest = {
        "name": name,
        "author": "MIKATA",
        "license": "Apache-2.0 (Mika, dessinée par MIKATA)",
        "states": states,
    }
    (folder / "skin.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return folder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="nom du dossier du skin (assets/mascot/<nom>)")
    parser.add_argument("--size", type=int, default=128, help="côté d'une image (px)")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    folder = export(args.name, args.size, args.fps)
    print(f"Skin exporté : {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
