"""Thèmes de la mascotte (ajout phase 8) : un dossier par thème dans assets/mascot/.

    assets/mascot/<thème>/theme.json   manifeste (voir assets/mascot/README.md)
    assets/mascot/<thème>/*.png|gif    images : bandes d'images (spritesheets), GIF, APNG

Le thème `default` est dessiné en code (`builtin: true`, mikata/ui/qt/painter.py) : aucune
image. Les autres thèmes sont décodés ici avec Pillow (déjà une dépendance), sans Qt : la
fenêtre convertit ensuite les images. Un état absent d'un thème prend l'animation d'un état
proche (`FALLBACKS`), jusqu'à `idle`, seul état obligatoire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence

from mikata.ui.model import VISUAL_STATES

MANIFEST = "theme.json"
DEFAULT_THEME = "default"
ONE_SHOT = ("wake", "success", "error")  # joués une fois (dernière image gardée)
FALLBACKS = {
    "acting": "thinking",
    "follow_up": "listening",
    "wake": "listening",
    "success": "speaking",
    "error": "idle",
    "mic_off": "idle",
    "listening": "idle",
    "thinking": "idle",
    "speaking": "idle",
}
MAX_FRAMES = 240
MAX_SIDE = 4096
DEFAULT_FPS = 8.0


class ThemeError(ValueError):
    """Thème illisible : manifeste ou image invalide (message en français)."""


@dataclass
class Animation:
    frames: list[Image.Image]  # RGBA
    durations_ms: list[int]
    loop: bool

    def frame_at(self, elapsed_s: float) -> Image.Image:
        """Image à afficher `elapsed_s` après le début de l'animation."""
        total = sum(self.durations_ms)
        t = elapsed_s * 1000
        if self.loop:
            t %= max(total, 1)
        elif t >= total:
            return self.frames[-1]
        for frame, duration in zip(self.frames, self.durations_ms, strict=True):
            if t < duration:
                return frame
            t -= duration
        return self.frames[-1]


@dataclass
class Theme:
    key: str
    name: str
    path: Path
    builtin: bool = False
    author: str = ""
    license: str = ""
    animations: dict[str, Animation] = field(default_factory=dict)

    def animation(self, state: str) -> Animation | None:
        """Animation de `state`, ou celle de l'état de repli le plus proche."""
        return self.animations.get(resolve_state(self.animations, state))


def resolve_state(available: dict[str, Any] | set[str], state: str) -> str:
    seen = set()
    while state not in available and state in FALLBACKS and state not in seen:
        seen.add(state)
        state = FALLBACKS[state]
    return state if state in available else "idle"


def list_themes(root: Path) -> list[tuple[str, str]]:
    """(clé, nom) des thèmes de `root`, le thème par défaut en premier."""
    themes = []
    for manifest in sorted(root.glob(f"*/{MANIFEST}")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            name = str(data.get("name", manifest.parent.name))
        except (OSError, ValueError):
            continue
        themes.append((manifest.parent.name, name))
    themes.sort(key=lambda t: (t[0] != DEFAULT_THEME, t[1].casefold()))
    return themes


def load_theme(root: Path, key: str) -> Theme:
    folder = root / key
    try:
        data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ThemeError(f"thème « {key} » introuvable : {folder / MANIFEST}") from exc
    except (OSError, ValueError) as exc:
        raise ThemeError(f"{folder / MANIFEST} illisible : {exc}") from exc
    if not isinstance(data, dict):
        raise ThemeError(f"{folder / MANIFEST} : un objet JSON est attendu")
    theme = Theme(
        key=key,
        name=str(data.get("name", key)),
        path=folder,
        builtin=bool(data.get("builtin", False)),
        author=str(data.get("author", "")),
        license=str(data.get("license", "")),
    )
    if theme.builtin:
        return theme
    states = data.get("states")
    if not isinstance(states, dict) or "idle" not in states:
        raise ThemeError(f"{folder / MANIFEST} : « states » doit décrire au moins « idle »")
    for state, spec in states.items():
        if state not in VISUAL_STATES:
            raise ThemeError(f"état inconnu « {state} » (attendus : {', '.join(VISUAL_STATES)})")
        if not isinstance(spec, dict):
            raise ThemeError(f"« {state} » : un objet est attendu")
        theme.animations[state] = _load_animation(folder, state, spec)
    return theme


def _load_animation(folder: Path, state: str, spec: dict[str, Any]) -> Animation:
    name = spec.get("file")
    if not isinstance(name, str) or not name:
        raise ThemeError(f"« {state} » : « file » manquant")
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ThemeError(f"« {state} » : {name} doit être dans le dossier du thème")
    try:
        image = Image.open(path)
    except (OSError, ValueError) as exc:
        raise ThemeError(f"« {state} » : image illisible ({name}) : {exc}") from exc
    if max(image.size) > MAX_SIDE:
        raise ThemeError(f"« {state} » : {name} dépasse {MAX_SIDE} px")
    loop = bool(spec.get("loop", state not in ONE_SHOT))
    fps = spec.get("fps")
    count = int(spec.get("frames", 1))
    if getattr(image, "is_animated", False):  # GIF ou APNG : images et durées du fichier
        frames, durations = [], []
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGBA"))
            durations.append(int(frame.info.get("duration") or 100))
            if len(frames) > MAX_FRAMES:
                raise ThemeError(f"« {state} » : plus de {MAX_FRAMES} images")
        if fps:
            durations = [round(1000 / float(fps))] * len(frames)
        return Animation(frames, [max(d, 10) for d in durations], loop)
    if not 1 <= count <= MAX_FRAMES:
        raise ThemeError(f"« {state} » : « frames » entre 1 et {MAX_FRAMES}")
    columns = int(spec.get("columns", count))
    rows = -(-count // max(columns, 1))
    sheet = image.convert("RGBA")
    width, height = sheet.width // max(columns, 1), sheet.height // rows
    if width < 1 or height < 1 or width * columns != sheet.width or height * rows != sheet.height:
        raise ThemeError(
            f"« {state} » : {name} ({sheet.width}×{sheet.height}) ne se découpe pas en "
            f"{count} image(s) sur {columns} colonne(s)"
        )
    frames = [
        sheet.crop(((i % columns) * width, (i // columns) * height,
                    (i % columns + 1) * width, (i // columns + 1) * height))
        for i in range(count)
    ]  # fmt: skip
    duration = round(1000 / float(fps or DEFAULT_FPS))
    return Animation(frames, [duration] * count, loop)
