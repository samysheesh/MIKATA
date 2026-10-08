"""Skins de la mascotte (ajout phase 8) : un dossier par skin dans assets/mascot/.

    assets/mascot/<skin>/skin.json   manifeste : nom, auteur, licence, images par état
    assets/mascot/<skin>/*.png ...   une image par état (animée par effets), ou plusieurs
                                     (bande d'images, GIF, APNG) jouées image par image

Un skin ne fait que fournir les images : les états (`model.VISUAL_STATES`) et leur logique
ne changent pas. Le skin `default` (Mika) est dessiné en code (`builtin: true`,
mikata/ui/qt/painter.py). Les autres sont décodés ici avec Pillow, sans Qt.

- **Une image par état** : le programme l'anime avec des effets (`EFFECTS` : respiration,
  rebond, clignement, lueur, balancement), réglés par état dans le manifeste (amplitude,
  vitesse) ; sans réglage, ceux de `DEFAULT_EFFECTS`.
- **Plusieurs images** : jouées image par image (durées du GIF/APNG, ou `fps`) ; des effets
  peuvent s'y ajouter.
- **État absent** : l'image de repos (`idle`, seul état obligatoire), avec un avertissement
  dans le journal. Un réglage invalide est ignoré avec un avertissement ; seul un skin
  inutilisable (manifeste ou image de repos illisible) est refusé (`SkinError`).
- `import_skin_zip` : installe un skin depuis un .zip, validé en mémoire avant d'écrire
  quoi que ce soit ; `delete_skin` : à la corbeille (jamais Mika).
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from PIL import Image, ImageSequence

from mikata.ui.model import STATE_LABELS, VISUAL_STATES

MANIFEST = "skin.json"
DEFAULT_SKIN = "default"
ONE_SHOT = ("wake", "success", "error")  # joués une fois (dernière image gardée)
MAX_FRAMES = 240
MAX_SIDE = 4096
DEFAULT_FPS = 8.0
IMAGE_SUFFIXES = (".png", ".apng", ".gif", ".webp", ".jpg", ".jpeg")
IMAGE_FORMATS = ("PNG", "GIF", "WEBP", "JPEG")
TEXT_SUFFIXES = (".json", ".txt", ".md")

# Import d'un .zip : limites contre les archives piégées.
ZIP_MAX_FILES = 300
ZIP_MAX_FILE_BYTES = 16 * 1024 * 1024
ZIP_MAX_TOTAL_BYTES = 64 * 1024 * 1024
ZIP_IGNORED = ("__MACOSX/", ".DS_Store", "Thumbs.db", "desktop.ini")

# Effets d'une image fixe. amplitude : fraction de la taille (breathe, bounce, blink),
# degrés (sway) ou intensité 0-1 (glow) ; speed : cycles par seconde.
EFFECTS = ("breathe", "bounce", "blink", "glow", "sway")
EFFECT_LABELS = {
    "breathe": "respiration",
    "bounce": "petit rebond",
    "blink": "clignement",
    "glow": "lueur",
    "sway": "balancement",
}
AMPLITUDE_MAX = {"breathe": 0.2, "bounce": 0.3, "blink": 0.5, "glow": 1.0, "sway": 30.0}
SPEED_MAX = 10.0
BLINK_S = 0.14  # durée d'un clignement
STATE_COLORS = {  # couleur de la lueur par défaut (celles de Mika, painter.COLORS)
    "idle": "#7c8aa5",
    "wake": "#f5b301",
    "listening": "#2f7de1",
    "thinking": "#f28c28",
    "acting": "#f28c28",
    "speaking": "#2fb350",
    "success": "#22c55e",
    "error": "#d93838",
    "follow_up": "#14b8a6",
    "mic_off": "#6b7280",
}


def _fx(**effects: tuple[float, float]) -> dict[str, dict[str, float]]:
    return {name: {"amplitude": a, "speed": f} for name, (a, f) in effects.items()}


# Effets d'une image fixe quand le manifeste n'en donne pas : (amplitude, vitesse).
DEFAULT_EFFECTS: dict[str, dict[str, dict[str, float]]] = {
    "idle": _fx(breathe=(0.025, 0.22)),  # il dort : respiration lente
    "wake": _fx(bounce=(0.10, 2.2), glow=(0.8, 3.0)),
    "listening": _fx(breathe=(0.02, 0.5), blink=(0.08, 0.3), glow=(0.5, 1.0)),
    "thinking": _fx(sway=(4.0, 0.5), blink=(0.08, 0.2)),
    "acting": _fx(bounce=(0.03, 2.0), sway=(3.0, 1.0)),
    "speaking": _fx(bounce=(0.025, 3.0), glow=(0.4, 1.5)),
    "success": _fx(bounce=(0.07, 1.8), glow=(0.8, 2.0)),
    "error": _fx(sway=(4.0, 3.0), glow=(0.7, 2.0)),
    "follow_up": _fx(breathe=(0.02, 0.6), glow=(0.5, 0.8)),
    "mic_off": _fx(breathe=(0.012, 0.2)),
}


class SkinError(ValueError):
    """Skin inutilisable (manifeste, image de repos, archive) ; message en français."""


@dataclass(frozen=True)
class Effect:
    amplitude: float
    speed: float
    color: str | None = None  # lueur uniquement (« #rrggbb »)


@dataclass(frozen=True)
class Pose:
    """Déformation de l'image à un instant (calculée par `pose`, appliquée par Qt)."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    dy: float = 0.0  # fraction de la hauteur, négatif = vers le haut
    angle: float = 0.0  # degrés, autour du bas de l'image
    glow: float = 0.0  # 0 : pas de lueur
    glow_color: str = "#ffffff"
    blink: bool = False  # en plein clignement (image `blink_file` si fournie)


def pose(effects: Mapping[str, Effect], t: float, color: str = "#ffffff") -> Pose:
    """Pose de l'image `t` secondes après le début (horloge continue)."""
    sx = sy = 1.0
    dy = angle = glow = 0.0
    glow_color = color
    blink = False
    for name, effect in effects.items():
        a, f = effect.amplitude, effect.speed
        wave = math.sin(2 * math.pi * f * t)
        if name == "breathe":
            sy *= 1 + a * wave
            sx *= 1 - a * wave / 2
        elif name == "bounce":
            dy -= a * abs(math.sin(math.pi * f * t))
        elif name == "sway":
            angle += a * wave
        elif name == "glow":
            glow = max(glow, a * (0.65 + 0.35 * wave))
            glow_color = effect.color or color
        elif name == "blink" and f > 0:
            phase = t % (1 / f)
            if phase < BLINK_S:
                blink = True
                sy *= 1 - a * math.sin(math.pi * phase / BLINK_S)
    return Pose(sx, sy, dy, angle, glow, glow_color, blink)


@dataclass
class StateArt:
    """Images d'un état : une seule (animée par effets) ou une animation image par image."""

    frames: list[Image.Image]  # RGBA
    durations_ms: list[int]
    loop: bool
    effects: dict[str, Effect] = field(default_factory=dict)
    blink_frame: Image.Image | None = None  # yeux fermés, montrée pendant `blink`

    def frame_at(self, elapsed_s: float) -> Image.Image:
        """Image à afficher `elapsed_s` après le début de l'état."""
        if len(self.frames) == 1:
            return self.frames[0]
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
class Skin:
    key: str
    name: str
    path: Path
    builtin: bool = False
    author: str = ""
    license: str = ""
    states: dict[str, StateArt] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def art(self, state: str) -> StateArt | None:
        """Images de `state`, ou celles du repos si le skin ne le fournit pas."""
        return self.states.get(state) or self.states.get("idle")

    @property
    def missing(self) -> list[str]:
        if self.builtin:
            return []
        return [state for state in VISUAL_STATES if state not in self.states]


@dataclass(frozen=True)
class SkinInfo:
    key: str
    name: str
    author: str
    license: str
    builtin: bool


# -- liste, chargement -----------------------------------------------------------------------


def list_skins(root: Path) -> list[SkinInfo]:
    """Skins installés (manifeste lisible), Mika en premier puis par nom."""
    skins = []
    for manifest in sorted(root.glob(f"*/{MANIFEST}")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        key = manifest.parent.name
        skins.append(
            SkinInfo(
                key=key,
                name=str(data.get("name") or key),
                author=str(data.get("author") or ""),
                license=str(data.get("license") or ""),
                builtin=bool(data.get("builtin", False)),
            )
        )
    skins.sort(key=lambda s: (s.key != DEFAULT_SKIN, s.name.casefold()))
    return skins


def load_skin(root: Path, key: str) -> Skin:
    folder = root / key
    try:
        raw = (folder / MANIFEST).read_bytes()
    except FileNotFoundError as exc:
        raise SkinError(f"skin « {key} » introuvable : {folder / MANIFEST}") from exc
    except OSError as exc:
        raise SkinError(f"{folder / MANIFEST} illisible : {exc}") from exc

    def read(name: str) -> bytes:
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise SkinError(f"{name} doit être dans le dossier du skin")
        return path.read_bytes()

    skin = parse_skin(key, folder, raw, read)
    for warning in skin.warnings:
        logger.warning("Skin « {} » : {}", key, warning)
    return skin


def parse_skin(key: str, folder: Path, manifest: bytes, read: Callable[[str], bytes]) -> Skin:
    """Skin décrit par `manifest` ; `read(nom)` donne le contenu d'un fichier du skin."""
    try:
        data = json.loads(manifest.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SkinError(f"{MANIFEST} illisible : {exc}") from exc
    if not isinstance(data, dict):
        raise SkinError(f"{MANIFEST} : un objet JSON est attendu")
    skin = Skin(
        key=key,
        name=str(data.get("name") or key),
        path=folder,
        builtin=bool(data.get("builtin", False)),
        author=str(data.get("author") or ""),
        license=str(data.get("license") or ""),
    )
    if skin.builtin:
        return skin
    states = data.get("states")
    if not isinstance(states, dict) or not isinstance(states.get("idle"), dict):
        raise SkinError(f"{MANIFEST} : « states » doit décrire au moins « idle » (repos)")
    for state, spec in states.items():
        if state not in VISUAL_STATES:
            skin.warnings.append(f"état inconnu « {state} » ignoré ({', '.join(VISUAL_STATES)})")
            continue
        if not isinstance(spec, dict):
            skin.warnings.append(f"« {state} » : un objet est attendu, état ignoré")
            continue
        try:
            skin.states[state] = _load_state(state, spec, read, skin.warnings)
        except (TypeError, ValueError) as exc:  # SkinError, ou un nombre invalide (fps...)
            message = str(exc) if isinstance(exc, SkinError) else f"« {state} » : {exc}"
            if state == "idle":
                raise SkinError(message) from exc
            skin.warnings.append(f"{message} : image de repos utilisée")
    for state in skin.missing:
        if state not in states:
            skin.warnings.append(
                f"état « {state} » ({STATE_LABELS[state]}) absent : image de repos utilisée"
            )
    return skin


def _open_image(name: str, state: str, read: Callable[[str], bytes]) -> Image.Image:
    if not name.lower().endswith(IMAGE_SUFFIXES):
        raise SkinError(f"« {state} » : {name} n'est pas une image PNG, GIF, APNG, WebP ou JPEG")
    try:
        image = Image.open(io.BytesIO(read(name)))
        image.load()
    except SkinError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SkinError(f"« {state} » : image illisible ({name}) : {exc}") from exc
    if image.format not in IMAGE_FORMATS:
        raise SkinError(f"« {state} » : {name} : format {image.format} non pris en charge")
    if max(image.size) > MAX_SIDE:
        raise SkinError(f"« {state} » : {name} dépasse {MAX_SIDE} px")
    return image


def _load_state(
    state: str, spec: dict[str, Any], read: Callable[[str], bytes], warnings: list[str]
) -> StateArt:
    name = spec.get("file")
    if not isinstance(name, str) or not name:
        raise SkinError(f"« {state} » : « file » manquant")
    image = _open_image(name, state, read)
    loop = bool(spec.get("loop", state not in ONE_SHOT))
    fps = spec.get("fps")
    if getattr(image, "n_frames", 1) > 1:  # GIF, APNG, WebP animé : images et durées du fichier
        frames, durations = [], []
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGBA"))
            durations.append(int(frame.info.get("duration") or 100))
            if len(frames) > MAX_FRAMES:
                raise SkinError(f"« {state} » : plus de {MAX_FRAMES} images")
        if fps:
            durations = [round(1000 / float(fps))] * len(frames)
        art = StateArt(frames, [max(d, 10) for d in durations], loop)
    else:
        art = StateArt(_split_sheet(state, name, image, spec), [], loop)
        art.durations_ms = [round(1000 / float(fps or DEFAULT_FPS))] * len(art.frames)
    default = DEFAULT_EFFECTS.get(state, {}) if len(art.frames) == 1 else {}
    art.effects = _effects(state, spec.get("effects", default), warnings)
    blink = spec.get("blink_file")
    if isinstance(blink, str) and blink:
        try:
            art.blink_frame = _open_image(blink, state, read).convert("RGBA")
        except SkinError as exc:
            warnings.append(f"{exc} : clignement sans image")
        else:  # image yeux fermés : on la montre au lieu d'écraser l'image
            speed = art.effects["blink"].speed if "blink" in art.effects else 0.25
            art.effects["blink"] = Effect(0.0, speed)
    return art


def _split_sheet(state: str, name: str, image: Image.Image, spec: dict[str, Any]) -> list[Any]:
    """Bande d'images (spritesheet) : `frames` images sur `columns` colonnes."""
    count = int(spec.get("frames", 1))
    if not 1 <= count <= MAX_FRAMES:
        raise SkinError(f"« {state} » : « frames » entre 1 et {MAX_FRAMES}")
    columns = max(int(spec.get("columns", count)), 1)
    rows = -(-count // columns)
    sheet = image.convert("RGBA")
    width, height = sheet.width // columns, sheet.height // rows
    if width < 1 or height < 1 or width * columns != sheet.width or height * rows != sheet.height:
        raise SkinError(
            f"« {state} » : {name} ({sheet.width}×{sheet.height}) ne se découpe pas en "
            f"{count} image(s) sur {columns} colonne(s)"
        )
    if count == 1:
        return [sheet]
    return [
        sheet.crop(((i % columns) * width, (i // columns) * height,
                    (i % columns + 1) * width, (i // columns + 1) * height))
        for i in range(count)
    ]  # fmt: skip


def _effects(state: str, raw: Any, warnings: list[str]) -> dict[str, Effect]:
    if not isinstance(raw, dict):
        warnings.append(f"« {state} » : « effects » doit être un objet, effets ignorés")
        return {}
    effects = {}
    for name, params in raw.items():
        if name not in EFFECTS:
            warnings.append(f"« {state} » : effet inconnu « {name} » ignoré ({', '.join(EFFECTS)})")
            continue
        if not isinstance(params, dict):
            warnings.append(f"« {state} » : réglages de « {name} » invalides, effet ignoré")
            continue
        try:
            amplitude = float(params.get("amplitude", 0.0))
            speed = float(params.get("speed", 1.0))
        except (TypeError, ValueError):
            warnings.append(f"« {state} » : « {name} » : amplitude et speed sont des nombres")
            continue
        top = AMPLITUDE_MAX[name]
        if not (0 <= amplitude <= top and 0 <= speed <= SPEED_MAX):
            warnings.append(
                f"« {state} » : « {name} » ramené dans les limites "
                f"(amplitude 0 à {top:g}, speed 0 à {SPEED_MAX:g})"
            )
            amplitude, speed = min(max(amplitude, 0.0), top), min(max(speed, 0.0), SPEED_MAX)
        color = params.get("color")
        if color is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)):
            warnings.append(f"« {state} » : couleur « {color} » invalide (#rrggbb attendu)")
            color = None
        effects[name] = Effect(amplitude, speed, str(color) if color else None)
    return effects


# -- import et suppression -------------------------------------------------------------------


def skin_key(name: str) -> str:
    """Nom de dossier d'un skin : minuscules, chiffres, - et _ (« Sakura Chibi » ->
    sakura-chibi)."""
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    key = re.sub(r"[^a-z0-9_-]+", "-", ascii_name.lower()).strip("-_")[:40]
    return key or "skin"


def import_skin_zip(zip_path: Path, root: Path) -> Skin:
    """Installe le skin d'un .zip dans `root` ; rien n'est écrit si l'archive est invalide."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise SkinError(f"{zip_path.name} n'est pas un fichier .zip lisible ({exc})") from exc
    with archive:
        infos = [
            info
            for info in archive.infolist()
            if not info.is_dir() and not any(part in info.filename for part in ZIP_IGNORED)
        ]
        if len(infos) > ZIP_MAX_FILES:
            raise SkinError(f"archive trop grande : plus de {ZIP_MAX_FILES} fichiers")
        if sum(info.file_size for info in infos) > ZIP_MAX_TOTAL_BYTES:
            raise SkinError(f"archive trop grande : plus de {ZIP_MAX_TOTAL_BYTES // 2**20} Mo")
        manifests = [i.filename for i in infos if PurePosixPath(i.filename).name == MANIFEST]
        if len(manifests) != 1 or len(PurePosixPath(manifests[0]).parts) > 2:
            raise SkinError(
                f"l'archive doit contenir un seul {MANIFEST}, à la racine ou dans un dossier"
            )
        prefix = PurePosixPath(manifests[0]).parent
        files: dict[str, bytes] = {}
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise SkinError(f"chemin refusé dans l'archive : {info.filename}")
            if prefix.parts and path.parts[: len(prefix.parts)] != prefix.parts:
                continue  # hors du dossier du skin : ignoré
            relative = PurePosixPath(*path.parts[len(prefix.parts) :]).as_posix()
            if not relative.lower().endswith(IMAGE_SUFFIXES + TEXT_SUFFIXES):
                raise SkinError(
                    f"fichier refusé : {relative} (seulement des images PNG, GIF, APNG, WebP, "
                    "JPEG, et des fichiers .json, .txt, .md)"
                )
            if info.file_size > ZIP_MAX_FILE_BYTES:
                raise SkinError(f"{relative} : plus de {ZIP_MAX_FILE_BYTES // 2**20} Mo")
            data = archive.read(info)
            if len(data) != info.file_size:
                raise SkinError(f"{relative} : taille incohérente dans l'archive")
            files[relative] = data
    manifest = files[MANIFEST]
    try:
        declared = json.loads(manifest.decode("utf-8-sig")).get("name")
    except (UnicodeDecodeError, ValueError, AttributeError) as exc:
        raise SkinError(f"{MANIFEST} illisible : {exc}") from exc
    key = skin_key(str(declared or (prefix.name if prefix.parts else zip_path.stem)))
    if key == DEFAULT_SKIN or (root / key).exists():
        raise SkinError(f"un skin « {key} » existe déjà : supprime-le d'abord, ou renomme-le")

    def read(name: str) -> bytes:
        normalized = PurePosixPath(name).as_posix()
        if normalized not in files:
            raise SkinError(f"{name} absent de l'archive")
        return files[normalized]

    skin = parse_skin(key, root / key, manifest, read)
    if skin.builtin:
        raise SkinError("un skin importé ne peut pas être « builtin »")
    for relative, data in files.items():
        target = root / key / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    logger.info("Skin « {} » importé depuis {} ({} fichiers).", key, zip_path, len(files))
    for warning in skin.warnings:
        logger.warning("Skin « {} » : {}", key, warning)
    return skin


def delete_skin(root: Path, key: str) -> None:
    """Met le dossier d'un skin à la corbeille (jamais Mika, le skin par défaut)."""
    from send2trash import send2trash

    folder = root / key
    if key == DEFAULT_SKIN:
        raise SkinError("Mika, le skin par défaut, ne peut pas être supprimé")
    if not key or folder.resolve().parent != root.resolve():
        raise SkinError(f"nom de skin invalide : « {key} »")
    try:
        builtin = json.loads((folder / MANIFEST).read_text(encoding="utf-8")).get("builtin")
    except (OSError, ValueError, AttributeError):
        builtin = False
    if builtin:
        raise SkinError("un skin intégré ne peut pas être supprimé")
    if not folder.is_dir():
        raise SkinError(f"skin « {key} » introuvable")
    send2trash(str(folder))
    logger.info("Skin « {} » mis à la corbeille.", key)
