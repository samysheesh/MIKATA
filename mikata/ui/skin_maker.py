"""Fabrication d'un skin à partir d'images brutes (ajout phase 8), utilisée par
scripts/make_skin.py. Sans Qt : Pillow, numpy et scipy.

Les générateurs d'images (Nano Banana...) rendent un fond uni ou un faux damier de
transparence, jamais une vraie transparence. Pour chaque image :

1. **Fond** : couleur(s) relevées sur le pourtour de l'image ; une couleur dominante = fond
   uni, deux couleurs qui se partagent le bord = damier. Est retiré tout ce qui ressemble au
   fond et touche le bord (composantes connexes), plus, pour un damier, les trous fermés qui
   montrent encore le damier (entre un bras et le corps). Le blanc des yeux d'un personnage
   sur fond blanc reste donc intact. Une ombre au sol (le fond assombri sous le personnage)
   devient une ombre noire translucide, sans le damier. Une image déjà transparente est
   gardée telle quelle.
2. **Bords propres** : sur la bande de contour, l'opacité suit la distance à la couleur du
   fond et la couleur est « décontaminée » (le fond mélangé au bord est retiré), puis un léger
   flou adoucit le contour (anticrénelage) sans halo.
3. **Recadrage et taille** : même échelle pour tous les états (le personnage ne change pas
   de taille d'un état à l'autre), centré horizontalement, posé en bas du cadre (ou centré),
   redimensionné en alpha prémultiplié (pas de liseré sombre).
4. **Manifeste** : skin.json avec nom, auteur, licence et les effets par défaut de chaque
   état (`skins.DEFAULT_EFFECTS`), à retoucher à la main.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence
from scipy import ndimage

from mikata.ui.model import STATE_LABELS, VISUAL_STATES
from mikata.ui.skins import DEFAULT_EFFECTS, IMAGE_SUFFIXES, MANIFEST, ONE_SHOT, SkinError

# Noms de fichiers acceptés pour chaque état (sans accents, minuscules, « _ » ou « - »).
STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "idle": ("idle", "repos", "dort", "sleep", "sleeping"),
    "wake": ("wake", "activation", "mot_activation", "reveil", "wake_word"),
    "listening": ("listening", "ecoute", "listen"),
    "thinking": ("thinking", "reflexion", "think"),
    "acting": ("acting", "action"),
    "speaking": ("speaking", "parole", "speak", "talking"),
    "success": ("success", "succes", "fait", "c_est_fait", "done"),
    "error": ("error", "erreur"),
    "follow_up": ("follow_up", "suivi", "fenetre_de_suivi"),
    "mic_off": ("mic_off", "micro_coupe", "micro_off", "muted"),
}
BLINK_SUFFIXES = ("blink", "clin", "yeux_fermes")
DEFAULT_SIZE = 384
DEFAULT_TOLERANCE = 38.0
DEFAULT_MARGIN = 0.06
DEFAULT_FPS = 8.0


@dataclass
class Cutout:
    """Une image détourée et ce qu'on a détecté."""

    image: Image.Image  # RGBA
    background: str  # « uni », « damier » ou « transparent »
    colors: list[tuple[int, int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StateSource:
    state: str
    frames: list[Path | Image.Image]
    durations_ms: list[int] | None = None
    blink: Path | None = None


@dataclass
class MakeReport:
    folder: Path
    lines: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


# -- noms de fichiers ------------------------------------------------------------------------


def _normalize(stem: str) -> str:
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_stem.lower()).strip("_")


def state_of(stem: str) -> tuple[str, int | None, bool] | None:
    """(état, numéro d'image ou None, image de clignement ?) d'après un nom de fichier.
    « Repos », « idle_02 », « ecoute-blink » -> ("idle", None, False), ("idle", 2, False),
    ("listening", None, True)."""
    name = _normalize(stem)
    blink = False
    for suffix in BLINK_SUFFIXES:
        if name.endswith("_" + suffix):
            name, blink = name[: -len(suffix) - 1], True
            break
    number = None
    match = re.fullmatch(r"(.+?)_?(\d+)", name)
    if match and not blink:
        name, number = match.group(1), int(match.group(2))
    for state, aliases in STATE_ALIASES.items():
        if name in aliases:
            return state, number, blink
    return None


def collect_sources(folder: Path) -> tuple[dict[str, StateSource], list[str]]:
    """Images de `folder` rangées par état ; renvoie aussi les fichiers ignorés."""
    sources: dict[str, StateSource] = {}
    numbered: dict[str, list[tuple[int, Path]]] = {}
    ignored = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        found = state_of(path.stem)
        if found is None:
            ignored.append(path.name)
            continue
        state, number, blink = found
        source = sources.setdefault(state, StateSource(state, []))
        if blink:
            source.blink = path
        elif number is None:
            source.frames.insert(0, path)
        else:
            numbered.setdefault(state, []).append((number, path))
    for state, items in numbered.items():
        source = sources[state]
        if not source.frames:  # « idle.png » prime sur « idle_01.png... »
            source.frames = [path for _, path in sorted(items)]
    for state in [s for s, src in sources.items() if not src.frames]:  # clignement seul
        ignored.append(f"{state} (clignement sans image)")
        del sources[state]
    return sources, ignored


# -- détourage -------------------------------------------------------------------------------


def _border_pixels(rgb: np.ndarray, width: int = 2) -> np.ndarray:
    return np.concatenate(
        [
            rgb[:width].reshape(-1, 3),
            rgb[-width:].reshape(-1, 3),
            rgb[:, :width].reshape(-1, 3),
            rgb[:, -width:].reshape(-1, 3),
        ]
    )


def background_colors(rgb: np.ndarray, tolerance: float) -> tuple[list[np.ndarray], str, str]:
    """Couleur(s) du fond relevées sur le bord : ([couleurs], « uni »|« damier », alerte)."""
    border = _border_pixels(rgb)
    quantized = (border // 16).astype(np.int32)
    codes = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1]
    first = border[codes == values[order[0]]].mean(axis=0)
    near_first = np.linalg.norm(border - first, axis=1) < tolerance
    share_first = near_first.mean()
    rest = border[~near_first]
    if len(rest):
        rest_q = (rest // 16).astype(np.int32)
        rest_codes = rest_q[:, 0] * 256 + rest_q[:, 1] * 16 + rest_q[:, 2]
        rest_values, rest_counts = np.unique(rest_codes, return_counts=True)
        second = rest[rest_codes == rest_values[np.argmax(rest_counts)]].mean(axis=0)
        near_second = np.linalg.norm(border - second, axis=1) < tolerance
        share_second = (near_second & ~near_first).mean()
        if share_second >= 0.2 and share_first + share_second >= 0.85:
            return [first, second], "damier", ""
    warning = ""
    if share_first < 0.6:
        warning = (
            f"fond peu uniforme ({share_first:.0%} du bord) : vérifie le détourage dans "
            "apercu.png, ou régénère l'image sur un fond uni"
        )
    return [first], "uni", warning


def checker_cell(nearest: np.ndarray) -> int:
    """Côté d'une case du damier (px), d'après l'alternance des couleurs sur le bord ; 0 pour
    un fond uni."""
    runs = []
    for line in (nearest[0], nearest[-1], nearest[:, 0], nearest[:, -1]):
        changes = np.nonzero(np.diff(line.astype(np.int8)))[0]
        runs.extend(np.diff(changes).tolist())
    return int(np.median(runs)) if len(runs) >= 4 else 0


def shadow_region(
    rgb: np.ndarray, background: np.ndarray, colors: list[np.ndarray], nearest: np.ndarray
) -> np.ndarray:
    """Ombre au sol : zone épaisse et aplatie, sous le personnage, collée au fond, dont la
    couleur est celle du fond assombrie (même teinte, plus sombre). Prudente : seuls les
    gris neutres proches du fond s'y ajoutent, pour ne pas manger un personnage gris."""
    ratio = rgb / np.maximum(np.stack(colors)[nearest], 1.0)  # par rapport au fond le plus proche
    k = ratio.mean(axis=-1)
    flat = ~background & (np.ptp(ratio, axis=-1) < 0.06) & (k > 0.45) & (k < 0.97)
    # Ombre pâle, jusqu'à la couleur du fond (sur un damier, une case claire enfermée par
    # l'ombre n'est pas reliée au fond) : seule à pouvoir prolonger ou remplir l'ombre.
    faint = ~background & (np.ptp(ratio, axis=-1) < 0.06) & (k > 0.7)
    reach = max(4, min(background.shape) // 48, checker_cell(nearest) // 2 + 2)
    pieces_open = ndimage.binary_opening(flat, iterations=3)  # pas les traits fins
    # Sur un damier, l'ombre est découpée en cases : on les réunit avant de juger sa forme.
    square = np.ones((3, 3), dtype=bool)  # une croix ne relierait pas les cases
    merged = ndimage.binary_closing(pieces_open, square, iterations=reach) & (background | flat)
    labels, count = ndimage.label(merged | pieces_open)
    foreground_rows = np.nonzero((~background).any(axis=1))[0]
    if not count or not len(foreground_rows):
        return np.zeros_like(background)
    top, bottom = foreground_rows[0], foreground_rows[-1]
    touching = np.unique(labels[ndimage.binary_dilation(background, iterations=2)])
    region = np.zeros_like(background)
    for label, (rows, cols) in enumerate(ndimage.find_objects(labels), start=1):
        height, width = rows.stop - rows.start, cols.stop - cols.start
        center = (rows.start + rows.stop) / 2
        if label in touching and height < 0.6 * width and center > top + 0.7 * (bottom - top):
            region |= labels == label
    region &= flat
    if not region.any():
        return region
    region = ndimage.binary_dilation(region, iterations=2) & flat
    # Bouts pâles de l'ombre (trop fins pour être trouvés) : morceaux d'ombre pâle tout près,
    # de proche en proche (sur un damier, les cases les séparent les uns des autres).
    rows, _ = np.nonzero(region)
    below = np.zeros_like(background)
    below[max(0, rows.min() - reach) :] = True  # pas plus haut que l'ombre trouvée
    pieces, _ = ndimage.label(faint & below)
    for _ in range(8):
        near = ndimage.binary_dilation(region, square, iterations=reach) & (pieces > 0)
        grown = region | np.isin(pieces, np.unique(pieces[near]))
        if (grown == region).all():
            break
        region = grown
    # L'ombre posée sur une case claire du damier a la couleur d'une case foncée : ces
    # morceaux ont été pris pour du fond. On referme la zone de l'ombre et on la remplit.
    area = ndimage.binary_fill_holes(ndimage.binary_closing(region, square, iterations=reach))
    rows, cols = np.nonzero(region)
    box = np.zeros_like(background)
    box[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
    return region | (area & box & (background | faint))


def remove_background(
    image: Image.Image,
    tolerance: float = DEFAULT_TOLERANCE,
    holes: str = "auto",
    shadow: str = "clean",
) -> Cutout:
    """Détoure `image` (voir le docstring du module). `holes` : « auto » (trous fermés retirés
    pour un damier), « on » (aussi pour un fond uni) ou « off ». `shadow` (ombre au sol) :
    « clean » (gardée, en noir translucide : plus de damier dedans), « remove » ou « keep »."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    rgb, alpha_in = rgba[..., :3], rgba[..., 3]
    if (alpha_in < 250).mean() > 0.01:  # déjà transparente : rien à retirer
        return Cutout(image.convert("RGBA"), "transparent")
    colors, kind, warning = background_colors(rgb, tolerance)
    distances = np.stack([np.linalg.norm(rgb - c, axis=-1) for c in colors])
    nearest = np.argmin(distances, axis=0)
    distance = distances.min(axis=0)
    candidate = distance < tolerance
    labels, count = ndimage.label(candidate)
    edge = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )  # composantes qui touchent le bord
    background = np.isin(labels, edge[edge > 0])
    remove_holes = holes == "on" or (holes == "auto" and kind == "damier")
    if remove_holes and count:
        min_area = max(64, int(candidate.size * 0.0005))
        areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, count + 1))
        for label in np.nonzero(areas >= min_area)[0] + 1:
            if label in edge:
                continue
            region = labels == label
            if kind == "damier":  # un vrai trou montre les deux cases du damier
                shares = np.bincount(nearest[region], minlength=2) / region.sum()
                if shares.min() < 0.2:
                    continue
            background |= region
    shadow_mask = np.zeros_like(background)
    if shadow != "keep":
        shadow_mask = shadow_region(rgb, background, colors, nearest)
        if shadow == "remove":
            background |= shadow_mask
    # Bande de contour : chaque pixel y est un mélange c = a·F + (1 - a)·B du personnage (F,
    # couleur du pixel intérieur le plus proche) et du fond (B) ; on en tire l'opacité a,
    # puis la couleur sans le fond.
    band = ndimage.binary_dilation(background, iterations=2) & ~background
    bg_rgb = np.stack(colors)[nearest]
    core = ndimage.binary_erosion(~background, iterations=3)
    if core.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
        diff = rgb[iy, ix] - bg_rgb
        norm = (diff**2).sum(axis=-1)
        mix = ((rgb - bg_rgb) * diff).sum(axis=-1) / np.maximum(norm, 1.0)
        soft = np.where(norm < tolerance**2, 1.0, np.clip(mix, 0.0, 1.0))
    else:
        soft = np.clip((distance - tolerance * 0.5) / (tolerance * 2.5), 0.0, 1.0)
    alpha = np.where(background, 0.0, 1.0)
    alpha = np.where(band, soft, alpha)
    safe = np.maximum(alpha, 1e-3)[..., None]
    clean = np.where(band[..., None], bg_rgb + (rgb - bg_rgb) / safe, rgb)
    if shadow == "clean" and shadow_mask.any():  # ombre noire translucide, sans le fond
        lum = rgb.mean(axis=-1)
        if len(colors) == 1:  # fond uni : assombrissement exact, pixel par pixel
            darkness = 1 - lum / max(float(colors[0].mean()), 1.0)
        else:  # damier : assombrissement moyen sur quelques cases, là où il y a de l'ombre
            sigma = max(2.0, min(rgb.shape[:2]) / 64)
            reference = float(np.mean([c.mean() for c in colors]))
            darkness = 1 - ndimage.gaussian_filter(lum, sigma) / max(reference, 1.0)
            darker = lum / np.maximum(np.stack(colors)[nearest].mean(axis=-1), 1.0) < 0.97
            coverage = ndimage.gaussian_filter((shadow_mask & darker).astype(np.float32), sigma)
            darkness *= np.clip(coverage * 2.2, 0.0, 1.0)
        alpha = np.where(shadow_mask, np.clip(darkness, 0.0, 0.85), alpha)
        clean = np.where(shadow_mask[..., None], 0.0, clean)
    alpha = np.minimum(alpha, ndimage.gaussian_filter(alpha, 0.7))  # anticrénelage, sans halo
    clean = np.where((alpha > 0)[..., None], clean, 0.0)  # pixels transparents : noirs
    out = np.dstack([np.clip(clean, 0, 255), np.clip(alpha * 255, 0, 255)]).astype(np.uint8)
    rounded = [tuple(int(v) for v in np.round(c)) for c in colors]
    return Cutout(Image.fromarray(out, "RGBA"), kind, rounded, [warning] if warning else [])


# -- recadrage, taille -------------------------------------------------------------------------


def alpha_bbox(image: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(image.getchannel("A"))
    rows, cols = np.nonzero(alpha > threshold)
    if not len(rows):
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def place(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    scale: float,
    size: int,
    margin: float,
    align: str,
) -> Image.Image:
    """`frame` recadrée sur `box`, mise à l'échelle et posée sur une image size×size."""
    crop = frame.crop(box).convert(
        "RGBa"
    )  # alpha prémultiplié : pas de liseré au redimensionnement
    width = max(1, round((box[2] - box[0]) * scale))
    height = max(1, round((box[3] - box[1]) * scale))
    resized = crop.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - width) // 2
    y = size - round(size * margin) - height if align == "bottom" else (size - height) // 2
    canvas.alpha_composite(resized, (x, max(0, y)))
    return canvas


# -- skin complet ------------------------------------------------------------------------------


def _frames_of(source: StateSource) -> tuple[list[Image.Image], list[int] | None]:
    frames: list[Image.Image] = []
    durations: list[int] | None = None
    for item in source.frames:
        image = Image.open(item) if isinstance(item, Path) else item
        if getattr(image, "n_frames", 1) > 1 and len(source.frames) == 1:
            durations = []
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA"))
                durations.append(int(frame.info.get("duration") or 100))
        else:
            frames.append(image.convert("RGBA"))
    return frames, durations


def make_skin(
    sources: dict[str, StateSource],
    folder: Path,
    *,
    name: str,
    author: str = "",
    license: str = "",
    size: int = DEFAULT_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
    margin: float = DEFAULT_MARGIN,
    align: str = "bottom",
    holes: str = "auto",
    shadow: str = "clean",
    fps: float = DEFAULT_FPS,
) -> MakeReport:
    """Écrit le skin dans `folder` (qui ne doit pas exister) : images détourées, skin.json,
    apercu.png (tous les états sur un damier, pour vérifier le détourage)."""
    if "idle" not in sources:
        raise SkinError("il faut au moins l'image de repos (idle.png ou repos.png)")
    if folder.exists():
        raise SkinError(f"{folder} existe déjà")
    report = MakeReport(folder)
    cut: dict[str, tuple[list[Image.Image], list[int] | None, Image.Image | None]] = {}
    fractions = []
    for state in VISUAL_STATES:
        source = sources.get(state)
        if source is None:
            report.missing.append(state)
            continue
        raw_frames, durations = _frames_of(source)
        frames, kinds = [], []
        for raw in raw_frames:
            cutout = remove_background(raw, tolerance, holes, shadow)
            frames.append(cutout.image)
            kinds.append(cutout.background)
            for warning in cutout.warnings:
                report.lines.append(f"{state} : {warning}")
        kind = kinds[0]
        blink = None
        if source.blink is not None:
            blink = remove_background(Image.open(source.blink), tolerance, holes, shadow).image
        boxes = [b for b in (alpha_bbox(f) for f in [*frames, *([blink] if blink else [])]) if b]
        if not boxes:
            raise SkinError(f"« {state} » : l'image est vide une fois le fond retiré")
        box = _union(boxes)
        side = max(frames[0].size)
        fractions.append(max(box[2] - box[0], box[3] - box[1]) / side)
        cut[state] = (frames, durations, blink)
        report.lines.append(
            f"{state} ({STATE_LABELS[state]}) : {len(frames)} image(s), fond {kind}"
            + (", clignement" if blink is not None else "")
        )
    inner = size * (1 - 2 * margin)
    largest = max(fractions)
    folder.mkdir(parents=True)
    states: dict[str, Any] = {}
    sheet = []
    for state, (frames, durations, blink) in cut.items():
        boxes = [b for b in (alpha_bbox(f) for f in [*frames, *([blink] if blink else [])]) if b]
        box = _union(boxes)
        scale = inner / (largest * max(frames[0].size))
        placed = [place(f, box, scale, size, margin, align) for f in frames]
        spec: dict[str, Any] = {"file": f"{state}.png"}
        if len(placed) == 1:
            placed[0].save(folder / f"{state}.png", optimize=True)
            spec["effects"] = DEFAULT_EFFECTS.get(state, {})
        else:
            delays = durations or [round(1000 / fps)] * len(placed)
            placed[0].save(
                folder / f"{state}.png",
                save_all=True,
                append_images=placed[1:],
                duration=delays,
                loop=0,
                disposal=1,
                blend=0,
            )
            if state in ONE_SHOT:
                spec["loop"] = False
        if blink is not None:
            place(blink, box, scale, size, margin, align).save(folder / f"{state}_blink.png")
            spec["blink_file"] = f"{state}_blink.png"
            spec.setdefault("effects", {})["blink"] = {"amplitude": 0.0, "speed": 0.25}
        states[state] = spec
        sheet.append((state, placed[0]))
    manifest = {
        "name": name,
        "author": author,
        "license": license,
        "size": size,
        "generator": "scripts/make_skin.py",
        "states": states,
    }
    (folder / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    contact_sheet(sheet).save(folder / "apercu.png")
    for state in report.missing:
        report.lines.append(
            f"{state} ({STATE_LABELS[state]}) : absent, l'image de repos sera utilisée"
        )
    return report


def checkerboard(width: int, height: int, cell: int = 12) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    light = ((x // cell + y // cell) % 2 == 0)[..., None]
    pixels = np.where(light, [236, 239, 244], [206, 212, 222]).astype(np.uint8)
    return Image.fromarray(pixels, "RGB").convert("RGBA")


def contact_sheet(images: list[tuple[str, Image.Image]], cell: int = 160) -> Image.Image:
    """Planche de tous les états sur un damier (vérification du détourage)."""
    from PIL import ImageDraw

    columns = 5
    rows = -(-len(images) // columns)
    sheet = checkerboard(columns * cell, rows * (cell + 18))
    draw = ImageDraw.Draw(sheet)
    for i, (state, image) in enumerate(images):
        x, y = (i % columns) * cell, (i // columns) * (cell + 18)
        thumb = image.resize((cell, cell), Image.Resampling.LANCZOS)
        sheet.alpha_composite(thumb, (x, y))
        draw.text((x + 6, y + cell + 2), state, fill=(31, 36, 48, 255))
    return sheet
