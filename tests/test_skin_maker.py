"""Phase 8 : fabrication d'un skin à partir d'images brutes (scripts/make_skin.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scripts_loader import load_script

from mikata.ui.model import VISUAL_STATES
from mikata.ui.skin_maker import (
    StateSource,
    background_colors,
    collect_sources,
    make_skin,
    remove_background,
    state_of,
)
from mikata.ui.skins import DEFAULT_EFFECTS, SkinError, load_skin

SIZE = 200
RED = (210, 40, 50)


def _background(kind: str) -> Image.Image:
    if kind == "uni":
        return Image.new("RGB", (SIZE, SIZE), (250, 250, 250))
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    light = ((x // 10 + y // 10) % 2 == 0)[..., None]
    return Image.fromarray(np.where(light, 255, 204).repeat(3, -1).astype(np.uint8), "RGB")


def _character(
    kind: str = "uni",
    box: tuple[int, int, int, int] = (50, 40, 150, 140),
    ring: bool = False,
    shadow: bool = False,
    jpeg: bool = False,
) -> Image.Image:
    """Un disque rouge cerné de noir, avec un œil blanc ; fond uni ou damier."""
    image = _background(kind)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if shadow:  # ombre au sol : fond assombri, aplati, sous le personnage
        shade = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(shade).ellipse(
            (60, box[3] + 4, 140, box[3] + 22), fill=(0, 0, 0, 52)
        )  # case claire assombrie = case foncée
        image = Image.alpha_composite(image.convert("RGBA"), shade).convert("RGB")
    draw.ellipse(box, fill=(*RED, 255), outline=(20, 20, 30, 255), width=4)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    if ring:  # trou au milieu : on voit le fond à travers
        draw.ellipse(
            (cx - 22, cy - 22, cx + 22, cy + 22),
            fill=(0, 0, 0, 0),
            outline=(20, 20, 30, 255),
            width=4,
        )
    else:
        draw.ellipse((cx - 14, cy - 24, cx + 14, cy + 4), fill=(255, 255, 255, 255))  # œil
    big = layer.resize((SIZE * 4, SIZE * 4), Image.Resampling.NEAREST)
    smooth = big.resize((SIZE, SIZE), Image.Resampling.LANCZOS)  # bords anticrénelés
    if ring:
        hole = np.asarray(smooth).copy()
        mask = np.zeros(hole.shape[:2], bool)
        yy, xx = np.ogrid[:SIZE, :SIZE]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 < 17**2] = True
        hole[mask] = 0
        smooth = Image.fromarray(hole, "RGBA")
    out = Image.alpha_composite(image.convert("RGBA"), smooth).convert("RGB")
    if jpeg:
        buffer = io.BytesIO()
        out.save(buffer, "JPEG", quality=85)
        out = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return out


def _alpha(image: Image.Image) -> np.ndarray:
    return np.asarray(image.getchannel("A")).astype(int)


# -- noms de fichiers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("idle", ("idle", None, False)),
        ("Repos", ("idle", None, False)),
        ("écoute", ("listening", None, False)),
        ("Réflexion", ("thinking", None, False)),
        ("c'est fait", ("success", None, False)),
        ("micro-coupé", ("mic_off", None, False)),
        ("idle_02", ("idle", 2, False)),
        ("parole3", ("speaking", 3, False)),
        ("ecoute_blink", ("listening", None, True)),
        ("repos-yeux-fermes", ("idle", None, True)),
        ("portrait", None),
    ],
)
def test_state_names(stem: str, expected: Any) -> None:
    assert state_of(stem) == expected


def test_collect_sources(tmp_path: Path) -> None:
    for name in ("repos.png", "repos_blink.png", "parole_1.png", "parole_2.png", "erreur.jpg",
                 "notes.txt", "fond.png", "ecoute_blink.png"):  # fmt: skip
        (tmp_path / name).write_bytes(b"")
    sources, ignored = collect_sources(tmp_path)
    assert set(sources) == {"idle", "speaking", "error"}
    assert sources["idle"].blink == tmp_path / "repos_blink.png"
    assert [p.name for p in sources["speaking"].frames] == ["parole_1.png", "parole_2.png"]
    assert "fond.png" in ignored and any("listening" in i for i in ignored)


# -- détourage ------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["uni", "damier"])
def test_background_kind_is_detected(kind: str) -> None:
    rgb = np.asarray(_character(kind), dtype=np.float32)
    colors, found, warning = background_colors(rgb, 38)
    assert found == kind and warning == "" and len(colors) == (1 if kind == "uni" else 2)


@pytest.mark.parametrize("kind", ["uni", "damier"])
@pytest.mark.parametrize("jpeg", [False, True])
def test_background_removed_character_kept(kind: str, jpeg: bool) -> None:
    cut = remove_background(_character(kind, jpeg=jpeg))
    alpha = _alpha(cut.image)
    assert cut.background == kind
    assert alpha[:30].max() == 0 and alpha[:, :30].max() == 0  # tout le fond est parti
    assert alpha[110, 100] == 255  # le corps
    assert alpha[80, 100] == 255  # l'œil blanc, même sur fond blanc
    edge = (alpha > 0) & (alpha < 255)
    assert edge.sum() > 50  # bords adoucis (anticrénelage)
    rgb = np.asarray(cut.image)[..., :3].astype(int)
    assert rgb[edge & (alpha > 60)].mean() < 120  # pas de liseré clair : le contour reste sombre


def test_holes_showing_the_checkerboard_are_removed() -> None:
    alpha = _alpha(remove_background(_character("damier", ring=True)).image)
    assert alpha[90, 100] == 0 and alpha[45, 100] > 200  # le trou, l'anneau


def test_holes_on_a_plain_background_only_on_request() -> None:
    raw = _character("uni", ring=True)
    assert _alpha(remove_background(raw).image)[90, 100] == 255  # auto : gardé (œil blanc ?)
    assert _alpha(remove_background(raw, holes="on").image)[90, 100] == 0


def test_ground_shadow_becomes_clean() -> None:
    raw = _character("damier", shadow=True)
    clean = remove_background(raw, shadow="clean").image
    alpha, rgb = _alpha(clean), np.asarray(clean)[..., :3].astype(int)
    shade = alpha[148:158, 80:120]
    assert 15 < shade.mean() < 160 and shade.std() < 25  # translucide et uniforme : plus de damier
    assert rgb[148:158, 80:120].max() < 40  # noir
    assert _alpha(remove_background(raw, shadow="remove").image)[148:158, 80:120].max() < 40
    kept = _alpha(remove_background(raw, shadow="keep").image)[148:158, 80:120]
    assert kept.max() == 255 and kept.std() > 50  # telle quelle : le damier y reste
    assert _alpha(clean)[110, 100] == 255  # le personnage n'a pas bougé


def test_already_transparent_images_are_kept() -> None:
    image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((5, 5, 35, 35), fill=(*RED, 255))
    cut = remove_background(image)
    assert cut.background == "transparent" and np.array_equal(_alpha(cut.image), _alpha(image))


def width(alpha: np.ndarray) -> int:
    return int(np.ptp(np.nonzero(alpha.max(axis=0) > 128)[0]))


def bottom(alpha: np.ndarray) -> int:
    return int(np.nonzero(alpha.max(axis=1) > 128)[0].max())


# -- skin complet ----------------------------------------------------------------------------------


def test_make_skin(tmp_path: Path) -> None:
    sources = {
        "idle": StateSource("idle", [_character("damier")]),
        "speaking": StateSource("speaking", [_character("uni", box=(30, 20, 170, 160))]),
        "wake": StateSource("wake", [_character("damier", box=(70, 10, 130, 70))] * 3),
    }
    report = make_skin(
        sources, tmp_path / "rond", name="Rond", author="Moi", license="usage personnel", size=128
    )
    folder = tmp_path / "rond"
    manifest = json.loads((folder / "skin.json").read_text(encoding="utf-8"))
    assert (manifest["name"], manifest["author"], manifest["license"]) == (
        "Rond", "Moi", "usage personnel",
    )  # fmt: skip
    assert manifest["states"]["idle"]["effects"] == DEFAULT_EFFECTS["idle"]
    assert (
        "effects" not in manifest["states"]["wake"] and manifest["states"]["wake"]["loop"] is False
    )
    assert (folder / "apercu.png").is_file()
    assert set(report.missing) == set(VISUAL_STATES) - set(sources)
    skin = load_skin(tmp_path, "rond")
    assert len(skin.states["wake"].frames) == 3  # APNG
    idle, speaking = skin.states["idle"].frames[0], skin.states["speaking"].frames[0]
    assert idle.size == (128, 128)
    a_idle, a_speak = _alpha(idle), _alpha(speaking)
    assert width(a_speak) > width(a_idle) * 1.2  # même échelle : le grand reste plus grand
    assert abs(bottom(a_idle) - bottom(a_speak)) <= 2  # posés au même niveau
    cols = np.nonzero(a_idle.max(axis=0) > 128)[0]
    assert abs((cols.min() + cols.max()) / 2 - 64) <= 2  # centré


def test_make_skin_errors(tmp_path: Path) -> None:
    with pytest.raises(SkinError, match="repos"):
        make_skin({"wake": StateSource("wake", [_character()])}, tmp_path / "a", name="A")
    (tmp_path / "b").mkdir()
    with pytest.raises(SkinError, match="existe déjà"):
        make_skin({"idle": StateSource("idle", [_character()])}, tmp_path / "b", name="B")
    empty = Image.new("RGB", (50, 50), (250, 250, 250))
    with pytest.raises(SkinError, match="vide"):
        make_skin({"idle": StateSource("idle", [empty])}, tmp_path / "c", name="C")


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    raw, out = tmp_path / "images", tmp_path / "skins"
    raw.mkdir()
    _character("damier").save(raw / "repos.png")
    _character("uni", jpeg=True).save(raw / "ecoute.jpg")
    _character("uni").save(raw / "repos_blink.png")
    script = load_script("make_skin")
    args = [str(raw), "--name", "Ma Mascotte", "--author", "Moi", "--out", str(out), "--size", "96"]
    assert script.main(args) == 0
    skin = load_skin(out, "ma-mascotte")
    assert skin.author == "Moi" and skin.states["idle"].blink_frame is not None
    assert "listening" in skin.states and "absent" in capsys.readouterr().out
    assert script.main(args) == 2  # existe déjà
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", trashed.append)
    # Corbeille simulée : l'ancien dossier reste là, d'où le refus (code 1) qui suit.
    assert script.main([*args, "--replace"]) == 1
    assert trashed == [str(out / "ma-mascotte")]
    assert script.main([str(tmp_path / "absent"), "--out", str(out)]) == 2
    assert script.main([str(raw), "--name", "default", "--out", str(out)]) == 2
