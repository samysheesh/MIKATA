"""Phase 8 : skins de la mascotte (manifeste, une image animée par effets, image par image,
repli sur le repos, import .zip, suppression)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from scripts_loader import load_script

from mikata.ui.model import VISUAL_STATES
from mikata.ui.skins import (
    DEFAULT_EFFECTS,
    EFFECTS,
    Effect,
    SkinError,
    delete_skin,
    import_skin_zip,
    list_skins,
    load_skin,
    pose,
    skin_key,
)

ROOT = Path(__file__).resolve().parent.parent
COLORS = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 128)]


def _png(color: tuple[int, ...] = (200, 30, 30, 255), size: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (size, size), color).save(buffer, "PNG")
    return buffer.getvalue()


def _skin(root: Path, key: str, states: dict[str, Any], **extra: Any) -> Path:
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"name": key.capitalize(), "states": states, **extra}
    (folder / "skin.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def _sheet(path: Path, count: int, columns: int, size: int = 8) -> None:
    rows = -(-count // columns)
    sheet = Image.new("RGBA", (columns * size, rows * size))
    for i in range(count):
        tile = Image.new("RGBA", (size, size), COLORS[i % len(COLORS)])
        sheet.paste(tile, ((i % columns) * size, (i // columns) * size))
    sheet.save(path)


def _animated(path: Path, durations: list[int]) -> None:
    frames = [Image.new("RGBA", (8, 8), COLORS[i]) for i in range(len(durations))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


# -- skins installés ---------------------------------------------------------------------------


def test_mika_is_the_builtin_default() -> None:
    skin = load_skin(ROOT / "assets" / "mascot", "default")
    assert skin.builtin and skin.states == {} and skin.missing == []
    assert "Apache-2.0" in skin.license
    assert list_skins(ROOT / "assets" / "mascot")[0].key == "default"


def test_list_skins_reads_name_author_license(tmp_path: Path) -> None:
    _skin(tmp_path, "zz", {}, name="Abeille", author="Moi", license="usage personnel")
    _skin(tmp_path, "default", {}, name="Mika", builtin=True)
    _skin(tmp_path, "chat", {}, name="Chat")
    (tmp_path / "casse").mkdir()
    (tmp_path / "casse" / "skin.json").write_text("{", encoding="utf-8")
    skins = list_skins(tmp_path)
    assert [s.key for s in skins] == ["default", "zz", "chat"]
    assert (skins[1].author, skins[1].license) == ("Moi", "usage personnel")


# -- une image par état ------------------------------------------------------------------------


def test_one_image_per_state_gets_default_effects(tmp_path: Path) -> None:
    folder = _skin(tmp_path, "fixe", {s: {"file": f"{s}.png"} for s in VISUAL_STATES})
    for state in VISUAL_STATES:
        (folder / f"{state}.png").write_bytes(_png())
    skin = load_skin(tmp_path, "fixe")
    assert skin.missing == [] and skin.warnings == []
    for state in VISUAL_STATES:
        art = skin.art(state)
        assert art is not None and len(art.frames) == 1
        assert set(art.effects) == set(DEFAULT_EFFECTS[state])


def test_effects_from_the_manifest(tmp_path: Path) -> None:
    effects = {
        "bounce": {"amplitude": 0.05, "speed": 2},
        "glow": {"amplitude": 0.5, "speed": 1, "color": "#ff00ff"},
        "dance": {"amplitude": 1},
        "sway": {"amplitude": 90, "speed": 1},
    }
    folder = _skin(tmp_path, "fx", {"idle": {"file": "idle.png", "effects": effects}})
    (folder / "idle.png").write_bytes(_png())
    skin = load_skin(tmp_path, "fx")
    art = skin.states["idle"]
    assert art.effects["bounce"] == Effect(0.05, 2.0)
    assert art.effects["glow"].color == "#ff00ff"
    assert art.effects["sway"].amplitude == 30.0  # ramené dans les limites
    assert "dance" not in art.effects
    assert any("dance" in w for w in skin.warnings) and any("limites" in w for w in skin.warnings)
    folder = _skin(tmp_path, "sans", {"idle": {"file": "idle.png", "effects": {}}})
    (folder / "idle.png").write_bytes(_png())
    assert load_skin(tmp_path, "sans").states["idle"].effects == {}  # aucun effet voulu


def test_pose_of_each_effect() -> None:
    assert pose({}, 1.23) == pose({}, 0)  # sans effet : immobile
    breathe = pose({"breathe": Effect(0.1, 0.25)}, 1.0)  # sommet de la respiration
    assert breathe.scale_y == pytest.approx(1.1) and breathe.scale_x == pytest.approx(0.95)
    bounce = pose({"bounce": Effect(0.1, 1.0)}, 0.5)
    assert bounce.dy == pytest.approx(-0.1) and pose({"bounce": Effect(0.1, 1.0)}, 0).dy == 0
    assert pose({"sway": Effect(5, 0.25)}, 1.0).angle == pytest.approx(5)
    glow = pose({"glow": Effect(0.8, 1.0, "#00ff00")}, 0.25, color="#123456")
    assert glow.glow == pytest.approx(0.8) and glow.glow_color == "#00ff00"
    assert pose({"glow": Effect(0.8, 1.0)}, 0, color="#123456").glow_color == "#123456"
    blink = {"blink": Effect(0.2, 0.5)}  # un clignement toutes les 2 s
    assert pose(blink, 0.07).blink and pose(blink, 0.07).scale_y == pytest.approx(0.8)
    assert not pose(blink, 1.0).blink and pose(blink, 2.07).blink
    assert set(EFFECTS) == {"breathe", "bounce", "blink", "glow", "sway"}


def test_blink_image_replaces_the_squash(tmp_path: Path) -> None:
    folder = _skin(tmp_path, "yeux", {"idle": {"file": "idle.png", "blink_file": "idle_blink.png"}})
    (folder / "idle.png").write_bytes(_png())
    (folder / "idle_blink.png").write_bytes(_png((0, 0, 0, 255)))
    art = load_skin(tmp_path, "yeux").states["idle"]
    assert art.blink_frame is not None and art.effects["blink"].amplitude == 0
    breathing_only = {"breathe": art.effects["breathe"]}  # pas d'écrasement en plus
    assert pose(art.effects, 0.05).blink
    assert pose(art.effects, 0.05).scale_y == pose(breathing_only, 0.05).scale_y


def test_missing_states_fall_back_to_idle_with_a_warning(
    tmp_path: Path, caplog_loguru: list[str]
) -> None:
    folder = _skin(
        tmp_path,
        "mini",
        {"idle": {"file": "a.png"}, "thinking": {"file": "b.png"}, "error": {"file": "x.png"}},
    )
    (folder / "a.png").write_bytes(_png())
    (folder / "b.png").write_bytes(_png((0, 0, 255, 255)))
    skin = load_skin(tmp_path, "mini")  # error.png manque : pas de plantage
    assert skin.art("thinking") is skin.states["thinking"]
    for state in ("acting", "success", "follow_up", "wake", "error", "mic_off"):
        assert skin.art(state) is skin.states["idle"]
    assert set(skin.missing) == set(VISUAL_STATES) - {"idle", "thinking"}
    assert any("« listening »" in m and "repos" in m for m in caplog_loguru)
    assert any("x.png" in m for m in caplog_loguru)


# -- image par image ----------------------------------------------------------------------------


def test_spritesheet_grid(tmp_path: Path) -> None:
    spec = {"file": "idle.png", "frames": 3, "columns": 2, "fps": 10}
    folder = _skin(tmp_path, "grille", {"idle": spec})
    _sheet(folder / "idle.png", 3, 2)
    art = load_skin(tmp_path, "grille").states["idle"]
    assert len(art.frames) == 3 and art.loop and art.effects == {}  # pas d'effet par défaut
    assert [art.frame_at(t).getpixel((4, 4)) for t in (0, 0.15, 0.25, 0.31)] == [
        COLORS[0], COLORS[1], COLORS[2], COLORS[0],  # 100 ms par image, en boucle
    ]  # fmt: skip


def test_gif_and_apng_keep_their_durations(tmp_path: Path) -> None:
    folder = _skin(tmp_path, "anime", {"idle": {"file": "idle.gif"}, "wake": {"file": "wake.png"}})
    _animated(folder / "idle.gif", [100, 300])
    _animated(folder / "wake.png", [50, 50, 200])  # APNG
    skin = load_skin(tmp_path, "anime")
    idle, wake = skin.states["idle"], skin.states["wake"]
    assert idle.durations_ms == [100, 300] and idle.loop
    assert wake.durations_ms == [50, 50, 200] and not wake.loop  # wake : joué une fois
    assert wake.frame_at(10).getpixel((0, 0)) == COLORS[2]  # dernière image gardée


# -- erreurs ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ({"thinking": {"file": "a.png"}}, "idle"),
        ({"idle": {}}, "file"),
        ({"idle": {"file": "../secret.png"}}, "dossier du skin"),
        ({"idle": {"file": "absent.png"}}, "absent.png"),
        ({"idle": {"file": "a.png", "frames": 3}}, "ne se découpe pas"),
        ({"idle": {"file": "a.png", "frames": 0}}, "frames"),
        ({"idle": {"file": "a.png", "fps": "vite"}}, "vite"),
        ({"idle": {"file": "notes.txt"}}, "n'est pas une image"),
    ],
)
def test_unusable_skins_are_explained(tmp_path: Path, states: dict[str, Any], message: str) -> None:
    folder = _skin(tmp_path, "casse", states)
    _sheet(folder / "a.png", 2, 2)
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    _sheet(tmp_path / "secret.png", 1, 1)
    with pytest.raises(SkinError, match=message):
        load_skin(tmp_path, "casse")


def test_a_broken_optional_state_is_only_a_warning(tmp_path: Path) -> None:
    folder = _skin(tmp_path, "presque", {"idle": {"file": "a.png"}, "wake": {"file": "z.png"},
                                          "dancing": {"file": "a.png"}})  # fmt: skip
    (folder / "a.png").write_bytes(_png())
    skin = load_skin(tmp_path, "presque")
    assert skin.art("wake") is skin.states["idle"]
    assert any("dancing" in w for w in skin.warnings)


def test_unreadable_manifests(tmp_path: Path) -> None:
    with pytest.raises(SkinError, match="introuvable"):
        load_skin(tmp_path, "absent")
    (tmp_path / "vide").mkdir()
    (tmp_path / "vide" / "skin.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SkinError):
        load_skin(tmp_path, "vide")


# -- import .zip ---------------------------------------------------------------------------------


def _zip(path: Path, files: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _manifest(name: str = "Sakura Chibi", **states: str) -> str:
    states = states or {"idle": "idle.png"}
    return json.dumps(
        {
            "name": name,
            "author": "Moi",
            "license": "usage personnel",
            "states": {s: {"file": f} for s, f in states.items()},
        }
    )


def test_import_a_zip(tmp_path: Path) -> None:
    root = tmp_path / "skins"
    root.mkdir()
    archive = _zip(
        tmp_path / "sakura.zip",
        {
            "sakura/skin.json": _manifest(idle="idle.png", wake="anim/wake.png"),
            "sakura/idle.png": _png(),
            "sakura/anim/wake.png": _png((0, 255, 0, 255)),
            "sakura/LISEZMOI.txt": "merci",
            "__MACOSX/sakura/._idle.png": b"\x00",
        },
    )
    skin = import_skin_zip(archive, root)
    assert skin.key == "sakura-chibi" and skin.name == "Sakura Chibi" and skin.author == "Moi"
    assert (root / "sakura-chibi" / "anim" / "wake.png").is_file()
    assert not (root / "sakura-chibi" / "__MACOSX").exists()
    assert load_skin(root, "sakura-chibi").art("wake") is not None
    with pytest.raises(SkinError, match="existe déjà"):
        import_skin_zip(archive, root)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"idle.png": _png()}, "un seul skin.json"),
        ({"a/skin.json": _manifest(), "b/skin.json": _manifest()}, "un seul skin.json"),
        ({"a/b/skin.json": _manifest(), "a/b/idle.png": _png()}, "un seul skin.json"),
        ({"skin.json": _manifest(), "idle.png": _png(), "virus.exe": b"MZ"}, "virus.exe"),
        ({"skin.json": _manifest(), "idle.png": _png(), "../evil.png": _png()}, "refusé"),
        ({"skin.json": _manifest()}, "idle.png absent"),
        ({"skin.json": "{pas du json", "idle.png": _png()}, "illisible"),
        ({"skin.json": _manifest(), "idle.png": b"pas une image"}, "illisible"),
        ({"skin.json": _manifest(name="default"), "idle.png": _png()}, "existe déjà"),
    ],
)
def test_bad_zips_are_refused_and_nothing_is_written(
    tmp_path: Path, files: dict[str, Any], message: str
) -> None:
    root = tmp_path / "skins"
    root.mkdir()
    with pytest.raises(SkinError, match=message):
        import_skin_zip(_zip(tmp_path / "x.zip", files), root)
    assert list(root.iterdir()) == []


def test_not_a_zip(tmp_path: Path) -> None:
    fake = tmp_path / "skin.zip"
    fake.write_text("bonjour", encoding="utf-8")
    with pytest.raises(SkinError, match=r"pas un fichier \.zip"):
        import_skin_zip(fake, tmp_path)


def test_zip_bomb_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mikata.ui import skins

    monkeypatch.setattr(skins, "ZIP_MAX_TOTAL_BYTES", 1000)
    archive = _zip(tmp_path / "gros.zip", {"skin.json": _manifest(), "idle.png": b"\0" * 5000})
    with pytest.raises(SkinError, match="trop grande"):
        import_skin_zip(archive, tmp_path)


def test_skin_key() -> None:
    assert skin_key("Sakura Chibi") == "sakura-chibi"
    assert skin_key("Été ☀ 2") == "ete-2"
    assert skin_key("***") == "skin"


# -- suppression ----------------------------------------------------------------------------------


def test_delete_goes_to_the_trash_never_mika(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", trashed.append)
    _skin(tmp_path, "default", {}, builtin=True)
    _skin(tmp_path, "integre", {}, builtin=True)
    _skin(tmp_path, "sakura", {"idle": {"file": "idle.png"}})
    delete_skin(tmp_path, "sakura")
    assert trashed == [str(tmp_path / "sakura")]
    for key, message in (("default", "Mika"), ("integre", "intégré"), ("absent", "introuvable"),
                         ("../x", "invalide")):  # fmt: skip
        with pytest.raises(SkinError, match=message):
            delete_skin(tmp_path, key)
    assert len(trashed) == 1


def test_exported_mika_is_a_loadable_skin(tmp_path: Path, qapp: Any) -> None:
    folder = load_script("export_mascot_skin").export("mika-images", size=32, fps=5, root=tmp_path)
    skin = load_skin(tmp_path, "mika-images")
    assert folder == tmp_path / "mika-images" and not skin.builtin and skin.missing == []
    idle = skin.states["idle"]
    assert idle.frames[0].size == (32, 32) and idle.loop
    assert idle.frames[0].getpixel((0, 0))[3] == 0  # fond transparent
    assert idle.frames[0].getpixel((16, 18))[3] == 255  # le personnage
    assert not skin.states["success"].loop
    with pytest.raises(SystemExit):
        load_script("export_mascot_skin").export("mika-images", root=tmp_path)  # jamais écrasé
