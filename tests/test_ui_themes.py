"""Phase 8 : thèmes de la mascotte (manifeste, spritesheets, GIF, APNG, replis)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from scripts_loader import load_script

from mikata.ui.model import VISUAL_STATES
from mikata.ui.themes import (
    FALLBACKS,
    Animation,
    ThemeError,
    list_themes,
    load_theme,
    resolve_state,
)

ROOT = Path(__file__).resolve().parent.parent
COLORS = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 128)]


def _theme(root: Path, key: str, states: dict[str, Any], **extra: Any) -> Path:
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"name": key.capitalize(), "states": states, **extra}
    (folder / "theme.json").write_text(json.dumps(manifest), encoding="utf-8")
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


def test_the_repository_default_theme_is_builtin() -> None:
    theme = load_theme(ROOT / "assets" / "mascot", "default")
    assert theme.builtin and theme.animations == {} and "Apache-2.0" in theme.license
    assert list_themes(ROOT / "assets" / "mascot")[0][0] == "default"


def test_spritesheet_grid(tmp_path: Path) -> None:
    folder = _theme(tmp_path, "grille", {"idle": {"file": "idle.png", "frames": 3, "columns": 2,
                                                  "fps": 10}})  # fmt: skip
    _sheet(folder / "idle.png", 3, 2)
    animation = load_theme(tmp_path, "grille").animation("idle")
    assert animation is not None and len(animation.frames) == 3 and animation.loop
    assert [animation.frame_at(t).getpixel((4, 4)) for t in (0, 0.15, 0.25, 0.31)] == [
        COLORS[0], COLORS[1], COLORS[2], COLORS[0],  # 100 ms par image, en boucle
    ]  # fmt: skip


def test_gif_and_apng_keep_their_durations(tmp_path: Path) -> None:
    folder = _theme(tmp_path, "anime", {"idle": {"file": "idle.gif"},
                                        "wake": {"file": "wake.png"}})  # fmt: skip
    _animated(folder / "idle.gif", [100, 300])
    _animated(folder / "wake.png", [50, 50, 200])  # APNG
    theme = load_theme(tmp_path, "anime")
    idle, wake = theme.animations["idle"], theme.animations["wake"]
    assert idle.durations_ms == [100, 300] and idle.loop
    assert wake.durations_ms == [50, 50, 200] and not wake.loop  # wake : joué une fois
    assert wake.frame_at(10).getpixel((0, 0)) == COLORS[2]  # dernière image gardée


def test_missing_states_fall_back(tmp_path: Path) -> None:
    folder = _theme(tmp_path, "mini", {"idle": {"file": "a.png"}, "thinking": {"file": "b.png"}})
    _sheet(folder / "a.png", 1, 1)
    _sheet(folder / "b.png", 1, 1)
    theme = load_theme(tmp_path, "mini")
    assert theme.animation("acting") is theme.animations["thinking"]
    for state in ("success", "follow_up", "wake", "error", "mic_off"):
        assert theme.animation(state) is theme.animations["idle"]
    assert set(FALLBACKS) == set(VISUAL_STATES) - {"idle"}
    assert resolve_state({"idle", "listening"}, "follow_up") == "listening"


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ({"thinking": {"file": "a.png"}}, "idle"),
        ({"idle": {"file": "a.png"}, "dancing": {"file": "a.png"}}, "état inconnu"),
        ({"idle": {}}, "file"),
        ({"idle": {"file": "../secret.png"}}, "dossier du thème"),
        ({"idle": {"file": "absent.png"}}, "illisible"),
        ({"idle": {"file": "a.png", "frames": 3}}, "ne se découpe pas"),
        ({"idle": {"file": "a.png", "frames": 0}}, "frames"),
    ],
)
def test_invalid_themes_are_explained(tmp_path: Path, states: dict[str, Any], message: str) -> None:
    folder = _theme(tmp_path, "casse", states)
    _sheet(folder / "a.png", 2, 2)
    _sheet(tmp_path / "secret.png", 1, 1)
    with pytest.raises(ThemeError, match=message):
        load_theme(tmp_path, "casse")


def test_unreadable_manifests(tmp_path: Path) -> None:
    with pytest.raises(ThemeError, match="introuvable"):
        load_theme(tmp_path, "absent")
    (tmp_path / "vide").mkdir()
    (tmp_path / "vide" / "theme.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ThemeError):
        load_theme(tmp_path, "vide")
    (tmp_path / "vide" / "theme.json").write_text("{", encoding="utf-8")
    assert list_themes(tmp_path) == []  # ignoré par la liste


def test_list_themes_sorts_default_first(tmp_path: Path) -> None:
    for key, name in (("zz", "Abeille"), ("default", "Mika"), ("chat", "Chat")):
        _theme(tmp_path, key, {}, name=name)
    assert list_themes(tmp_path) == [("default", "Mika"), ("zz", "Abeille"), ("chat", "Chat")]


def test_one_frame_animation() -> None:
    frame = Image.new("RGBA", (2, 2))
    animation = Animation([frame], [100], loop=False)
    assert animation.frame_at(0) is frame and animation.frame_at(99) is frame


def test_export_gives_a_loadable_theme(tmp_path: Path, qapp: Any) -> None:
    script = load_script("export_mascot_theme")
    folder = script.export("mon_theme", size=32, fps=5, root=tmp_path)
    theme = load_theme(tmp_path, "mon_theme")
    assert folder == tmp_path / "mon_theme" and not theme.builtin
    assert set(theme.animations) == set(VISUAL_STATES)
    idle = theme.animations["idle"]
    assert idle.frames[0].size == (32, 32) and idle.loop
    assert idle.frames[0].getpixel((0, 0))[3] == 0  # fond transparent
    assert idle.frames[0].getpixel((16, 18))[3] == 255  # le personnage
    assert not theme.animations["success"].loop
    with pytest.raises(SystemExit):
        script.export("mon_theme", root=tmp_path)  # jamais écrasé
