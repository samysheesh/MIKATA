"""Phase 8 : onglet « Skins » des réglages (hors écran) : choix immédiat gardé dans
config.yaml, aperçus, import d'un .zip, mise à la corbeille."""

from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

pytestmark = pytest.mark.usefixtures("qapp")
ROOT = Path(__file__).resolve().parent.parent


def _png(color: tuple[int, ...]) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (24, 24), color), (4, 4))
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _install(root: Path, key: str, name: str, color: tuple[int, ...] = (0, 120, 255, 255)) -> None:
    folder = root / key
    folder.mkdir(parents=True)
    (folder / "idle.png").write_bytes(_png(color))
    manifest = {"name": name, "author": "Moi", "license": "usage personnel",
                "states": {"idle": {"file": "idle.png"}}}  # fmt: skip
    (folder / "skin.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def base(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "config.yaml", tmp_path / "config.yaml")
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    # le skin choisi dans le dépôt (femme) ne doit pas changer ces tests : on repart de Mika
    (tmp_path / "config.yaml").write_text(
        re.sub(r"(?m)^(  skin: )\S+", r"\g<1>default", text), encoding="utf-8"
    )
    shutil.copytree(  # seul le skin par défaut : les tests ne dépendent pas des skins du dépôt
        ROOT / "assets" / "mascot" / "default", tmp_path / "assets" / "mascot" / "default"
    )
    root = tmp_path / "assets" / "mascot"
    _install(root, "sakura", "Sakura")
    (root / "casse").mkdir()
    (root / "casse" / "skin.json").write_text('{"name": "Cassé", "states": {}}', encoding="utf-8")
    return tmp_path


class Recorder:
    def __init__(self) -> None:
        self.applied: list[dict[str, Any]] = []
        self.status: list[str] = []
        self.zip: Path | None = None
        self.confirm = True


def _tab(base: Path, rec: Recorder) -> Any:
    from mikata.ui.qt.skins_tab import SkinsTab

    return SkinsTab(
        base / "assets" / "mascot",
        base / "config.yaml",
        on_applied=rec.applied.append,
        on_status=rec.status.append,
        ask_zip=lambda parent: rec.zip,
        confirm=lambda parent, text: rec.confirm,
    )


def _texts(tab: Any) -> list[str]:
    return [tab.list.item(i).text() for i in range(tab.list.count())]


def _row(tab: Any, key: str) -> int:
    from PySide6.QtCore import Qt

    keys = [tab.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(tab.list.count())]
    return keys.index(key)


def test_lists_installed_skins_with_previews(base: Path) -> None:
    tab = _tab(base, Recorder())
    assert _texts(tab) == ["Mika (par défaut) (en cours)", "Cassé", "Sakura"]
    for i in range(tab.list.count()):
        assert not tab.list.item(i).icon().isNull()
    assert tab.selected_key() == "default" and not tab.delete_button.isEnabled()
    assert "inutilisable" in tab.describe("casse")
    assert "Auteur : Moi" in tab.describe("sakura") and "États absents" in tab.describe("sakura")


def test_a_click_applies_the_skin_and_keeps_it(base: Path) -> None:
    before = (base / "config.yaml").read_text(encoding="utf-8").splitlines()
    rec = Recorder()
    tab = _tab(base, rec)
    tab.list.setCurrentRow(_row(tab, "sakura"))  # un clic
    assert rec.applied == [{"ui.skin": "sakura"}]
    assert rec.status[-1] == "Skin « Sakura » appliqué."
    after = (base / "config.yaml").read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert len(changed) == 1 and changed[0][1].startswith("  skin: sakura ")
    assert changed[0][1].split("#")[1] == changed[0][0].split("#")[1]  # commentaire gardé
    assert "Sakura (en cours)" in _texts(tab) and "Mika (par défaut)" in _texts(tab)
    assert _tab(base, Recorder()).selected_key() == "sakura"  # gardé à la réouverture
    assert tab.delete_button.isEnabled()


def test_an_unusable_skin_is_not_applied(base: Path) -> None:
    rec = Recorder()
    tab = _tab(base, rec)
    tab.list.setCurrentRow(_row(tab, "casse"))
    assert rec.applied == [] and "inutilisable" in rec.status[-1]
    assert "skin: default" in (base / "config.yaml").read_text(encoding="utf-8")


def test_import_a_zip(base: Path, tmp_path: Path) -> None:
    rec = Recorder()
    tab = _tab(base, rec)
    rec.zip = tmp_path / "chat.zip"
    with zipfile.ZipFile(rec.zip, "w") as archive:
        archive.writestr("chat/skin.json", json.dumps(
            {"name": "Chat", "states": {"idle": {"file": "idle.png"}}}))  # fmt: skip
        archive.writestr("chat/idle.png", _png((255, 140, 0, 255)))
    assert tab.import_zip() == "chat"
    assert "Chat (en cours)" in _texts(tab) and rec.applied[-1] == {"ui.skin": "chat"}
    assert "importé" in rec.status[-1] and "États absents" in rec.status[-1]
    assert tab.import_zip() is None and "existe déjà" in rec.status[-1]  # deux fois : refusé
    rec.zip = tmp_path / "faux.zip"
    rec.zip.write_text("pas un zip", encoding="utf-8")
    assert tab.import_zip() is None and rec.status[-1].startswith("Import refusé")
    rec.zip = None  # dialogue annulé
    assert tab.import_zip() is None


def test_delete_puts_the_skin_in_the_trash(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trashed: list[str] = []

    def fake_trash(path: str) -> None:
        trashed.append(path)
        shutil.move(path, str(base / "corbeille"))

    monkeypatch.setattr("send2trash.send2trash", fake_trash)
    rec = Recorder()
    tab = _tab(base, rec)
    tab.list.setCurrentRow(_row(tab, "sakura"))
    rec.confirm = False
    assert not tab.delete_selected() and trashed == []
    rec.confirm = True
    assert tab.delete_selected()
    assert trashed == [str(base / "assets" / "mascot" / "sakura")]
    assert rec.applied[-1] == {"ui.skin": "default"}  # c'était le skin en cours : retour à Mika
    assert "Sakura" not in " ".join(_texts(tab)) and "corbeille" in rec.status[-1]
    tab.list.setCurrentRow(_row(tab, "default"))
    assert not tab.delete_button.isEnabled() and not tab.delete_selected()  # jamais Mika


def test_settings_window_has_the_skins_tab_and_switches_the_mascot(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mikata.config import load_config
    from mikata.ui.qt import app as ui_app

    monkeypatch.setattr(ui_app.QApplication, "quit", lambda: None)
    config = load_config(base / "config.yaml")
    controller = ui_app.UiController(config, base / "config.yaml")
    controller.open_settings()
    settings = controller.settings
    assert settings is not None
    tabs = [settings.tabs.tabText(i) for i in range(settings.tabs.count())]
    assert tabs == ["Audio", "Mot d'activation", "Conversation", "En cours", "Coût et clés",
                    "Mascotte", "Skins", "Journal"]  # fmt: skip
    assert controller.mascot.painter.skin is not None and controller.mascot.painter.skin.builtin
    settings.skins_tab.list.setCurrentRow(_row(settings.skins_tab, "sakura"))
    assert controller.mascot.painter.skin.key == "sakura"  # tout de suite, sans redémarrer
    assert "Sakura" in settings.status.text()
    controller.client.close()
    settings.close()
    controller.mascot.bubble.close()
    controller.mascot.close()


def test_preview_window_shows_each_state(base: Path) -> None:
    """make_skin.py --preview : la mascotte avec le skin, et une liste des états."""
    from mikata.ui.model import VISUAL_STATES
    from mikata.ui.qt.preview import CYCLE_S, PreviewPanel
    from mikata.ui.skins import load_skin

    now = [100.0]
    panel = PreviewPanel(load_skin(base / "assets" / "mascot", "sakura"), clock=lambda: now[0])
    panel.start()
    assert panel.mascot.isVisible() and panel.mascot.painter.skin.key == "sakura"
    assert panel.states.count() == len(VISUAL_STATES)
    assert "absent, image de repos" in panel.states.item(1).text()  # sakura n'a que le repos
    for row, state in enumerate(VISUAL_STATES):
        panel.states.setCurrentRow(row)
        assert panel.model.visual() == state
    panel.states.setCurrentRow(VISUAL_STATES.index("success"))
    now[0] += 5  # « c'est fait » est fini : rejoué tant qu'il est choisi
    panel.tick()
    assert panel.model.visual() == "success"
    panel.cycle.setChecked(True)
    now[0] += CYCLE_S
    panel.tick()
    assert panel.state == VISUAL_STATES[VISUAL_STATES.index("success") + 1]
    panel.close()
    assert not panel.mascot.isVisible()
