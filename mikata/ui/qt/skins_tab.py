"""Onglet « Skins » des réglages (ajout phase 8) : skins installés avec un aperçu ; un clic
change de skin tout de suite (gardé dans config.yaml, `ui.skin`), import d'un .zip, mise à
la corbeille d'un skin (jamais Mika, le skin par défaut)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QRectF, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mikata.ui.qt.painter import Frame
from mikata.ui.settings_store import ConfigEditor, EditorError
from mikata.ui.skins import (
    DEFAULT_SKIN,
    Skin,
    SkinError,
    delete_skin,
    import_skin_zip,
    list_skins,
    load_skin,
)

PREVIEW_PX = 96


def render_preview(skin: Skin | None, size: int = PREVIEW_PX, state: str = "idle") -> QPixmap:
    """Aperçu d'un skin : son image de repos (Mika si `skin` est None ou intégré)."""
    from mikata.ui.qt.mascot import SkinPainter

    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    SkinPainter(skin).paint(painter, QRectF(0, 0, size, size), Frame(state, 1.3, 0.45, None))
    painter.end()
    return QPixmap.fromImage(image)


def _ask_zip(parent: QWidget) -> Path | None:
    name, _ = QFileDialog.getOpenFileName(parent, "Importer un skin", "", "Skin (*.zip)")
    return Path(name) if name else None


def _confirm(parent: QWidget, text: str) -> bool:
    answer = QMessageBox.question(parent, "Supprimer le skin", text)
    return answer == QMessageBox.StandardButton.Yes


class SkinsTab(QWidget):
    def __init__(
        self,
        root: Path,
        config_path: Path,
        on_applied: Callable[[dict[str, Any]], None],
        on_status: Callable[[str], None],
        ask_zip: Callable[[QWidget], Path | None] = _ask_zip,
        confirm: Callable[[QWidget, str], bool] = _confirm,
    ) -> None:
        super().__init__()
        self.root = root
        self.config_path = config_path
        self._on_applied = on_applied
        self._on_status = on_status
        self._ask_zip = ask_zip
        self._confirm = confirm
        self.skins: dict[str, Skin | None] = {}  # None : skin inutilisable
        self.errors: dict[str, str] = {}

        self.list = QListWidget()
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setIconSize(QSize(PREVIEW_PX, PREVIEW_PX))
        self.list.setGridSize(QSize(PREVIEW_PX + 44, PREVIEW_PX + 40))
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setWordWrap(True)
        self.list.currentItemChanged.connect(self._on_current)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.import_button = QPushButton("Importer un .zip…")
        self.import_button.clicked.connect(self.import_zip)
        self.delete_button = QPushButton("Supprimer")
        self.delete_button.clicked.connect(self.delete_selected)
        folder = QPushButton("Ouvrir le dossier")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root))))
        refresh = QPushButton("Actualiser")
        refresh.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        for button in (self.import_button, self.delete_button, folder, refresh):
            buttons.addWidget(button)
        buttons.addStretch(1)
        hint = QLabel(
            "Un clic change de skin tout de suite. Créer un skin : docs/MASCOT_SKIN_GUIDE.md "
            "(scripts/make_skin.py)."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.details)
        layout.addLayout(buttons)
        layout.addWidget(hint)
        self.refresh()

    # -- liste -------------------------------------------------------------------------------

    def current_key(self) -> str:
        try:
            return str(ConfigEditor(self.config_path).get("ui.skin") or DEFAULT_SKIN)
        except (OSError, EditorError, ValueError):
            return DEFAULT_SKIN

    def refresh(self) -> None:
        """Relit assets/mascot/ : skins, aperçus, skin en cours sélectionné."""
        self.skins.clear()
        self.errors.clear()
        self.list.blockSignals(True)
        self.list.clear()
        active = self.current_key()
        for info in list_skins(self.root):
            try:
                skin: Skin | None = load_skin(self.root, info.key)
            except SkinError as exc:
                skin = None
                self.errors[info.key] = str(exc)
            self.skins[info.key] = skin
            label = info.name + (" (en cours)" if info.key == active else "")
            state = "error" if info.key in self.errors else "idle"  # inutilisable : Mika en erreur
            item = QListWidgetItem(QIcon(render_preview(skin, state=state)), label)
            item.setData(Qt.ItemDataRole.UserRole, info.key)
            item.setToolTip(self.describe(info.key))
            if skin is None and not info.builtin:
                item.setForeground(QColor("#b91c1c"))
            self.list.addItem(item)
            if info.key == active:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self._show_details(self.selected_key())

    def selected_key(self) -> str | None:
        item = self.list.currentItem()
        return None if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def describe(self, key: str) -> str:
        if key in self.errors:
            return f"Skin inutilisable : {self.errors[key]}"
        skin = self.skins.get(key)
        if skin is None:
            return "Mika, dessinée en code."
        lines = [skin.name]
        if skin.author:
            lines.append(f"Auteur : {skin.author}")
        if skin.license:
            lines.append(f"Licence : {skin.license}")
        if skin.builtin:
            lines.append("Dessinée en code, toujours disponible.")
        elif skin.missing:
            lines.append(f"États absents (image de repos utilisée) : {', '.join(skin.missing)}")
        else:
            lines.append("Tous les états sont fournis.")
        if skin.warnings and not skin.builtin:
            lines.append(f"{len(skin.warnings)} avertissement(s), voir data/logs/ui.log.")
        return "\n".join(lines)

    def _show_details(self, key: str | None) -> None:
        self.details.setText("" if key is None else self.describe(key))
        self.delete_button.setEnabled(key is not None and key != DEFAULT_SKIN)

    def _on_current(self, item: QListWidgetItem | None, _previous: Any = None) -> None:
        if item is None:
            return
        key = str(item.data(Qt.ItemDataRole.UserRole))
        self._show_details(key)
        self.apply(key)

    # -- actions -----------------------------------------------------------------------------

    def apply(self, key: str) -> bool:
        """Change de skin tout de suite et le garde dans config.yaml (`ui.skin`)."""
        if key in self.errors:
            self._on_status(f"Skin « {key} » inutilisable : {self.errors[key]}")
            return False
        try:
            editor = ConfigEditor(self.config_path)
            if editor.get("ui.skin") != key:
                editor.set("ui.skin", key)
                editor.save()
        except (OSError, EditorError) as exc:
            self._on_status(f"Skin non enregistré : {exc}")
            return False
        self._on_applied({"ui.skin": key})
        skin = self.skins.get(key)
        name = skin.name if skin is not None else "Mika"
        self._on_status(f"Skin « {name} » appliqué.")
        logger.info("Skin choisi : {}", key)
        for row in range(self.list.count()):  # « (en cours) » sur le bon skin
            item = self.list.item(row)
            item_key = str(item.data(Qt.ItemDataRole.UserRole))
            base = item.text().removesuffix(" (en cours)")
            item.setText(base + (" (en cours)" if item_key == key else ""))
        return True

    def import_zip(self) -> str | None:
        path = self._ask_zip(self)
        if path is None:
            return None
        try:
            skin = import_skin_zip(path, self.root)
        except SkinError as exc:
            self._on_status(f"Import refusé : {exc}")
            return None
        self.refresh()
        self._select(skin.key)
        missing = f" États absents : {', '.join(skin.missing)}." if skin.missing else ""
        self._on_status(f"Skin « {skin.name} » importé et appliqué.{missing}")
        return skin.key

    def delete_selected(self) -> bool:
        key = self.selected_key()
        if key is None or key == DEFAULT_SKIN:
            return False
        skin = self.skins.get(key)
        name = skin.name if skin is not None else key
        if not self._confirm(self, f"Mettre le skin « {name} » à la corbeille ?"):
            return False
        try:
            delete_skin(self.root, key)
        except (SkinError, OSError) as exc:
            self._on_status(f"Skin non supprimé : {exc}")
            return False
        if self.current_key() == key:
            self.apply(DEFAULT_SKIN)
        self.refresh()
        self._on_status(f"Skin « {name} » mis à la corbeille.")
        return True

    def _select(self, key: str) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == key:
                self.list.setCurrentItem(item)  # -> _on_current -> apply
                return
