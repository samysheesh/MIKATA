"""Aperçu d'un skin dans la vraie mascotte (ajout phase 8), sans MIKATA :
`scripts/make_skin.py --preview` ou `python -m mikata.ui --preview <skin>`.

La mascotte s'affiche comme d'habitude (sans bordure, au-dessus, glissable) avec le skin
demandé ; une petite fenêtre liste les états : un clic montre l'état, « Défilement » les
enchaîne. Les états joués une fois (mot d'activation, c'est fait, erreur) sont rejoués tant
qu'ils sont choisis. Rien n'est écrit dans config.yaml.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mikata.ui.model import STATE_LABELS, VISUAL_STATES, MascotModel, demo_messages
from mikata.ui.prefs import MascotPrefs
from mikata.ui.qt.mascot import MascotWindow
from mikata.ui.skins import Skin, load_skin

CYCLE_S = 2.5  # « Défilement » : durée de chaque état


class PreviewPanel(QWidget):
    def __init__(self, skin: Skin, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self.setWindowTitle(f"Aperçu du skin « {skin.name} »")
        self.model = MascotModel(clock=clock)
        self.model.handle({"type": "hello", "state": "idle", "mic": "ok"})
        prefs_path = Path(tempfile.gettempdir()) / "mikata_preview_state.json"
        self.mascot = MascotWindow(
            self.model, MascotPrefs(size="grande"), prefs_path, skin, lambda m: False, self.raise_
        )
        self.states = QListWidget()
        for state in VISUAL_STATES:
            item = QListWidgetItem(f"{STATE_LABELS[state]} ({state})")
            item.setData(Qt.ItemDataRole.UserRole, state)
            if state in skin.missing:
                item.setText(item.text() + " : absent, image de repos")
            self.states.addItem(item)
        self.states.currentItemChanged.connect(lambda item, _=None: self.show_state(self.state))
        self.cycle = QCheckBox("Défilement automatique")
        close = QPushButton("Fermer")
        close.clicked.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{skin.name} — {skin.author or 'auteur non indiqué'}"))
        layout.addWidget(self.states, 1)
        layout.addWidget(self.cycle)
        layout.addWidget(close)
        self._shown_at = clock()
        self._clock = clock
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self.states.setCurrentRow(0)

    @property
    def state(self) -> str:
        item = self.states.currentItem()
        return "idle" if item is None else str(item.data(Qt.ItemDataRole.UserRole))

    def show_state(self, state: str) -> None:
        self.model.disconnected()  # oublie l'état précédent (« ! », coche encore affichés)
        self.model.handle({"type": "hello", "state": "idle", "mic": "ok"})
        for message in demo_messages(state):
            self.model.handle(message)
        self.model.note(f"Aperçu : {STATE_LABELS[state]}")
        self._shown_at = self._clock()

    def tick(self) -> None:
        elapsed = self._clock() - self._shown_at
        if self.cycle.isChecked() and elapsed >= CYCLE_S:
            self.states.setCurrentRow((self.states.currentRow() + 1) % self.states.count())
        elif self.model.visual() != self.state:  # état joué une fois : rejoué
            self.show_state(self.state)

    def start(self) -> None:
        self.mascot.start()
        self.show()
        self._timer.start(200)

    def closeEvent(self, event: object) -> None:
        self._timer.stop()
        self.mascot.bubble.close()
        self.mascot.close()
        super().closeEvent(event)  # type: ignore[arg-type]


def preview_skin(root: Path, key: str) -> int:
    """Ouvre l'aperçu et attend sa fermeture ; code de sortie (0 si tout va bien)."""
    app = QApplication.instance() or QApplication([])
    skin = load_skin(root, key)
    panel = PreviewPanel(skin)
    panel.start()
    QApplication.setQuitOnLastWindowClosed(True)
    return int(app.exec())
