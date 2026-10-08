"""Captures PNG de l'interface, sans écran (ajout phase 8).

    uv run python scripts/ui_screenshots.py                  # -> docs/ui/
    uv run python scripts/ui_screenshots.py --out data/captures --gif

Rend avec Qt hors écran (QT_QPA_PLATFORM=offscreen) :
- mascotte_etats.png : les dix états de la mascotte ;
- mascotte_bulle.png : la mascotte qui parle, avec sa bulle, sur un faux bureau ;
- mascotte_menu.png : le menu du clic droit ;
- reglages_<onglet>.png : chaque onglet des réglages (config.yaml du dépôt, données fictives :
  aucun secret réel n'est lu) ;
- avec --gif : mascotte_animation.gif (tous les états animés).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "docs" / "ui"
SETTINGS_TABS = ("audio", "mot_activation", "conversation", "cout_cles", "mascotte", "journal")
_FAKE_EVENTS = (
    ("09:12:01.204", "INFO", "MainThread", "run:352", "MIKATA démarré, en attente."),
    ("09:12:44.913", "INFO", "mikata-wake", "_on_wake:640",
     "Mot d'activation détecté (score 0.92, voix 0.88) : bip ≈ 131 ms après la fin du mot."),
    ("09:12:46.102", "SUCCESS", "MainThread", "_transcribe:930",
     "Commande #1 : « quelle heure est-il à Tokyo »"),
    ("09:12:46.988", "INFO", "MainThread", "_respond:1050",
     "Appel LLM 1 (réflexion none) : premier élément 612 ms, total 840 ms, outils : aucun."),
    ("09:12:47.415", "INFO", "MainThread", "_log_response:1001",
     "Réponse #1 : « À Tokyo, il est dix-sept heures douze. »"),
    ("09:13:02.001", "WARNING", "MainThread", "_mic_watchdog:410",
     "Micro : aucune trame depuis 2 s (occupé ?)."),
)  # fmt: skip
FAKE_LOG = "".join(
    f"2026-10-08 {t} | {level:<8} | {thread} | mikata.app:{where} - {text}\n"
    for t, level, thread, where, text in _FAKE_EVENTS
)


def _fake_base(tmp: Path) -> Path:
    """Dossier de démonstration : config.yaml du dépôt, coût, journal et .env fictifs."""
    shutil.copy2(PROJECT_ROOT / "config.yaml", tmp / "config.yaml")
    (tmp / "data" / "logs").mkdir(parents=True)
    (tmp / "data" / "logs" / "mikata.log").write_text(FAKE_LOG, encoding="utf-8")
    usage = {"cost_usd": 0.4213, "commands": 318, "since": "2026-09-30T10:02:11"}
    (tmp / "data" / "llm_usage.json").write_text(json.dumps(usage), encoding="utf-8")
    (tmp / ".env").write_text("LLM_API_KEY=demo-cle-fictive-pour-capture-9f3c\n", encoding="utf-8")
    shutil.copytree(PROJECT_ROOT / "assets" / "mascot", tmp / "assets" / "mascot")
    return tmp


def _desktop(width: int, height: int) -> object:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0, QColor("#1e3a5f"))
    gradient.setColorAt(1, QColor("#6d4c8f"))
    painter.fillRect(QRectF(0, 0, width, height), gradient)
    painter.setBrush(QColor(245, 246, 248, 235))  # une fenêtre ouverte derrière la mascotte
    painter.setPen(QColor(0, 0, 0, 40))
    painter.drawRoundedRect(QRectF(40, 60, width * 0.55, height - 120), 8, 8)
    painter.end()
    return image


def render(out: Path, gif: bool = False) -> list[Path]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from mikata.config import load_config
    from mikata.ui.model import STATE_LABELS, VISUAL_STATES, MascotModel
    from mikata.ui.prefs import MascotPrefs
    from mikata.ui.qt.mascot import BubbleWindow, MascotWindow
    from mikata.ui.qt.painter import Frame, paint_mascot
    from mikata.ui.qt.settings import SettingsWindow

    app = QApplication.instance() or QApplication([])
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. les dix états
    cell, cols = 170, 5
    grid = QImage(cols * cell, 2 * (cell + 30), QImage.Format.Format_ARGB32)
    grid.fill(QColor("#eef2f7"))
    painter = QPainter(grid)
    font = QFont()
    font.setPointSizeF(10)
    painter.setFont(font)
    for i, state in enumerate(VISUAL_STATES):
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        frame = Frame(state, t=1.3, age=0.45, follow_up_left=0.6)
        paint_mascot(painter, QRectF(x + 15, y + 10, cell - 30, cell - 30), frame)
        painter.setPen(QColor("#1f2430"))
        painter.drawText(
            QRectF(x, y + cell - 18, cell, 24), Qt.AlignmentFlag.AlignCenter, STATE_LABELS[state]
        )
    painter.end()
    written.append(out / "mascotte_etats.png")
    grid.save(str(written[-1]))

    # 2. la mascotte qui parle, avec la bulle, sur un faux bureau
    clock = [100.0]
    model = MascotModel(clock=lambda: clock[0])
    for message in (
        {"type": "hello", "state": "idle", "mic": "ok"},
        {"type": "wake", "score": 0.93},
        {"type": "heard", "id": 1, "text": "quelle heure est-il à Tokyo ?"},
        {"type": "state", "state": "speaking"},
        {"type": "say", "id": 1, "text": "À Tokyo, il est dix-sept heures douze."},
    ):
        model.handle(message)
    clock[0] += 1.5
    tmp_prefs = Path(tempfile.mkdtemp()) / "ui_state.json"
    mascot = MascotWindow(
        model, MascotPrefs(size="grande"), tmp_prefs, None, lambda m: True, lambda: None
    )
    mascot.move(560, 330)
    bubble = BubbleWindow()
    content = model.bubble()
    assert content is not None
    bubble.set_content(content.heard, content.said, content.note)
    desktop = _desktop(900, 560)
    painter = QPainter(desktop)  # type: ignore[arg-type]
    painter.drawPixmap(QPoint(mascot.x(), mascot.y()), mascot.grab())
    painter.drawPixmap(
        QPoint(
            mascot.x() + mascot.width() // 2 - bubble.width() // 2, mascot.y() - bubble.height() - 2
        ),
        bubble.grab(),
    )
    painter.end()
    written.append(out / "mascotte_bulle.png")
    desktop.save(str(written[-1]))  # type: ignore[attr-defined]

    # 3. le menu du clic droit
    menu = mascot.build_menu()
    menu.ensurePolished()
    menu.adjustSize()
    written.append(out / "mascotte_menu.png")
    menu.grab().save(str(written[-1]))

    # 4. les réglages, onglet par onglet
    with tempfile.TemporaryDirectory() as tmp:
        base = _fake_base(Path(tmp))
        config = load_config(base / "config.yaml", env_file=base / ".env")
        devices = (
            ["Microphone (RØDE NT-USB+)", "Microphone (Realtek(R) Audio)"],
            ["Casque (AirPods)", "Haut-parleurs (Realtek(R) Audio)"],
            None,
        )
        window = SettingsWindow(
            base / "config.yaml", config.base_dir, lambda m: True, devices=lambda: devices
        )
        window.resize(700, 560)
        window.status.setText(
            "Enregistré dans config.yaml. Tout de suite : Seuil du mot d'activation."
        )
        for index, name in enumerate(SETTINGS_TABS):
            window.tabs.setCurrentIndex(index)
            app.processEvents()
            written.append(out / f"reglages_{name}.png")
            window.grab().save(str(written[-1]))
        window.close()

    if gif:
        written.append(_render_gif(out / "mascotte_animation.gif"))
    return written


def _render_gif(path: Path, size: int = 128, fps: int = 12) -> Path:
    from PIL import Image
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter

    from mikata.ui.model import VISUAL_STATES
    from mikata.ui.qt.painter import Frame, paint_mascot

    frames = []
    for state in VISUAL_STATES:
        for i in range(int(1.4 * fps)):
            t = i / fps
            image = QImage(size, size, QImage.Format.Format_RGB32)
            image.fill(QColor("#eef2f7"))
            painter = QPainter(image)
            left = 1 - i / (1.4 * fps) if state == "follow_up" else None
            paint_mascot(painter, QRectF(0, 0, size, size), Frame(state, t, t, left))
            painter.end()
            raw = bytes(image.constBits())  # BGRA
            frames.append(
                Image.frombuffer("RGBA", (size, size), raw, "raw", "BGRA", 0, 1).convert("P")
            )
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gif", action="store_true", help="aussi un GIF animé des états")
    args = parser.parse_args(argv)
    for path in render(args.out, args.gif):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
