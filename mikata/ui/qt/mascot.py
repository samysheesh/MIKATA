"""Mascotte de bureau (ajout phase 8) : petite fenêtre sans bordure, transparente, toujours
au-dessus, qui ne prend jamais le focus (le jeu ou la fenêtre active le garde).

- Glisser avec le bouton gauche, sur n'importe quel écran : position et écran mémorisés
  (data/ui_state.json) ; un écran débranché ramène la mascotte sur l'écran principal.
- Clic droit : taille, mode traversable (les clics passent à travers), bulle on/off, pause
  de l'écoute, réglages, masquer. En mode traversable la fenêtre ne reçoit plus de clic :
  on le désactive depuis l'icône MIKATA (« Afficher la mascotte ») ou les réglages.
- Double-clic : réglages.
- La bulle est une seconde fenêtre (toujours traversable) placée au-dessus de la mascotte,
  ou en dessous près du haut de l'écran.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QMenu, QWidget

from mikata.ui.model import MascotModel
from mikata.ui.prefs import SIZES, MascotPrefs, save_prefs
from mikata.ui.qt.painter import Frame, paint_mascot
from mikata.ui.themes import Theme

FAST_MS = 33  # ~30 images/s pendant une animation vive
SLOW_MS = 80  # repos et micro coupé : animations lentes, moins de CPU
MARGIN = 24
BUBBLE_WIDTH = 300
CLICK_THROUGH_NOTE = (
    "Mode traversable : les clics passent à travers. Pour le quitter : icône MIKATA > "
    "Afficher la mascotte, ou les réglages."
)

_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool  # pas de bouton dans la barre des tâches
    | Qt.WindowType.WindowDoesNotAcceptFocus
    | Qt.WindowType.NoDropShadowWindowHint
)


def pil_to_pixmap(image: Any) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())


class ThemePainter:
    """Dessine un état : thème par défaut en code, ou images d'un thème (mises en cache)."""

    def __init__(self, theme: Theme | None) -> None:
        self.theme = theme
        self._cache: dict[tuple[int, int], QPixmap] = {}

    def paint(self, painter: QPainter, rect: QRectF, frame: Frame) -> None:
        animation = None
        if self.theme is not None and not self.theme.builtin:
            animation = self.theme.animation(frame.state)
        if animation is None:
            paint_mascot(painter, rect, frame)
            return
        image = animation.frame_at(frame.age)
        key = (id(image), int(rect.width()))
        pixmap = self._cache.get(key)
        if pixmap is None:
            pixmap = pil_to_pixmap(image).scaled(
                int(rect.width()),
                int(rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._cache[key] = pixmap
        x = rect.x() + (rect.width() - pixmap.width()) / 2
        y = rect.y() + (rect.height() - pixmap.height()) / 2
        painter.drawPixmap(int(x), int(y), pixmap)


class BubbleWindow(QWidget):
    """Bulle de texte : ce que MIKATA a compris (« Tu ») et ce qu'il dit."""

    def __init__(self) -> None:
        super().__init__(None, _FLAGS | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._lines: list[tuple[str, str]] = []  # (style, texte)
        self.tail_down = True  # pointe vers le bas (bulle au-dessus de la mascotte)
        self.font_main = QFont()
        self.font_main.setPointSizeF(10)

    def set_content(self, heard: str, said: str, note: str) -> bool:
        lines = []
        if note:
            lines.append(("note", note))
        if heard:
            lines.append(("heard", f"Tu : {heard}"))
        if said:
            lines.append(("said", said))
        changed = lines != self._lines
        self._lines = lines
        if changed:
            self._layout()
        return bool(lines)

    def _layout(self) -> None:
        metrics = QFontMetrics(self.font_main)
        height = 16
        for _, text in self._lines:
            box = metrics.boundingRect(
                QRect(0, 0, BUBBLE_WIDTH - 24, 2000), Qt.TextFlag.TextWordWrap, text
            )
            height += min(box.height(), metrics.lineSpacing() * 6) + 4
        self.resize(BUBBLE_WIDTH, height + 12)
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(1, 1 if self.tail_down else 11, self.width() - 2, self.height() - 12)
        path = QPainterPath()
        path.addRoundedRect(body, 12, 12)
        tail = QPainterPath()
        cx = self.width() / 2
        if self.tail_down:
            tail.moveTo(cx - 9, body.bottom() - 1)
            tail.lineTo(cx, self.height() - 1)
            tail.lineTo(cx + 9, body.bottom() - 1)
        else:
            tail.moveTo(cx - 9, body.top() + 1)
            tail.lineTo(cx, 1)
            tail.lineTo(cx + 9, body.top() + 1)
        path = path.united(tail)
        painter.setPen(QPen(QColor(31, 36, 48, 90), 1.2))
        painter.setBrush(QColor(255, 255, 255, 242))
        painter.drawPath(path)
        y = body.top() + 8
        for style, text in self._lines:
            font = QFont(self.font_main)
            color = QColor("#1f2430")
            if style == "heard":
                font.setItalic(True)
                color = QColor("#5b6475")
            elif style == "note":
                color = QColor("#b45309")
            painter.setFont(font)
            painter.setPen(color)
            metrics = QFontMetrics(font)
            area = QRect(12, int(y), BUBBLE_WIDTH - 24, metrics.lineSpacing() * 6)
            box = metrics.boundingRect(area, Qt.TextFlag.TextWordWrap, text)
            painter.drawText(area, Qt.TextFlag.TextWordWrap, text)
            y += min(box.height(), area.height()) + 4
        painter.end()


class MascotWindow(QWidget):
    def __init__(
        self,
        model: MascotModel,
        prefs: MascotPrefs,
        prefs_path: Path,
        theme: Theme | None,
        send: Callable[[dict[str, Any]], bool],
        open_settings: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("MIKATA")
        self.model = model
        self.prefs = prefs
        self.prefs_path = prefs_path
        self.painter = ThemePainter(theme)
        self._send = send
        self._open_settings = open_settings
        self._clock = clock
        self._start = clock()
        self._drag: QPoint | None = None
        self._moved = False
        self.bubble = BubbleWindow()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self.apply_size()
        self._apply_click_through(prefs.click_through, show=False)
        self.restore_position()

    # -- préférences ------------------------------------------------------------------------

    def set_theme(self, theme: Theme | None) -> None:
        self.painter = ThemePainter(theme)
        self.update()

    def apply_size(self) -> None:
        side = self.prefs.pixels
        self.setFixedSize(side, side)

    def set_size(self, name: str) -> None:
        if name not in SIZES:
            return
        bottom_center = self.geometry().center().x(), self.geometry().bottom()
        self.prefs.size = name
        self.apply_size()
        self.move(bottom_center[0] - self.width() // 2, bottom_center[1] - self.height() + 1)
        self.clamp_to_screen()
        self.save()

    def set_click_through(self, enabled: bool) -> None:
        self.prefs.click_through = enabled
        self._apply_click_through(enabled, show=not self.prefs.hidden)
        if enabled:
            self.model.note(CLICK_THROUGH_NOTE)
        self.save()

    def _apply_click_through(self, enabled: bool, show: bool) -> None:
        # Changer ce drapeau recrée la fenêtre native (WS_EX_TRANSPARENT sous Windows) : il
        # faut la réafficher.
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        if show:
            self.show()

    def set_bubble(self, enabled: bool) -> None:
        self.prefs.bubble = enabled
        if not enabled:
            self.bubble.hide()
        self.save()

    def set_hidden(self, hidden: bool) -> None:
        self.prefs.hidden = hidden
        if hidden:
            self.hide()
            self.bubble.hide()
            self._timer.stop()
        else:
            self.show()
            self.raise_()
            self._timer.start(FAST_MS)
        self.save()

    def reveal(self) -> None:
        """« Afficher la mascotte » (icône MIKATA, réglages) : visible et cliquable."""
        if self.prefs.click_through:
            self._apply_click_through(False, show=False)
            self.prefs.click_through = False
        self.set_hidden(False)
        self.clamp_to_screen()

    def save(self) -> None:
        screen = self.screen()
        self.prefs.screen = screen.name() if screen is not None else ""
        self.prefs.x, self.prefs.y = self.x(), self.y()
        try:
            save_prefs(self.prefs_path, self.prefs)
        except OSError as exc:
            logger.warning("Préférences de la mascotte non enregistrées : {}", exc)

    # -- position ---------------------------------------------------------------------------

    def restore_position(self) -> None:
        screens = QGuiApplication.screens()
        target = next((s for s in screens if s.name() == self.prefs.screen), None)
        if target is None or self.prefs.x is None or self.prefs.y is None:
            target = QGuiApplication.primaryScreen()
            area = target.availableGeometry()
            self.move(area.right() - self.width() - MARGIN, area.bottom() - self.height() - MARGIN)
            return
        self.move(self.prefs.x, self.prefs.y)
        self.clamp_to_screen(target.availableGeometry())

    def clamp_to_screen(self, area: QRect | None = None) -> None:
        if area is None:
            screen = QGuiApplication.screenAt(self.geometry().center()) or self.screen()
            area = screen.availableGeometry()
        x = min(max(self.x(), area.left()), area.right() - self.width() + 1)
        y = min(max(self.y(), area.top()), area.bottom() - self.height() + 1)
        self.move(x, y)

    # -- animation --------------------------------------------------------------------------

    def start(self) -> None:
        if not self.prefs.hidden:
            self.show()
            self._timer.start(FAST_MS)

    def tick(self) -> None:
        state = self.model.visual()
        interval = SLOW_MS if state in ("idle", "mic_off") else FAST_MS
        if self._timer.interval() != interval:
            self._timer.setInterval(interval)
        self.update()
        self._update_bubble()

    def current_frame(self) -> Frame:
        return Frame(
            state=self.model.visual(),
            t=self._clock() - self._start,
            age=self.model.visual_age(),
            follow_up_left=self.model.follow_up_left(),
        )

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        self.painter.paint(painter, QRectF(self.rect()), self.current_frame())
        painter.end()

    def _update_bubble(self) -> None:
        content = self.model.bubble() if self.prefs.bubble and self.isVisible() else None
        if content is None or not self.bubble.set_content(
            content.heard, content.said, content.note
        ):
            self.bubble.hide()
            return
        self.place_bubble()
        if not self.bubble.isVisible():
            self.bubble.show()

    def place_bubble(self) -> None:
        screen = QGuiApplication.screenAt(self.geometry().center()) or self.screen()
        area = screen.availableGeometry()
        x = self.geometry().center().x() - self.bubble.width() // 2
        x = min(max(x, area.left()), area.right() - self.bubble.width() + 1)
        above = self.y() - self.bubble.height() - 2
        self.bubble.tail_down = above >= area.top()
        y = above if self.bubble.tail_down else self.geometry().bottom() + 2
        self.bubble.move(x, y)

    # -- souris -----------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._moved = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)
            self._moved = True
            if self.bubble.isVisible():
                self.place_bubble()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            self._drag = None
            if self._moved:
                self.clamp_to_screen()
                self.save()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_settings()

    def contextMenuEvent(self, event: Any) -> None:
        self.build_menu().popup(event.globalPos())

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        sizes = menu.addMenu("Taille")
        group = QActionGroup(sizes)
        for name, pixels in SIZES.items():
            action = QAction(f"{name.capitalize()} ({pixels} px)", sizes, checkable=True)
            action.setChecked(self.prefs.size == name)
            action.triggered.connect(lambda _=False, n=name: self.set_size(n))
            group.addAction(action)
            sizes.addAction(action)
        through = QAction("Mode traversable (les clics passent à travers)", menu, checkable=True)
        through.setChecked(self.prefs.click_through)
        through.triggered.connect(self.set_click_through)
        menu.addAction(through)
        bubble = QAction("Bulle de texte", menu, checkable=True)
        bubble.setChecked(self.prefs.bubble)
        bubble.triggered.connect(self.set_bubble)
        menu.addAction(bubble)
        menu.addSeparator()
        pause = QAction("Mettre en pause l'écoute", menu, checkable=True)
        pause.setChecked(self.model.mic == "paused")
        pause.setEnabled(self.model.connected)
        pause.triggered.connect(lambda checked: self._send({"type": "pause", "paused": checked}))
        menu.addAction(pause)
        settings = QAction("Réglages…", menu)
        settings.triggered.connect(self._open_settings)
        menu.addAction(settings)
        menu.addSeparator()
        hide = QAction("Masquer", menu)
        hide.triggered.connect(lambda: self.set_hidden(True))
        menu.addAction(hide)
        return menu
