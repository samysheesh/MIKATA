"""Thème par défaut de la mascotte, dessiné en code (QPainter) : aucune image (ajout phase 8).

« Mika » : une petite boule ronde avec une antenne, des yeux et des joues. Chaque état a sa
couleur (proches de celles de l'icône de la zone de notification) et son animation :
- idle : gris bleuté, yeux fermés, respire lentement, des « z » s'envolent (il dort) ;
- wake : jaune, sursaut, grands yeux, « ! » ;
- listening : bleu, regarde en l'air, ondes qui arrivent vers l'antenne ;
- thinking : orange, regard en coin, trois points tournent au-dessus de sa tête ;
- acting : orange, un engrenage tourne ;
- speaking : vert, la bouche bouge, ondes qui partent ;
- success : vert vif, yeux rieurs, coche qui apparaît (« c'est fait ») ;
- error : rouge, yeux en croix, secousse, « ! » ;
- follow_up : turquoise, anneau qui se vide pendant la fenêtre de suivi ;
- mic_off : gris, paupières mi-closes, micro barré.

Tout est dessiné dans une boîte de 100 × 100 mise à l'échelle : net à toutes les tailles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

COLORS = {
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
INK = QColor("#1f2430")


@dataclass
class Frame:
    """Ce qu'il faut pour dessiner une image : état, temps, âge de l'état, suivi restant."""

    state: str
    t: float  # secondes (horloge continue : respiration, clignements)
    age: float  # secondes depuis le début de l'état (sursaut, apparition)
    follow_up_left: float | None = None


def _round_pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    return pen


def _ease_out_back(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def paint_mascot(painter: QPainter, rect: QRectF, frame: Frame) -> None:
    """Dessine la mascotte dans `rect` (carré conseillé)."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    side = min(rect.width(), rect.height())
    painter.translate(rect.center().x() - side / 2, rect.center().y() - side / 2)
    painter.scale(side / 100, side / 100)
    _Mika(painter, frame).draw()
    painter.restore()


class _Mika:
    def __init__(self, painter: QPainter, frame: Frame) -> None:
        self.p = painter
        self.f = frame
        self.color = QColor(COLORS.get(frame.state, COLORS["idle"]))

    # -- ensemble ---------------------------------------------------------------------------

    def draw(self) -> None:
        f, t = self.f, self.f.t
        dx = dy = 0.0
        squash = 1.0
        rotation = 0.0
        if f.state == "idle":
            squash = 1 + 0.035 * math.sin(2 * math.pi * t / 3.5)
        elif f.state == "wake":
            dy = -9 * math.sin(math.pi * min(f.age / 0.45, 1.0))
        elif f.state in ("thinking", "acting"):
            rotation = 3 * math.sin(2 * math.pi * t / 2.2)
        elif f.state == "success":
            dy = -6 * abs(math.sin(math.pi * min(f.age / 0.5, 2.0)))
        elif f.state == "error":
            dx = 4 * math.sin(t * 38) * math.exp(-f.age * 2.5)
        elif f.state == "listening":
            dy = -1.5 * math.sin(2 * math.pi * t / 1.6)
        self._shadow(dy)
        self.p.save()
        self.p.translate(50 + dx, 92 + dy)
        self.p.rotate(rotation)
        self.p.scale(1 / squash, squash)
        self.p.translate(-50, -92)
        if f.state == "follow_up":
            self._follow_up_ring()
        self._antenna()
        self._body()
        self._face()
        self.p.restore()
        self._extras()

    def _shadow(self, dy: float) -> None:
        shrink = 1 + dy / 40
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(QColor(0, 0, 0, 55))
        self.p.drawEllipse(QPointF(50, 93), 24 * shrink, 3.5 * shrink)

    def _body(self) -> None:
        base = self.color
        gradient = QRadialGradient(QPointF(40, 45), 42)
        gradient.setColorAt(0.0, base.lighter(150))
        gradient.setColorAt(0.55, base)
        gradient.setColorAt(1.0, base.darker(125))
        path = QPainterPath()
        path.addEllipse(QPointF(50, 60), 33, 31)
        self.p.setPen(QPen(base.darker(170), 1.6))
        self.p.setBrush(QBrush(gradient))
        self.p.drawPath(path)
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(QColor(255, 255, 255, 70))
        self.p.drawEllipse(QPointF(41, 44), 10, 6)  # reflet
        self.p.setBrush(QColor(255, 120, 150, 70))
        self.p.drawEllipse(QPointF(31, 67), 5, 3)  # joues
        self.p.drawEllipse(QPointF(69, 67), 5, 3)

    def _antenna(self) -> None:
        state = self.f.state
        bright = state not in ("idle", "mic_off")
        self.p.setPen(_round_pen(self.color.darker(160), 2.2))
        path = QPainterPath(QPointF(50, 31))
        path.quadTo(QPointF(53, 22), QPointF(50, 16))
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        self.p.drawPath(path)
        glow = 0.5 + 0.5 * math.sin(self.f.t * 5) if bright else 0.2
        bulb = self.color.lighter(170 if bright else 110)
        halo = QColor(bulb)
        halo.setAlphaF(0.35 * glow)
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.setBrush(halo)
        self.p.drawEllipse(QPointF(50, 14), 7 + 2 * glow, 7 + 2 * glow)
        self.p.setBrush(bulb)
        self.p.setPen(QPen(self.color.darker(160), 1.2))
        self.p.drawEllipse(QPointF(50, 14), 4, 4)

    # -- visage -----------------------------------------------------------------------------

    def _face(self) -> None:
        state = self.f.state
        if state == "idle":
            self._closed_eyes()
        elif state == "success":
            self._happy_eyes()
        elif state == "error":
            self._cross_eyes()
        else:
            self._open_eyes()
        self._mouth()

    def _gaze(self) -> tuple[float, float]:
        state, t = self.f.state, self.f.t
        if state == "listening":
            return 0.0, -2.2
        if state in ("thinking", "acting"):
            return 2.2, -2.0
        if state == "speaking":
            return 0.8 * math.sin(t * 1.3), 0.0
        if state == "mic_off":
            return 0.0, 1.5
        return 0.0, 0.0

    def _blink(self) -> float:
        """1 : œil ouvert ; 0 : fermé (clignement de 140 ms toutes les ~4 s)."""
        phase = (self.f.t + 0.7) % 4.1
        return 0.1 if phase < 0.14 else 1.0

    def _open_eyes(self) -> None:
        state = self.f.state
        big = state == "wake"
        open_ = self._blink() * (0.55 if state == "mic_off" else 1.0)
        gx, gy = self._gaze()
        for x in (39, 61):
            self.p.setPen(QPen(INK, 1.2))
            self.p.setBrush(QColor("white"))
            ry = (9.5 if big else 8) * open_
            self.p.drawEllipse(QPointF(x, 56), 7 if not big else 8, max(ry, 0.8))
            if open_ > 0.3:
                self.p.setPen(Qt.PenStyle.NoPen)
                self.p.setBrush(INK)
                pupil = 2.6 if big else 3.6
                self.p.drawEllipse(QPointF(x + gx, 56 + gy * open_), pupil, pupil * open_)
                self.p.setBrush(QColor("white"))
                self.p.drawEllipse(QPointF(x + gx + 1.3, 54.5 + gy * open_), 1.1, 1.1)
            if state == "mic_off":  # paupière lourde
                self.p.setPen(QPen(INK, 1.6))
                self.p.drawLine(QPointF(x - 7, 53), QPointF(x + 7, 53))

    def _closed_eyes(self) -> None:
        self.p.setPen(_round_pen(INK, 1.8))
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        for x in (39, 61):
            self.p.drawArc(QRectF(x - 6, 52, 12, 7), 200 * 16, 140 * 16)

    def _happy_eyes(self) -> None:
        self.p.setPen(_round_pen(INK, 2.0))
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        for x in (39, 61):
            self.p.drawArc(QRectF(x - 6, 52, 12, 9), 20 * 16, 140 * 16)

    def _cross_eyes(self) -> None:
        self.p.setPen(_round_pen(INK, 2.0))
        for x in (39, 61):
            self.p.drawLine(QPointF(x - 4, 52), QPointF(x + 4, 60))
            self.p.drawLine(QPointF(x + 4, 52), QPointF(x - 4, 60))

    def _mouth(self) -> None:
        state, t = self.f.state, self.f.t
        self.p.setPen(_round_pen(INK, 1.6))
        if state == "speaking":
            opening = 1.5 + 4.5 * abs(math.sin(t * 9.0)) * (0.6 + 0.4 * math.sin(t * 2.3))
            self.p.setBrush(QColor("#7a2531"))
            self.p.drawEllipse(QPointF(50, 70), 4.2, opening)
        elif state in ("success", "wake"):
            self.p.setBrush(QColor("#7a2531"))
            path = QPainterPath(QPointF(44, 68))
            path.quadTo(QPointF(50, 77 if state == "success" else 74), QPointF(56, 68))
            path.closeSubpath()
            self.p.drawPath(path)
        elif state == "error":
            self.p.setBrush(Qt.BrushStyle.NoBrush)
            self.p.drawArc(QRectF(44, 69, 12, 7), 20 * 16, 140 * 16)
        elif state == "idle":
            self.p.setBrush(Qt.BrushStyle.NoBrush)
            self.p.drawEllipse(QPointF(50, 70), 1.8, 1.4)
        else:
            self.p.setBrush(Qt.BrushStyle.NoBrush)
            self.p.drawArc(QRectF(45, 64, 10, 7), 200 * 16, 140 * 16)

    # -- décors -----------------------------------------------------------------------------

    def _follow_up_ring(self) -> None:
        left = self.f.follow_up_left if self.f.follow_up_left is not None else 1.0
        track = QColor(self.color)
        track.setAlpha(50)
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        self.p.setPen(QPen(track, 3.5))
        self.p.drawEllipse(QPointF(50, 60), 41, 39)
        self.p.setPen(_round_pen(self.color.darker(115), 3.5))
        self.p.drawArc(QRectF(9, 21, 82, 78), 90 * 16, int(360 * 16 * left))

    def _extras(self) -> None:
        state, t, age = self.f.state, self.f.t, self.f.age
        if state == "idle":
            self._zzz(t)
        elif state == "wake":
            self._badge_text("!", QColor("#f5b301"), 80, 22, _ease_out_back(age / 0.3))
        elif state == "listening":
            self._ripples_in(t)
        elif state == "thinking":
            self._thought_dots(t)
        elif state == "acting":
            self._gear(t)
        elif state == "speaking":
            self._waves_out(t)
        elif state == "success":
            self._check(_ease_out_back(age / 0.35))
        elif state == "error":
            self._badge_text("!", QColor("#d93838"), 80, 22, _ease_out_back(age / 0.3))
        elif state == "mic_off":
            self._muted_mic()

    def _zzz(self, t: float) -> None:
        font = QFont()
        font.setBold(True)
        for i in range(3):
            phase = ((t / 2.6) + i / 3) % 1.0
            alpha = math.sin(math.pi * phase)
            color = QColor(self.color.darker(140))
            color.setAlphaF(max(alpha, 0.0))
            font.setPointSizeF(6 + 6 * phase)
            self.p.setFont(font)
            self.p.setPen(color)
            self.p.drawText(QPointF(66 + 16 * phase, 32 - 24 * phase), "z")

    def _ripples_in(self, t: float) -> None:
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(3):
            phase = ((t / 1.2) + i / 3) % 1.0
            radius = 4 + 18 * (1 - phase)
            color = QColor(self.color)
            color.setAlphaF(0.75 * phase)
            self.p.setPen(QPen(color, 1.6))
            self.p.drawArc(
                QRectF(50 - radius, 14 - radius, 2 * radius, 2 * radius), 20 * 16, 140 * 16
            )

    def _thought_dots(self, t: float) -> None:
        self.p.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            angle = t * 2.4 + i * 2 * math.pi / 3
            x, y = 50 + 16 * math.cos(angle), 9 + 4 * math.sin(angle)
            color = QColor(self.color.darker(120))
            color.setAlphaF(0.55 + 0.45 * (math.sin(angle) + 1) / 2)
            self.p.setBrush(color)
            self.p.drawEllipse(QPointF(x, y), 3, 3)

    def _gear(self, t: float) -> None:
        self.p.save()
        self.p.translate(80, 22)
        self.p.rotate(t * 120)
        self.p.setPen(QPen(self.color.darker(150), 1.2))
        self.p.setBrush(self.color.lighter(130))
        path = QPainterPath()
        for i in range(16):
            r = 8 if i % 2 == 0 else 5.5
            a = i * math.pi / 8
            point = QPointF(r * math.cos(a), r * math.sin(a))
            path.moveTo(point) if i == 0 else path.lineTo(point)
        path.closeSubpath()
        self.p.drawPath(path)
        self.p.setBrush(QColor("white"))
        self.p.drawEllipse(QPointF(0, 0), 2.4, 2.4)
        self.p.restore()

    def _waves_out(self, t: float) -> None:
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(3):
            phase = ((t / 0.9) + i / 3) % 1.0
            radius = 5 + 12 * phase
            color = QColor(self.color.darker(110))
            color.setAlphaF(0.85 * (1 - phase))
            self.p.setPen(_round_pen(color, 1.8))
            self.p.drawArc(
                QRectF(82 - radius, 64 - radius, 2 * radius, 2 * radius), -45 * 16, 90 * 16
            )

    def _check(self, scale: float) -> None:
        self.p.save()
        self.p.translate(80, 24)
        self.p.scale(scale, scale)
        self.p.setPen(QPen(QColor("white"), 1.5))
        self.p.setBrush(QColor("#16a34a"))
        self.p.drawEllipse(QPointF(0, 0), 10, 10)
        self.p.setPen(_round_pen(QColor("white"), 2.6))
        self.p.drawPolyline([QPointF(-5, 0), QPointF(-1.5, 4), QPointF(5.5, -4)])
        self.p.restore()

    def _badge_text(self, text: str, color: QColor, x: float, y: float, scale: float) -> None:
        self.p.save()
        self.p.translate(x, y)
        self.p.scale(scale, scale)
        self.p.setPen(QPen(QColor("white"), 1.5))
        self.p.setBrush(color)
        self.p.drawEllipse(QPointF(0, 0), 9, 9)
        font = QFont()
        font.setBold(True)
        font.setPointSizeF(11)
        self.p.setFont(font)
        self.p.setPen(QColor("white"))
        self.p.drawText(QRectF(-9, -9, 18, 18), Qt.AlignmentFlag.AlignCenter, text)
        self.p.restore()

    def _muted_mic(self) -> None:
        self.p.save()
        self.p.translate(80, 80)
        self.p.setPen(QPen(QColor("white"), 1.5))
        self.p.setBrush(QColor("#4b5563"))
        self.p.drawEllipse(QPointF(0, 0), 10, 10)
        self.p.setPen(_round_pen(QColor("white"), 1.6))
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        self.p.drawRoundedRect(QRectF(-2.5, -6, 5, 8), 2.5, 2.5)
        self.p.drawArc(QRectF(-4.5, -3, 9, 7), 200 * 16, 140 * 16)
        self.p.drawLine(QPointF(0, 4), QPointF(0, 6))
        self.p.setPen(_round_pen(QColor("#f87171"), 2.0))
        self.p.drawLine(QPointF(-6, -6), QPointF(6, 6))
        self.p.restore()
