"""Fenêtre de réglages (ajout phase 8, refaite phase 15b), ouverte depuis l'icône MIKATA, la
mascotte (double-clic ou clic droit) ou `python -m mikata.ui --settings`.

Une barre latérale et six pages faites de cartes :
- **Accueil** : le navigateur de travail (onglets fermables un par un, pause, afficher ou
  masquer la fenêtre ; il reste lancé en fond), les dépenses, la mascotte ;
- **Son et voix** : micro, sortie (changée sans redémarrer), tests, mot d'activation ;
- **Comportement** : réflexion, fenêtre de suivi, historique, accès complet ;
- **Apparence** : mascotte et personnage (`skins_tab`) ;
- **Clés et coût** : coût cumulé, crédit, voix ElevenLabs, clés masquées ;
- **Journal** : log en direct.

Seuls les réglages pris en compte au redémarrage portent un badge ; les autres s'appliquent
dès l'enregistrement. Enregistrer (bouton ou Ctrl+S, actif seulement s'il y a des
modifications) réécrit config.yaml en gardant commentaires et ordre
(`settings_store.ConfigEditor`, sauvegarde dans data/backups), puis demande à MIKATA de le
relire (`reload_config`). Les clés restent dans .env et ne sont jamais affichées en clair.
"""

from __future__ import annotations

import queue
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mikata.ui.qt.skins_tab import SkinsTab, render_preview
from mikata.ui.settings_store import (
    BACKUP_DIR,
    SECRET_NAMES,
    SPECS,
    USAGE_FILE,
    VOICE_USAGE_FILE,
    ConfigEditor,
    EditorError,
    LogTail,
    list_audio_devices,
    read_secrets,
    read_usage,
    voice_usage_text,
    write_secret,
)

REASONING = {
    "none": "Rapide (sans réflexion)",
    "low": "Légère",
    "medium": "Moyenne",
    "high": "Approfondie (plus lent)",
}
NARRATION = {
    "bavard": "Bavard : dit ce qu'il fait",
    "discret": "Discret : seulement le résultat",
}
HISTORY_PRESETS = {
    1: "1 heure",
    12: "12 heures",
    24: "1 jour",
    72: "3 jours",
    168: "1 semaine",
    720: "1 mois",
}
LOG_LEVELS = {"Tout": 0, "Infos et plus": 1, "Avertissements et erreurs": 2}
_LEVEL_RANK = {"TRACE": 0, "DEBUG": 0, "INFO": 1, "SUCCESS": 1, "WARNING": 2, "ERROR": 2}
MAX_LOG_LINES = 2000
COST_REFRESH_POLLS = 10  # coûts relus toutes les 5 s (relevé toutes les 500 ms)
TASKS_REFRESH_POLLS = 6  # accueil affiché : onglets du navigateur relus toutes les 3 s
TAB_TITLE_MAX = 70
PAGES = ("Accueil", "Son et voix", "Comportement", "Apparence", "Clés et coût", "Journal")
UNSAVED = "Modifications non enregistrées."

ACCENT = "#6d5dfc"
ICONS = (Path(__file__).resolve().parent / "icons").as_posix()
STYLE = f"""
QWidget {{ font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; font-size: 10pt;
          color: #1d1f2b; }}
QWidget#content, QScrollArea, QWidget#page {{ background: #f4f5f9; }}
QScrollArea {{ border: none; }}
QWidget#sidebar {{ background: #191a24; }}
QLabel#brand {{ color: #ffffff; font-size: 15pt; font-weight: 600; }}
QLabel#brand_sub {{ color: #8b8fa3; font-size: 9pt; }}
QListWidget#nav {{ background: transparent; border: none; outline: none; color: #c9cbd6;
                  font-size: 10.5pt; }}
QListWidget#nav::item {{ padding: 7px 14px; margin: 1px 10px; border-radius: 8px; }}
QListWidget#nav::item:hover:!selected {{ background: #262836; }}
QListWidget#nav::item:selected {{ background: {ACCENT}; color: #ffffff; }}
QLabel#page_title {{ font-size: 18pt; font-weight: 600; }}
QLabel#page_sub, QLabel#hint {{ color: #6b7083; }}
QFrame#card {{ background: #ffffff; border: 1px solid #e3e5ee; border-radius: 12px; }}
QLabel#card_title {{ font-size: 11.5pt; font-weight: 600; }}
QLabel#row_label {{ font-weight: 600; }}
QLabel#row_help {{ color: #6b7083; font-size: 9pt; }}
QFrame#separator {{ background: #eceef4; border: none; }}
QLabel#badge_restart {{ background: #fff1e6; color: #a1460f; border-radius: 9px;
                       padding: 1px 9px; font-size: 8.5pt; }}
QLabel#badge_live {{ background: #e5f8ec; color: #17693a; border-radius: 9px;
                    padding: 1px 9px; font-size: 8.5pt; }}
QLabel#badge_idle {{ background: #eef0f5; color: #555a6e; border-radius: 9px;
                    padding: 1px 9px; font-size: 8.5pt; }}
QLabel#big_number {{ font-size: 16pt; font-weight: 600; }}
QLabel#status {{ color: #3a3e50; }}
QWidget#footer {{ background: #ffffff; border-top: 1px solid #e3e5ee; }}
QPushButton {{ background: #ffffff; border: 1px solid #d6d9e4; border-radius: 8px;
              padding: 6px 14px; }}
QPushButton:hover {{ background: #f2f3f8; }}
QPushButton:pressed {{ background: #e7e9f1; }}
QPushButton:disabled {{ color: #a3a7b7; background: #f6f7fa; }}
QPushButton:checked {{ background: #ece9ff; border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: #ffffff; border: none; padding: 7px 20px;
                      font-weight: 600; }}
QPushButton#primary:hover {{ background: #5b4bea; }}
QPushButton#primary:disabled {{ background: #c9c4f5; color: #ffffff; }}
QPushButton#ghost {{ border: none; background: transparent; color: #5b4bea; padding: 4px 8px; }}
QPushButton#ghost:hover {{ background: #f0eeff; }}
QComboBox, QDoubleSpinBox, QLineEdit {{ background: #ffffff; border: 1px solid #d6d9e4;
    border-radius: 8px; padding: 5px 8px; min-height: 20px; }}
QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: url({ICONS}/chevron-down.svg); width: 12px; height: 12px; }}
QDoubleSpinBox {{ padding-right: 22px; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ border: none; width: 22px;
    background: transparent; }}
QDoubleSpinBox::up-arrow {{ image: url({ICONS}/chevron-up.svg); width: 10px; height: 10px; }}
QDoubleSpinBox::down-arrow {{ image: url({ICONS}/chevron-down.svg); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: #ffffff; selection-background-color: #ece9ff;
                              selection-color: #1d1f2b; border: 1px solid #d6d9e4; }}
QProgressBar {{ background: #eceef4; border: none; border-radius: 4px; max-height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QPlainTextEdit {{ background: #14151d; color: #d7d9e4; border: none; border-radius: 10px;
                 padding: 8px; font-family: "Cascadia Mono", Consolas, monospace;
                 font-size: 9pt; }}
QListWidget {{ background: #ffffff; border: 1px solid #e3e5ee; border-radius: 10px; }}
"""


class Toggle(QCheckBox):
    """Interrupteur : une case à cocher dessinée comme un bouton glissant."""

    def __init__(self, checked: bool = False) -> None:
        super().__init__()
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(44, 24)

    def hitButton(self, pos: Any) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        track = QRectF(1, 2, 42, 20)
        painter.setBrush(QColor(ACCENT if self.isChecked() else "#cfd2dd"))
        painter.drawRoundedRect(track, 10, 10)
        painter.setBrush(QColor("#ffffff"))
        x = track.right() - 18 if self.isChecked() else track.left() + 2
        painter.drawEllipse(QRectF(x, track.top() + 2, 16, 16))
        painter.end()


def badge(live: bool) -> QLabel:
    label = QLabel("tout de suite" if live else "au redémarrage")
    label.setObjectName("badge_live" if live else "badge_restart")
    label.setToolTip(
        "MIKATA applique ce réglage sans redémarrer."
        if live
        else "Pris en compte au prochain démarrage de MIKATA."
    )
    return label


class AudioTester:
    """Tests audio de la page Son et voix (sounddevice, importé à la demande)."""

    def __init__(self) -> None:
        self._stream: Any = None
        self._levels: queue.Queue[float] = queue.Queue()

    def _device(self, name: str, kind: str) -> int | None:
        import sounddevice as sd

        from mikata.audio.devices import collect_devices, match_devices

        if not name or name == "default":
            return None
        matches = match_devices(collect_devices(sd), name, kind)  # type: ignore[arg-type]
        return matches[0].index if matches else None

    def start_mic(self, name: str) -> None:
        import sounddevice as sd

        self.stop_mic()
        levels = self._levels

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            levels.put_nowait(float(np.abs(indata).max()))  # rien d'autre dans le callback

        self._stream = sd.InputStream(
            device=self._device(name, "input"), channels=1, callback=callback
        )
        self._stream.start()

    def level(self) -> float | None:
        """Crête depuis le dernier appel (0 à 1), None si rien de nouveau."""
        peak = None
        while not self._levels.empty():
            value = self._levels.get_nowait()
            peak = value if peak is None else max(peak, value)
        return peak

    def stop_mic(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def play_output(self, name: str) -> None:
        import sounddevice as sd

        from mikata.audio.earcons import synthesize

        rate = 48_000
        sd.play(synthesize("timer", rate) * 0.5, rate, device=self._device(name, "output"))


class SettingsWindow(QWidget):
    def __init__(
        self,
        config_path: Path,
        base_dir: Path,
        send: Callable[[dict[str, Any]], bool],
        on_applied: Callable[[dict[str, Any]], None] = lambda values: None,
        on_show_mascot: Callable[[], None] = lambda: None,
        devices: Callable[[], tuple[list[str], list[str], str | None]] = list_audio_devices,
        tester: AudioTester | None = None,
    ) -> None:
        super().__init__()
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "setColorScheme"):  # fenêtre dessinée en clair, même en mode sombre
            hints.setColorScheme(Qt.ColorScheme.Light)
        self.setWindowTitle("MIKATA — Réglages")
        self.setStyleSheet(STYLE)
        self.resize(960, 680)
        self.setMinimumSize(780, 540)
        self.config_path = config_path
        self.base_dir = base_dir
        self._send = send
        self._on_applied = on_applied
        self._on_show_mascot = on_show_mascot
        self._devices = devices
        self._tester = tester or AudioTester()
        self.editor = ConfigEditor(config_path)
        self.fields: dict[str, Any] = {}
        self.tail = LogTail(self._log_path())
        self._log_lines: list[str] = []
        self._polls = 0
        self._browser: dict[str, Any] = {}

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pages = QStackedWidget()
        self.skins_tab = SkinsTab(
            base_dir / "assets" / "mascot",
            config_path,
            on_applied=self._skin_applied,
            on_status=lambda text: self.status.setText(text),
        )
        self._add_page(PAGES[0], "Ce que MIKATA fait en ce moment.", self._home_page())
        self._add_page(PAGES[1], "Micro, casque et mot d'activation.", self._sound_page())
        self._add_page(PAGES[2], "Comment MIKATA réfléchit et retient.", self._behavior_page())
        self._add_page(PAGES[3], "La mascotte et son personnage.", self._look_page())
        self._add_page(PAGES[4], "Dépenses et clés d'accès.", self._keys_page())
        self._add_page(PAGES[5], "Ce que MIKATA écrit dans son journal.", self._log_page())
        self.nav.currentRowChanged.connect(self._page_changed)

        self.save_button = QPushButton("Enregistrer")
        self.save_button.setObjectName("primary")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Ctrl+S")
        self.save_button.clicked.connect(self.save)
        shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        shortcut.activated.connect(self.save)
        self.restart_button = QPushButton("Redémarrer MIKATA")
        self.restart_button.setToolTip(
            "Arrête MIKATA proprement puis le relance (modèles et réglages rechargés)."
        )
        self.restart_button.clicked.connect(self.restart_mikata)
        footer = QWidget()
        footer.setObjectName("footer")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        buttons = QHBoxLayout(footer)
        buttons.setContentsMargins(24, 12, 24, 12)
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.restart_button)
        buttons.addWidget(self.save_button)

        content = QWidget()
        content.setObjectName("content")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        right = QVBoxLayout(content)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self.pages, 1)
        right.addWidget(footer)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._sidebar())
        layout.addWidget(content, 1)
        self.nav.setCurrentRow(0)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(500)
        self._poll()

    # -- charpente --------------------------------------------------------------------------

    def _sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        side.setFixedWidth(210)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(0, 22, 0, 16)
        self.avatar = QLabel()
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refresh_avatar()
        brand = QLabel("MIKATA")
        brand.setObjectName("brand")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Réglages")
        sub.setObjectName("brand_sub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.avatar)
        layout.addWidget(brand)
        layout.addWidget(sub)
        layout.addSpacing(18)
        layout.addWidget(self.nav, 1)
        return side

    def _refresh_avatar(self) -> None:
        skin = self.skins_tab.skins.get(self.skins_tab.current_key())
        self.avatar.setPixmap(render_preview(skin, 72))

    def _add_page(self, title: str, subtitle: str, body: QWidget) -> None:
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        heading = QLabel(title)
        heading.setObjectName("page_title")
        sub = QLabel(subtitle)
        sub.setObjectName("page_sub")
        layout.addWidget(heading)
        layout.addWidget(sub)
        layout.addSpacing(4)
        layout.addWidget(body, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        QListWidgetItem(title, self.nav)

    def page_names(self) -> list[str]:
        return [self.nav.item(i).text() for i in range(self.nav.count())]

    def show_page(self, name: str) -> None:
        self.nav.setCurrentRow(self.page_names().index(name))

    def _page_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self._refresh_tasks()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._refresh_tasks()

    def _card(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("card_title")
        layout.addWidget(heading)
        if subtitle:
            layout.addWidget(self._hint(subtitle))
        return card, layout

    @staticmethod
    def _stack(*cards: QWidget) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        for card in cards:
            layout.addWidget(card)
        layout.addStretch(1)
        return body

    def _setting(self, layout: QVBoxLayout, key: str, widget: QWidget, help_text: str) -> None:
        """Une ligne de réglage : libellé et explication à gauche, contrôle à droite ; un
        badge seulement si le réglage attend le redémarrage."""
        spec = SPECS[key]
        self.fields[key] = widget
        rows = int(layout.property("rows") or 0)
        layout.setProperty("rows", rows + 1)
        if rows:
            separator = QFrame()
            separator.setObjectName("separator")
            separator.setFixedHeight(1)
            layout.addWidget(separator)
        text = QVBoxLayout()
        text.setSpacing(2)
        label = QLabel(spec.label)
        label.setObjectName("row_label")
        text.addWidget(label)
        explanation = QLabel(help_text)
        explanation.setObjectName("row_help")
        explanation.setWordWrap(True)
        text.addWidget(explanation)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addLayout(text, 1)
        if not spec.live:
            row.addWidget(badge(False))
        row.addWidget(widget)
        layout.addLayout(row)
        for signal in ("valueChanged", "toggled", "currentIndexChanged", "editTextChanged"):
            if hasattr(widget, signal):
                getattr(widget, signal).connect(self._edited)

    def _spin(
        self, key: str, low: float, high: float, step: float, decimals: int, suffix: str = ""
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(float(self.editor.get(key) or 0))
        spin.setSuffix(suffix)
        spin.setMinimumWidth(110)
        return spin

    def _choice(self, key: str, choices: dict[Any, str]) -> QComboBox:
        combo = QComboBox()
        current = self.editor.get(key)
        if current is not None and current not in choices:  # valeur réglée à la main
            choices = {**choices, current: str(current)}
        for value, text in choices.items():
            combo.addItem(text, value)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.setMinimumWidth(220)
        return combo

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    # -- pages -------------------------------------------------------------------------------

    def _home_page(self) -> QWidget:
        card, layout = self._card(
            "Navigateur de travail",
            "MIKATA s'en sert pour YouTube, les recherches et les sites. Il reste lancé en fond, "
            "invisible ; affiche-le pour voir ce qu'il fait ou te connecter à un compte.",
        )
        head = QHBoxLayout()
        self.browser_state = QLabel("…")
        self.browser_state.setObjectName("badge_idle")
        head.addWidget(self.browser_state)
        head.addStretch(1)
        self.browser_start = QPushButton("Démarrer")
        self.browser_start.clicked.connect(lambda: self._browser_command("start"))
        self.browser_pause = QPushButton("Pause")
        self.browser_pause.setToolTip("Met en pause les vidéos et musiques de tous les onglets.")
        self.browser_pause.clicked.connect(lambda: self._browser_command("pause"))
        self.browser_toggle = QPushButton("Afficher")
        self.browser_toggle.clicked.connect(self._toggle_browser)
        for button in (self.browser_start, self.browser_pause, self.browser_toggle):
            head.addWidget(button)
        layout.addLayout(head)
        self.browser_empty = self._hint("Aucun onglet ouvert.")
        layout.addWidget(self.browser_empty)
        self.browser_tabs = QVBoxLayout()
        self.browser_tabs.setSpacing(2)
        layout.addLayout(self.browser_tabs)

        spend, spend_layout = self._card("Dépenses")
        self.home_cost = QLabel("")
        self.home_cost.setObjectName("big_number")
        self.home_cost_detail = self._hint("")
        spend_layout.addWidget(self.home_cost)
        spend_layout.addWidget(self.home_cost_detail)

        mascot, mascot_layout = self._card("Mascotte")
        line = QHBoxLayout()
        line.addWidget(self._hint("Cachée, ou perdue hors de l'écran ?"), 1)
        show = QPushButton("Afficher la mascotte")
        show.clicked.connect(self._on_show_mascot)
        line.addWidget(show)
        mascot_layout.addLayout(line)
        self._apply_browser_state()
        return self._stack(card, spend, mascot)

    def _sound_page(self) -> QWidget:
        inputs, outputs, error = self._devices()
        mic, mic_layout = self._card("Écoute")
        self._setting(
            mic_layout,
            "audio.input_device",
            self._device_combo("audio.input_device", inputs),
            "Le micro qu'écoute MIKATA.",
        )
        self.mic_level = QProgressBar()
        self.mic_level.setRange(0, 100)
        self.mic_level.setTextVisible(False)
        self.mic_button = QPushButton("Tester le micro")
        self.mic_button.setCheckable(True)
        self.mic_button.toggled.connect(self._toggle_mic_test)
        tests = QHBoxLayout()
        tests.addWidget(self.mic_button)
        tests.addWidget(self.mic_level, 1)
        mic_layout.addLayout(tests)
        mic_layout.addWidget(
            self._hint(
                error or "Le test ouvre le micro en mode partagé : MIKATA continue d'écouter."
            )
        )

        out, out_layout = self._card("Voix de MIKATA")
        self._setting(
            out_layout,
            "audio.output_device",
            self._device_combo("audio.output_device", outputs),
            "Où MIKATA parle. Change sans redémarrer ; « default » suit Windows.",
        )
        out_button = QPushButton("Tester la sortie")
        out_button.clicked.connect(self._test_output)
        line = QHBoxLayout()
        line.addWidget(self._hint("Joue le son du minuteur sur la sortie choisie."), 1)
        line.addWidget(out_button)
        out_layout.addLayout(line)

        wake, wake_layout = self._card(
            "Mot d'activation",
            "Plus haut : moins de faux déclenchements, mais plus de ratés. Valeurs conseillées : "
            "scripts/eval_wakeword.py.",
        )
        self._setting(
            wake_layout,
            "wake.threshold",
            self._spin("wake.threshold", 0, 1, 0.01, 2),
            f"Sensibilité à « MIKATA » (modèle {self.editor.get('wake.model')}).",
        )
        self._setting(
            wake_layout,
            "wake.verifier_threshold",
            self._spin("wake.verifier_threshold", 0, 1, 0.01, 2),
            "Ressemblance demandée avec ta voix enregistrée "
            f"({self.editor.get('wake.verifier') or 'aucun vérificateur'}).",
        )
        return self._stack(mic, out, wake)

    def _device_combo(self, key: str, names: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # sous-chaîne du nom acceptée, comme dans config.yaml
        combo.addItems(names)
        combo.setCurrentText(str(self.editor.get(key)))
        combo.setMinimumWidth(300)
        return combo

    def _behavior_page(self) -> QWidget:
        answers, answers_layout = self._card("Réponses")
        self._setting(
            answers_layout,
            "llm.reasoning_effort",
            self._choice("llm.reasoning_effort", REASONING),
            "« Réfléchis bien » à la voix pousse au maximum pour une seule demande.",
        )
        self._setting(
            answers_layout,
            "agent.narration",
            self._choice("agent.narration", NARRATION),
            "Sur une demande qui dure (YouTube, recherche, fichiers), MIKATA dit ce qu'il fait "
            "et ce qu'il trouve, au lieu d'attendre en silence.",
        )
        self._setting(
            answers_layout,
            "agent.follow_up_s",
            self._spin("agent.follow_up_s", 0, 30, 1, 0, " s"),
            "Après une réponse, MIKATA écoute encore sans « MIKATA » (0 : jamais).",
        )
        memory, memory_layout = self._card("Mémoire")
        self._setting(
            memory_layout,
            "agent.history_persist_h",
            self._choice("agent.history_persist_h", HISTORY_PRESETS),
            "Combien de temps MIKATA se souvient de la conversation. « Nouvelle "
            "conversation » à la voix efface tout.",
        )
        safety, safety_layout = self._card("Sécurité")
        self._setting(
            safety_layout,
            "guard.full_access",
            Toggle(bool(self.editor.get("guard.full_access"))),
            "Plus aucune confirmation après une page web ou un fichier lu. Les suppressions "
            "vont toujours à la corbeille.",
        )
        return self._stack(answers, memory, safety)

    def _look_page(self) -> QWidget:
        mascot, layout = self._card("Mascotte")
        self._setting(
            layout,
            "ui.bubble_s",
            self._spin("ui.bubble_s", 1, 120, 1, 0, " s"),
            "Après la dernière phrase. Taille de la mascotte : clic droit dessus.",
        )
        self._setting(
            layout,
            "ui.mascot",
            Toggle(bool(self.editor.get("ui.mascot"))),
            "Sinon, seule l'icône près de l'horloge.",
        )
        skins, skins_layout = self._card("Personnage")
        self.skins_tab.layout().setContentsMargins(0, 0, 0, 0)
        self.skins_tab.setMinimumHeight(340)
        skins_layout.addWidget(self.skins_tab)
        return self._stack(mascot, skins)

    def _keys_page(self) -> QWidget:
        llm, llm_layout = self._card("Modèle de langage")
        self.cost_label = QLabel("")
        self.cost_label.setWordWrap(True)
        self.credit_bar = QProgressBar()
        self.credit_bar.setRange(0, 1000)
        self.credit_bar.setTextVisible(False)
        llm_layout.addWidget(self.cost_label)
        llm_layout.addWidget(self.credit_bar)

        voice, voice_layout = self._card(
            "Voix ElevenLabs",
            "Plafond, seuil et coût : section tts.elevenlabs de config.yaml. Sous le seuil, au "
            "plafond ou sans connexion, MIKATA passe sur sa voix de base (Kokoro).",
        )
        self.voice_label = QLabel("")  # ajout phase 11 : crédits ElevenLabs
        self.voice_label.setWordWrap(True)
        self.voice_bar = QProgressBar()
        self.voice_bar.setRange(0, 1000)
        self.voice_bar.setTextVisible(False)
        voice_layout.addWidget(self.voice_label)
        voice_layout.addWidget(self.voice_bar)

        keys, keys_layout = self._card(
            "Clés",
            "Rangées dans .env, jamais dans config.yaml ni dans le dépôt. Une clé remplacée sert "
            "au prochain démarrage de MIKATA.",
        )
        self.secret_labels: dict[str, QLabel] = {}
        for name, label in SECRET_NAMES.items():
            text = QVBoxLayout()
            text.setSpacing(2)
            title = QLabel(label)
            title.setObjectName("row_label")
            shown = QLabel("")
            shown.setObjectName("row_help")
            shown.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            text.addWidget(title)
            text.addWidget(shown)
            replace = QPushButton("Remplacer…")
            replace.setToolTip(f"{name} dans .env")
            replace.clicked.connect(lambda _=False, n=name: self._replace_secret(n))
            row = QHBoxLayout()
            row.setSpacing(12)
            row.addLayout(text, 1)
            row.addWidget(replace)
            keys_layout.addLayout(row)
            self.secret_labels[name] = shown
        self._refresh_cost()
        return self._stack(llm, voice, keys)

    def _log_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.log_filter = QComboBox()
        self.log_filter.addItems(LOG_LEVELS)
        self.log_filter.setCurrentIndex(1)
        self.log_filter.currentIndexChanged.connect(self._render_log)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setMinimumHeight(380)
        top = QHBoxLayout()
        path = self._log_path()
        shown = path.relative_to(self.base_dir) if path.is_relative_to(self.base_dir) else path
        top.addWidget(self._hint(shown.as_posix()), 1)
        top.addWidget(self.log_filter)
        layout.addLayout(top)
        layout.addWidget(self.log_view, 1)
        return page

    # -- valeurs ----------------------------------------------------------------------------

    def values(self) -> dict[str, Any]:
        """Valeurs saisies, par clé de config.yaml."""
        out: dict[str, Any] = {}
        for key, widget in self.fields.items():
            if isinstance(widget, QDoubleSpinBox):
                value: Any = widget.value()
                out[key] = round(value, widget.decimals()) if widget.decimals() else int(value)
            elif isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                data = widget.currentData()
                out[key] = data if data is not None else widget.currentText().strip()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
        return out

    def pending(self) -> list[str]:
        """Réglages modifiés dans la fenêtre et pas encore enregistrés."""
        return [key for key, value in self.values().items() if self.editor.get(key) != value]

    def _edited(self, *_: Any) -> None:
        dirty = bool(self.pending())
        self.save_button.setEnabled(dirty)
        if dirty:
            self.status.setText(UNSAVED)
        elif self.status.text() == UNSAVED:
            self.status.setText("")

    def save(self) -> list[str]:
        """Écrit les réglages modifiés ; renvoie les clés changées."""
        self.editor = ConfigEditor(self.config_path)  # relu : le fichier a pu changer
        try:
            for key, value in self.values().items():
                if self.editor.get(key) != value:
                    self.editor.set(key, value)
            changed = self.editor.changed()
            if not changed:
                self.status.setText("Rien à enregistrer.")
                self.save_button.setEnabled(False)
                return []
            self.editor.save(self.base_dir / BACKUP_DIR)
        except (EditorError, OSError) as exc:
            self.status.setText(f"Non enregistré : {exc}")
            return []
        self.editor = ConfigEditor(self.config_path)  # référence des prochaines modifications
        self.save_button.setEnabled(False)
        live = [k for k in changed if k in SPECS and SPECS[k].live]
        restart = [k for k in changed if k not in live]
        sent = self._send({"type": "reload_config"})
        text = "Enregistré."
        if live:
            text += f" Tout de suite : {', '.join(SPECS[k].label for k in live)}."
        if restart:
            text += f" Au redémarrage : {', '.join(SPECS[k].label for k in restart if k in SPECS)}."
        if not sent:
            text += " (MIKATA n'est pas lancé : appliqué à son prochain démarrage.)"
        self.status.setText(text)
        logger.info("Réglages enregistrés : {}", ", ".join(changed))
        self._on_applied({k: self.values().get(k) for k in changed})
        return changed

    def restart_mikata(self) -> bool:
        """Demande à MIKATA de redémarrer ; la fenêtre se ferme avec lui (« bye »)."""
        if not self._send({"type": "restart"}):
            self.status.setText("MIKATA n'est pas lancé : rien à redémarrer.")
            return False
        self.restart_button.setEnabled(False)
        self.status.setText("Redémarrage de MIKATA… (la mascotte revient dans une minute)")
        return True

    def _skin_applied(self, values: dict[str, Any]) -> None:
        """Skin choisi : la mascotte change tout de suite, et MIKATA relit config.yaml pour
        prendre la voix du profil (phase 11)."""
        self._on_applied(values)
        self._send({"type": "reload_config"})
        self._refresh_avatar()

    def on_voice(self, message: dict[str, Any]) -> None:
        """Voix de MIKATA changée (secours Kokoro, retour ElevenLabs, profil)."""
        self.status.setText(str(message.get("message", "")))
        self._refresh_cost()

    def on_config_reply(self, message: dict[str, Any]) -> None:
        """Réponse de MIKATA à reload_config."""
        if message.get("error"):
            self.status.setText(f"MIKATA refuse la configuration : {message['error']}")

    def on_tasks(self, message: dict[str, Any]) -> None:
        """Ce qui tourne en arrière-plan (phase 13f) ; phase 15b : état du navigateur de
        travail et ses onglets, sur l'accueil."""
        tasks = message.get("tasks")
        if not isinstance(tasks, list):
            return
        browser = next((t for t in tasks if isinstance(t, dict) and t.get("id") == "browser"), {})
        self._browser = browser
        self._apply_browser_state()

    # -- navigateur de travail (accueil) ----------------------------------------------------

    def _refresh_tasks(self) -> None:
        """Demande l'état du navigateur, seulement quand l'accueil est à l'écran."""
        if not self.isVisible() or self.nav.currentRow() != 0:
            return
        if not self._send({"type": "list_tasks"}):
            self._browser = {"offline": True}
            self._apply_browser_state()

    def _browser_command(self, action: str, **fields: Any) -> None:
        if not self._send({"type": "browser", "action": action, **fields}):
            self.status.setText("MIKATA n'est pas lancé.")

    def _toggle_browser(self) -> None:
        self._browser_command("hide" if self._browser.get("visible") else "show")

    def _apply_browser_state(self) -> None:
        info = self._browser
        running = bool(info.get("running"))
        visible = bool(info.get("visible"))
        tabs = [tab for tab in info.get("tabs") or [] if isinstance(tab, dict)]
        if info.get("offline"):
            text, style = "MIKATA n'est pas lancé", "badge_idle"
        elif not info:
            text, style = "…", "badge_idle"
        elif not running:
            text, style = "en veille, démarre à la première demande", "badge_idle"
        else:
            text, style = ("affiché à l'écran" if visible else "actif, en fond"), "badge_live"
        self.browser_state.setText(text)
        self.browser_state.setObjectName(style)
        self.browser_state.style().unpolish(self.browser_state)
        self.browser_state.style().polish(self.browser_state)
        self.browser_start.setHidden(not info or running or bool(info.get("offline")))
        self.browser_toggle.setHidden(not running)
        self.browser_toggle.setText("Masquer" if visible else "Afficher")
        self.browser_pause.setHidden(not running)
        while self.browser_tabs.count():
            item = self.browser_tabs.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.browser_empty.setHidden(not running or bool(tabs))
        for tab in tabs:
            self.browser_tabs.addWidget(self._tab_row(tab))

    def _tab_row(self, tab: dict[str, Any]) -> QWidget:
        title = str(tab.get("title") or tab.get("url") or "Onglet")
        if len(title) > TAB_TITLE_MAX:
            title = title[: TAB_TITLE_MAX - 1] + "…"
        label = QLabel(title)
        label.setToolTip(str(tab.get("url", "")))
        label.setMinimumWidth(1)  # un titre long ne pousse jamais la carte hors de la fenêtre
        close_button = QPushButton("Fermer")
        close_button.setObjectName("ghost")
        close_button.setToolTip("Ferme cet onglet ; le navigateur de travail reste prêt.")
        tab_id = str(tab.get("id", ""))

        def close() -> None:
            close_button.setEnabled(False)
            self._browser_command("close_tab", tab=tab_id)

        close_button.clicked.connect(close)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(label, 1)
        row.addWidget(close_button)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # -- actions ----------------------------------------------------------------------------

    def _replace_secret(self, name: str) -> None:
        value, ok = QInputDialog.getText(
            self,
            "Remplacer la clé",
            f"Nouvelle valeur de {name} (écrite dans .env, prise en compte au redémarrage) :",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not value:
            return
        try:
            write_secret(self.base_dir / ".env", name, value)
        except (EditorError, OSError) as exc:
            self.status.setText(f"Clé non enregistrée : {exc}")
            return
        self.status.setText(f"{name} remplacée dans .env : redémarre MIKATA pour l'utiliser.")
        self._refresh_cost()

    def _toggle_mic_test(self, on: bool) -> None:
        try:
            if on:
                self._tester.start_mic(self.values()["audio.input_device"])
                self.mic_button.setText("Arrêter le test")
            else:
                self._tester.stop_mic()
                self.mic_button.setText("Tester le micro")
                self.mic_level.setValue(0)
        except Exception as exc:
            self.status.setText(f"Test du micro impossible : {exc}")
            self.mic_button.setChecked(False)

    def _test_output(self) -> None:
        try:
            self._tester.play_output(self.values()["audio.output_device"])
        except Exception as exc:
            self.status.setText(f"Test de la sortie impossible : {exc}")

    # -- rafraîchissement -------------------------------------------------------------------

    def _log_path(self) -> Path:
        path = Path(str(self.editor.get("logging.file")))
        return path if path.is_absolute() else self.base_dir / path

    def _poll(self) -> None:
        self._polls += 1
        if self._polls % COST_REFRESH_POLLS == 0:  # compteur ElevenLabs écrit par MIKATA
            self._refresh_cost()
        if self._polls % TASKS_REFRESH_POLLS == 0:
            self._refresh_tasks()
        lines = self.tail.poll()
        if lines:
            self._log_lines = (self._log_lines + lines)[-MAX_LOG_LINES:]
            self._render_log()
        if self.mic_button.isChecked():
            level = self._tester.level()
            if level is not None:
                self.mic_level.setValue(min(100, int(level * 100)))

    def _render_log(self) -> None:
        minimum = list(LOG_LEVELS.values())[self.log_filter.currentIndex()]
        shown = [short_line(line) for line in self._log_lines if _line_rank(line) >= minimum]
        bar = self.log_view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.log_view.setPlainText("\n".join(shown[-500:]))
        if at_bottom:
            bar.setValue(bar.maximum())

    def _refresh_cost(self) -> None:
        credit = float(self.editor.get("llm.credit_usd") or 0)
        usage = read_usage(self.base_dir / USAGE_FILE, credit)
        since = f" depuis le {usage.since[:10]}" if usage.since else ""
        self.cost_label.setText(
            f"Coût LLM cumulé : {usage.cost_usd:.4f} $ ({usage.commands} commande(s){since}) — "
            f"crédit restant ≈ {usage.remaining_usd:.2f} $ sur {usage.credit_usd:.2f} $"
        )
        share = usage.remaining_usd / usage.credit_usd if usage.credit_usd else 0.0
        self.credit_bar.setValue(int(1000 * min(max(share, 0.0), 1.0)))
        self.home_cost.setText(f"{usage.cost_usd:.2f} $")
        self.home_cost_detail.setText(
            f"{usage.commands} commande(s){since} · crédit restant ≈ {usage.remaining_usd:.2f} $"
        )
        text, voice_share = voice_usage_text(self.base_dir / VOICE_USAGE_FILE)
        self.voice_label.setText(text)
        self.voice_bar.setVisible(voice_share is not None)
        self.voice_bar.setValue(int(1000 * (voice_share or 0.0)))
        for name, shown in read_secrets(self.base_dir / ".env").items():
            self.secret_labels[name].setText(shown)

    def closeEvent(self, event: Any) -> None:
        if self.mic_button.isChecked():
            self.mic_button.setChecked(False)  # arrête le test du micro
        super().closeEvent(event)


def short_line(line: str) -> str:
    """« 2026-10-08 09:12:01.204 | INFO | thread | module:ligne - message » ->
    « 09:12:01  INFO      message » (ligne inchangée si elle n'a pas ce format)."""
    parts = line.split(" | ", 3)
    if len(parts) < 4 or " - " not in parts[3]:
        return line
    clock = parts[0].split(" ")[-1][:8]
    return f"{clock}  {parts[1].strip():<8}  {parts[3].split(' - ', 1)[1]}"


def _line_rank(line: str) -> int:
    parts = line.split("|")
    if len(parts) > 1:
        return _LEVEL_RANK.get(parts[1].strip(), 1)
    return 1
