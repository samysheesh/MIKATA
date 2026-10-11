"""Fenêtre de réglages (ajout phase 8), ouverte depuis l'icône MIKATA, la mascotte (double-
clic ou clic droit) ou `python -m mikata.ui --settings`.

Onglets : Audio (micro, sortie, tests), Mot d'activation (seuils), Conversation (réflexion,
fenêtre de suivi, historique), Coût et clés (coût cumulé, crédit, clés masquées), Mascotte
(bulle, démarrage), Skins (choix immédiat, import, suppression : `skins_tab`), Journal (log en
direct).

Chaque réglage porte un badge : « tout de suite » (MIKATA le relit sans redémarrer) ou « au
redémarrage ». Enregistrer réécrit config.yaml en gardant commentaires et ordre
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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mikata.ui.qt.skins_tab import SkinsTab
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

REASONING = ("none", "low", "medium", "high")
LOG_LEVELS = {"Tout": 0, "Infos et plus": 1, "Avertissements et erreurs": 2}
_LEVEL_RANK = {"TRACE": 0, "DEBUG": 0, "INFO": 1, "SUCCESS": 1, "WARNING": 2, "ERROR": 2}
MAX_LOG_LINES = 2000
COST_REFRESH_POLLS = 10  # coûts relus toutes les 5 s (relevé toutes les 500 ms)

STYLE = """
QWidget { font-size: 10pt; }
QLabel#badge_live { background: #dcfce7; color: #166534; border-radius: 8px; padding: 1px 8px; }
QLabel#badge_restart { background: #ffedd5; color: #9a3412; border-radius: 8px; padding: 1px 8px; }
QLabel#hint { color: #5b6475; }
QLabel#status { color: #1f2430; }
QPushButton#primary { background: #2f7de1; color: white; border-radius: 4px; padding: 5px 16px; }
"""


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
    """Tests audio de l'onglet Audio (sounddevice, importé à la demande)."""

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
        self.setWindowTitle("MIKATA — Réglages")
        self.setStyleSheet(STYLE)
        self.resize(640, 560)
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

        self.tabs = QTabWidget()
        self.tabs.addTab(self._audio_tab(), "Audio")
        self.tabs.addTab(self._wake_tab(), "Mot d'activation")
        self.tabs.addTab(self._conversation_tab(), "Conversation")
        self.tasks_tab_index = 3
        self.tabs.addTab(self._tasks_tab(), "En cours")  # phase 13f : arrière-plan fermable
        self.tabs.addTab(self._cost_tab(), "Coût et clés")
        self.tabs.addTab(self._mascot_tab(), "Mascotte")
        self.skins_tab = SkinsTab(
            base_dir / "assets" / "mascot",
            config_path,
            on_applied=self._skin_applied,
            on_status=lambda text: self.status.setText(text),
        )
        self.tabs.addTab(self.skins_tab, "Skins")
        self.tabs.addTab(self._log_tab(), "Journal")
        self.tabs.currentChanged.connect(self._tab_changed)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        save = QPushButton("Enregistrer")
        save.setObjectName("primary")
        save.clicked.connect(self.save)
        close = QPushButton("Fermer")
        close.clicked.connect(self.close)
        self.restart_button = QPushButton("Redémarrer MIKATA")
        self.restart_button.setToolTip(
            "Arrête MIKATA proprement puis le relance (modèles et réglages rechargés)."
        )
        self.restart_button.clicked.connect(self.restart_mikata)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.restart_button)
        buttons.addWidget(close)
        buttons.addWidget(save)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(500)
        self._poll()

    # -- construction -----------------------------------------------------------------------

    def _row(self, form: QFormLayout, key: str, widget: QWidget) -> None:
        spec = SPECS[key]
        self.fields[key] = widget
        line = QHBoxLayout()
        line.addWidget(widget, 1)
        line.addWidget(badge(spec.live))
        holder = QWidget()
        holder.setLayout(line)
        line.setContentsMargins(0, 0, 0, 0)
        label = QLabel(spec.label)
        if spec.help:
            label.setToolTip(spec.help)
            widget.setToolTip(spec.help)
        form.addRow(label, holder)

    def _spin(
        self, key: str, low: float, high: float, step: float, decimals: int
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(float(self.editor.get(key) or 0))
        return spin

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    def _audio_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        inputs, outputs, error = self._devices()
        for key, names in (("audio.input_device", inputs), ("audio.output_device", outputs)):
            combo = QComboBox()
            combo.setEditable(True)  # sous-chaîne du nom acceptée, comme dans config.yaml
            combo.addItems(names)
            combo.setCurrentText(str(self.editor.get(key)))
            self._row(form, key, combo)
        self.mic_level = QProgressBar()
        self.mic_level.setRange(0, 100)
        self.mic_level.setTextVisible(False)
        self.mic_button = QPushButton("Tester le micro")
        self.mic_button.setCheckable(True)
        self.mic_button.toggled.connect(self._toggle_mic_test)
        out_button = QPushButton("Tester la sortie")
        out_button.clicked.connect(self._test_output)
        tests = QHBoxLayout()
        tests.addWidget(self.mic_button)
        tests.addWidget(self.mic_level, 1)
        tests.addWidget(out_button)
        holder = QWidget()
        holder.setLayout(tests)
        tests.setContentsMargins(0, 0, 0, 0)
        form.addRow("Tests", holder)
        form.addRow(
            self._hint(
                error
                or "Le test du micro l'ouvre en mode partagé : MIKATA continue d'écouter. "
                "Le test de sortie joue le son du minuteur."
            )
        )
        return page

    def _wake_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._row(form, "wake.threshold", self._spin("wake.threshold", 0, 1, 0.01, 2))
        self._row(
            form, "wake.verifier_threshold", self._spin("wake.verifier_threshold", 0, 1, 0.01, 2)
        )
        model = QLabel(str(self.editor.get("wake.model")))
        verifier = QLabel(str(self.editor.get("wake.verifier") or "aucun"))
        form.addRow("Modèle", model)
        form.addRow("Vérificateur", verifier)
        form.addRow(
            self._hint(
                "Plus haut : moins de faux déclenchements, plus de ratés. Les valeurs "
                "conseillées sont données par scripts/eval_wakeword.py (README, phase 7)."
            )
        )
        return page

    def _conversation_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        combo = QComboBox()
        combo.addItems(REASONING)
        combo.setCurrentText(str(self.editor.get("llm.reasoning_effort") or "none"))
        self._row(form, "llm.reasoning_effort", combo)
        self._row(form, "agent.follow_up_s", self._spin("agent.follow_up_s", 0, 30, 1, 0))
        self._row(
            form, "agent.history_persist_h", self._spin("agent.history_persist_h", 1, 8760, 12, 0)
        )
        form.addRow(
            self._hint(
                "Durée de l'historique : comme « garde la conversation une semaine » à la voix "
                "(168 h = 7 jours). « Nouvelle conversation » vide toujours tout."
            )
        )
        return page

    def _cost_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.cost_label = QLabel("")
        self.cost_label.setFont(QFont(self.font().family(), 12))
        self.credit_bar = QProgressBar()
        self.credit_bar.setRange(0, 1000)
        layout.addWidget(self.cost_label)
        layout.addWidget(self.credit_bar)
        layout.addSpacing(12)
        self.voice_label = QLabel("")  # ajout phase 11 : crédits ElevenLabs
        self.voice_label.setWordWrap(True)
        self.voice_bar = QProgressBar()
        self.voice_bar.setRange(0, 1000)
        layout.addWidget(self.voice_label)
        layout.addWidget(self.voice_bar)
        layout.addWidget(
            self._hint(
                "Plafond, seuil et coût : section tts.elevenlabs de config.yaml. Sous le seuil, "
                "au plafond ou sans connexion, MIKATA passe sur sa voix de base (Kokoro)."
            )
        )
        layout.addSpacing(12)
        form = QFormLayout()
        self.secret_labels: dict[str, QLabel] = {}
        for name, label in SECRET_NAMES.items():
            shown = QLabel("")
            shown.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            replace = QPushButton("Remplacer…")
            replace.clicked.connect(lambda _=False, n=name: self._replace_secret(n))
            line = QHBoxLayout()
            line.addWidget(shown, 1)
            line.addWidget(badge(False))
            line.addWidget(replace)
            holder = QWidget()
            holder.setLayout(line)
            line.setContentsMargins(0, 0, 0, 0)
            form.addRow(f"{label} ({name})", holder)
            self.secret_labels[name] = shown
        layout.addLayout(form)
        layout.addWidget(
            self._hint("Les clés sont dans .env (jamais dans config.yaml ni dans le dépôt).")
        )
        layout.addStretch(1)
        self._refresh_cost()
        return page

    def _mascot_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self._row(form, "ui.bubble_s", self._spin("ui.bubble_s", 1, 120, 1, 0))
        startup = QCheckBox()
        startup.setChecked(bool(self.editor.get("ui.mascot")))
        self._row(form, "ui.mascot", startup)
        show = QPushButton("Afficher la mascotte")
        show.clicked.connect(self._on_show_mascot)
        form.addRow(show)
        form.addRow(
            self._hint(
                "Taille et bulle : clic droit sur la mascotte. Pour cliquer sur ce qu'il y a "
                "dessous, déplace-la. Changer de personnage : onglet Skins."
            )
        )
        return page

    def _log_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.log_filter = QComboBox()
        self.log_filter.addItems(LOG_LEVELS)
        self.log_filter.setCurrentIndex(1)
        self.log_filter.currentIndexChanged.connect(self._render_log)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSizeF(9)
        self.log_view.setFont(mono)
        top = QHBoxLayout()
        path = self._log_path()
        shown = path.relative_to(self.base_dir) if path.is_relative_to(self.base_dir) else path
        top.addWidget(QLabel(shown.as_posix()), 1)
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
                return []
            self.editor.save(self.base_dir / BACKUP_DIR)
        except (EditorError, OSError) as exc:
            self.status.setText(f"Non enregistré : {exc}")
            return []
        live = [k for k in changed if k in SPECS and SPECS[k].live]
        restart = [k for k in changed if k not in live]
        sent = self._send({"type": "reload_config"})
        text = f"Enregistré dans {self.config_path.name}."
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

    def on_voice(self, message: dict[str, Any]) -> None:
        """Voix de MIKATA changée (secours Kokoro, retour ElevenLabs, profil)."""
        self.status.setText(str(message.get("message", "")))
        self._refresh_cost()

    def on_config_reply(self, message: dict[str, Any]) -> None:
        """Réponse de MIKATA à reload_config."""
        if message.get("error"):
            self.status.setText(f"MIKATA refuse la configuration : {message['error']}")

    def on_tasks(self, message: dict[str, Any]) -> None:
        """Phase 13f : la liste de ce qui tourne en arrière-plan, avec « Fermer »."""
        tasks = message.get("tasks")
        if not isinstance(tasks, list):
            return
        self._rebuild_tasks(tasks)

    # -- onglet « En cours » (phase 13f) ----------------------------------------------------

    def _tasks_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(
            self._hint("Ce que MIKATA fait tourner en arrière-plan, fermable d'un clic.")
        )
        self.tasks_empty = QLabel("Rien ne tourne en arrière-plan.")
        self.tasks_empty.setObjectName("hint")
        layout.addWidget(self.tasks_empty)
        self.tasks_rows = QVBoxLayout()
        layout.addLayout(self.tasks_rows)
        layout.addStretch(1)
        return page

    def _tab_changed(self, index: int) -> None:
        if index == self.tasks_tab_index:
            self._send({"type": "list_tasks"})

    def _refresh_tasks(self) -> None:
        if self.tabs.currentIndex() == self.tasks_tab_index:
            self._send({"type": "list_tasks"})

    def _rebuild_tasks(self, tasks: list[dict[str, Any]]) -> None:
        while self.tasks_rows.count():
            item = self.tasks_rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        active = [task for task in tasks if task.get("running")]
        self.tasks_empty.setVisible(not active)
        for task in active:
            task_id = task.get("id")
            row = QHBoxLayout()
            label = QLabel(str(task.get("label", task.get("id", ""))))
            state = QLabel("actif")
            state.setObjectName("badge_live")
            close_button = QPushButton("Fermer")
            close_button.clicked.connect(
                lambda _checked=False, task_id=task_id: self._send(
                    {"type": "close_task", "id": task_id}
                )
            )
            row.addWidget(label, 1)
            row.addWidget(state)
            row.addWidget(close_button)
            holder = QWidget()
            holder.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            self.tasks_rows.addWidget(holder)

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
        if self._polls % 20 == 0:  # toutes les 10 s : onglet « En cours »
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
        text, share = voice_usage_text(self.base_dir / VOICE_USAGE_FILE)
        self.voice_label.setText(text)
        self.voice_bar.setVisible(share is not None)
        self.voice_bar.setValue(int(1000 * (share or 0.0)))
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
