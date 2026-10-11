"""Processus de l'interface (ajout phase 8) : `python -m mikata.ui [--settings | --preview SKIN]`.

Lancé par MIKATA au démarrage (`ui.mascot`) ou à la demande (menu de l'icône). Une seule
instance : une deuxième transmet sa demande (réglages, afficher la mascotte) à la première
puis se termine. L'interface se ferme quand MIKATA s'arrête (`bye`) ; si MIKATA plante,
elle attend qu'il revienne (mascotte grise, bulle « MIKATA n'est pas lancé »).
Journal à part : data/logs/ui.log. Priorité basse : le jeu passe devant.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from mikata.config import Config, ConfigError, load_config, resolve_config_path
from mikata.ui.model import MascotModel
from mikata.ui.prefs import STATE_FILE, load_prefs
from mikata.ui.protocol import CHANNEL_FILE
from mikata.ui.qt.client import MikataClient
from mikata.ui.qt.mascot import MascotWindow
from mikata.ui.qt.settings import SettingsWindow
from mikata.ui.skins import DEFAULT_SKIN, Skin, SkinError, load_skin

LOG_FILE = "data/logs/ui.log"


def instance_name() -> str:
    return f"mikata-ui-{getpass.getuser()}"


def load_skin_or_default(root: Path, key: str) -> Skin | None:
    """Le skin `key`, ou Mika (None si même son manifeste manque) : jamais d'exception."""
    try:
        return load_skin(root, key)
    except SkinError as exc:
        logger.warning("Skin « {} » ignoré : {} (Mika, le skin par défaut).", key, exc)
        return None if key == DEFAULT_SKIN else load_skin_or_default(root, DEFAULT_SKIN)


class UiController(QObject):
    def __init__(self, config: Config, config_path: Path) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.base_dir = config.base_dir
        self.skin_root = self.base_dir / "assets" / "mascot"  # comme les réglages
        self.model = MascotModel(bubble_s=config.ui.bubble_s)
        self.model.disconnected()
        self.client = MikataClient(config.resolve(CHANNEL_FILE), self)
        self.client.message.connect(self.on_message)
        self.client.disconnected.connect(self.model.disconnected)
        prefs_path = config.resolve(STATE_FILE)
        self.mascot = MascotWindow(
            self.model,
            load_prefs(prefs_path),
            prefs_path,
            load_skin_or_default(self.skin_root, config.ui.skin),
            self.client.send,
            self.open_settings,
        )
        self.settings: SettingsWindow | None = None

    def start(self, open_settings: bool = False) -> None:
        self.mascot.start()
        self.client.connect_to_mikata()
        if open_settings:
            self.open_settings()

    def on_message(self, message: dict[str, Any]) -> None:
        self.model.handle(message)
        kind = message.get("type")
        if kind == "open_settings":
            self.open_settings()
        elif kind == "show_mascot":
            self.mascot.reveal()
        elif kind == "config" and self.settings is not None:
            self.settings.on_config_reply(message)
        elif kind == "voice" and self.settings is not None:
            self.settings.on_voice(message)
        elif kind == "tasks" and self.settings is not None:
            self.settings.on_tasks(message)
        elif kind == "bye":
            logger.info("MIKATA s'arrête : l'interface aussi.")
            QApplication.quit()

    def on_request(self, request: str) -> None:
        """Demande d'une autre instance (`--settings`) ou du menu."""
        if request == "settings":
            self.open_settings()
        else:
            self.mascot.reveal()

    def open_settings(self) -> None:
        if self.settings is None:
            self.settings = SettingsWindow(
                self.config_path,
                self.base_dir,
                self.client.send,
                on_applied=self.apply,
                on_show_mascot=self.mascot.reveal,
            )
        self.settings.show()
        self.settings.raise_()
        self.settings.activateWindow()

    def apply(self, values: dict[str, Any]) -> None:
        """Réglages de l'interface elle-même, appliqués tout de suite."""
        if "ui.skin" in values:
            self.mascot.set_skin(load_skin_or_default(self.skin_root, str(values["ui.skin"])))
        if "ui.bubble_s" in values:
            self.model.bubble_s = float(values["ui.bubble_s"])


def _setup_logging(base_dir: Path) -> None:
    from mikata.logging_setup import trash_old_logs

    logger.remove()
    if sys.stderr is not None:
        logger.add(sys.stderr, level="INFO")
    logger.add(
        base_dir / LOG_FILE,
        level="INFO",
        rotation=1024 * 1024,
        retention=trash_old_logs(2),
        encoding="utf-8",
    )


def _forward_to_running(request: str) -> bool:
    """Une interface tourne déjà : lui transmettre la demande. True si c'est le cas."""
    socket = QLocalSocket()
    socket.connectToServer(instance_name())
    if not socket.waitForConnected(500):
        return False
    socket.write(request.encode() + b"\n")
    socket.waitForBytesWritten(500)
    socket.disconnectFromServer()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mikata.ui", description="Interface de MIKATA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--settings", action="store_true", help="ouvrir les réglages")
    parser.add_argument("--preview", metavar="SKIN", help="aperçu d'un skin, sans MIKATA")
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)
    try:
        config = load_config(config_path)
    except ConfigError as exc:  # réglages illisibles : valeurs par défaut, MIKATA dira l'erreur
        print(exc, file=sys.stderr)
        config = Config()
        config._base_dir = config_path.parent
    _setup_logging(config.base_dir)
    if args.preview:
        from mikata.ui.qt.preview import preview_skin

        try:
            return preview_skin(config.base_dir / "assets" / "mascot", args.preview)
        except SkinError as exc:
            print(f"Aperçu impossible : {exc}", file=sys.stderr)
            return 1
    app = QApplication(sys.argv[:1])
    app.setApplicationName("MIKATA")
    app.setQuitOnLastWindowClosed(False)  # mascotte masquée, réglages fermés : on reste là
    request = "settings" if args.settings else "show"
    if _forward_to_running(request):
        logger.info("Interface déjà lancée : demande « {} » transmise.", request)
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(instance_name())  # nom resté d'une instance plantée (Unix)
    server.listen(instance_name())
    try:
        from mikata.governor import lower_priority

        lower_priority()
    except Exception as exc:
        logger.debug("Priorité inchangée : {}", exc)
    controller = UiController(config, config_path)

    def on_connection() -> None:
        socket = server.nextPendingConnection()
        socket.waitForReadyRead(500)
        controller.on_request(bytes(socket.readAll().data()).decode().strip())
        socket.disconnectFromServer()

    server.newConnection.connect(on_connection)
    controller.start(open_settings=args.settings)
    logger.info("Interface de MIKATA lancée.")
    return app.exec()
