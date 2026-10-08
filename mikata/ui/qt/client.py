"""Côté interface du canal local (ajout phase 8) : client TCP Qt, intégré à la boucle Qt.

Lit data/ui_channel.json (port et jeton écrits par MIKATA), se connecte, envoie `hello`,
puis émet `message(dict)` pour chaque ligne reçue. Si MIKATA n'est pas lancé ou s'arrête,
réessaie toutes les `RETRY_MS` (le fichier est relu : le port change à chaque démarrage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QAbstractSocket, QTcpSocket

from mikata.ui.protocol import (
    MAX_LINE_BYTES,
    MIKATA_EVENTS,
    ProtocolError,
    decode,
    encode,
    read_channel,
)

RETRY_MS = 2000


class MikataClient(QObject):
    message = Signal(dict)
    connected = Signal()
    disconnected = Signal()

    def __init__(self, channel_path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.channel_path = channel_path
        self._socket = QTcpSocket(self)
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.readyRead.connect(self._on_ready_read)
        self._socket.errorOccurred.connect(self._on_error)
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.timeout.connect(self.connect_to_mikata)
        self._buffer = b""
        self._token = ""
        self.is_connected = False

    def connect_to_mikata(self) -> None:
        if self._socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            return
        channel = read_channel(self.channel_path)
        if channel is None:
            self._retry.start(RETRY_MS)
            return
        self._token = channel.token
        self._socket.connectToHost("127.0.0.1", channel.port)

    def send(self, message: dict[str, Any]) -> bool:
        if not self.is_connected:
            return False
        self._socket.write(encode(message))
        return True

    def close(self) -> None:
        self._retry.stop()
        self._socket.abort()

    def _on_connected(self) -> None:
        self._socket.write(encode({"type": "hello", "token": self._token}))
        self.is_connected = True
        self._buffer = b""
        logger.info("Connectée à MIKATA.")
        self.connected.emit()

    def _on_disconnected(self) -> None:
        was = self.is_connected
        self.is_connected = False
        if was:
            logger.info("MIKATA déconnecté.")
            self.disconnected.emit()
        self._retry.start(RETRY_MS)

    def _on_error(self, _error: QAbstractSocket.SocketError) -> None:
        if not self.is_connected:  # MIKATA pas (encore) lancé : on réessaie en silence
            self._socket.abort()
            self._retry.start(RETRY_MS)

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._socket.readAll().data())
        *lines, self._buffer = self._buffer.split(b"\n")
        if len(self._buffer) > MAX_LINE_BYTES:  # ligne sans fin : jetée
            self._buffer = b""
        for line in lines:
            if not line.strip():
                continue
            try:
                self.message.emit(decode(line, MIKATA_EVENTS))
            except ProtocolError as exc:
                logger.debug("Message de MIKATA ignoré : {}", exc)
