"""Côté MIKATA du canal local vers l'interface (ajout phase 8, voir `ui.protocol`).

- `publish` ne fait qu'ajouter un message à la file de chaque interface connectée : pas
  d'encodage ni d'écriture dans l'appel (boucle asyncio, jamais le callback audio). Une
  tâche par client écrit ensuite les lignes.
- File bornée par client : une interface qui ne lit plus perd les plus vieux messages ;
  une interface bloquée plus de `WRITE_TIMEOUT_S` est déconnectée.
- Aucune interface, ou une interface qui plante : MIKATA continue comme avant.
- `launch_ui` lance l'interface (`python -m mikata.ui`) dans un processus séparé.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.ui.protocol import (
    PROTOCOL_VERSION,
    UI_COMMANDS,
    Channel,
    ProtocolError,
    decode,
    encode,
    write_channel,
)

QUEUE_SIZE = 256
HELLO_TIMEOUT_S = 3.0
WRITE_TIMEOUT_S = 2.0


class _Client:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.queue: deque[dict[str, Any]] = deque(maxlen=QUEUE_SIZE)
        self.ready = asyncio.Event()


class UiBridge:
    def __init__(
        self,
        channel_path: Path,
        on_command: Callable[[dict[str, Any]], None],
        snapshot: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """`on_command` : commande reçue d'une interface (appelé dans la boucle asyncio).
        `snapshot` : champs du message `hello` envoyé à chaque nouvelle interface."""
        self.channel_path = channel_path
        self._on_command = on_command
        self._snapshot = snapshot or dict
        self._clients: set[_Client] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._server: asyncio.base_events.Server | None = None
        self.channel: Channel | None = None
        self._closing = False

    @property
    def connected(self) -> bool:
        return bool(self._clients)

    async def start(self) -> Channel:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.channel = Channel.new(port)
        await asyncio.to_thread(write_channel, self.channel_path, self.channel)
        logger.info("Canal de l'interface : 127.0.0.1:{} ({}).", port, self.channel_path)
        return self.channel

    def publish(self, message: dict[str, Any]) -> None:
        """Message pour toutes les interfaces connectées. Ne bloque jamais."""
        for client in self._clients:
            client.queue.append(message)
            client.ready.set()

    async def close(self) -> None:
        """Prévient les interfaces (`bye`), puis ferme le canal."""
        self.publish({"type": "bye"})
        self._closing = True
        for client in list(self._clients):
            with contextlib.suppress(Exception):
                async with asyncio.timeout(WRITE_TIMEOUT_S):
                    await self._flush(client)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for client in list(self._clients):
            client.writer.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # -- connexion d'une interface ---------------------------------------------------------

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        client: _Client | None = None
        try:
            async with asyncio.timeout(HELLO_TIMEOUT_S):
                line = await reader.readline()
            hello = decode(line, ("hello",))
            assert self.channel is not None
            if not secrets.compare_digest(str(hello.get("token", "")), self.channel.token):
                logger.warning("Interface refusée : jeton invalide.")
                return
            client = _Client(writer)
            self._clients.add(client)
            client.queue.append({"type": "hello", "version": PROTOCOL_VERSION, **self._snapshot()})
            client.ready.set()
            sender = asyncio.create_task(self._send_loop(client), name="ui-envoi")
            self._tasks.add(sender)
            sender.add_done_callback(self._tasks.discard)
            logger.info("Interface connectée ({} en tout).", len(self._clients))
            while line := await reader.readline():
                try:
                    command = decode(line, UI_COMMANDS)
                except ProtocolError as exc:
                    logger.debug("Commande d'interface ignorée : {}", exc)
                    continue
                try:
                    self._on_command(command)
                except Exception:
                    logger.exception("Commande d'interface en échec : {}", command.get("type"))
            sender.cancel()
        except (TimeoutError, ProtocolError, ConnectionError, OSError) as exc:
            logger.debug("Connexion d'interface fermée : {!r}", exc)
        finally:
            if client is not None:
                self._clients.discard(client)
                logger.info("Interface déconnectée.")
            writer.close()
            if task is not None:
                self._tasks.discard(task)

    async def _send_loop(self, client: _Client) -> None:
        # asyncio.timeout et non wait_for : sous Python 3.11, wait_for peut avaler une
        # annulation qui tombe pendant l'écriture, et close() attendrait cette tâche.
        try:
            while not self._closing:
                await client.ready.wait()
                async with asyncio.timeout(WRITE_TIMEOUT_S):
                    await self._flush(client)
        except (TimeoutError, ConnectionError, OSError):
            logger.warning("Interface trop lente ou partie : déconnectée.")
            self._clients.discard(client)
            client.writer.close()

    @staticmethod
    async def _flush(client: _Client) -> None:
        client.ready.clear()
        while client.queue:
            message = client.queue.popleft()
            try:
                client.writer.write(encode(message))
            except ProtocolError as exc:
                logger.debug("Message d'interface non envoyé : {}", exc)
        await client.writer.drain()


def ui_command(argv: Sequence[str] = ()) -> list[str]:
    """`python -m mikata.ui` avec le même interpréteur (pythonw.exe : pas de console)."""
    return [sys.executable, "-m", "mikata.ui", *argv]


def launch_ui(cwd: Path, argv: Sequence[str] = ()) -> subprocess.Popen[bytes] | None:
    """Lance l'interface dans un processus séparé ; None si impossible (jamais bloquant)."""
    try:
        return subprocess.Popen(
            ui_command(argv),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        logger.warning("Interface non lancée : {}", exc)
        return None
