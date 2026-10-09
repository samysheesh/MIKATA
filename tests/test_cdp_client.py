"""Le vrai client CDP contre un faux serveur WebSocket (phase 12b).

Le serveur répond d'après un script et envoie `Page.loadEventFired` après chaque
navigation : on vérifie ici le transport (numérotation, erreurs, événements), pas le
navigateur.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import pytest
from websockets.asyncio.server import serve as ws_serve

from mikata.browser.cdp import CdpClient, CdpClosed, CdpError


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeCdpServer:
    def __init__(
        self, responses: dict[str, Any] | None = None, silent: bool = False, drop: bool = False
    ) -> None:
        self.responses = responses or {}
        self.silent = silent  # True : ne répond jamais (commande laissée en attente)
        self.drop = drop  # True : ferme la connexion après le premier message
        self.received: list[tuple[int, str, dict[str, Any]]] = []

    async def handler(self, ws: Any) -> None:
        async for raw in ws:
            message = json.loads(raw)
            if "id" not in message and "method" not in message:
                continue  # bruit : ignoré, comme le client l'ignore
            self.received.append(
                (message.get("id"), message.get("method", ""), message.get("params", {}))
            )
            if self.drop:
                await ws.close()
                return
            if self.silent:
                continue
            if message["method"] in self.responses:
                response = self.responses[message["method"]]
                if isinstance(response, Exception):
                    await ws.send(
                        json.dumps({"id": message["id"], "error": {"message": str(response)}})
                    )
                else:
                    await ws.send(json.dumps({"id": message["id"], "result": response}))
            else:
                await ws.send(json.dumps({"id": message["id"], "result": {}}))
            if message["method"] == "Page.navigate":
                await ws.send(json.dumps({"method": "Page.loadEventFired", "params": {}}))


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _scenario(server: FakeCdpServer) -> tuple[FakeCdpServer, CdpClient]:
    async with ws_serve(server.handler, "127.0.0.1", _free_port()) as srv:
        assert srv.sockets is not None
        port = srv.sockets[0].getsockname()[1]
        client = CdpClient(f"ws://127.0.0.1:{port}", timeout_s=3)
        await client.connect()
        try:
            yield server, client
        finally:
            await client.close()


def test_command_roundtrip() -> None:
    async def run() -> None:
        server = FakeCdpServer({"Page.enable": {"ok": True}})
        async for _, client in _scenario(server):
            assert await client.command("Page.enable") == {"ok": True}
            assert server.received == [(1, "Page.enable", {})]

    _run(run())


def test_error_becomes_cdp_error_in_french() -> None:
    async def run() -> None:
        server = FakeCdpServer({"Runtime.evaluate": Exception("boom")})
        async for _, client in _scenario(server):
            with pytest.raises(CdpError, match="boom"):
                await client.command("Runtime.evaluate", expression="1")

    _run(run())


def test_navigate_waits_until_the_dom_is_ready() -> None:
    async def run() -> None:
        server = FakeCdpServer(
            {
                "Page.navigate": {},
                "Runtime.evaluate": {"result": {"value": "complete"}},
            }
        )
        async for _, client in _scenario(server):
            await client.navigate("https://example.com", timeout_s=2)
            methods = [method for _, method, _ in server.received]
            assert "Page.enable" in methods and "Page.navigate" in methods
            assert "Runtime.evaluate" in methods  # sondage du readyState

    _run(run())


def test_evaluate_returns_the_value() -> None:
    async def run() -> None:
        server = FakeCdpServer({"Runtime.evaluate": {"result": {"value": {"title": "Exemple"}}}})
        async for _, client in _scenario(server):
            assert await client.evaluate("document.title") == {"title": "Exemple"}

    _run(run())


def test_close_releases_pending_commands() -> None:
    async def run() -> None:
        server = FakeCdpServer(silent=True)  # aucune réponse : la commande restera en attente
        async for _, client in _scenario(server):
            task = asyncio.create_task(client.command("Page.enable"))
            await asyncio.sleep(0.2)
            await client.close()
            with pytest.raises(CdpClosed, match="fermée"):
                await task

    _run(run())


def test_broken_connection_fails_pending_commands_fast() -> None:
    async def run() -> None:
        server = FakeCdpServer(drop=True)  # le navigateur coupe la connexion sans prévenir
        async for _, client in _scenario(server):
            task = asyncio.create_task(client.command("Page.enable"))
            with pytest.raises(CdpClosed, match="fermée"):
                await asyncio.wait_for(task, timeout=1.0)  # pas le délai complet (10 s)
            await asyncio.sleep(0.1)  # le reader a fini son ménage
            with pytest.raises(CdpClosed, match="fermée"):
                await client.command("Page.enable")  # les suivantes échouent tout de suite

    _run(run())


def test_unknown_messages_are_ignored() -> None:
    async def run() -> None:
        server = FakeCdpServer({"Page.enable": {"ok": 1}})
        async with ws_serve(server.handler, "127.0.0.1", _free_port()) as srv:
            assert srv.sockets is not None
            port = srv.sockets[0].getsockname()[1]
            client = CdpClient(f"ws://127.0.0.1:{port}", timeout_s=3)
            await client.connect()
            try:
                first = await client.command("Page.enable")
                # un message sans id ni méthode (bruit) ne casse rien
                await client._ws.send(json.dumps({"bonjour": True}))
                second = await client.command("Page.enable")
                assert first == second == {"ok": 1}
            finally:
                await client.close()

    _run(run())
