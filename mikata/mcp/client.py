"""Client MCP minimal (phase 16f), transport stdio.

Le serveur est un programme lancé par MIKATA ; les messages sont du JSON-RPC 2.0, un par
ligne, sur son entrée et sa sortie standard (sa sortie d'erreur va au journal). MIKATA
n'utilise que trois méthodes : `initialize` (puis la notification `initialized`),
`tools/list` (avec pagination) et `tools/call`. Une demande du serveur (`ping`,
échantillonnage...) reçoit une réponse vide ou « méthode inconnue » : MIKATA ne prête
jamais son modèle à un serveur.

Pas de dépendance : le SDK officiel tire un serveur web entier pour trois méthodes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from mikata import __version__

PROTOCOL_VERSION = "2025-06-18"
MAX_LINE = 4 * 1024 * 1024  # un message plus long est refusé
START_TIMEOUT_S = 180.0  # premier lancement par uvx ou npx : le serveur se télécharge


class McpError(Exception):
    pass


@dataclass
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


class McpServer:
    """Un serveur MCP lancé en sous-processus, et sa session."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.timeout_s = timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self.server_info: dict[str, Any] = {}

    async def start(self) -> None:
        flags = 0x08000000 if sys.platform == "win32" else 0  # sans console
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
            limit=MAX_LINE,
            creationflags=flags,
        )
        self._reader = asyncio.create_task(self._read(), name=f"mcp {self.name}")
        self._stderr = asyncio.create_task(self._log_stderr(), name=f"mcp {self.name} (err)")
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mikata", "version": __version__},
            },
            timeout_s=max(self.timeout_s, START_TIMEOUT_S),
        )
        self.server_info = dict(result.get("serverInfo") or {})
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def list_tools(self) -> list[McpTool]:
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(50):  # pagination bornée
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            for raw in result.get("tools") or []:
                if isinstance(raw, dict) and raw.get("name"):
                    tools.append(
                        McpTool(
                            name=str(raw["name"]),
                            description=str(raw.get("description") or ""),
                            input_schema=dict(raw.get("inputSchema") or {}),
                            annotations=dict(raw.get("annotations") or {}),
                        )
                    )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """(réussi, texte rendu par l'outil)."""
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(f"({item.get('type', 'contenu')} non affichable)")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        return not bool(result.get("isError")), "\n".join(parts).strip()

    async def request(
        self, method: str, params: dict[str, Any], timeout_s: float | None = None
    ) -> dict[str, Any]:
        if self._process is None or self._process.returncode is not None:
            raise McpError(f"serveur MCP « {self.name} » arrêté")
        self._next_id += 1
        ident = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[ident] = future
        try:
            await self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
            async with asyncio.timeout(timeout_s or self.timeout_s):
                message = await future
        except TimeoutError:
            raise McpError(f"« {self.name} » ne répond pas à {method}") from None
        finally:
            self._pending.pop(ident, None)
        if "error" in message:
            error = message["error"] or {}
            raise McpError(f"« {self.name} » : {error.get('message', 'erreur')}")
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    async def _send(self, message: dict[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        # ASCII échappé (\u00e9) : lisible par un serveur quel que soit son encodage.
        data = (json.dumps(message, ensure_ascii=True) + "\n").encode("ascii")
        async with self._write_lock:
            self._process.stdin.write(data)
            await self._process.stdin.drain()

    async def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        while True:
            try:
                line = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError):
                logger.warning("MCP « {} » : message trop long, ignoré.", self.name)
                continue
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue  # une ligne de journal égarée sur la sortie standard
            if not isinstance(message, dict):
                continue
            if "method" in message and "id" in message:
                try:
                    await self._answer_server(message)
                except (OSError, RuntimeError) as exc:  # serveur parti entre-temps
                    logger.debug("MCP « {} » : réponse non envoyée ({})", self.name, exc)
            elif "id" in message:
                future = self._pending.get(message["id"])
                if future is not None and not future.done():
                    future.set_result(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError(f"serveur MCP « {self.name} » arrêté"))

    async def _answer_server(self, message: dict[str, Any]) -> None:
        """Une demande du serveur : `ping` reçoit une réponse vide ; le reste (dont
        l'échantillonnage, qui emprunterait le modèle de MIKATA) est refusé."""
        if message.get("method") == "ping":
            await self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
            return
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "méthode non prise en charge par MIKATA"},
            }
        )

    async def _log_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            logger.debug("MCP « {} » : {}", self.name, line.decode("utf-8", "replace").rstrip())

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                async with asyncio.timeout(3):
                    await process.wait()
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        for task in (self._reader, self._stderr):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
