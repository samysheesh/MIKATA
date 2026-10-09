"""Client minimal du protocole Chrome DevTools (phase 12b).

Parle à Chromium (Brave) par WebSocket : découverte des cibles sur
`http://127.0.0.1:{port}/json`, commandes numérotées avec réponse, événements. Pas de
Playwright ni de moteur téléchargé : une WebSocket (`websockets`) et du JSON. Le
JavaScript envoyé au navigateur est fixe et écrit ici, jamais fourni par le LLM.

Testé sous Linux contre un faux serveur WebSocket (tests/test_cdp_client.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any

import httpx2
from websockets.asyncio.client import connect as ws_connect

LOAD_TIMEOUT_S = 8.0  # plafond : on attend une page stable, pas tout le chargement
MAX_WS_BYTES = 2**26  # 64 Mo : une grosse page ne casse pas la connexion


class CdpError(Exception):
    """Échec du protocole DevTools (message en français)."""


class CdpClosed(CdpError):
    """Connexion au navigateur fermée (fenêtre fermée, processus mort) : le contrôleur
    jette sa connexion et se rattachera au prochain appel."""


@dataclass(frozen=True)
class Target:
    """Un onglet du navigateur."""

    title: str
    url: str
    ws_url: str
    kind: str = "page"


async def list_targets(port: int, timeout_s: float = 3.0) -> list[Target]:
    """Onglets du navigateur, lus sur le point d'entrée HTTP de DevTools."""
    try:
        async with httpx2.AsyncClient(timeout=timeout_s) as http:
            response = await http.get(f"http://127.0.0.1:{port}/json")
            response.raise_for_status()
            raw = response.json()
    except Exception as exc:
        raise CdpError(f"navigateur injoignable sur le port {port} : {exc}") from exc
    targets: list[Target] = []
    for entry in raw:
        ws_url = entry.get("webSocketDebuggerUrl") or ""
        if not ws_url:
            continue
        targets.append(
            Target(
                title=entry.get("title", ""),
                url=entry.get("url", ""),
                ws_url=ws_url,
                kind=entry.get("type", "page"),
            )
        )
    return targets


class CdpClient:
    """Une connexion WebSocket vers un onglet : commandes et événements numérotés."""

    def __init__(self, ws_url: str, timeout_s: float = 10.0) -> None:
        self._ws_url = ws_url
        self._timeout = timeout_s
        self._ws: Any = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._closed = False  # True : close() en cours (le reader ne doit pas y toucher)

    async def connect(self) -> None:
        self._ws = await ws_connect(self._ws_url, open_timeout=self._timeout, max_size=MAX_WS_BYTES)
        self._reader = asyncio.create_task(self._reader_loop())

    async def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CdpClosed("connexion au navigateur fermée"))
        self._pending.clear()

    async def __aenter__(self) -> CdpClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                call_id = message.get("id")
                if call_id is None:
                    continue  # événement du navigateur : ignoré (on sonde, on n'écoute pas)
                future = self._pending.pop(call_id, None)
                if future is not None and not future.done():
                    if "error" in message:
                        error = message["error"]
                        future.set_exception(
                            CdpError(error.get("message") or "erreur du navigateur")
                        )
                    else:
                        future.set_result(message.get("result") or {})
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            # Fin de la connexion (fermeture propre ou brutale) : les commandes en attente
            # échouent tout de suite, pas après le délai, et les suivantes verront une
            # connexion morte (CdpClosed) au lieu d'un envoi sur une WebSocket fermée.
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(CdpClosed("connexion au navigateur fermée"))
            self._pending.clear()
            if not self._closed:
                self._ws = None

    async def command(self, method: str, **params: Any) -> dict[str, Any]:
        if self._ws is None:
            raise CdpClosed("connexion au navigateur fermée")
        self._next_id += 1
        call_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        await self._ws.send(json.dumps({"id": call_id, "method": method, "params": params}))
        return await asyncio.wait_for(future, self._timeout)

    # -- opérations de haut niveau ----------------------------------------------------

    async def navigate(self, url: str, timeout_s: float = LOAD_TIMEOUT_S) -> None:
        """Va à `url` et attend que la page soit lisible (plafond `timeout_s`)."""
        with contextlib.suppress(CdpError):
            await self.command("Page.enable")
        await self.command("Page.navigate", url=url)
        await self.wait_ready(timeout_s)

    async def wait_ready(self, timeout_s: float = LOAD_TIMEOUT_S) -> None:
        """Attend que la page soit lisible : DOM interactif, puis texte stable.

        `loadEventFired` attend tout (images, pubs...) ; `readyState` seul ne garantit pas
        que le contenu chargé après coup (résultats de recherche) soit rendu. On attend
        donc deux lectures identiques du texte de la page : plus rien ne change, la page a
        fini de se construire. Plafond `timeout_s` : une page vide ou très lente n'arrête
        pas la commande.
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        last = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                state = await self.evaluate("document.readyState")
            except CdpError:
                state = None
            if state in ("interactive", "complete"):
                try:
                    text = await self.evaluate("document.body ? document.body.innerText : ''")
                except CdpError:
                    text = None
                if isinstance(text, str) and text.strip():
                    if text == last:
                        return  # deux lectures identiques : la page est stable
                    last = text
            await asyncio.sleep(0.2)

    async def evaluate(self, expression: str) -> Any:
        """Évalue du JavaScript fixe et renvoie la valeur (JSON) du navigateur."""
        result = await self.command("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            description = details.get("exception", {}).get("description") or "erreur JavaScript"
            raise CdpError(f"lecture de la page impossible : {description}")
        return result.get("result", {}).get("value")
