"""Le contrôleur de navigateur contre un faux serveur CDP (phase 12b) : ouverture,
lecture, et reprise quand la première cible meurt (onglet qui se ferme)."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

from websockets.asyncio.server import serve as ws_serve

from mikata.browser.cdp import Target
from mikata.browser.controller import BrowserController
from mikata.config import Config


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeCdpServer:
    """Répond aux commandes CDP (le détail du transport est testé dans test_cdp_client.py)."""

    def __init__(self) -> None:
        self.received: list[tuple[int, str, dict[str, Any]]] = []

    async def handler(self, ws: Any) -> None:
        async for raw in ws:
            message = json.loads(raw)
            self.received.append(
                (message.get("id"), message.get("method", ""), message.get("params", {}))
            )
            if message["method"] == "Page.navigate":
                await ws.send(json.dumps({"id": message["id"], "result": {}}))
                await ws.send(json.dumps({"method": "Page.loadEventFired", "params": {}}))
            elif message["method"] == "Runtime.evaluate":
                expression = message.get("params", {}).get("expression", "")
                if "document.readyState" in expression:
                    value = "complete"  # sondage du DOM : la page simulée est prête
                elif expression.strip() == "location.href":
                    value = "https://x"  # l'URL ne change jamais dans la page simulée
                else:
                    value = json.dumps(
                        {
                            "ok": True,
                            "label": "X",
                            "tag": "A",
                            "href": "https://x",
                            "title": "X",
                            "url": "https://x",
                            "text": "Contenu.",
                            "links": [],
                            "buttons": [],
                        }
                    )
                await ws.send(
                    json.dumps({"id": message["id"], "result": {"result": {"value": value}}})
                )
            else:
                await ws.send(json.dumps({"id": message["id"], "result": {}}))


class NoWindows:
    def windows(self) -> list[Any]:
        return []


def _config(tmp_path: Path) -> Config:
    config = Config()
    config._base_dir = tmp_path
    return config


def _target(port: int) -> Target:
    return Target(title="Onglet", url="https://x", ws_url=f"ws://127.0.0.1:{port}")


def test_js_arguments_cannot_collide_with_placeholders() -> None:
    """Un texte utilisateur contenant les anciens marqueurs (ou les nouveaux) ne casse
    pas l'injection : aucun placeholder ne doit rester dans le JavaScript final."""
    from mikata.browser.controller import _FIND_JS, _TYPE_JS, ARG_FIELD, ARG_NTH, ARG_TEXT, _js

    find = _js(_FIND_JS, text="titre avec ARG2 dedans", nth=2)
    assert ARG_TEXT not in find and ARG_NTH not in find  # aucun placeholder ne reste
    assert '"titre avec ARG2 dedans"' in find  # le texte de l'utilisateur est injecté tel quel
    assert "[2 - 1]" in find  # et le n-ième est bien à sa place
    typed = _js(_TYPE_JS, field="ARG2", text="bonjour tout le monde")
    assert ARG_FIELD not in typed and ARG_TEXT not in typed
    assert '"bonjour tout le monde"' in typed


def test_click_without_navigation_does_not_wait(monkeypatch: Any, tmp_path: Path) -> None:
    """Un clic qui ne navigue pas (même URL avant/après) ne sonde pas la stabilité."""
    port = _free_port()
    server = FakeCdpServer()

    async def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return [_target(port)]

    monkeypatch.setattr("mikata.browser.controller.list_targets", fake_targets)

    async def run() -> None:
        async with ws_serve(server.handler, "127.0.0.1", port):
            controller = BrowserController(_config(tmp_path), NoWindows())
            try:
                await controller.open_url("https://x")
                ready_before = sum(
                    1
                    for _, method, params in server.received
                    if method == "Runtime.evaluate"
                    and "document.readyState" in params.get("expression", "")
                )
                assert "cliqué" in await controller.click("Lecture", 1)
                ready_after = sum(
                    1
                    for _, method, params in server.received
                    if method == "Runtime.evaluate"
                    and "document.readyState" in params.get("expression", "")
                )
                # Le clic lit location.href deux fois (même URL) mais ne sonde pas le DOM.
                assert ready_after == ready_before
            finally:
                await controller.aclose()

    asyncio.run(run())


def test_controller_open_and_read(monkeypatch: Any, tmp_path: Path) -> None:
    port = _free_port()
    server = FakeCdpServer()

    async def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return [_target(port)]

    monkeypatch.setattr("mikata.browser.controller.list_targets", fake_targets)

    async def run() -> None:
        async with ws_serve(server.handler, "127.0.0.1", port):
            controller = BrowserController(_config(tmp_path), NoWindows())
            try:
                assert "Page ouverte" in await controller.open_url("https://x")
                assert "Titre : X" in await controller.read(4000)
            finally:
                await controller.aclose()

    asyncio.run(run())


def test_controller_retries_when_the_first_target_dies(monkeypatch: Any, tmp_path: Path) -> None:
    dead_port = _free_port()  # rien n'écoute : la connexion échoue tout de suite
    live_port = _free_port()
    server = FakeCdpServer()
    calls = {"n": 0}

    def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        calls["n"] += 1
        return [_target(dead_port if calls["n"] == 1 else live_port)]

    async def async_fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return fake_targets(port_arg, timeout_s)

    monkeypatch.setattr("mikata.browser.controller.list_targets", async_fake_targets)

    async def run() -> None:
        async with ws_serve(server.handler, "127.0.0.1", live_port):
            controller = BrowserController(_config(tmp_path), NoWindows())
            try:
                assert "Page ouverte" in await controller.open_url("https://x")
                assert calls["n"] == 2  # premier essai raté : on reprend avec des cibles fraîches
            finally:
                await controller.aclose()

    asyncio.run(run())


# -- phase 12f : profil de travail dédié, en arrière-plan --------------------------------


class BraveOpen:
    """Un Brave déjà ouvert (sans port de contrôle)."""

    def windows(self) -> list[Any]:
        return [type("W", (), {"process": "chrome", "title": "Brave"})()]


def test_own_profile_launches_a_minimized_dedicated_profile(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Port muet : MIKATA lance SON profil (user-data-dir sous data/), fenêtre minimisée —
    jamais le Brave de l'utilisateur."""
    import sys as _sys

    from mikata.browser import controller as controller_module

    port = _free_port()
    server = FakeCdpServer()
    launches: list[dict[str, Any]] = []
    calls = {"n": 0}

    def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        calls["n"] += 1
        return [] if calls["n"] == 1 else [_target(port)]

    async def async_fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return fake_targets(port_arg, timeout_s)

    def fake_popen(argv: list[str], **kwargs: Any) -> None:
        launches.append({"argv": argv, **kwargs})

    minimized = {"n": 0}
    monkeypatch.setattr(
        "mikata.browser.controller.BrowserController._minimize_work_window",
        lambda self: minimized.__setitem__("n", minimized["n"] + 1),
    )
    monkeypatch.setattr("mikata.browser.controller.list_targets", async_fake_targets)
    monkeypatch.setattr("mikata.browser.controller.subprocess.Popen", fake_popen)

    async def run() -> None:
        async with ws_serve(server.handler, "127.0.0.1", port):
            controller = BrowserController(_config(tmp_path), NoWindows())
            try:
                assert "Page ouverte" in await controller.open_url("https://x")
            finally:
                await controller.aclose()

    asyncio.run(run())
    assert len(launches) == 1
    assert any(
        arg.startswith("--user-data-dir=") and "brave_profile" in arg
        for arg in launches[0]["argv"]
    )
    assert "--autoplay-policy=no-user-gesture-required" in launches[0]["argv"]
    assert minimized["n"] == 1  # la fenêtre de travail est re-minimisée à l'attache
    if _sys.platform == "win32":
        startupinfo = launches[0]["startupinfo"]
        assert startupinfo is not None
        assert startupinfo.dwFlags & controller_module.subprocess.STARTF_USESHOWWINDOW
        assert startupinfo.wShowWindow == 6  # SW_MINIMIZE


def test_own_profile_attaches_without_relaunching(monkeypatch: Any, tmp_path: Path) -> None:
    """Le port répond déjà (profil de travail vivant) : on s'attache, zéro lancement."""
    port = _free_port()
    server = FakeCdpServer()
    launched = {"n": 0}

    def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return [_target(port)]

    async def async_fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return fake_targets(port_arg, timeout_s)

    def fake_popen(argv: list[str], **kwargs: Any) -> None:
        launched["n"] += 1

    monkeypatch.setattr(
        "mikata.browser.controller.BrowserController._minimize_work_window",
        lambda self: None,
    )
    monkeypatch.setattr("mikata.browser.controller.list_targets", async_fake_targets)
    monkeypatch.setattr("mikata.browser.controller.subprocess.Popen", fake_popen)

    async def run() -> None:
        async with ws_serve(server.handler, "127.0.0.1", port):
            controller = BrowserController(_config(tmp_path), NoWindows())
            try:
                assert "Page ouverte" in await controller.open_url("https://x")
            finally:
                await controller.aclose()

    asyncio.run(run())
    assert launched["n"] == 0


def test_legacy_mode_still_refuses_politely(monkeypatch: Any, tmp_path: Path) -> None:
    """own_profile: false garde l'ancien comportement : refus poli, jamais de fermeture."""
    from mikata.tools.base import ToolError

    config = _config(tmp_path)
    config = config.model_copy(
        update={"browser": config.browser.model_copy(update={"own_profile": False})}
    )
    launched = {"n": 0}

    def fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return []

    async def async_fake_targets(port_arg: int, timeout_s: float = 3) -> list[Target]:
        return fake_targets(port_arg, timeout_s)

    def fake_popen(argv: list[str], **kwargs: Any) -> None:
        launched["n"] += 1

    monkeypatch.setattr("mikata.browser.controller.list_targets", async_fake_targets)
    monkeypatch.setattr("mikata.browser.controller.subprocess.Popen", fake_popen)

    async def run() -> None:
        controller = BrowserController(config, BraveOpen())
        try:
            try:
                await controller.open_url("https://x")
                raise AssertionError("un refus était attendu")
            except ToolError as exc:
                assert "Brave est ouvert sans son port de contrôle" in str(exc)
        finally:
            await controller.aclose()

    asyncio.run(run())
    assert launched["n"] == 0  # jamais de lancement ni de fermeture forcée
