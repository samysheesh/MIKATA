"""Phase 17e : les applis tenues en arrière-plan dans l'interface (accueil des réglages, menu
de l'icône) — les voir, les afficher, Pause / Reprendre, les fermer — et les sessions média
de Windows qui disent ce qu'elles jouent. Rien n'y est propre à Spotify."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any

import pytest
from test_devtools_apps import FakeApps, Proc, _controller, ax, system  # noqa: F401
from test_tray import win32  # noqa: F401
from tool_fakes import FakeUi, make_config, make_ctx

from mikata.browser import apps as apps_mod
from mikata.browser.apps import DevtoolsApps
from mikata.browser.cdp import Target
from mikata.tools import media_sessions
from mikata.tools.base import UiWindowInfo
from mikata.tools.media_sessions import MediaSession, matching
from mikata.tools.uia import app_launch_limit
from mikata.uia.controller import renamed

HIDDEN = UiWindowInfo("Spotify", "Spotify", 0, "devtools")

# -- sessions média de Windows -----------------------------------------------------------------


class FakeSession:
    def __init__(self, app: str, status: int, title: str = "", artist: str = "") -> None:
        self.source_app_user_model_id = app
        self.status, self.title, self.artist = status, title, artist
        self.calls: list[str] = []

    def get_playback_info(self) -> Any:
        controls = types.SimpleNamespace(
            is_pause_enabled=self.status == 4, is_play_enabled=self.status == 5
        )
        return types.SimpleNamespace(playback_status=self.status, controls=controls)

    async def try_get_media_properties_async(self) -> Any:
        return types.SimpleNamespace(title=self.title, artist=self.artist)

    async def try_pause_async(self) -> bool:
        self.calls.append("pause")
        return True

    async def try_play_async(self) -> bool:
        self.calls.append("play")
        return True


def _windows_media(monkeypatch: pytest.MonkeyPatch, sessions: list[FakeSession]) -> None:
    class Manager:
        @staticmethod
        async def request_async() -> Any:
            return types.SimpleNamespace(get_sessions=lambda: sessions)

    monkeypatch.setattr(media_sessions, "_manager", lambda: Manager)


def test_media_sessions_tell_what_each_app_plays(monkeypatch: pytest.MonkeyPatch) -> None:
    spotify = FakeSession("Spotify.exe", 4, "Jet Lag", "Tiakola")
    brave = FakeSession("Brave", 5, "One Piece OP")
    _windows_media(monkeypatch, [brave, spotify])
    found = media_sessions.sessions()
    assert [s.app for s in found] == ["Brave", "Spotify.exe"]
    session = matching("Spotify", found)
    assert session is not None and session.playing and session.label == "Jet Lag · Tiakola"
    assert matching("spotify", [MediaSession("SpotifyAB.SpotifyMusic_zpd!Spotify", False)])
    assert matching("Discord", found) is None


def test_pause_and_resume_target_that_app_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les touches média allaient à l'appli « courante » ; ici, seule l'appli visée."""
    spotify = FakeSession("Spotify.exe", 4, "Jet Lag")
    brave = FakeSession("Brave", 4, "Vidéo")
    _windows_media(monkeypatch, [brave, spotify])
    assert media_sessions.toggle("Spotify") == "pause"
    assert spotify.calls == ["pause"] and brave.calls == []
    spotify.status = 5
    assert media_sessions.toggle("Spotify") == "lecture" and spotify.calls[-1] == "play"
    assert media_sessions.toggle("Discord") is None


def test_media_sessions_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise OSError("winrt absent")

    monkeypatch.setattr(media_sessions, "_manager", broken)
    assert media_sessions.sessions() == [] and media_sessions.toggle("Spotify") is None


# -- état et actions des applis ---------------------------------------------------------------


def test_status_reads_without_ever_launching(
    system: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Target("Jet Lag • Tiakola", "https://xpui.app.spotify.com/index.html", "ws://x")
    monkeypatch.setattr(apps_mod, "list_targets_sync", lambda port, timeout: [target])
    apps = DevtoolsApps(["Spotify", "Code"], None)
    system["procs"] = [Proc(1, ["Spotify.exe", "--remote-debugging-port=41000"])]
    status = apps.status()
    assert status[0] == {
        "name": "Spotify",
        "open": True,
        "driven": True,
        "visible": False,
        "title": "Jet Lag • Tiakola",
    }
    assert system["launched"] == [] and system["killed"] == []


def test_show_hide_and_close_from_the_interface(
    system: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[set[int]] = []
    hidden: list[set[int]] = []
    monkeypatch.setattr(apps_mod, "_show_windows", shown.append)
    monkeypatch.setattr(apps_mod, "_hide_windows", hidden.append)
    apps = DevtoolsApps(["Spotify"], None)
    assert apps.act("Spotify", "show") == "Spotify est déjà fermé."
    system["procs"] = [Proc(7, ["Spotify.exe", "--remote-debugging-port=41000"])]
    assert apps.act("spotify", "show") == "Spotify affiché." and shown == [{7}]
    assert apps.act("Spotify", "hide") == "Spotify caché." and hidden == [{7}]
    assert apps.act("Spotify", "close") == "Spotify fermé." and system["killed"] == [1]
    with pytest.raises(Exception, match="pas une appli tenue"):
        apps.act("Discord", "close")


# -- outils : délai et nom d'un bouton après le clic --------------------------------------------


def test_a_tool_that_may_launch_a_hidden_app_waits_for_it(tmp_path: Path) -> None:
    plain = make_ctx(tmp_path)
    assert app_launch_limit(plain) is None and app_launch_limit(plain, 25.0) == 25.0
    listed = make_ctx(
        tmp_path,
        config=make_config(
            tmp_path, uia={"devtools_apps": ["Spotify"], "devtools_launch_timeout_s": 25}
        ),
    )
    timeout = listed.config.tools.timeout_s
    assert app_launch_limit(listed) == timeout + 25
    assert app_launch_limit(listed, 100.0) == 100.0


def test_a_click_says_what_a_multi_state_button_became() -> None:
    """La répétition de Spotify : « Enable repeat » (liste) puis « Enable repeat one »."""
    assert renamed("Enable repeat", "Enable repeat one") == (
        " ; il s'appelle maintenant « Enable repeat one »"
    )
    assert renamed("Mute", "mute") == "" and renamed("Play", "") == ""
    apps = FakeApps(
        [
            ax(1, "RootWebArea", "Spotify", [2]),
            ax(2, "checkbox", "Enable repeat", props={"checked": "false"}),
        ]
    )
    apps.partial = {"nodes": [ax(2, "checkbox", "Enable repeat one", props={"checked": "true"})]}
    result = _controller(apps).click(HIDDEN, "Enable repeat", None, 1)
    assert result == (
        "« Enable repeat » basculé : maintenant activé ; "
        "il s'appelle maintenant « Enable repeat one »."
    )


# -- menu de l'icône et accueil des réglages ----------------------------------------------------


def test_the_tray_menu_closes_each_hidden_app(
    fake_pystray: types.ModuleType,
    win32: None,  # noqa: F811
) -> None:
    from test_tray import FakeIcon

    from mikata.ui.tray import TrayIcon

    actions: list[str] = []
    tray = TrayIcon()
    tray.start(on_quit=lambda: None, on_action=actions.append, apps=["Spotify"])
    icon = FakeIcon.instances[-1]
    item = next(item for item in icon.menu.items if item.text == "Fermer Spotify")
    item.action(icon, item)
    assert actions == ["app_close:Spotify"]
    tray.stop()


def test_the_app_reports_and_drives_hidden_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mikata import app as app_module
    from mikata.app import App

    config = make_config(tmp_path, uia={"devtools_apps": ["Spotify"]})
    ui = FakeUi(windows=[UiWindowInfo("Jet Lag • Tiakola", "Spotify", 0, "devtools")])
    ctx = make_ctx(tmp_path, config=config, uia=ui)
    app = App(config)
    app._agent = types.SimpleNamespace(tools=types.SimpleNamespace(ctx=ctx))  # type: ignore[assignment]
    published: list[dict[str, Any]] = []
    app._publish_ui = lambda kind, **fields: published.append({"type": kind, **fields})  # type: ignore[method-assign]
    monkeypatch.setattr(
        app_module, "_media_sessions", lambda: [MediaSession("Spotify.exe", True, "Jet Lag")]
    )
    monkeypatch.setattr(app_module.sys, "platform", "win32")
    toggled: list[str] = []
    monkeypatch.setattr(media_sessions, "toggle", lambda name: toggled.append(name) or "pause")

    async def scenario() -> None:
        await app._publish_browser_state()
        await app._app_action({"name": "Spotify", "action": "pause"})
        await app._app_action({"name": "Spotify", "action": "show"})
        app.ui_action("app_close:Spotify")  # menu de l'icône : planifié dans la boucle
        for _ in range(50):
            if ("background_action", ("Spotify", "close")) in ui.log:
                break
            await asyncio.sleep(0.02)

    asyncio.run(scenario())
    task = next(t for t in published[0]["tasks"] if t["id"] == "app:Spotify")
    assert task["running"] and task["driven"] and not task["visible"]
    assert task["media"] == "playing" and task["now"] == "Jet Lag"
    assert toggled == ["Spotify"]
    assert ("background_action", ("Spotify", "show")) in ui.log
    assert ("background_action", ("Spotify", "close")) in ui.log
