"""Faux bureau Windows, faux volume et fausses notifications pour les tests des outils."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from mikata.agent.memory import LongTermMemory
from mikata.config import Config
from mikata.tools.base import AppEntry, ToolContext, ToolError, WindowInfo

NOW = datetime(2026, 10, 7, 15, 42, 7)


class FakeDesktop:
    def __init__(
        self,
        apps: list[AppEntry] | None = None,
        windows: list[WindowInfo] | None = None,
        late_apps: list[AppEntry] | None = None,
    ) -> None:
        self.apps = (
            apps
            if apps is not None
            else [
                AppEntry("Spotify", r"C:\Start\Spotify.lnk"),
                AppEntry("Brave", r"C:\Start\Brave.lnk"),
                AppEntry("Steam", r"C:\Start\Steam\Steam.lnk"),
                AppEntry("Google Chrome", r"C:\Start\Google Chrome.lnk"),
                AppEntry("Chrome Remote Desktop", r"C:\Start\Chrome Remote Desktop.lnk"),
            ]
        )
        self.late_apps = late_apps or []  # apparaissent au rafraîchissement de l'index
        self.win = (
            windows
            if windows is not None
            else [
                WindowInfo(11, "Spotify Premium", "Spotify"),
                WindowInfo(12, "Nouvel onglet - Brave", "brave"),
                WindowInfo(13, "Discussion - Brave", "brave"),
            ]
        )
        self.log: list[tuple[str, Any]] = []
        self.clipboard = ""
        self.fail: set[str] = set()

    def _record(self, name: str, value: Any = None) -> None:
        if name in self.fail:
            raise ToolError(f"échec simulé de {name}")
        self.log.append((name, value))

    def list_apps(self, refresh: bool = False) -> list[AppEntry]:
        self.log.append(("list_apps", refresh))
        return self.apps + (self.late_apps if refresh else [])

    def launch(self, app: AppEntry) -> None:
        self._record("launch", app.name)

    def open_target(self, target: str) -> None:
        self._record("open", target)

    def windows(self) -> list[WindowInfo]:
        return list(self.win)

    def close_window(self, window: WindowInfo) -> None:
        self._record("close", window.handle)

    def focus_window(self, window: WindowInfo) -> None:
        self._record("focus", window.handle)

    def media_key(self, action: str) -> None:
        self._record("media", action)

    def lock(self) -> None:
        self._record("lock")

    def clipboard_read(self) -> str:
        return self.clipboard

    def clipboard_write(self, text: str) -> None:
        self._record("clipboard", text)
        self.clipboard = text


class FakeAudio:
    def __init__(self, volume: float = 50.0) -> None:
        self.volume = volume
        self.muted = False
        self.default = "{id-speakers}"
        self.devices = [
            ("{id-speakers}", "Haut-parleurs (Realtek(R) Audio)"),
            ("{id-airpods}", "Casque (AirPods de Samy)"),
        ]

    def get_volume(self) -> float:
        return self.volume

    def set_volume(self, percent: float) -> None:
        self.volume = percent

    def set_mute(self, mute: bool) -> None:
        self.muted = mute

    def output_devices(self) -> list[tuple[str, str]]:
        return list(self.devices)

    def set_default_output(self, device_id: str) -> None:
        self.default = device_id


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.sent.append((title, body))


def make_config(tmp_path: Path, **sections: dict[str, Any]) -> Config:
    data = Config().model_dump(mode="json", exclude={"secrets"})
    for section, values in sections.items():
        data[section].update(values)
    config = Config.model_validate(data)
    config._base_dir = tmp_path
    return config


def make_ctx(
    tmp_path: Path,
    *,
    config: Config | None = None,
    desktop: Any = None,
    audio: Any = None,
    http: Any = None,
    search_api_key: str | None = None,
    announce: Callable[[str], Any] | None = None,
) -> ToolContext:
    async def no_wait(seconds: float) -> None:
        await asyncio.sleep(0)

    return ToolContext(
        config or make_config(tmp_path),
        desktop=desktop,
        audio=audio,
        notifier=FakeNotifier(),
        http=http,
        search_api_key=search_api_key,
        memory=LongTermMemory(tmp_path / "memory.json"),
        announce=announce,
        clock=lambda: NOW,
        sleep=no_wait,
    )
