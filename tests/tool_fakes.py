"""Faux bureau Windows, faux volume et fausses notifications pour les tests des outils."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from mikata.agent.guard import Guard
from mikata.agent.memory import LongTermMemory
from mikata.config import Config
from mikata.tools.base import (
    AppEntry,
    ToolContext,
    ToolError,
    UiControlInfo,
    UiWindowInfo,
    WindowInfo,
)

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
                AppEntry("Photos", "shell:AppsFolder\\Microsoft.Windows.Photos!App"),
                AppEntry("Visual Studio Code", r"C:\Start\Visual Studio Code.lnk"),
                AppEntry("Counter-Strike 2", r"C:\Start\Steam\Counter-Strike 2.url"),
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

    def open_folder(self, path: str, select: bool = False) -> None:
        self._record("reveal", (path, select))

    def windows(self) -> list[WindowInfo]:
        return list(self.win)

    def foreground_window(self) -> WindowInfo | None:
        return self.win[0] if self.win else None

    def cached_apps(self) -> list[AppEntry]:
        return list(self.apps)

    def close_window(self, window: WindowInfo) -> None:
        self._record("close", window.handle)

    def focus_window(self, window: WindowInfo) -> None:
        self._record("focus", window.handle)

    def media_key(self, action: str) -> None:
        self._record("media", action)

    def lock(self) -> None:
        self._record("lock")

    def power(self, action: str) -> None:
        self._record("power", action)

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


class FakeTrash:
    """Corbeille de test : déplace l'élément dans `folder` (jamais de suppression)."""

    def __init__(self, folder: Path, fail: bool = False) -> None:
        self.folder = folder
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, path: str) -> None:
        self.calls.append(path)
        if self.fail:
            raise OSError("corbeille pleine (simulé)")
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.move(path, self.folder / f"{len(self.calls)}-{Path(path).name}")


class FakeRunner:
    """Lanceur de commandes de test : enregistre, ne lance rien."""

    def __init__(self, code: int = 0, output: str = "ok", error: Exception | None = None) -> None:
        self.code, self.output, self.error = code, output, error
        self.calls: list[tuple[list[str], float, Path | None]] = []

    async def run(
        self, argv: list[str], timeout_s: float, cwd: Path | None = None
    ) -> tuple[int, str]:
        self.calls.append((argv, timeout_s, cwd))
        if self.error is not None:
            raise self.error
        return self.code, self.output


class FakeInput:
    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []

    def type_text(self, text: str) -> None:
        self.log.append(("type", text))

    def press_keys(self, keys: list[str]) -> None:
        self.log.append(("keys", keys))

    def mouse(self, action: str, x: int | None, y: int | None, amount: int) -> None:
        self.log.append(("mouse", (action, x, y, amount)))


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.sent.append((title, body))


class FakeUi:
    """Pilote UIA de test (phase 12a) : fenêtres et contrôles scriptés, appels enregistrés."""

    def __init__(
        self,
        windows: list[UiWindowInfo] | None = None,
        controls: list[UiControlInfo] | None = None,
        text: str = "",
        state: str = "",
    ) -> None:
        self.windows = windows or [UiWindowInfo("Discord", "Discord")]
        self.scripted_controls = (
            controls if controls is not None else [UiControlInfo("Rechercher", "bouton")]
        )
        self.text = text
        self.state = state
        self.paused: list[bool] = []
        self.log: list[tuple[str, Any]] = []
        self.fail: set[str] = set()

    def _record(self, name: str, value: Any = None) -> None:
        if name in self.fail:
            raise ToolError(f"échec simulé de {name}")
        self.log.append((name, value))

    def window(self, title: str | None, process: str | None) -> UiWindowInfo:
        self._record("window", (title, process))
        return self.windows[0]

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        self._record("controls", (window.title, max_controls))
        return list(self.scripted_controls)

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        self._record("read", (window.title, control, max_chars))
        return self.text

    def click(self, window: UiWindowInfo, control: str, kind: str | None, nth: int) -> str:
        self._record("click", (window.title, control, kind, nth))
        return f"« {control} » cliqué."

    def set_text(self, window: UiWindowInfo, control: str, text: str) -> str:
        self._record("set_text", (window.title, control, text))
        return f"« {control} » rempli."

    def state_summary(self) -> str:
        self._record("state")
        return self.state

    def pause_state(self, paused: bool) -> None:
        self.paused.append(paused)

    def aclose(self) -> None:
        pass


class FakeIndex:
    """Index de fichiers de test (phase 12d) : résultats scriptés, jamais de vrai disque."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        folder: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ready_state = True
        self.results = (
            results
            if results is not None
            else [
                {
                    "path": "D:\\Steam\\steamapps\\common\\Cyberpunk 2077.exe",
                    "name": "Cyberpunk 2077.exe",
                    "is_dir": False,
                    "mtime": 1700000000.0,
                },
            ]
        )
        self.folder = folder if folder is not None else []
        self.queries: list[tuple[str, int]] = []

    def ready(self) -> bool:
        return self.ready_state

    def stats(self) -> dict[str, Any]:
        return {"ready": self.ready_state, "entries": len(self.results)}

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self.queries.append((query, max_results))
        return [dict(r) for r in self.results[:max_results]]

    def list_folder(self, folder: str, limit: int) -> list[dict[str, Any]]:
        self.queries.append((folder, limit))
        return [dict(r) for r in self.folder[:limit]]

    def aclose(self) -> None:
        pass


class FakeBrowser:
    """Contrôleur de navigateur de test (phase 12b) : pages scriptées, appels enregistrés."""

    def __init__(self, tabs: str = "", page: str = "", href: str = "") -> None:
        self.tabs_text = tabs or "- Onglet — https://example.com"
        self.page = page or "Titre : Exemple\nURL : https://example.com\nTexte :\nBonjour."
        self.href = href  # lien renvoyé par find (téléchargement simulé si .exe/.zip...)
        self.log: list[tuple[str, Any]] = []
        self.fail: set[str] = set()
        self.running = False  # phase 13f : onglet « En cours »

    async def _record(self, name: str, value: Any = None) -> None:
        if name in self.fail:
            raise ToolError(f"échec simulé de {name}")
        self.log.append((name, value))

    async def tabs(self) -> str:
        await self._record("tabs")
        return self.tabs_text

    async def open_url(self, url: str) -> str:
        await self._record("open", url)
        self.running = True
        return f"Page ouverte : {url}"

    async def read(self, max_chars: int) -> str:
        await self._record("read", max_chars)
        return self.page

    async def find(self, text: str, nth: int) -> dict[str, Any]:
        await self._record("find", (text, nth))
        return {"ok": True, "label": text, "tag": "A", "href": self.href}

    async def click(self, text: str, nth: int) -> str:
        await self._record("click", (text, nth))
        return f"« {text} » cliqué."

    async def type_text(self, field: str, text: str) -> str:
        await self._record("type", (field, text))
        return f"Texte écrit dans le champ « {field} »."

    async def back(self) -> str:
        self.log.append(("back", ()))
        return "Page précédente."

    async def quit(self) -> str:
        self.log.append(("quit", ()))
        self.running = False
        return "Navigateur de travail fermé."

    def keep_hidden(self) -> None:
        """Le focus_window du code généré est suivi d'un re-cache de la fenêtre de travail."""
        self.log.append(("keep_hidden", ()))

    async def aclose(self) -> None:
        pass


def make_config(tmp_path: Path, **sections: dict[str, Any]) -> Config:
    """Config de test ; le garde-fou n'autorise que `tmp_path/home` (le « dossier perso »)."""
    data = Config().model_dump(mode="json", exclude={"secrets"})
    data["guard"]["allowed_dirs"] = [str(tmp_path / "home")]
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
    trash: FakeTrash | None = None,
    shell: Any = None,
    input: Any = None,
    uia: Any = None,
    browser: Any = None,
    skills: Any = None,
    skills_capture: Callable[[str], Any] | None = None,
    index: Any = None,
) -> ToolContext:
    async def no_wait(seconds: float) -> None:
        await asyncio.sleep(0)

    config = config or make_config(tmp_path)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    guard = Guard(
        config, home=home, trash=trash or FakeTrash(tmp_path / "Trash"), clock=lambda: NOW
    )
    return ToolContext(
        config,
        desktop=desktop,
        audio=audio,
        notifier=FakeNotifier(),
        http=http,
        search_api_key=search_api_key,
        memory=LongTermMemory(tmp_path / "memory.json"),
        announce=announce,
        clock=lambda: NOW,
        sleep=no_wait,
        guard=guard,
        shell=shell,
        input=input,
        uia=uia,
        browser=browser,
        skills=skills,
        skills_capture=skills_capture,
        index=index,
    )
