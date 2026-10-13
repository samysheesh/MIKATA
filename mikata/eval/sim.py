"""Banc d'essai : un PC simulé, avec un état (docs/AGENT_CORE.md §3.10).

Chaque service implémente l'interface de `mikata.tools.base` et garde un **état** que les
correcteurs relisent après la demande (volume, fenêtres ouvertes, page affichée...) : on note
l'état final, pas le chemin suivi. Rien ne touche le vrai PC ni le réseau ; seuls les fichiers
sont réels, dans un dossier temporaire qui sert de « dossier perso ».
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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

NOW = datetime(2026, 10, 13, 21, 30, 0)  # un mardi soir : les réponses datées sont stables

DEFAULT_APPS = (
    "Spotify",
    "Brave",
    "Discord",
    "Steam",
    "Visual Studio Code",
    "Bloc-notes",
    "Calculatrice",
    "Paramètres",
    "League of Legends",
    "OBS Studio",
)
_PROCESS = {
    "Bloc-notes": "Notepad",
    "Calculatrice": "CalculatorApp",
    "Visual Studio Code": "Code",
    "Paramètres": "SystemSettings",
    "League of Legends": "LeagueClient",
    "OBS Studio": "obs64",
}
WATCH_RE = re.compile(r"youtube\.com/watch", re.IGNORECASE)


@dataclass
class Page:
    """Une page du faux web : titre, texte et liens (libellé -> adresse)."""

    title: str
    text: str = ""
    links: dict[str, str] = field(default_factory=dict)


def find_page(pages: dict[str, Page], url: str) -> Page | None:
    r"""Page à cette adresse : clé exacte (au « / » final près), sinon première clé prise
    comme motif (ex. « youtube\.com/results » pour toute recherche YouTube)."""
    bare = url.replace("://www.", "://", 1)
    for key in (url, url.rstrip("/"), url + "/", bare, bare.rstrip("/"), bare + "/"):
        if key in pages:
            return pages[key]
    for pattern, page in pages.items():
        try:
            if re.search(pattern, url, re.IGNORECASE):
                return page
        except re.error:
            continue
    return None


class SimDesktop:
    """Bureau : lancer une appli ouvre une fenêtre, la fermer la retire."""

    def __init__(self, apps: list[str], open_apps: list[str]) -> None:
        self.apps = [AppEntry(name, f"C:\\Start\\{name}.lnk") for name in apps]
        self.win: list[WindowInfo] = []
        self._handle = 100
        self.clipboard = ""
        self.opened: list[str] = []  # adresses et fichiers ouverts par le programme par défaut
        self.revealed: list[str] = []  # dossiers montrés dans l'explorateur
        self.media: list[str] = []
        self.power_actions: list[str] = []
        self.locked = False
        for name in open_apps:
            self._open_window(name)

    def _open_window(self, name: str) -> None:
        self._handle += 1
        self.win.insert(0, WindowInfo(self._handle, name, _PROCESS.get(name, name)))

    def list_apps(self, refresh: bool = False) -> list[AppEntry]:
        return list(self.apps)

    def cached_apps(self) -> list[AppEntry]:
        return list(self.apps)

    def launch(self, app: AppEntry) -> None:
        self._open_window(app.name)

    def open_target(self, target: str) -> None:
        self.opened.append(target)

    def open_folder(self, path: str, select: bool = False) -> None:
        self.revealed.append(path)

    def windows(self) -> list[WindowInfo]:
        return list(self.win)

    def foreground_window(self) -> WindowInfo | None:
        return self.win[0] if self.win else None

    def close_window(self, window: WindowInfo) -> None:
        self.win = [w for w in self.win if w.handle != window.handle]

    def focus_window(self, window: WindowInfo) -> None:
        self.win = [w for w in self.win if w.handle == window.handle] + [
            w for w in self.win if w.handle != window.handle
        ]

    def media_key(self, action: str) -> None:
        self.media.append(action)

    def lock(self) -> None:
        self.locked = True

    def power(self, action: str) -> None:
        self.power_actions.append(action)

    def clipboard_read(self) -> str:
        return self.clipboard

    def clipboard_write(self, text: str) -> None:
        self.clipboard = text


class SimAudio:
    def __init__(self, volume: float) -> None:
        self.volume = volume
        self.muted = False
        self.devices = [("{hp}", "Haut-parleurs (Realtek)"), ("{casque}", "Casque (USB)")]
        self.default = "{casque}"

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


class SimInput:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.keys: list[list[str]] = []
        self.mouse_actions: list[tuple[str, int | None, int | None, int]] = []

    def type_text(self, text: str) -> None:
        self.typed.append(text)

    def press_keys(self, keys: list[str]) -> None:
        self.keys.append(list(keys))

    def mouse(self, action: str, x: int | None, y: int | None, amount: int) -> None:
        self.mouse_actions.append((action, x, y, amount))


class SimShell:
    """Aucune commande n'est lancée : elles sont notées et reçoivent une sortie scriptée."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.commands: list[str] = []

    async def run(
        self, argv: list[str], timeout_s: float, cwd: Path | None = None
    ) -> tuple[int, str]:
        command = " ".join(argv)
        self.commands.append(command)
        for pattern, output in self.outputs.items():
            if re.search(pattern, command, re.IGNORECASE):
                return 0, output
        return 0, ""


class SimNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, body: str) -> None:
        self.sent.append((title, body))


class SimUi:
    """Arbre d'accessibilité : des contrôles par appli ; un clic est noté."""

    def __init__(self, controls: dict[str, list[tuple[str, str, str]]]) -> None:
        self.controls_by_app = controls
        self.clicked: list[tuple[str, str]] = []
        self.texts: list[tuple[str, str, str]] = []

    def _match(self, title: str | None, process: str | None) -> str:
        wanted = (title or process or "").casefold()
        for app in self.controls_by_app:
            if not wanted or wanted in app.casefold() or app.casefold() in wanted:
                return app
        raise ToolError(f"aucune fenêtre ne correspond à « {title or process} »")

    def window(self, title: str | None, process: str | None) -> UiWindowInfo:
        app = self._match(title, process)
        return UiWindowInfo(app, _PROCESS.get(app, app))

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        rows = self.controls_by_app.get(window.title, [])
        return [UiControlInfo(name, kind, text) for name, kind, text in rows[:max_controls]]

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        rows = self.controls_by_app.get(window.title, [])
        if control is None:
            return "\n".join(f"{name} : {text}" for name, _, text in rows if text)[:max_chars]
        for name, _, text in rows:
            if control.casefold() in name.casefold():
                return text[:max_chars]
        raise ToolError(f"contrôle introuvable : « {control} »")

    def click(self, window: UiWindowInfo, control: str, kind: str | None, nth: int) -> str:
        for name, _, _ in self.controls_by_app.get(window.title, []):
            if control.casefold() in name.casefold():
                self.clicked.append((window.title, name))
                return f"« {name} » cliqué."
        raise ToolError(f"contrôle introuvable : « {control} »")

    def set_text(self, window: UiWindowInfo, control: str, text: str) -> str:
        self.texts.append((window.title, control, text))
        return f"« {control} » rempli."

    def state_summary(self) -> str:
        return ""

    def pause_state(self, paused: bool) -> None:
        pass

    def aclose(self) -> None:
        pass


class SimIndex:
    """Index des fichiers : les entrées déclarées, plus les vrais fichiers du dossier perso."""

    def __init__(self, entries: list[str], home: Path | None = None) -> None:
        self._declared = entries
        self._home = home

    @property
    def entries(self) -> list[str]:
        found = [str(p) for p in self._home.rglob("*")] if self._home is not None else []
        return [*self._declared, *sorted(found)]

    def ready(self) -> bool:
        return True

    def stats(self) -> dict[str, Any]:
        return {"ready": True, "entries": len(self.entries)}

    @staticmethod
    def _row(path: str) -> dict[str, Any]:
        name = path.replace("/", "\\").rsplit("\\", 1)[-1]
        return {"path": path, "name": name, "is_dir": "." not in name, "mtime": 1_760_000_000.0}

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        words = query.casefold().replace("*", " ").split()
        hits = [p for p in self.entries if all(w in p.casefold() for w in words)]
        return [self._row(p) for p in hits[:max_results]]

    def list_folder(self, folder: str, limit: int) -> list[dict[str, Any]]:
        prefix = folder.replace("\\", "/").rstrip("/").casefold() + "/"
        hits = []
        for path in self.entries:
            flat = path.replace("\\", "/")
            if flat.casefold().startswith(prefix) and "/" not in flat[len(prefix) :]:
                hits.append(path)  # enfants directs seulement
        return [self._row(p) for p in hits[:limit]]

    def aclose(self) -> None:
        pass


class SimBrowser:
    """Navigateur de travail sur le faux web : ouvrir, lire, cliquer un lien, revenir."""

    def __init__(self, pages: dict[str, Page]) -> None:
        self.pages = pages
        self.url = ""
        self.history: list[str] = []
        self.typed: list[tuple[str, str]] = []
        self.running = False

    def _page(self) -> Page:
        if not self.url:
            raise ToolError("aucune page ouverte")
        return find_page(self.pages, self.url) or Page("Page", "(page vide)")

    @property
    def playing(self) -> bool:
        return bool(WATCH_RE.search(self.url))

    async def tabs(self) -> str:
        return f"- {self._page().title} — {self.url}" if self.url else "Aucun onglet."

    async def open_url(self, url: str) -> str:
        if self.url:
            self.history.append(self.url)
        self.url, self.running = url, True
        return f"Page ouverte : {url}"

    async def read(self, max_chars: int) -> str:
        page = self._page()
        lines = [f"Titre : {page.title}", f"URL : {self.url}", f"Texte :\n{page.text[:max_chars]}"]
        if page.links:
            lines.append("Liens : " + " | ".join(page.links))
        return "\n".join(lines)

    def _link(self, text: str, nth: int) -> tuple[str, str]:
        hits = [
            (label, href)
            for label, href in self._page().links.items()
            if text.casefold() in label.casefold()
        ]
        if len(hits) < max(nth, 1):
            raise ToolError(f"aucun élément « {text} » sur la page")
        return hits[max(nth, 1) - 1]

    async def find(self, text: str, nth: int) -> dict[str, Any]:
        label, href = self._link(text, nth)
        return {"ok": True, "label": label, "tag": "A", "href": href}

    async def click(self, text: str, nth: int) -> str:
        label, href = self._link(text, nth)
        await self.open_url(href)
        return f"« {label} » cliqué."

    async def type_text(self, field: str, text: str) -> str:
        self.typed.append((field, text))
        return f"Texte écrit dans le champ « {field} »."

    async def back(self) -> str:
        if self.history:
            self.url = self.history.pop()
        return "Page précédente."

    async def quit(self) -> str:
        self.url, self.running = "", False
        return "Navigateur de travail fermé."

    def keep_hidden(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class _Response:
    def __init__(self, status: int, body: Any, content_type: str) -> None:
        self.status_code = status
        self._body = body
        self.headers = {"content-type": content_type}
        self.text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        self.content = self.text.encode("utf-8")
        self.encoding = "utf-8"

    def json(self) -> Any:
        return self._body


class SimHttp:
    """Faux réseau : recherche (Tavily), météo (Open-Meteo) et pages du faux web."""

    def __init__(
        self,
        pages: dict[str, Page],
        search: dict[str, list[dict[str, str]]],
        weather: dict[str, float] | None = None,
    ) -> None:
        self.pages = pages
        self.search = search
        self.weather = {city.casefold(): temp for city, temp in (weather or {}).items()}
        self._cities: list[str] = []  # la latitude renvoyée encode la ville demandée
        self.requests: list[str] = []

    async def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        await asyncio.sleep(0)
        self.requests.append(url)
        host = urlsplit(url).hostname or ""
        if "tavily" in host:
            query = str((kwargs.get("json") or {}).get("query", "")).casefold()
            for pattern, results in self.search.items():
                if re.search(pattern, query, re.IGNORECASE):
                    rows = [
                        {"title": r["title"], "content": r["snippet"], "url": r["url"]}
                        for r in results
                    ]
                    return _Response(200, {"answer": None, "results": rows}, "application/json")
            return _Response(200, {"answer": None, "results": []}, "application/json")
        if "geocoding-api.open-meteo" in host:
            params = kwargs.get("params") or parse_qs(urlsplit(url).query)
            name = params.get("name", "Paris")
            name = name[0] if isinstance(name, list) else name
            if str(name) not in self._cities:
                self._cities.append(str(name))
            latitude = 40.0 + self._cities.index(str(name))
            place = {"name": name, "country": "France", "latitude": latitude, "longitude": 2.35}
            return _Response(200, {"results": [place]}, "application/json")
        if "api.open-meteo" in host:
            params = kwargs.get("params") or {}
            try:
                city = self._cities[int(float(params.get("latitude", 40.0))) - 40]
            except (IndexError, ValueError, TypeError):
                city = ""
            temperature = self.weather.get(city.casefold(), 14)
            body = {
                "current": {
                    "temperature_2m": temperature,
                    "apparent_temperature": 12,
                    "weather_code": 3,
                    "wind_speed_10m": 9,
                    "relative_humidity_2m": 71,
                },
                "daily": {
                    "time": ["2026-10-13", "2026-10-14", "2026-10-15"],
                    "weather_code": [3, 61, 1],
                    "temperature_2m_min": [9, 8, 7],
                    "temperature_2m_max": [16, 13, 15],
                    "precipitation_probability_max": [20, 80, 10],
                },
            }
            return _Response(200, body, "application/json")
        page = find_page(self.pages, url)
        if page is None:
            return _Response(404, "", "text/html")
        head = f"<head><title>{page.title}</title></head>"
        html = f"<html>{head}<body><p>{page.text}</p></body></html>"
        return _Response(200, html, "text/html; charset=utf-8")

    async def aclose(self) -> None:
        pass


class _Trash:
    """Corbeille du monde simulé : l'élément est déplacé, jamais supprimé."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.items: list[str] = []

    def __call__(self, path: str) -> None:
        import shutil

        self.items.append(path)
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.move(path, self.folder / f"{len(self.items)}-{Path(path).name}")


@dataclass
class WorldSpec:
    """État de départ d'une tâche (tout est facultatif)."""

    apps: list[str] = field(default_factory=lambda: list(DEFAULT_APPS))
    open_apps: list[str] = field(default_factory=list)
    volume: float = 50
    clipboard: str = ""
    files: dict[str, str] = field(default_factory=dict)  # chemin relatif au dossier perso
    index: list[str] = field(default_factory=list)
    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    search: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    controls: dict[str, list[list[str]]] = field(default_factory=dict)
    shell: dict[str, str] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    weather: dict[str, float] = field(default_factory=dict)  # ville -> température


class SimWorld:
    """Le PC simulé d'une tâche : services, dossier perso, et `ToolContext` prêt à l'emploi."""

    def __init__(self, spec: WorldSpec, root: Path, config: Config) -> None:
        self.root = root
        self.home = root / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        for relative, content in spec.files.items():
            target = self.home / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        pages = {url: Page(**raw) for url, raw in spec.pages.items()}
        self.desktop = SimDesktop(spec.apps, spec.open_apps)
        self.desktop.clipboard = spec.clipboard
        self.audio = SimAudio(spec.volume)
        self.input = SimInput()
        self.shell = SimShell(spec.shell)
        self.notifier = SimNotifier()
        self.browser = SimBrowser(pages)
        self.http = SimHttp(pages, spec.search, spec.weather)
        self.index = SimIndex(spec.index, self.home)
        self.ui = SimUi(
            {app: [(r[0], r[1], r[2] if len(r) > 2 else "") for r in rows]
             for app, rows in spec.controls.items()}
        )  # fmt: skip
        self.trash = _Trash(root / "Corbeille")
        self.spoken: list[str] = []
        data = config.model_dump(mode="json", exclude={"secrets"})
        data["guard"]["allowed_dirs"] = [str(self.home)]
        self.config = Config.model_validate(data)
        self.config._base_dir = root
        memory = LongTermMemory(root / "memory.json")
        for fact in spec.facts:
            memory.remember(fact)

        async def no_wait(seconds: float) -> None:
            await asyncio.sleep(0)

        async def announce(text: str) -> None:
            self.spoken.append(text)

        folders = {
            "desktop": self.home / "Bureau",
            "documents": self.home / "Documents",
            "downloads": self.home / "Téléchargements",
        }
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
        self.ctx = ToolContext(
            self.config,
            desktop=self.desktop,
            audio=self.audio,
            notifier=self.notifier,
            http=self.http,
            search_api_key="sim",
            memory=memory,
            announce=announce,
            clock=lambda: NOW,
            sleep=no_wait,
            guard=Guard(
                self.config, home=self.home, trash=self.trash, clock=lambda: NOW, folders=folders
            ),
            input=self.input,
            shell=self.shell,
            uia=self.ui,
            browser=self.browser,
            index=self.index,
        )
