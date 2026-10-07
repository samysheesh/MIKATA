"""Outils applications et fenêtres (§7) : open_app ★, close_app ★, focus_window ★,
list_windows.

Les noms dits à voix haute sont approximatifs : correspondance floue (`rapidfuzz`) avec
l'index du menu Démarrer (`tools.desktop`) ou les fenêtres ouvertes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, Field

from mikata.tools.base import AppEntry, BaseTool, NoParams, ToolError, WindowInfo
from mikata.tools.registry import register_tool

APP_CUTOFF = 75.0
PROCESS_CUTOFF = 80.0
TITLE_CUTOFF = 90.0
TIE_MARGIN = 3.0  # scores presque égaux : le nom le plus court gagne (« Google Chrome »)


def fold(text: str) -> str:
    """Forme de comparaison : minuscules, sans accents, sans « .exe » ni ponctuation."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\.exe\b", "", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def find_app(query: str, apps: Sequence[AppEntry], cutoff: float = APP_CUTOFF) -> AppEntry | None:
    """Application la plus proche de `query` : nom exact, sinon meilleur score flou."""
    from rapidfuzz import fuzz, process

    wanted = fold(query)
    if not wanted or not apps:
        return None
    names = [fold(app.name) for app in apps]
    for app, name in zip(apps, names, strict=True):
        if name == wanted:
            return app
    matches = process.extract(wanted, names, scorer=fuzz.WRatio, limit=8, score_cutoff=cutoff)
    if not matches:
        return None
    best = max(score for _, score, _ in matches)
    close = [index for _, score, index in matches if score >= best - TIE_MARGIN]
    return apps[min(close, key=lambda index: (len(names[index]), index))]


def find_windows(query: str, windows: Sequence[WindowInfo]) -> list[WindowInfo]:
    """Fenêtres de l'appli la plus proche de `query` (nom du processus, sinon titre)."""
    from rapidfuzz import fuzz

    wanted = fold(query)
    if not wanted or not windows:
        return []
    by_process: dict[str, list[WindowInfo]] = {}
    for window in windows:
        by_process.setdefault(fold(window.process), []).append(window)
    scored = [(fuzz.WRatio(wanted, name), name) for name in by_process if name]
    if scored:
        score, name = max(scored)
        if score >= PROCESS_CUTOFF:
            return by_process[name]
    titled = [(fuzz.partial_ratio(wanted, fold(w.title)), i) for i, w in enumerate(windows)]
    score, index = max(titled)
    return [windows[index]] if score >= TITLE_CUTOFF and len(wanted) >= 3 else []


def open_apps(windows: Sequence[WindowInfo], limit: int = 8) -> list[str]:
    """Noms courts des applis ouvertes (contexte dynamique du LLM, §6)."""
    names: list[str] = []
    for window in windows:
        name = window.process or window.title
        if name and name.casefold() not in (n.casefold() for n in names):
            names.append(name)
    return names[:limit]


class AppName(BaseModel):
    name: str = Field(
        min_length=1,
        description="Nom de l'application tel qu'affiché par Windows (ex. « Spotify », "
        "« Brave », « Steam »), corrigé si la transcription l'a déformé",
    )


@register_tool
class OpenApp(BaseTool[AppName]):
    name = "open_app"
    description = "Ouvre (ou ramène au premier plan) une application ou un jeu installé."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        app = find_app(args.name, await self.ctx.run_blocking(desktop.list_apps))
        if app is None:  # installée depuis le dernier index ?
            app = find_app(args.name, await self.ctx.run_blocking(desktop.list_apps, True))
        if app is None:
            raise ToolError(f"aucune application « {args.name} » dans le menu Démarrer")
        await self.ctx.run_blocking(desktop.launch, app)
        return f"{app.name} ouvert."


@register_tool
class CloseApp(BaseTool[AppName]):
    name = "close_app"
    description = "Ferme une application ouverte (toutes ses fenêtres), comme avec la croix."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        windows = find_windows(args.name, await self.ctx.run_blocking(desktop.windows))
        if not windows:
            raise ToolError(f"aucune fenêtre ouverte pour « {args.name} »")
        for window in windows:
            await self.ctx.run_blocking(desktop.close_window, window)
        label = windows[0].process or windows[0].title
        return f"{label} fermé ({len(windows)} fenêtre(s))."


@register_tool
class FocusWindow(BaseTool[AppName]):
    name = "focus_window"
    description = "Met au premier plan la fenêtre d'une application déjà ouverte."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        windows = find_windows(args.name, await self.ctx.run_blocking(desktop.windows))
        if not windows:
            raise ToolError(f"aucune fenêtre ouverte pour « {args.name} »")
        await self.ctx.run_blocking(desktop.focus_window, windows[0])
        return f"« {windows[0].title} » au premier plan."


@register_tool
class ListWindows(BaseTool[NoParams]):
    name = "list_windows"
    description = "Liste les fenêtres ouvertes (application et titre)."
    params = NoParams

    async def run(self, args: NoParams) -> str:
        desktop = self.ctx.require_desktop()
        windows = await self.ctx.run_blocking(desktop.windows)
        if not windows:
            return "Aucune fenêtre ouverte."
        lines = [f"- {w.process or '?'} : {w.title}" for w in windows[:30]]
        return "Fenêtres ouvertes :\n" + "\n".join(lines)
