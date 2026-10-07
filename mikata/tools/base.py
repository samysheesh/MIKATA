"""Interface d'un outil appelable par le LLM (docs/ARCHITECTURE.md §7 et §9).

terminal=True : si l'outil réussit, pas de second appel LLM (« C'est fait » en cache).

Ajouts phase 4 : `BaseTool` (classe de base des outils enregistrés), `ToolError` (échec
expliqué au LLM) et `ToolContext` (services partagés : bureau Windows, volume, web,
mémoire...). Le code propre à Windows est derrière les interfaces `Desktop`, `AudioControl`
et `Notifier`, simulées dans les tests.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from loguru import logger
from pydantic import BaseModel

from mikata.config import Config

P = TypeVar("P", bound=BaseModel)
R = TypeVar("R")


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str  # en français, 1 phrase, lue par le LLM
    params: type[BaseModel]  # schéma JSON généré via model_json_schema()
    terminal: bool

    async def run(self, args: BaseModel) -> str:
        """Texte court renvoyé au LLM."""
        ...


class NoParams(BaseModel):
    """Outil sans argument."""


class BaseTool(Generic[P]):
    """Classe de base d'un outil : attributs de classe + `run`, instancié avec le contexte."""

    name: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[type[BaseModel]]
    terminal: ClassVar[bool] = False

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    async def run(self, args: P) -> str:
        raise NotImplementedError


class ToolError(Exception):
    """Échec attendu d'un outil (appli introuvable, clé absente...) : message en français
    renvoyé au LLM, qui l'explique à l'utilisateur."""


@dataclass
class AppEntry:
    """Application lançable : raccourci du menu Démarrer ou appli du Store."""

    name: str
    target: str  # chemin d'un .lnk / .url, ou « shell:AppsFolder\\<AppID> »


@dataclass
class WindowInfo:
    handle: int
    title: str
    process: str  # nom de l'exécutable, sans « .exe »


class Desktop(Protocol):
    """Bureau Windows (tools/desktop.py), simulé dans les tests."""

    def list_apps(self, refresh: bool = False) -> list[AppEntry]: ...
    def launch(self, app: AppEntry) -> None: ...
    def open_target(self, target: str) -> None: ...  # URL ou fichier, programme par défaut
    def windows(self) -> list[WindowInfo]: ...
    def close_window(self, window: WindowInfo) -> None: ...
    def focus_window(self, window: WindowInfo) -> None: ...
    def media_key(self, action: str) -> None: ...  # play_pause | next | previous | stop
    def lock(self) -> None: ...
    def clipboard_read(self) -> str: ...
    def clipboard_write(self, text: str) -> None: ...


class AudioControl(Protocol):
    """Volume et sortie audio de Windows (tools/media.py, pycaw), simulé dans les tests."""

    def get_volume(self) -> float: ...  # 0..100
    def set_volume(self, percent: float) -> None: ...
    def set_mute(self, mute: bool) -> None: ...
    def output_devices(self) -> list[tuple[str, str]]: ...  # (identifiant, nom)
    def set_default_output(self, device_id: str) -> None: ...


class Notifier(Protocol):
    def notify(self, title: str, body: str) -> None: ...


@dataclass
class ToolContext:
    """Services partagés par les outils ; un outil absent d'un service répond « indisponible »."""

    config: Config
    desktop: Desktop | None = None
    audio: AudioControl | None = None
    notifier: Notifier | None = None
    http: Any = None  # httpx2.AsyncClient partagé (outils web)
    search_api_key: str | None = None
    memory: Any = None  # agent.memory.LongTermMemory
    announce: Callable[[str], Awaitable[None]] | None = None  # dire un texte (minuteurs)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)  # minuteurs en cours
    clock: Callable[[], datetime] = datetime.now  # remplaçables dans les tests
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)

    async def run_blocking(self, fn: Callable[..., R], *args: Any) -> R:
        """Appel bloquant (Win32, COM, presse-papiers) dans le thread dédié aux outils.

        Un seul thread, initialisé pour COM sous Windows : pycaw et les API du bureau y
        trouvent toujours le même appartement COM.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mikata-tools", initializer=_com_init
            )
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def require_desktop(self) -> Desktop:
        if self.desktop is None:
            raise ToolError("le pilotage du bureau Windows n'est pas disponible ici")
        return self.desktop

    def require_audio(self) -> AudioControl:
        if self.audio is None:
            raise ToolError("le contrôle du volume Windows n'est pas disponible ici")
        return self.audio

    def spawn(self, coro: Awaitable[Any], name: str) -> None:
        """Tâche de fond gardée en vie (minuteurs, rappels)."""
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def aclose(self) -> None:
        """Arrêt : minuteurs annulés, client HTTP fermé, thread des outils arrêté."""
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.http is not None:
            try:
                await self.http.aclose()
            except Exception as exc:
                logger.debug("Fermeture du client HTTP des outils : {}", exc)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def _com_init() -> None:
    if sys.platform != "win32":
        return
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception as exc:  # pycaw dira la vraie erreur au premier appel
        logger.warning("Initialisation COM impossible : {}", exc)
