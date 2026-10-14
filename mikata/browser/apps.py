"""Applis Chromium (Electron, CEF) pilotées en arrière-plan par DevTools (phase 17e).

Spotify, Discord, VS Code... sont des pages web dans Chromium. Lancées avec
`--remote-debugging-port`, elles se pilotent par le protocole DevTools, comme le navigateur
de travail : leur arbre d'accessibilité (celui que Chromium donne aussi à UIA, lu ici en
~0,1 s au lieu de plusieurs secondes) et des clics en JavaScript qui prennent même fenêtre
cachée. L'appli tourne cachée (SW_HIDE) : rien n'apparaît jamais à l'écran. Mesuré sur
Spotify le 14/10 : lancement, recherche et lecture sans aucune fenêtre visible (relevés
toutes les 10 ms), alors qu'en UIA Spotify ignorait « Play » fenêtre réduite.

Seulement les applis listées dans `uia.devtools_apps`, par choix de l'utilisateur : tant
qu'une appli tourne ainsi, tout programme du PC peut la piloter et lire sa session par ce
port. Il n'écoute que sur 127.0.0.1, son numéro est tiré au hasard, et Chromium refuse les
pages web (en-tête Origin, sans `--remote-allow-origins`).

Une appli listée est :
- fermée : lancée cachée avec le port ;
- ouverte avec le port (lancée par MIKATA, même avant un redémarrage de MIKATA : le port se
  relit dans sa ligne de commande) : rejointe ;
- ouverte sans le port : relancée cachée avec, seulement si elle ne joue aucun son et n'est
  pas la fenêtre devant l'utilisateur (`uia.devtools_relaunch`) ; sinon, repli sur UIA
  (fenêtre réduite, montrée le temps d'un clic, comme avant).

L'utilisateur la retrouve comme d'habitude : son raccourci la remet à l'écran (vérifié sur
Spotify). Méthodes bloquantes : appelées depuis le thread des outils `uia_*`.
"""

from __future__ import annotations

import contextlib
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.browser.cdp import CdpError, CdpSyncClient, Target, list_targets_sync
from mikata.tools.base import Desktop, ToolError, UiWindowInfo

PORT_FLAG = "--remote-debugging-port="
_PORT_RE = re.compile(r"^--remote-debugging-port=(\d{2,5})$")
# Cibles internes de Chromium (omnibox de CEF, outils de développement) : jamais la page.
_INTERNAL_URLS = ("chrome://", "chrome-untrusted://", "devtools://", "chrome-extension://")
# Résumé de la page : vide tant qu'elle n'est pas chargée ; ensuite, le nombre d'éléments et
# la taille du texte, arrondis. Deux résumés identiques à 0,5 s d'écart : l'appli a fini de se
# construire. Mesuré sur Spotify : texte affiché à +0,5 s mais barre de lecture (« Mute »)
# absente, page stable à +2 s ; agir avant, c'était cliquer dans le vide.
_READY_JS = (
    "(() => { if (document.readyState !== 'complete' || !document.body) return ''; "
    "const text = document.body.innerText.trim(); if (!text) return ''; "
    "return Math.round(document.getElementsByTagName('*').length / 50) + '|' "
    "+ Math.round(text.length / 200); })()"
)
JOIN_TIMEOUT_S = 4.0  # une appli déjà lancée avec le port répond tout de suite
BUSY_LOG_S = 60.0  # « occupée, pas relancée » : une ligne de journal par minute au plus
_SW_HIDE = 0
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000  # l'appli survit à l'arrêt de la tâche MIKATA


def fold(text: str) -> str:
    """Minuscules sans accents ni « .exe » : « Spotify.exe » vaut « spotify »."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c)).strip()
    return text.removesuffix(".exe").strip()


def debug_port(cmdlines: list[list[str]]) -> int | None:
    """Le port DevTools d'une appli, lu dans les lignes de commande de ses processus."""
    for cmdline in cmdlines:
        for arg in cmdline:
            match = _PORT_RE.match(arg)
            if match:
                return int(match.group(1))
    return None


def main_page(targets: list[Target]) -> Target | None:
    """La page de l'appli parmi les cibles DevTools : pas l'omnibox ni les outils internes."""
    for target in targets:
        if target.kind == "page" and not target.url.startswith(_INTERNAL_URLS):
            return target
    return None


def free_port() -> int:
    """Un port libre de 127.0.0.1, tiré par le système."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _Session:
    process: str  # nom réel du processus (« Spotify »)
    port: int
    title: str
    client: CdpSyncClient


class DevtoolsApps:
    """Les applis listées dans `uia.devtools_apps`, tenues cachées et pilotées par DevTools."""

    def __init__(
        self,
        names: list[str],
        desktop: Desktop | None,
        relaunch: bool = True,
        launch_timeout_s: float = 25.0,
        command_timeout_s: float = 10.0,
    ) -> None:
        self._names = [name.strip() for name in names if name.strip()]
        self._desktop = desktop
        self._relaunch = relaunch
        self._launch_timeout = launch_timeout_s
        self._command_timeout = command_timeout_s
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}  # clé : fold(nom listé)
        self._exes: dict[str, str] = {}  # programmes déjà vus, pour relancer une appli fermée
        self._busy_logged: dict[str, float] = {}

    # -- appariement ----------------------------------------------------------------------

    def serves(self, name: str | None) -> str | None:
        """Le nom listé qui désigne `name` (processus, avec ou sans « .exe », ou titre de
        fenêtre « Spotify Free »), sinon None."""
        if not name:
            return None
        wanted = fold(name)
        for app in self._names:
            key = fold(app)
            if key and (wanted == key or wanted.startswith(key + " ")):
                return app
        return None

    # -- fenêtre et commandes -------------------------------------------------------------

    def window(self, app: str) -> UiWindowInfo | None:
        """La fenêtre de l'appli listée `app`, lue par DevTools ; l'appli est lancée (ou
        relancée) cachée s'il le faut. None : pas pilotable ainsi -> UIA, comme avant."""
        with self._lock:
            session = self._session(app)
            if session is None:
                return None
            return UiWindowInfo(session.title or session.process, session.process, 0, "devtools")

    def call(self, process: str, method: str, **params: Any) -> dict[str, Any]:
        """Une commande DevTools vers l'appli ; une connexion tombée (appli relancée,
        page rechargée) est rétablie une fois."""
        app = self.serves(process)
        if app is None:
            raise ToolError(f"« {process} » n'est pas une appli pilotée en arrière-plan")
        with self._lock:
            for _ in range(2):
                session = self._session(app)
                if session is None:
                    raise ToolError(f"{app} ne se laisse plus piloter en arrière-plan")
                try:
                    return session.client.command(method, **params)
                except CdpError as exc:
                    if session.client.connected:
                        raise ToolError(f"{app} : {exc}") from exc
                    self._drop(app)  # connexion perdue : on la rétablit
            raise ToolError(f"{app} a été fermé pendant l'action")

    def evaluate(self, process: str, expression: str) -> Any:
        """Du JavaScript fixe, écrit dans MIKATA (jamais fourni par le modèle)."""
        result = self.call(process, "Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            raise ToolError(f"{process} : lecture impossible")
        return result.get("result", {}).get("value")

    def background_apps(self) -> list[str]:
        """Les applis listées qui tournent cachées avec le port (pour le contexte du modèle :
        elles n'ont aucune fenêtre visible, il ne les verrait pas ouvertes)."""
        found: list[str] = []
        for app in self._names:
            procs = _processes(app)
            if (
                procs
                and debug_port(_cmdlines(procs)) is not None
                and not _visible_windows({p.pid for p in procs})
            ):
                found.append(app)
        return found

    def quit(self, name: str) -> str | None:
        """Ferme une appli listée qui tourne cachée (« ferme Spotify » : elle n'a aucune
        fenêtre à fermer). None : ce n'est pas l'une d'elles, ou elle est visible (la
        fermeture normale s'en charge)."""
        app = self.serves(name)
        if app is None:
            return None
        with self._lock:
            procs = _processes(app)
            if not procs or debug_port(_cmdlines(procs)) is None:
                return None
            if _visible_windows({p.pid for p in procs}):
                return None
            self._drop(app)
            _terminate(procs)
        logger.info("{} (caché) fermé à la demande.", app)
        return f"{app} fermé."

    def aclose(self) -> None:
        """Arrêt de MIKATA : les connexions se ferment, les applis continuent de tourner."""
        with self._lock:
            for app in list(self._sessions):
                self._drop(app)

    # -- interne --------------------------------------------------------------------------

    def _drop(self, app: str) -> None:
        session = self._sessions.pop(fold(app), None)
        if session is not None:
            session.client.close()

    def _session(self, app: str) -> _Session | None:
        session = self._sessions.get(fold(app))
        if session is not None and session.client.connected:
            return session
        self._drop(app)
        ready = self._port(app)  # (Windows : UiController ne crée ce gestionnaire que là)
        if ready is None:
            return None
        port, process, launched = ready
        timeout = self._launch_timeout if launched else JOIN_TIMEOUT_S
        found = self._page(app, port, timeout, hide=launched)
        if found is None:
            logger.warning("{} : page introuvable sur le port DevTools {}.", app, port)
            return None
        target, client = found
        session = _Session(process, port, target.title, client)
        self._sessions[fold(app)] = session
        return session

    def _port(self, app: str) -> tuple[int, str, bool] | None:
        """(port, processus, lancée à l'instant) de l'appli listée `app`, lancée ou relancée
        cachée s'il le faut ; None : elle n'est pas pilotable ainsi pour l'instant."""
        procs = _processes(app)
        if procs:
            process = _stem(procs)
            exe = _main_exe(procs)
            if exe:
                self._exes[fold(app)] = exe
            port = debug_port(_cmdlines(procs))
            if port is not None:
                return port, process, False
            if not self._relaunch or not exe:
                return None
            busy = _busy({p.pid for p in procs})
            if busy:
                now = time.monotonic()
                if now - self._busy_logged.get(app, -BUSY_LOG_S) >= BUSY_LOG_S:
                    self._busy_logged[app] = now
                    logger.info(
                        "{} ouvert sans l'accès DevTools et {} : pas relancé (UIA).", app, busy
                    )
                return None
            logger.info("{} relancé caché avec l'accès DevTools (il ne jouait rien).", app)
            _terminate(procs)
        else:
            exe = self._exes.get(fold(app)) or self._shortcut_exe(app)
            if not exe:
                logger.info("{} : programme introuvable, pas de lancement caché (UIA).", app)
                return None
            process = Path(exe).stem
        port = free_port()
        _launch_hidden(exe, port)
        logger.info("{} lancé caché (DevTools, 127.0.0.1:{}).", app, port)
        return port, process, True

    def _page(
        self, app: str, port: int, timeout_s: float, hide: bool
    ) -> tuple[Target, CdpSyncClient] | None:
        """La page de l'appli, une fois construite (lancée à l'instant : deux résumés
        identiques à 0,5 s). Pendant un lancement (`hide`), toute fenêtre de l'appli qui
        s'afficherait d'elle-même est recachée aussitôt ; une appli rejointe garde ses
        fenêtres telles quelles (l'utilisateur l'a peut-être affichée)."""
        deadline = time.monotonic() + timeout_s
        client: CdpSyncClient | None = None
        target: Target | None = None
        last = ""
        while True:
            if hide:
                _hide_windows({p.pid for p in _processes(app)})
            if client is None:
                with contextlib.suppress(CdpError):
                    target = main_page(list_targets_sync(port, 1.0))
                if target is not None:
                    client = CdpSyncClient(target.ws_url, timeout_s=self._command_timeout)
                    try:
                        client.connect()
                    except CdpError:
                        client = None
            if client is not None and target is not None:
                try:
                    summary = str(client.evaluate(_READY_JS) or "")
                except CdpError:
                    client.close()
                    client, summary = None, ""
                # Lancée à l'instant : il faut qu'elle ait fini de se construire ; rejointe,
                # elle est déjà prête (et une page qui joue bouge sans cesse).
                if client is not None and summary and (summary == last or not hide):
                    if hide:
                        _hide_windows({p.pid for p in _processes(app)})
                    return target, client
                last = summary
            if time.monotonic() >= deadline:
                if client is not None:
                    client.close()
                return None
            time.sleep(0.5 if client is not None else 0.25)

    def _shortcut_exe(self, app: str) -> str | None:
        """Le programme d'une appli fermée : la cible de son raccourci du menu Démarrer
        (une appli du Store n'en a pas : None, elle reste pilotée par UIA)."""
        if self._desktop is None:
            return None
        from mikata.tools.apps import find_app

        try:
            entry = find_app(app, self._desktop.list_apps())
        except Exception:
            return None
        if entry is None or not entry.target.casefold().endswith(".lnk"):
            return None
        try:
            import win32com.client  # type: ignore[import-untyped,unused-ignore]

            shell = win32com.client.Dispatch("WScript.Shell")
            target = str(shell.CreateShortCut(entry.target).Targetpath)
        except Exception:
            return None
        return target if target.casefold().endswith(".exe") and Path(target).is_file() else None


# -- processus et fenêtres (Windows) ----------------------------------------------------------


def _processes(app: str) -> list[Any]:
    """Les processus de l'appli (« Spotify.exe » et ses processus d'aide, même programme)."""
    import psutil

    wanted = fold(app)
    found: list[Any] = []
    for proc in psutil.process_iter(["name"]):
        name = proc.info.get("name") or ""
        if fold(name) == wanted:
            found.append(proc)
    return found


def _cmdlines(procs: list[Any]) -> list[list[str]]:
    lines: list[list[str]] = []
    for proc in procs:
        with contextlib.suppress(Exception):
            lines.append(list(proc.cmdline()))
    return lines


def _main_exe(procs: list[Any]) -> str:
    """Le programme du processus principal (celui sans « --type= »)."""
    for proc in procs:
        with contextlib.suppress(Exception):
            if not any(arg.startswith("--type=") for arg in proc.cmdline()):
                return str(proc.exe())
    return ""


def _stem(procs: list[Any]) -> str:
    for proc in procs:
        with contextlib.suppress(Exception):
            return Path(str(proc.info.get("name") or proc.name())).stem
    return ""


def _busy(pids: set[int]) -> str | None:
    """Pourquoi l'appli ne doit pas être relancée maintenant, ou None : elle est devant
    l'utilisateur, ou elle joue du son (ou on ne sait pas : dans le doute, on n'y touche pas)."""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    if pid.value in pids:
        return "devant l'utilisateur"
    playing = _plays_sound(pids)
    if playing is None:
        return "son illisible"
    return "en train de jouer" if playing else None


def _plays_sound(pids: set[int]) -> bool | None:
    """Une session audio active (son en cours, même coupé dans l'appli) ; None si illisible."""
    try:
        from pycaw.utils import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            if session.ProcessId in pids and session.State == 1:  # AudioSessionStateActive
                return True
    except Exception:
        return None
    return False


def _terminate(procs: list[Any]) -> None:
    """Arrête l'appli (tous ses processus) et attend leur fin, 5 s au plus."""
    import psutil

    for proc in procs:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    with contextlib.suppress(Exception):
        psutil.wait_procs(procs, timeout=5.0)


def _launch_hidden(exe: str, port: int) -> None:
    """Lance l'appli cachée avec le port DevTools, détachée de MIKATA (elle continue de
    tourner si MIKATA s'arrête)."""
    startup = subprocess.STARTUPINFO()  # type: ignore[attr-defined,unused-ignore]
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined,unused-ignore]
    startup.wShowWindow = _SW_HIDE
    argv = [exe, f"{PORT_FLAG}{port}"]
    flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    for extra in (_CREATE_BREAKAWAY_FROM_JOB, 0):  # sans, si le job de la tâche l'interdit
        try:
            subprocess.Popen(
                argv,
                cwd=str(Path(exe).parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startup,
                creationflags=flags | extra,
            )
            return
        except OSError as exc:
            if not extra:
                raise ToolError(f"lancement caché impossible : {exc}") from exc


def _top_windows(pids: set[int]) -> list[int]:
    """Fenêtres principales (sans propriétaire, plus grandes qu'une icône) des processus."""
    if sys.platform != "win32" or not pids:
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)  # type: ignore[attr-defined,unused-ignore]
    def visit(hwnd: int, _: int) -> bool:
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and not user32.GetWindow(hwnd, 4):  # GW_OWNER
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if rect.right - rect.left >= 60 and rect.bottom - rect.top >= 60:
                found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def _visible_windows(pids: set[int]) -> list[int]:
    if sys.platform != "win32":
        return []
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    return [h for h in _top_windows(pids) if user32.IsWindowVisible(h)]


def _hide_windows(pids: set[int]) -> None:
    """Recache les fenêtres visibles de l'appli que l'on vient de lancer cachée."""
    if sys.platform != "win32":
        return
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    for hwnd in _visible_windows(pids):
        user32.ShowWindow(hwnd, _SW_HIDE)
