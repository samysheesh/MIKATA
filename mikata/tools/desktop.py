"""Bureau Windows pour les outils (ajout phase 4) : applications, fenêtres, touches média,
verrouillage, presse-papiers.

Implémente l'interface `tools.base.Desktop`. Seul ce module (et `media.PycawAudio`,
`ui.notify`) appelle des API Windows ; imports paresseux, simulé dans les tests. Les
fonctions d'analyse (`scan_start_menu`, `parse_start_apps`, `merge_apps`) sont pures et
testées sous Linux.

Index des applications : raccourcis `.lnk` / `.url` du menu Démarrer (utilisateur + tous
les utilisateurs) puis `Get-StartApps` (applis du Store, jeux...). Il est construit au
premier besoin, gardé en mémoire et reconstruit si une appli demandée est introuvable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.tools.base import AppEntry, ToolError, WindowInfo

SHORTCUT_SUFFIXES = (".lnk", ".url")
# Raccourcis annexes du menu Démarrer, jamais ce qu'on veut ouvrir.
_SKIP_RE = re.compile(
    r"\b(uninstall|désinstall\w*|desinstall\w*|readme|read me|lisez[- ]moi|license|licence|"
    r"release notes|notes de version|help|aide|documentation|manual|manuel|website|site web)\b",
    re.IGNORECASE,
)
GET_START_APPS = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
)
APP_INDEX_MAX_AGE_S = 600.0
CREATE_NO_WINDOW = 0x08000000  # pas de console qui clignote sous pythonw

# Dossiers personnels (ajout phase 5) : leur vrai chemin, qui peut être dans OneDrive.
KNOWN_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
}

# Touches média virtuelles (Win32 VK_MEDIA_*).
MEDIA_KEYS = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}


def known_folders() -> dict[str, Path]:
    """Vrais chemins du Bureau, des Documents, Téléchargements... (SHGetKnownFolderPath) ;
    un dossier introuvable est simplement absent du résultat."""
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined,unused-ignore]
    ole32 = ctypes.windll.ole32  # type: ignore[attr-defined,unused-ignore]
    folders: dict[str, Path] = {}
    for name, text in KNOWN_FOLDER_IDS.items():
        u = uuid.UUID(text)
        guid = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        found = ctypes.c_wchar_p()
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(found)) == 0:
            if found.value:
                folders[name] = Path(found.value)
            ole32.CoTaskMemFree(found)
    return folders


def start_menu_dirs() -> list[Path]:
    dirs = []
    for env in ("APPDATA", "PROGRAMDATA"):
        root = os.environ.get(env)
        if root:
            dirs.append(Path(root) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def scan_start_menu(dirs: Iterable[Path]) -> list[AppEntry]:
    """Raccourcis du menu Démarrer, sans désinstalleurs ni pages d'aide."""
    apps: list[AppEntry] = []
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SHORTCUT_SUFFIXES or _SKIP_RE.search(path.stem):
                continue
            apps.append(AppEntry(path.stem, str(path)))
    return apps


def parse_start_apps(output: str) -> list[AppEntry]:
    """Sortie JSON de `Get-StartApps` -> applications lançables par `shell:AppsFolder`."""
    output = output.strip().lstrip("﻿")
    if not output:
        return []
    data = json.loads(output)
    if isinstance(data, dict):  # une seule appli : PowerShell ne renvoie pas de liste
        data = [data]
    apps = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name, app_id = str(item.get("Name") or "").strip(), str(item.get("AppID") or "").strip()
        if name and app_id and not _SKIP_RE.search(name):
            apps.append(AppEntry(name, f"shell:AppsFolder\\{app_id}"))
    return apps


def merge_apps(*sources: Iterable[AppEntry]) -> list[AppEntry]:
    """Une entrée par nom (insensible à la casse) ; la première source l'emporte."""
    seen: set[str] = set()
    merged: list[AppEntry] = []
    for source in sources:
        for app in source:
            key = app.name.casefold()
            if key not in seen:
                seen.add(key)
                merged.append(app)
    return merged


def _run_powershell(command: str, timeout: float = 10.0) -> str:
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        timeout=timeout,
        check=True,
        creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return completed.stdout.decode("utf-8", errors="replace")


APP_FRAME_HOST = "applicationframehost"

# -- macOS (darwin) -------------------------------------------------------------------
# Dossiers d'applications inspectés par MacDesktop.list_apps, avant le repli `mdfind`.
MAC_APP_DIRS = ("/Applications", "/System/Applications", "~/Applications")


def _run_osascript(script: str, timeout: float = 10.0) -> str:
    """Sortie texte d'un script AppleScript (`osascript -e`), sans fenêtre."""
    completed = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def scan_applications_dirs(dirs: Iterable[str | Path] | None = None) -> list[AppEntry]:
    """Paquets `.app` des dossiers d'applications macOS (nom sans extension -> chemin)."""
    roots = [Path(d).expanduser() for d in (dirs if dirs is not None else MAC_APP_DIRS)]
    apps: list[AppEntry] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.app")):
            apps.append(AppEntry(path.stem, str(path)))
    return apps


def parse_mdfind(output: str) -> list[AppEntry]:
    """Sortie de `mdfind 'kMDItemKind == Application'` -> applications lançables."""
    apps: list[AppEntry] = []
    for line in output.splitlines():
        line = line.strip()
        if line.endswith(".app"):
            apps.append(AppEntry(Path(line).stem, line))
    return apps


def _process_name(pid: int) -> str:
    """Nom du programme d'un processus, sans « .exe » (« Notepad ») ; "" si inconnu.

    psutil d'abord ; à défaut (ajout phase 6), `QueryFullProcessImageNameW` avec le droit
    minimal PROCESS_QUERY_LIMITED_INFORMATION, qui marche aussi pour les applis empaquetées
    (Bloc-notes de Windows 11).
    """
    import psutil

    try:
        return Path(psutil.Process(pid).name()).stem
    except (psutil.Error, OSError):
        pass
    if sys.platform != "win32":
        return ""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return Path(buffer.value).stem
    finally:
        kernel32.CloseHandle(handle)


def _hosted_process(hwnd: int, host_pid: int) -> str:
    """Appli UWP affichée dans un cadre ApplicationFrameHost : processus de la fenêtre
    enfant qui n'appartient pas au cadre (« CalculatorApp »...) ; "" si introuvable."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    found: list[str] = []

    def visit(child: int, _: Any) -> bool:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(child, ctypes.byref(pid))
        if pid.value and pid.value != host_pid:
            found.append(_process_name(pid.value))
            return False  # arrête l'énumération
        return True

    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(visit)
    user32.EnumChildWindows(hwnd, callback, 0)
    return next((name for name in found if name), "")


# -- liens d'applis et arrière-plan (phase 17a) --------------------------------------------

# Protocoles jamais ouverts par MIKATA : failles connues ou ouverture de documents distants
# (ms-msdt : « Follina », search-ms, protocoles Office), ou trop larges (fichiers, scripts).
DENIED_SCHEMES = frozenset(
    {
        "ms-msdt", "search-ms", "search", "ms-officecmd", "ms-appinstaller", "ms-word",
        "ms-excel", "ms-powerpoint", "ms-visio", "ms-access", "ms-project", "ms-publisher",
        "ms-spd", "ms-infopath", "ms-cxh", "ms-cxh-full", "file", "javascript", "vbscript",
        "res", "mk", "its", "ms-its", "mhtml", "hcp", "shell", "ldap", "microsoft-edge",
        "http", "https", "ftp",
    }
)  # fmt: skip
# Protocoles du système sans programme déclaré dans le registre classique, sûrs et utiles.
SYSTEM_LINKS = {"ms-settings": ("Paramètres Windows", "SystemSettings")}


def _registered_links() -> dict[str, tuple[str, str]]:
    """Schémas déclarés « URL Protocol » (HKCU puis HKLM) dont le programme est une appli
    installée : schéma -> (appli, processus). Pas les composants de Windows, ni les
    navigateurs (leurs pages s'ouvrent par `open_url`), ni les associations internes
    (« explorer.assocprotocol.search-ms »), ni les crochets de jeux (« discord-1443... »)."""
    import winreg

    found: dict[str, tuple[str, str]] = dict(SYSTEM_LINKS)
    windows_dir = os.environ.get("SYSTEMROOT", r"C:\Windows").casefold()
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            classes = winreg.OpenKey(hive, r"Software\Classes")
        except OSError:
            continue
        with classes:
            index = 0
            while True:
                try:
                    scheme = winreg.EnumKey(classes, index)
                except OSError:
                    break
                index += 1
                key = scheme.casefold()
                if key in found or not _linkable(key):
                    continue
                try:
                    with winreg.OpenKey(classes, scheme) as entry:
                        winreg.QueryValueEx(entry, "URL Protocol")
                    with winreg.OpenKey(classes, rf"{scheme}\shell\open\command") as cmd:
                        command = os.path.expandvars(str(winreg.QueryValueEx(cmd, "")[0]))
                except OSError:
                    continue
                match = re.match(r'\s*"([^"]+)"|\s*(\S+)', command)
                exe = (match.group(1) or match.group(2)) if match else ""
                stem = Path(exe).stem
                if (
                    not exe.casefold().endswith(".exe")
                    or exe.casefold().startswith(windows_dir)
                    or stem.casefold() in BROWSER_PROGRAMS
                ):
                    continue
                found[key] = (stem, stem)
    return found


BROWSER_PROGRAMS = frozenset({"brave", "chrome", "msedge", "iexplore", "firefox", "opera"})


def _linkable(scheme: str) -> bool:
    """Un schéma qu'une appli peut déclarer et que MIKATA peut ouvrir."""
    return (
        scheme not in DENIED_SCHEMES
        and re.fullmatch(r"[a-z][a-z0-9+.-]{1,40}", scheme) is not None
        and "assoc" not in scheme  # associations internes (dont « ...search-ms »)
        and not any(denied in scheme for denied in ("search-ms", "ms-msdt"))
        and re.search(r"-\d{6,}$", scheme) is None  # crochets de jeux
    )


def _keep_in_background(process: str, previous: int, timeout_s: float = 10.0) -> None:
    """L'appli ouverte pour y agir passe devant (Spotify ignore la demande d'ouverture
    minimisée, constaté le 14/10) : dès qu'une fenêtre de `process` prend le premier plan,
    elle est minimisée et la fenêtre de l'utilisateur revient devant. L'appli reste
    pilotable minimisée (arbre d'accessibilité, Invoke : vérifié sur Spotify). Rien d'autre
    n'est jamais touché : si l'utilisateur change lui-même de fenêtre, on n'agit pas."""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    wanted = process.casefold()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hwnd = user32.GetForegroundWindow()
        if hwnd and hwnd != previous and _window_process(hwnd).casefold() == wanted:
            user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
            if previous and user32.IsWindow(previous):
                user32.keybd_event(0x12, 0, 0, 0)  # Alt : lève le verrou de premier plan
                user32.keybd_event(0x12, 0, 0x0002, 0)
                user32.SetForegroundWindow(previous)
            return
        time.sleep(0.05)


def _window_process(hwnd: int) -> str:
    """Nom du programme d'une fenêtre (« Spotify »), "" si inconnu."""
    import ctypes

    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))  # type: ignore[attr-defined,unused-ignore]
    return _process_name(pid.value) if pid.value else ""


class WindowsDesktop:
    """Implémentation Windows de `tools.base.Desktop` (API Win32 via ctypes, pyperclip)."""

    def __init__(self) -> None:
        self._apps: list[AppEntry] = []
        self._apps_at = float("-inf")
        self._apps_lock = threading.Lock()
        self._links: dict[str, tuple[str, str]] | None = None

    # -- applications -----------------------------------------------------------------

    def list_apps(self, refresh: bool = False) -> list[AppEntry]:
        with self._apps_lock:
            stale = time.monotonic() - self._apps_at > APP_INDEX_MAX_AGE_S
            if refresh or stale or not self._apps:
                start = time.monotonic()
                shortcuts = scan_start_menu(start_menu_dirs())
                try:
                    store = parse_start_apps(_run_powershell(GET_START_APPS))
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    logger.warning("Get-StartApps indisponible : {}", exc)
                    store = []
                self._apps = merge_apps(shortcuts, store)
                self._apps_at = time.monotonic()
                logger.info(
                    "Index des applications : {} entrées en {:.0f} ms.",
                    len(self._apps),
                    (self._apps_at - start) * 1000,
                )
            return list(self._apps)

    def cached_apps(self) -> list[AppEntry]:
        """Index déjà construit (vide avant le premier `list_apps`) ; jamais bloquant."""
        return list(self._apps)

    def launch(self, app: AppEntry) -> None:
        self.open_target(app.target)

    def open_target(
        self, target: str, background: bool = False, process: str | None = None
    ) -> None:
        """Ouvre une URL, un lien d'appli ou un fichier. `background` (phase 17a) : l'appli
        `process` est ouverte pour y agir, sans rester devant l'utilisateur."""
        import ctypes

        previous = ctypes.windll.user32.GetForegroundWindow() if background else 0  # type: ignore[attr-defined,unused-ignore]
        try:
            os.startfile(target)  # type: ignore[attr-defined,unused-ignore]
        except OSError as exc:
            raise ToolError(f"Windows n'a pas pu ouvrir « {target} » : {exc}") from exc
        if background and process:
            threading.Thread(
                target=_keep_in_background,
                args=(process, previous),
                name="mikata-arriere-plan",
                daemon=True,
            ).start()

    def link_schemes(self) -> dict[str, tuple[str, str]]:
        """Phase 17a : les liens que savent ouvrir les applis installées (protocoles
        déclarés dans le registre) : schéma -> (nom de l'appli, nom de son processus)."""
        with self._apps_lock:
            if self._links is None:
                self._links = _registered_links()
            return dict(self._links)

    def open_folder(self, path: str, select: bool = False) -> None:
        """L'Explorateur sur `path` : le dossier s'ouvre, ou le fichier est sélectionné."""
        import subprocess

        argv = ["explorer.exe"] + (["/select,", str(path)] if select else [str(path)])
        try:
            subprocess.run(
                argv,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
        except OSError as exc:
            raise ToolError(f"Windows n'a pas pu ouvrir « {path} » : {exc}") from exc

    # -- fenêtres -----------------------------------------------------------------------

    def windows(self) -> list[WindowInfo]:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined,unused-ignore]
        gwl_exstyle, ws_ex_toolwindow, gw_owner, dwmwa_cloaked = -20, 0x80, 4, 14
        found: list[WindowInfo] = []

        def visit(hwnd: int, _: Any) -> bool:
            if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, gw_owner):
                return True
            if user32.GetWindowLongW(hwnd, gwl_exstyle) & ws_ex_toolwindow:
                return True
            cloaked = wintypes.DWORD()
            dwmapi.DwmGetWindowAttribute(
                hwnd, dwmwa_cloaked, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
            )
            if cloaked.value:  # fenêtres d'applis du Store en arrière-plan
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if not length:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = _process_name(pid.value)
            if process.casefold() == APP_FRAME_HOST:  # Calculatrice, Paramètres... (UWP)
                process = _hosted_process(int(hwnd), pid.value) or process
            found.append(WindowInfo(int(hwnd), buffer.value, process))
            return True

        callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(visit)
        user32.EnumWindows(callback, 0)
        return [w for w in found if w.process.casefold() not in ("explorer", "mikata")]

    def foreground_window(self) -> WindowInfo | None:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined,unused-ignore]
        if not hwnd:
            return None
        for window in self.windows():
            if window.handle == hwnd:
                return window
        return None  # bureau, barre des tâches, fenêtre de MIKATA : rien à fermer

    def close_window(self, window: WindowInfo) -> None:
        import ctypes

        wm_close = 0x0010
        ctypes.windll.user32.PostMessageW(window.handle, wm_close, 0, 0)  # type: ignore[attr-defined,unused-ignore]

    def focus_window(self, window: WindowInfo) -> None:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        sw_restore, vk_menu, keyup = 9, 0x12, 0x0002
        if user32.IsIconic(window.handle):
            user32.ShowWindow(window.handle, sw_restore)
        # Windows refuse le premier plan à un processus sans saisie récente : une pression
        # de la touche Alt simulée lève ce verrou.
        user32.keybd_event(vk_menu, 0, 0, 0)
        user32.keybd_event(vk_menu, 0, keyup, 0)
        if not user32.SetForegroundWindow(window.handle):
            raise ToolError(f"Windows a refusé de mettre « {window.title} » au premier plan")

    # -- divers -------------------------------------------------------------------------

    def media_key(self, action: str) -> None:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
        key = MEDIA_KEYS[action]
        extended, keyup = 0x0001, 0x0002
        user32.keybd_event(key, 0, extended, 0)
        user32.keybd_event(key, 0, extended | keyup, 0)

    def lock(self) -> None:
        import ctypes

        if not ctypes.windll.user32.LockWorkStation():  # type: ignore[attr-defined,unused-ignore]
            raise ToolError("Windows a refusé de verrouiller la session")

    def power(self, action: str) -> None:
        """Extinction, redémarrage (shutdown.exe, droits normaux) ou veille (powrprof)."""
        if action == "sleep":
            import ctypes

            # SetSuspendState(hibernate=False, force=False, wakeup_events_disabled=False)
            if not ctypes.windll.powrprof.SetSuspendState(0, 0, 0):  # type: ignore[attr-defined,unused-ignore]
                raise ToolError("Windows a refusé la mise en veille")
            return
        flag = {"shutdown": "/s", "restart": "/r"}[action]
        try:
            subprocess.run(
                ["shutdown.exe", flag, "/t", "0"],
                check=True,
                capture_output=True,
                timeout=10,
                creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"Windows a refusé : {exc}") from exc

    def clipboard_read(self) -> str:
        import pyperclip

        return str(pyperclip.paste() or "")

    def clipboard_write(self, text: str) -> None:
        import pyperclip

        pyperclip.copy(text)


class MacDesktop:
    """Implémentation macOS de `tools.base.Desktop` (NSWorkspace/open(1), osascript).

    Même noms et signatures de méthodes que `WindowsDesktop` (protocole
    `tools.base.Desktop`) : le reste du code ne distingue que par `sys.platform`.
    Imports lourds (pyobjc) paresseux, dans les fonctions.
    """

    def __init__(self) -> None:
        self._apps: list[AppEntry] = []
        self._apps_at = float("-inf")
        self._apps_lock = threading.Lock()

    # -- applications -----------------------------------------------------------------

    def list_apps(self, refresh: bool = False) -> list[AppEntry]:
        with self._apps_lock:
            stale = time.monotonic() - self._apps_at > APP_INDEX_MAX_AGE_S
            if refresh or stale or not self._apps:
                start = time.monotonic()
                local = scan_applications_dirs()
                try:
                    completed = subprocess.run(
                        ["mdfind", "kMDItemKind == 'Application'"],
                        capture_output=True,
                        timeout=15.0,
                        check=True,
                    )
                    spotlight = parse_mdfind(completed.stdout.decode("utf-8", errors="replace"))
                except (OSError, subprocess.SubprocessError) as exc:
                    logger.warning("mdfind indisponible : {}", exc)
                    spotlight = []
                self._apps = merge_apps(local, spotlight)
                self._apps_at = time.monotonic()
                logger.info(
                    "Index des applications : {} entrées en {:.0f} ms.",
                    len(self._apps),
                    (self._apps_at - start) * 1000,
                )
            return list(self._apps)

    def cached_apps(self) -> list[AppEntry]:
        """Index déjà construit (vide avant le premier `list_apps`) ; jamais bloquant."""
        return list(self._apps)

    def launch(self, app: AppEntry) -> None:
        self.open_target(app.target)

    def link_schemes(self) -> dict[str, tuple[str, str]]:
        return {}  # phase 17a : découverte des liens d'applis pas encore portée sur macOS

    def open_target(
        self, target: str, background: bool = False, process: str | None = None
    ) -> None:
        """Ouvre une appli, un fichier ou une URL avec l'appli par défaut (open(1))."""
        if target.startswith(("http://", "https://")) or Path(target).exists():
            argv = ["open", target]
        else:  # nom d'appli sans chemin (index) : laisse open(1) la résoudre
            argv = ["open", "-a", target]
        try:
            subprocess.run(argv, check=True, capture_output=True, timeout=15.0)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"macOS n'a pas pu ouvrir « {target} » : {exc}") from exc

    def launch_bundle(self, bundle_path: str) -> None:
        """Ouvre un paquet `.app` via NSWorkspace (pyobjc, import paresseux)."""
        from AppKit import NSWorkspace
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(bundle_path)
        if not NSWorkspace.sharedWorkspace().launchApplicationAtURL_options_configuration_error_(
            url, 0, None, None
        )[0]:
            raise ToolError(f"macOS n'a pas pu ouvrir « {bundle_path} »")

    def open_folder(self, path: str, select: bool = False) -> None:
        """Le Finder sur `path` : le dossier s'ouvre, ou le fichier est sélectionné."""
        argv = ["open", "-R", str(path)] if select else ["open", str(path)]
        try:
            subprocess.run(argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise ToolError(f"macOS n'a pas pu ouvrir « {path} » : {exc}") from exc

    # -- fenêtres -----------------------------------------------------------------------

    def windows(self) -> list[WindowInfo]:
        """Fenêtres visibles via System Events (nom du processus, titre)."""
        script = (
            'tell application "System Events" to get {name of every process whose '
            "background only is false, name of every window of every process whose "
            "background only is false}"
        )
        try:
            _run_osascript(script)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"liste des fenêtres indisponible : {exc}") from exc
        # AppleScript renvoie des listes imbriquées difficiles à apparier titre/processus
        # sans _hosted_process équivalent : on liste une entrée par processus visible.
        script2 = (
            'tell application "System Events" to get name of every process whose '
            "background only is false and number of windows is greater than 0"
        )
        try:
            output = _run_osascript(script2)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"liste des fenêtres indisponible : {exc}") from exc
        return [
            WindowInfo(0, name.strip(), name.strip()) for name in output.split(",") if name.strip()
        ]

    def foreground_window(self) -> WindowInfo | None:
        try:
            name = _run_osascript(
                'tell application "System Events" to get name '
                "of first process whose frontmost is true"
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return WindowInfo(0, name, name) if name else None

    def close_window(self, window: WindowInfo) -> None:
        """Ferme la fenêtre en quittant l'appli (Cmd+Q ciblé impossible sans handle)."""
        if not window.process:
            raise ToolError("aucun processus à fermer")
        try:
            _run_osascript(f'tell application "{window.process}" to quit')
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"macOS n'a pas pu fermer « {window.title} » : {exc}") from exc

    def focus_window(self, window: WindowInfo) -> None:
        if not window.process:
            raise ToolError("aucun processus à mettre au premier plan")
        try:
            _run_osascript(f'tell application "{window.process}" to activate')
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(
                f"macOS a refusé de mettre « {window.title} » au premier plan : {exc}"
            ) from exc

    # -- divers -------------------------------------------------------------------------

    def media_key(self, action: str) -> None:
        mapping = {"play_pause": "playpause", "next": "next track", "previous": "previous track"}
        if action == "stop":
            _run_osascript('tell application "Music" to stop')
            return
        try:
            _run_osascript(f'tell application "Music" to {mapping[action]}')
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"touche média indisponible : {exc}") from exc

    def lock(self) -> None:
        """Verrouille la session (CGSession, droits normaux)."""
        cgsession = (
            "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession"
        )
        try:
            subprocess.run([cgsession, "-suspend"], check=True, capture_output=True, timeout=10.0)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"macOS a refusé de verrouiller la session : {exc}") from exc

    def power(self, action: str) -> None:
        """Veille (pmset), extinction ou redémarrage (événement loginwindow)."""
        try:
            if action == "sleep":
                subprocess.run(["pmset", "sleepnow"], check=True, capture_output=True, timeout=10.0)
            elif action == "shutdown":
                _run_osascript('tell application "loginwindow" to «event aevtrsdn»')
            elif action == "restart":
                _run_osascript('tell application "loginwindow" to «event aevtrrst»')
            else:
                raise ToolError(f"action inconnue : {action}")
        except (OSError, subprocess.SubprocessError) as exc:
            raise ToolError(f"macOS a refusé : {exc}") from exc

    def clipboard_read(self) -> str:
        import pyperclip

        return str(pyperclip.paste() or "")

    def clipboard_write(self, text: str) -> None:
        import pyperclip

        pyperclip.copy(text)


def make_desktop() -> WindowsDesktop | MacDesktop:
    """Bureau du système courant (Windows ou macOS)."""
    if sys.platform == "darwin":
        return MacDesktop()
    return WindowsDesktop()
