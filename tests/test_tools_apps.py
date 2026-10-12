"""Outils applications / fenêtres et index du menu Démarrer (Windows simulé)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeDesktop, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools import desktop as desktop_module
from mikata.tools.apps import find_app, find_windows, fold, open_apps
from mikata.tools.base import AppEntry, WindowInfo
from mikata.tools.desktop import WindowsDesktop, merge_apps, parse_start_apps, scan_start_menu
from mikata.tools.registry import ToolRegistry, ToolResult

APPS = FakeDesktop().apps


def _run(tmp_path: Path, desktop: FakeDesktop, tool: str, **arguments: Any) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(make_ctx(tmp_path, desktop=desktop))
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Spotify", "Spotify"),
        ("spotify.exe", "Spotify"),
        ("brave", "Brave"),
        ("Steam", "Steam"),
        ("chrome", "Google Chrome"),  # le moins de mots en trop
        ("spotfy", "Spotify"),  # faute de frappe sur un mot long
        ("Spotify Music", "Spotify"),  # mot générique en trop
        ("l'application Brave", "Brave"),
        ("visual studio code", "Visual Studio Code"),
        ("vs code", "Visual Studio Code"),  # sigle
        ("VSCode", "Visual Studio Code"),
        ("cs2", "Counter-Strike 2"),
        ("photo", "Photos"),
        # Bug de la phase 4 : « ouvre Photoshop » (non installé) ouvrait Photos.
        ("Photoshop", None),
        ("Adobe Photoshop", None),
        ("Steam VR", None),  # mot en trop non générique : autre appli
        ("Spotifaille", None),  # déformation : le LLM corrige avant d'appeler l'outil
        ("Chromium", None),
        ("", None),
    ],
)
def test_find_app(query: str, expected: str | None) -> None:
    app = find_app(query, APPS)
    assert (app.name if app else None) == expected


def test_fold() -> None:
    assert fold("  Éditeur.EXE (Bêta) ") == "editeur beta"


def test_scan_start_menu(tmp_path: Path) -> None:
    user, common = tmp_path / "user", tmp_path / "common"
    for path in (
        user / "Spotify.lnk",
        user / "Steam" / "Counter-Strike 2.url",
        user / "Steam" / "Uninstall Steam.lnk",
        common / "Brave.lnk",
        common / "Outils" / "Lisez-moi.lnk",
        common / "Outils" / "notes.txt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    apps = scan_start_menu([user, common, tmp_path / "absent"])
    assert [a.name for a in apps] == ["Spotify", "Counter-Strike 2", "Brave"]
    assert apps[0].target == str(user / "Spotify.lnk")


def test_parse_start_apps() -> None:
    output = "﻿" + json.dumps(
        [
            {"Name": "Spotify", "AppID": "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"},
            {"Name": "Désinstaller Truc", "AppID": "x"},
            {"Name": "", "AppID": "y"},
        ]
    )
    apps = parse_start_apps(output)
    assert apps == [
        AppEntry("Spotify", "shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify")
    ]
    single = parse_start_apps(json.dumps({"Name": "Calculatrice", "AppID": "calc"}))
    assert single == [AppEntry("Calculatrice", "shell:AppsFolder\\calc")]
    assert parse_start_apps("  ") == []


def test_merge_apps_keeps_first_source() -> None:
    merged = merge_apps(
        [AppEntry("Spotify", "a.lnk")],
        [AppEntry("spotify", "shell:AppsFolder\\x"), AppEntry("Xbox", "shell:AppsFolder\\y")],
    )
    assert merged == [AppEntry("Spotify", "a.lnk"), AppEntry("Xbox", "shell:AppsFolder\\y")]


def test_windows_desktop_app_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "Brave.lnk").write_bytes(b"")
    calls: list[str] = []

    def fake_powershell(command: str, timeout: float = 10.0) -> str:
        calls.append(command)
        return json.dumps([{"Name": "Spotify", "AppID": "Spotify!App"}])

    monkeypatch.setattr(desktop_module, "start_menu_dirs", lambda: [tmp_path])
    monkeypatch.setattr(desktop_module, "_run_powershell", fake_powershell)
    desktop = WindowsDesktop()
    assert [a.name for a in desktop.list_apps()] == ["Brave", "Spotify"]
    desktop.list_apps()  # en cache
    assert len(calls) == 1 and "Get-StartApps" in calls[0]
    desktop.list_apps(refresh=True)
    assert len(calls) == 2

    def broken(command: str, timeout: float = 10.0) -> str:
        raise subprocess.CalledProcessError(1, "powershell")

    monkeypatch.setattr(desktop_module, "_run_powershell", broken)
    assert [a.name for a in desktop.list_apps(refresh=True)] == ["Brave"]  # raccourcis seuls


def test_start_menu_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", "/u/AppData/Roaming")
    monkeypatch.setenv("PROGRAMDATA", "/ProgramData")
    dirs = desktop_module.start_menu_dirs()
    assert dirs[0] == Path("/u/AppData/Roaming/Microsoft/Windows/Start Menu/Programs")
    assert dirs[1] == Path("/ProgramData/Microsoft/Windows/Start Menu/Programs")


def test_open_app_launches_best_match(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    result = _run(tmp_path, desktop, "open_app", name="spotify music")
    assert result.ok and result.content == "Spotify ouvert." and result.terminal
    assert desktop.log == [("list_apps", False), ("launch", "Spotify")]


def test_open_app_refreshes_index_before_giving_up(tmp_path: Path) -> None:
    desktop = FakeDesktop(late_apps=[AppEntry("Discord", "Discord.lnk")])
    result = _run(tmp_path, desktop, "open_app", name="Discord")
    assert result.ok and desktop.log[-2:] == [("list_apps", True), ("launch", "Discord")]


def test_open_app_never_opens_a_lookalike(tmp_path: Path) -> None:
    """« ouvre Photoshop » sans Photoshop installé : refus, Photos n'est pas lancé."""
    desktop = FakeDesktop()
    missing = _run(tmp_path, desktop, "open_app", name="Photoshop")
    assert not missing.ok and missing.content == "application introuvable : Photoshop"
    assert not [entry for entry in desktop.log if entry[0] == "launch"]
    assert desktop.log == [("list_apps", False), ("list_apps", True)]  # index rafraîchi


def test_find_windows_by_process_then_title() -> None:
    windows = [
        WindowInfo(1, "Spotify Premium", "Spotify"),
        WindowInfo(2, "Nouvel onglet - Brave", "brave"),
        WindowInfo(3, "Discussion - Brave", "brave"),
        WindowInfo(4, "main.py - Visual Studio Code", "Code"),
        WindowInfo(5, "Photos", "Photos"),
    ]
    assert [w.handle for w in find_windows("Brave", windows)] == [2, 3]
    assert [w.handle for w in find_windows("visual studio code", windows)] == [4]
    assert [w.handle for w in find_windows("vs code", windows)] == [4]
    assert find_windows("Discord", windows) == []
    assert find_windows("Photoshop", windows) == []  # « ferme Photoshop » ne ferme pas Photos
    assert open_apps(windows) == ["Spotify", "brave", "Code", "Photos"]


def test_close_focus_and_list_windows(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    closed = _run(tmp_path, desktop, "close_app", name="Brave")
    assert closed.ok and closed.content == "brave fermé (2 fenêtre(s))."
    assert desktop.log == [("close", 12), ("close", 13)]
    focused = _run(tmp_path, desktop, "focus_window", name="spotify")
    assert focused.ok and desktop.log[-1] == ("focus", 11)
    listed = _run(tmp_path, desktop, "list_windows")
    assert not listed.terminal and "Spotify : Spotify Premium" in listed.content
    nothing = _run(tmp_path, desktop, "close_app", name="Discord")
    assert not nothing.ok and "aucune fenêtre" in nothing.content
    empty = _run(tmp_path, FakeDesktop(windows=[]), "list_windows")
    assert empty.content == "Aucune fenêtre ouverte."


def test_focus_window_rehides_the_work_browser(tmp_path: Path) -> None:
    """Un focus_window du code généré peut montrer le navigateur de travail : l'outil le
    re-cache aussitôt (jamais le Brave de l'utilisateur)."""
    from agent2_fakes import make_registry
    from tool_fakes import FakeBrowser

    browser = FakeBrowser()

    async def scenario() -> ToolResult:
        registry = make_registry(tmp_path, browser=browser, desktop=FakeDesktop())
        try:
            return await registry.execute(ToolCallReq("1", "focus_window", {"name": "Spotify"}))
        finally:
            await registry.aclose()

    focused = asyncio.run(scenario())
    assert focused.ok
    assert ("keep_hidden", ()) in browser.log


def test_close_window_closes_only_one(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    active = _run(tmp_path, desktop, "close_window")  # « ferme cette fenêtre »
    assert active.ok and active.terminal and active.content == "Fenêtre « Spotify Premium » fermée."
    assert desktop.log == [("close", 11)]
    one = _run(tmp_path, desktop, "close_window", name="Brave")
    assert one.ok and desktop.log[-1] == ("close", 12)  # une seule des deux fenêtres Brave
    assert len([entry for entry in desktop.log if entry[0] == "close"]) == 2
    none = _run(tmp_path, FakeDesktop(windows=[]), "close_window")
    assert not none.ok and none.content == "aucune fenêtre active"


def test_close_app_by_french_title(tmp_path: Path) -> None:
    """« ferme le bloc-notes » : le processus s'appelle notepad, le titre dit Bloc-notes."""
    windows = [
        WindowInfo(21, "courses.txt - Bloc-notes", "notepad"),
        WindowInfo(22, "Sans titre - Bloc-notes", "notepad"),
        WindowInfo(23, "Bloc-notes et astuces - YouTube - Google Chrome", "chrome"),
    ]
    desktop = FakeDesktop(windows=windows)
    for name in ("Bloc-notes", "notepad"):
        result = _run(tmp_path, desktop, "close_app", name=name)
        assert result.ok
    closed = [value for kind, value in desktop.log if kind == "close"]
    assert set(closed) == {21, 22}  # l'onglet Chrome « Bloc-notes et astuces » reste ouvert
    youtube = _run(tmp_path, FakeDesktop(windows=windows), "close_window", name="YouTube")
    assert youtube.ok and "YouTube" in youtube.content  # sinon : un mot du titre suffit


def test_installed_app_names() -> None:
    from mikata.tools.apps import installed_app_names

    apps = [AppEntry(n, "x") for n in ("spotify", "Brave", "Spotify", "  Steam  ", "x" * 80)]
    assert installed_app_names(apps) == ["Brave", "spotify", "Steam"]


WIN11_NOTEPAD = [
    WindowInfo(31, "Sans titre - Bloc-notes", "Notepad"),  # Windows 11 : appli du Store
    WindowInfo(32, "YouTube - Google Chrome", "chrome"),
]


@pytest.mark.parametrize("name", ["notepad", "Notepad.exe", "Bloc-notes", "bloc notes"])
def test_notepad_of_windows_11_is_found(name: str) -> None:
    """Phase 6 : « Stop-Process -Name Notepad » / « ferme le bloc-notes » avec le Bloc-notes
    de Windows 11, nom de processus connu ou non."""
    assert [w.handle for w in find_windows(name, WIN11_NOTEPAD)] == [31]
    unknown = [WindowInfo(31, "Sans titre - Bloc-notes", ""), WIN11_NOTEPAD[1]]
    assert [w.handle for w in find_windows(name, unknown)] == [31]


def test_french_names_and_process_names_are_aliases() -> None:
    windows = [
        WindowInfo(41, "Calculatrice", "CalculatorApp"),
        WindowInfo(42, "Paramètres", "SystemSettings"),
        WindowInfo(43, "Document1 - Word", "WINWORD"),
        WindowInfo(44, "Sans titre - Paint", "mspaint"),
    ]
    assert [w.handle for w in find_windows("calculator", windows)] == [41]
    assert [w.handle for w in find_windows("Calculatrice", windows)] == [41]
    assert [w.handle for w in find_windows("paramètres", windows)] == [42]
    assert [w.handle for w in find_windows("winword", windows)] == [43]
    assert [w.handle for w in find_windows("Paint", windows)] == [44]
    assert find_windows("Excel", windows) == []
    apps = [AppEntry("Bloc-notes", "n"), AppEntry("Calculatrice", "c"), AppEntry("Paint", "p")]
    assert find_app("notepad", apps) == apps[0]
    assert find_app("calc", apps) == apps[1]
    assert find_app("mspaint", apps) == apps[2]
    assert find_app("Photoshop", apps) is None


def test_close_app_error_lists_open_windows(tmp_path: Path) -> None:
    """Rien ne correspond : le LLM reçoit la liste des fenêtres pour réessayer."""
    result = _run(tmp_path, FakeDesktop(windows=WIN11_NOTEPAD), "close_app", name="Wordpad")
    assert not result.ok
    assert "aucune fenêtre ouverte pour « Wordpad »" in result.content
    assert "Notepad : Sans titre - Bloc-notes" in result.content
    empty = _run(tmp_path, FakeDesktop(windows=[]), "close_window", name="x")
    assert "aucune fenêtre ouverte" in empty.content


def test_process_name_without_psutil_access(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def denied(pid: int) -> Any:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    if sys.platform == "win32":  # repli QueryFullProcessImageNameW, sur un PID qui existe
        assert desktop_module._process_name(os.getpid()).lower().startswith("python")
    # Hors Windows : pas d'autre moyen. (Sous Windows, un PID au hasard peut exister : les
    # 2 bits de poids faible sont ignorés, 1234 désignait le processus 1232 sur la CI.)
    monkeypatch.setattr(desktop_module.sys, "platform", "linux")
    assert desktop_module._process_name(1234) == ""

    class Proc:
        def __init__(self, pid: int) -> None:
            pass

        def name(self) -> str:
            return "Notepad.exe"

    monkeypatch.setattr(psutil, "Process", Proc)
    assert desktop_module._process_name(1) == "Notepad"
