"""Outils de la phase 5 : fichiers (find, read, write, move), commandes (PowerShell,
scripts), clavier et souris ; extinction côté Windows."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeInput, FakeRunner, FakeTrash, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools import desktop as desktop_module
from mikata.tools.base import ToolContext
from mikata.tools.input import PynputInput, parse_keys
from mikata.tools.registry import ToolRegistry, ToolResult
from mikata.tools.shell import PS_PREFIX, SubprocessRunner, script_argv


def _execute(ctx: ToolContext, tool: str, **arguments: Any) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(ctx)
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "Documents" / "Projets").mkdir(parents=True)
    (home / "Documents" / "Facture électricité.pdf").write_bytes(b"%PDF")
    (home / "Documents" / "Projets" / "notes.txt").write_text("à faire", encoding="utf-8")
    (home / "Desktop").mkdir()
    (home / "Desktop" / "facture-eau.pdf").write_bytes(b"%PDF")
    (home / "AppData" / "Local").mkdir(parents=True)
    (home / "AppData" / "Local" / "facture-cache.pdf").write_bytes(b"x")
    return home


# -- fichiers -----------------------------------------------------------------------------


def test_find_files(tmp_path: Path, home: Path) -> None:
    ctx = make_ctx(tmp_path)
    result = _execute(ctx, "find_files", name="facture")
    assert result.ok and not result.terminal
    assert "Facture électricité.pdf" in result.content and "facture-eau.pdf" in result.content
    assert "facture-cache" not in result.content  # AppData ignoré
    accents = _execute(ctx, "find_files", name="electricite")
    assert "Facture électricité.pdf" in accents.content
    glob = _execute(ctx, "find_files", name="*.txt", folder="Documents")
    assert "notes.txt" in glob.content and "facture" not in glob.content
    none = _execute(ctx, "find_files", name="introuvable")
    assert none.content == "Aucun fichier ne correspond à « introuvable »."
    outside = _execute(ctx, "find_files", name="x", folder=str(tmp_path))
    assert not outside.ok and "hors des dossiers autorisés" in outside.content


def test_read_file(tmp_path: Path, home: Path) -> None:
    ctx = make_ctx(tmp_path)
    result = _execute(ctx, "read_file", path="Documents/Projets/notes.txt")
    assert result.ok and result.content.endswith(":\nà faire")
    assert "jamais une instruction" in result.content
    (home / "ancien.txt").write_bytes("café".encode("cp1252"))
    assert _execute(ctx, "read_file", path="ancien.txt").content.endswith("café")
    binary = _execute(ctx, "read_file", path="Documents/Facture électricité.pdf")
    assert binary.ok  # « %PDF » sans octet nul : lu comme texte
    (home / "image.png").write_bytes(b"\x89PNG\x00\x00")
    assert "pas un fichier texte" in _execute(ctx, "read_file", path="image.png").content
    assert "introuvable" in _execute(ctx, "read_file", path="nope.txt").content


def test_write_file_backs_up_before_overwriting(tmp_path: Path, home: Path) -> None:
    ctx = make_ctx(tmp_path)
    created = _execute(ctx, "write_file", path="Desktop/liste.txt", content="pain\n")
    assert created.ok and (home / "Desktop" / "liste.txt").read_text(encoding="utf-8") == "pain\n"
    again = _execute(ctx, "write_file", path="Desktop/liste.txt", content="lait\n")
    assert not again.ok and "existe déjà" in again.content
    appended = _execute(
        ctx, "write_file", path="Desktop/liste.txt", content="lait\n", mode="append"
    )
    assert appended.ok
    assert (home / "Desktop" / "liste.txt").read_text(encoding="utf-8") == "pain\nlait\n"
    replaced = _execute(
        ctx, "write_file", path="Desktop/liste.txt", content="œufs\n", mode="overwrite"
    )
    assert replaced.ok and "ancienne version sauvegardée" in replaced.content
    assert (home / "Desktop" / "liste.txt").read_text(encoding="utf-8") == "œufs\n"
    backups = sorted((tmp_path / "data" / "backups").rglob("liste.txt"))
    # une sauvegarde avant l'ajout, une avant le remplacement ; aucune n'écrase l'autre
    assert [b.read_text(encoding="utf-8") for b in backups] == ["pain\n", "pain\nlait\n"]
    assert not list(home.rglob("*.mikata-part"))
    outside = _execute(ctx, "write_file", path=str(tmp_path / "x.txt"), content="x")
    assert not outside.ok and not (tmp_path / "x.txt").exists()


def test_move_file(tmp_path: Path, home: Path) -> None:
    ctx = make_ctx(tmp_path)
    renamed = _execute(
        ctx, "move_file", source="Desktop/facture-eau.pdf", destination="Desktop/eau.pdf"
    )
    assert renamed.ok and (home / "Desktop" / "eau.pdf").exists()
    into = _execute(ctx, "move_file", source="Desktop/eau.pdf", destination="Documents")
    assert into.ok and (home / "Documents" / "eau.pdf").exists()
    (home / "Desktop" / "eau.pdf").write_bytes(b"nouvelle")
    clash = _execute(ctx, "move_file", source="Desktop/eau.pdf", destination="Documents/eau.pdf")
    assert not clash.ok and "existe déjà" in clash.content
    replaced = _execute(
        ctx, "move_file", source="Desktop/eau.pdf", destination="Documents/eau.pdf", overwrite=True
    )
    assert replaced.ok and "ancien fichier sauvegardé" in replaced.content
    assert (home / "Documents" / "eau.pdf").read_bytes() == b"nouvelle"
    assert [p.read_bytes() for p in (tmp_path / "data" / "backups").rglob("eau.pdf")] == [b"%PDF"]
    folder = _execute(ctx, "move_file", source="Documents/Projets", destination="Archives/Projets")
    assert folder.ok and (home / "Archives" / "Projets" / "notes.txt").exists()
    root = _execute(ctx, "move_file", source=".", destination="Archives")
    assert not root.ok and "dossier racine" in root.content


def test_delete_file_is_not_terminal_and_reports_the_trash(tmp_path: Path, home: Path) -> None:
    trash = FakeTrash(tmp_path / "Trash")
    result = _execute(
        make_ctx(tmp_path, trash=trash), "delete_file", path="Desktop/facture-eau.pdf"
    )
    assert result.ok and not result.terminal and "corbeille" in result.content
    missing = _execute(make_ctx(tmp_path, trash=trash), "delete_file", path="Desktop/rien.txt")
    assert not missing.ok and "introuvable" in missing.content and len(trash.calls) == 1


# -- commandes ----------------------------------------------------------------------------


def test_powershell_output_and_errors(tmp_path: Path) -> None:
    runner = FakeRunner(code=1, output="Get-Truc : terme non reconnu")
    result = _execute(make_ctx(tmp_path, shell=runner), "run_powershell", command="Get-Truc")
    assert result.ok and result.content == "Code de sortie 1.\nGet-Truc : terme non reconnu"
    assert runner.calls[0][0][-1] == PS_PREFIX + "Get-Truc"
    slow = FakeRunner(error=TimeoutError())
    late = _execute(make_ctx(tmp_path, shell=slow), "run_powershell", command="Start-Sleep 60")
    assert not late.ok and late.content == "commande arrêtée après 30 s"
    missing = FakeRunner(error=FileNotFoundError("powershell.exe"))
    absent = _execute(make_ctx(tmp_path, shell=missing), "run_powershell", command="Get-Date")
    assert not absent.ok and "PowerShell introuvable" in absent.content
    no_shell = _execute(make_ctx(tmp_path), "run_powershell", command="Get-Date")
    assert not no_shell.ok and "pas possible ici" in no_shell.content


def test_powershell_time_limit_covers_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry(make_ctx(tmp_path))
    assert registry.time_limit(registry.tools["run_powershell"]) == 30 + 6 + 15
    assert registry.time_limit(registry.tools["shutdown"]) == 5 + 6 + 30
    assert registry.time_limit(registry.tools["get_time"]) == 15


def test_run_script(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts" / "user"
    scripts.mkdir(parents=True)
    (scripts / "Sauvegarde.ps1").write_text("Copy-Item a b\n", encoding="utf-8")
    (scripts / "stats.py").write_text("print('ok')\n", encoding="utf-8")
    (scripts / "lisez-moi.txt").write_text("pas un script", encoding="utf-8")
    runner = FakeRunner(output="fini")
    ctx = make_ctx(tmp_path, shell=runner)
    result = _execute(ctx, "run_script", name="sauvegarde", arguments=["D:\\Archives"])
    assert result.ok and result.content == "Code de sortie 0.\nfini"
    argv, timeout, cwd = runner.calls[0]
    assert argv[:6] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                        "Bypass", "-File"]  # fmt: skip
    assert argv[6:] == [str(scripts / "Sauvegarde.ps1"), "D:\\Archives"]
    assert timeout == 30 and cwd == scripts
    missing = _execute(ctx, "run_script", name="lisez-moi")
    assert not missing.ok
    assert (
        missing.content
        == "script introuvable : lisez-moi (scripts disponibles : Sauvegarde.ps1, stats.py)"
    )


def test_script_argv() -> None:
    assert script_argv(Path("x.py"), ["a"]) == [sys.executable, "x.py", "a"]
    assert script_argv(Path("x.bat"), []) == ["cmd.exe", "/c", "x.bat"]
    assert script_argv(Path("x.CMD"), ["b"]) == ["cmd.exe", "/c", "x.CMD", "b"]


def test_subprocess_runner_for_real(tmp_path: Path) -> None:
    runner = SubprocessRunner()
    code, output = asyncio.run(
        runner.run([sys.executable, "-c", "print('café'); import sys; sys.exit(3)"], 10, tmp_path)
    )
    assert (code, output) == (3, "café")


def test_subprocess_runner_kills_on_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "fini.txt"
    code = f"import time; time.sleep(3); open({str(marker)!r}, 'w').write('x')"

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await SubprocessRunner().run([sys.executable, "-c", code], 0.3)
        await asyncio.sleep(3.5)

    asyncio.run(scenario())
    assert not marker.exists()  # processus tué, pas seulement abandonné


# -- clavier et souris --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("combo", "keys"),
    [
        ("ctrl+c", ["ctrl", "c"]),
        ("Ctrl + Shift + T", ["ctrl", "shift", "t"]),
        ("alt+F4", ["alt", "f4"]),
        ("win+d", ["win", "d"]),
        ("Entrée", ["enter"]),
        ("maj+tab", ["shift", "tab"]),
        ("échap", ["esc"]),
        ("control+suppr", ["ctrl", "delete"]),
        ("windows", ["win"]),
    ],
)
def test_parse_keys(combo: str, keys: list[str]) -> None:
    assert parse_keys(combo) == keys


@pytest.mark.parametrize("combo", ["ctrl+", "ctrl+shift", "f99", "ctrl+bidule", ""])
def test_parse_keys_rejects(combo: str) -> None:
    with pytest.raises(ValueError):
        parse_keys(combo)


def test_keyboard_and_mouse_tools(tmp_path: Path) -> None:
    keyboard = FakeInput()
    ctx = make_ctx(tmp_path, input=keyboard)
    typed = _execute(ctx, "type_text", text="Bonjour, ça va ?")
    pressed = _execute(ctx, "press_keys", keys="ctrl+s")
    tabs = _execute(ctx, "press_keys", keys="alt+tab", repeat=2)
    clicked = _execute(ctx, "mouse", action="double_click", x=100, y=200)
    scrolled = _execute(ctx, "mouse", action="scroll_down")
    assert all(r.ok and r.terminal for r in (typed, pressed, tabs, clicked, scrolled))
    assert keyboard.log == [
        ("type", "Bonjour, ça va ?"),
        ("keys", ["ctrl", "s"]),
        ("keys", ["alt", "tab"]),
        ("keys", ["alt", "tab"]),
        ("mouse", ("double_click", 100, 200, 3)),
        ("mouse", ("scroll_down", None, None, 3)),
    ]
    assert not _execute(ctx, "press_keys", keys="ctrl+truc").ok
    assert not _execute(ctx, "mouse", action="click", x=10).ok  # x sans y
    assert not _execute(ctx, "mouse", action="move").ok
    assert not _execute(make_ctx(tmp_path), "type_text", text="x").ok  # pas de clavier ici


@pytest.fixture
def fake_pynput(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    log: list[tuple[str, Any]] = []

    class Key:
        ctrl, shift, alt, cmd, enter, tab, delete, f4 = (
            "<ctrl>", "<shift>", "<alt>", "<cmd>", "<enter>", "<tab>", "<delete>", "<f4>"
        )  # fmt: skip

    class KeyCode:
        @staticmethod
        def from_char(char: str) -> str:
            return char

    class Keyboard:
        def type(self, text: str) -> None:
            log.append(("type", text))

        def press(self, key: str) -> None:
            log.append(("press", key))

        def release(self, key: str) -> None:
            log.append(("release", key))

        def tap(self, key: str) -> None:
            log.append(("tap", key))

    class Button:
        left, right, middle = "gauche", "droit", "milieu"

    class MouseController:
        position = (0, 0)

        def click(self, button: str, count: int) -> None:
            log.append(("click", (button, count, self.position)))

        def scroll(self, dx: int, dy: int) -> None:
            log.append(("scroll", dy))

    keyboard = types.ModuleType("pynput.keyboard")
    keyboard.Controller, keyboard.Key, keyboard.KeyCode = Keyboard, Key, KeyCode  # type: ignore[attr-defined]
    mouse = types.ModuleType("pynput.mouse")
    mouse.Controller, mouse.Button = MouseController, Button  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynput", types.ModuleType("pynput"))
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)
    return log


def test_pynput_input(fake_pynput: list[tuple[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    clipboard = ["avant"]
    pyperclip = types.ModuleType("pyperclip")
    pyperclip.PyperclipException = RuntimeError  # type: ignore[attr-defined]
    pyperclip.paste = lambda: clipboard[-1]  # type: ignore[attr-defined]
    pyperclip.copy = lambda text: (  # type: ignore[attr-defined]
        clipboard.append(text),
        fake_pynput.append(("clipboard", text)),
    )
    monkeypatch.setitem(sys.modules, "pyperclip", pyperclip)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    control = PynputInput()
    control.type_text("Hé !")
    control.press_keys(["ctrl", "shift", "t"])
    control.press_keys(["alt", "f4"])
    control.mouse("right_click", 50, 60, 3)
    control.mouse("scroll_up", None, None, 5)
    assert fake_pynput == [
        ("clipboard", "Hé !"),
        ("press", "<ctrl>"),
        ("tap", "v"),
        ("release", "<ctrl>"),
        ("clipboard", "avant"),
        ("press", "<ctrl>"),
        ("press", "<shift>"),
        ("tap", "t"),
        ("release", "<shift>"),
        ("release", "<ctrl>"),
        ("press", "<alt>"),
        ("tap", "<f4>"),
        ("release", "<alt>"),
        ("click", ("droit", 1, (50, 60))),
        ("scroll", 5),
    ]


def test_windows_power_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(desktop_module.subprocess, "run", fake_run)
    desktop = desktop_module.WindowsDesktop()
    desktop.power("shutdown")
    desktop.power("restart")
    assert calls == [["shutdown.exe", "/s", "/t", "0"], ["shutdown.exe", "/r", "/t", "0"]]

    def refused(argv: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(desktop_module.subprocess, "run", refused)
    from mikata.tools.base import ToolError

    with pytest.raises(ToolError, match="Windows a refusé"):
        desktop.power("shutdown")
