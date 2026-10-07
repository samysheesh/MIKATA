"""Critère de la phase 5 (docs/ARCHITECTURE.md §11) : tests automatisés prouvant
1. qu'aucune commande de la liste noire ne s'exécute (PowerShell, scripts, texte tapé,
   raccourci Maj+Suppr) ;
2. que delete_file finit toujours dans la corbeille, et qu'aucune suppression définitive
   n'existe dans le code de MIKATA.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeInput, FakeRunner, FakeTrash, make_config, make_ctx

from mikata.config import PROJECT_ROOT
from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolContext
from mikata.tools.registry import ToolRegistry, ToolResult

# Commandes de suppression ou d'effacement définitif (§7), sous plusieurs formes.
BLACKLISTED = [
    # Remove-Item et ses alias (rm, ri, del, erase, rd, rmdir)
    r"Remove-Item C:\Users\samy\Documents\rapport.docx",
    r"Remove-Item -Path .\Photos -Recurse -Force",
    "remove-item notes.txt",
    "ReMoVe-ItEm notes.txt",
    "Get-ChildItem *.tmp | Remove-Item",
    "gci $env:TEMP | Remove-Item -Recurse",
    "rm notes.txt",
    "rm -r -fo Projets",
    "ri notes.txt",
    "del notes.txt",
    r"del /f /s /q C:\Users\samy\Downloads\*",
    "erase notes.txt",
    "rd /s /q Projets",
    "rmdir Projets",
    'cmd /c "del /q notes.txt"',
    'cmd.exe /c rd /s /q "C:\\Users\\samy\\Old"',
    "Start-Process cmd -ArgumentList '/c del notes.txt'",
    "if (Test-Path x) { Remove-Item x }",
    "Get-Date; Remove-Item notes.txt",
    "R`emove-Item notes.txt",  # accent grave PowerShell
    # Corbeille, formatage, disques, effacement sécurisé, clichés instantanés
    "Clear-RecycleBin -Force",
    "Clear-RecycleBin -DriveLetter C -Confirm:$false",
    "Format-Volume -DriveLetter D -FileSystem NTFS",
    "format D: /q",
    "diskpart /s script.txt",
    "Clear-Disk -Number 1 -RemoveData",
    "Remove-Partition -DriveLetter E",
    "cipher /w:C:\\",
    "sdelete -p 3 secret.txt",
    "sdelete64.exe -s Dossier",
    "vssadmin delete shadows /all /quiet",
    "wmic shadowcopy delete",
    "wbadmin delete catalog",
    "fsutil file setzerodata offset=0 length=100 x.bin",
    "robocopy C:\\Vide D:\\Archives /MIR",
    "robocopy src dst /purge",
    # Effacement de contenu
    "Clear-Content journal.txt",
    "clc journal.txt",
    "Clear-Item journal.txt",
    # Appels .NET .Delete()
    "[System.IO.File]::Delete('C:\\Users\\samy\\notes.txt')",
    "[IO.Directory]::Delete('C:\\Users\\samy\\Projets', $true)",
    "(Get-Item notes.txt).Delete()",
    "Get-ChildItem *.log | ForEach-Object { $_.Delete() }",
    "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('x.txt')",
    "gci *.tmp | % Delete",  # méthode .Delete appelée sans parenthèses
    "Get-ChildItem *.log | ForEach-Object Delete",
    # Autres outils qui suppriment définitivement
    "git clean -fdx",
    "python -c \"import shutil; shutil.rmtree('Projets')\"",
    "py -c \"import os; os.remove('notes.txt')\"",
    # Élévation, contournements, extinction hors délai d'annulation
    "Start-Process powershell -Verb RunAs",
    "runas /user:Administrateur cmd",
    "powershell -EncodedCommand UgBlAG0AbwB2AGUALQBJAHQAZQBtACAAeAA=",
    "powershell -enc UgBlAG0AbwB2AGUALQBJAHQAZQBtACAAeAA=",
    "[Convert]::FromBase64String('UgBlAG0AbwB2AGUA')",
    "Stop-Computer -Force",
    "Restart-Computer",
    "shutdown /s /t 0",
]


def _execute(ctx: ToolContext, tool: str, **arguments: Any) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(ctx)
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


@pytest.mark.parametrize("command", BLACKLISTED)
def test_blacklisted_powershell_never_runs(
    tmp_path: Path, command: str, caplog_loguru: list[str]
) -> None:
    runner = FakeRunner()
    result = _execute(make_ctx(tmp_path, shell=runner), "run_powershell", command=command)
    assert runner.calls == []  # rien n'a été lancé
    assert not result.ok and result.content.startswith("refusé par le garde-fou : ")
    assert any(line.startswith("REFUSÉ | run_powershell(") for line in caplog_loguru)


@pytest.mark.parametrize("command", BLACKLISTED)
def test_blacklisted_command_in_script_never_runs(tmp_path: Path, command: str) -> None:
    scripts = tmp_path / "scripts" / "user"
    scripts.mkdir(parents=True)
    (scripts / "menage.ps1").write_text(f"Write-Output 'début'\n{command}\n", encoding="utf-8")
    runner = FakeRunner()
    result = _execute(make_ctx(tmp_path, shell=runner), "run_script", name="menage")
    assert runner.calls == [] and not result.ok and "garde-fou" in result.content


@pytest.mark.parametrize("command", BLACKLISTED[:20])
def test_blacklisted_command_as_script_argument_never_runs(tmp_path: Path, command: str) -> None:
    scripts = tmp_path / "scripts" / "user"
    scripts.mkdir(parents=True)
    (scripts / "lance.ps1").write_text("param($c) Write-Output $c\n", encoding="utf-8")
    runner = FakeRunner()
    result = _execute(
        make_ctx(tmp_path, shell=runner), "run_script", name="lance", arguments=[command]
    )
    assert runner.calls == [] and not result.ok


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.remove('notes.txt')\n",
        "import os\nos.unlink(chemin)\n",
        "import shutil\nshutil.rmtree('Projets')\n",
        "from pathlib import Path\nPath('x').unlink()\n",
        "import os\nos.rmdir('vide')\n",
        "import subprocess\nsubprocess.run(['powershell', 'Remove-Item x'])\n",
    ],
)
def test_python_script_that_deletes_never_runs(tmp_path: Path, code: str) -> None:
    scripts = tmp_path / "scripts" / "user"
    scripts.mkdir(parents=True)
    (scripts / "nettoie.py").write_text(code, encoding="utf-8")
    runner = FakeRunner()
    result = _execute(make_ctx(tmp_path, shell=runner), "run_script", name="nettoie.py")
    assert runner.calls == [] and not result.ok and "garde-fou" in result.content


@pytest.mark.parametrize("command", BLACKLISTED[:20])
def test_blacklisted_command_is_never_typed(tmp_path: Path, command: str) -> None:
    """Taper « Remove-Item ... » dans un terminal contournerait le filtre."""
    keyboard = FakeInput()
    result = _execute(make_ctx(tmp_path, input=keyboard), "type_text", text=command)
    assert keyboard.log == [] and not result.ok and "garde-fou" in result.content


@pytest.mark.parametrize(
    "text",
    [
        "rm notes.txt",
        "cd Documents; rm -r Projets",
        "Get-ChildItem | Remove-Item",
        'cmd /c "del /q notes.txt"',
        "del /s *.docx\n",
    ],
)
def test_deletion_command_lines_are_never_typed(tmp_path: Path, text: str) -> None:
    keyboard = FakeInput()
    result = _execute(make_ctx(tmp_path, input=keyboard), "type_text", text=text)
    assert keyboard.log == [] and not result.ok


@pytest.mark.parametrize(
    "text", ["J'ai ri en lisant ton message.", "Le rd du rapport", "Merci, à demain !"]
)
def test_ordinary_text_is_typed(tmp_path: Path, text: str) -> None:
    """Les alias courts (ri, rd...) ne bloquent le texte qu'en position de commande."""
    keyboard = FakeInput()
    result = _execute(make_ctx(tmp_path, input=keyboard), "type_text", text=text)
    assert result.ok and keyboard.log == [("type", text)]


@pytest.mark.parametrize("keys", ["shift+delete", "Shift+Del", "maj+suppr", "ctrl+shift+delete"])
def test_shift_delete_is_never_pressed(tmp_path: Path, keys: str) -> None:
    keyboard = FakeInput()
    result = _execute(make_ctx(tmp_path, input=keyboard), "press_keys", keys=keys)
    assert keyboard.log == [] and not result.ok and "définitivement" in result.content


@pytest.mark.parametrize(
    "command",
    [
        "Get-ChildItem $HOME\\Documents",
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 5",
        "Get-Date",
        "ipconfig /all",
        "Get-Content notes.txt -TotalCount 20",
        "Write-Output 'model deleted'",  # « del » dans un mot : pas une commande
        "$rm = 3; $rm * 2",
        "Get-ChildItem -Recurse | Measure-Object -Property Length -Sum",
        "Test-Connection 8.8.8.8 -Count 2",
        "Copy-Item notes.txt notes-copie.txt",
        "Get-Volume",
        "Get-Content rd.txt",  # « rd » dans un nom de fichier
        "Get-ChildItem C:\\temp\\rd",
        "Get-ChildItem -Filter *.del",
        "git status",
    ],
)
def test_ordinary_commands_run(tmp_path: Path, command: str) -> None:
    runner = FakeRunner(output="résultat")
    result = _execute(make_ctx(tmp_path, shell=runner), "run_powershell", command=command)
    assert result.ok and result.content == "Code de sortie 0.\nrésultat"
    argv, timeout, cwd = runner.calls[0]
    assert argv[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert argv[4].endswith(command) and timeout == 30 and cwd == tmp_path / "home"


# -- delete_file : toujours la corbeille -------------------------------------------------


@pytest.fixture
def no_permanent_delete(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Toute suppression définitive pendant le test le fait échouer."""
    calls: list[str] = []

    def forbidden(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> None:
            calls.append(name)
            raise AssertionError(f"suppression définitive appelée : {name}{args}")

        return call

    for module, attribute in [
        (os, "remove"),
        (os, "unlink"),
        (os, "rmdir"),
        (os, "removedirs"),
        (shutil, "rmtree"),
        (Path, "unlink"),
        (Path, "rmdir"),
    ]:
        monkeypatch.setattr(module, attribute, forbidden(f"{module.__name__}.{attribute}"))
    return calls


@pytest.mark.parametrize("kind", ["file", "folder", "nested"])
def test_delete_file_always_ends_in_trash(
    tmp_path: Path, kind: str, no_permanent_delete: list[str], caplog_loguru: list[str]
) -> None:
    home = tmp_path / "home"
    if kind == "file":
        target = home / "rapport.txt"
        home.mkdir()
        target.write_text("contenu précieux", encoding="utf-8")
    else:
        target = home / "Projets" / "Ancien" if kind == "nested" else home / "Photos"
        (target / "sous").mkdir(parents=True)
        (target / "sous" / "a.jpg").write_bytes(b"\xff\xd8 image")
    trash = FakeTrash(tmp_path / "Trash")
    relative = target.relative_to(home).as_posix()
    result = _execute(make_ctx(tmp_path, trash=trash), "delete_file", path=relative)

    assert result.ok and result.content == f"Mis à la corbeille : {target}."
    assert trash.calls == [str(target)]  # send2trash, et rien d'autre
    assert not target.exists()
    in_trash = tmp_path / "Trash" / f"1-{target.name}"
    assert in_trash.exists()  # récupérable
    if kind == "file":
        assert in_trash.read_text(encoding="utf-8") == "contenu précieux"
    assert no_permanent_delete == []
    assert any(line.startswith("OK | delete_file(") for line in caplog_loguru)


def test_delete_file_failure_never_falls_back_to_permanent_delete(
    tmp_path: Path, no_permanent_delete: list[str]
) -> None:
    target = tmp_path / "home" / "rapport.txt"
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    trash = FakeTrash(tmp_path / "Trash", fail=True)
    result = _execute(make_ctx(tmp_path, trash=trash), "delete_file", path=str(target))
    assert not result.ok and "rien n'a été supprimé" in result.content
    assert target.exists() and no_permanent_delete == []


@pytest.mark.parametrize(
    "path",
    [
        "/etc/hosts",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "../../outside.txt",
        "..",
        "~",  # le dossier perso lui-même
        ".",
    ],
)
def test_delete_file_outside_allowed_dirs_is_refused(
    tmp_path: Path, path: str, no_permanent_delete: list[str]
) -> None:
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    trash = FakeTrash(tmp_path / "Trash")
    result = _execute(make_ctx(tmp_path, trash=trash), "delete_file", path=path)
    assert not result.ok and trash.calls == []
    assert (tmp_path / "outside.txt").exists()


def test_delete_file_refuses_what_windows_would_not_recycle(
    tmp_path: Path, no_permanent_delete: list[str]
) -> None:
    """Trop gros pour la corbeille : Windows supprimerait définitivement, donc refus."""
    target = tmp_path / "home" / "film.mkv"
    target.parent.mkdir()
    target.write_bytes(b"\0" * 2_000_000)
    config = make_config(tmp_path, guard={"trash_max_mb": 1})
    trash = FakeTrash(tmp_path / "Trash")
    result = _execute(
        make_ctx(tmp_path, config=config, trash=trash), "delete_file", path="film.mkv"
    )
    assert not result.ok and "supprimerait définitivement" in result.content
    assert trash.calls == [] and target.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="corbeille freedesktop (Linux)")
def test_delete_file_with_real_send2trash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_permanent_delete: list[str]
) -> None:
    """Le vrai send2trash : le fichier est dans la corbeille, avec de quoi le restaurer."""
    import send2trash.plat_other as freedesktop

    from mikata.agent.guard import Guard

    # send2trash lit ces chemins à l'import : on les remplace dans le module, et HOME
    # pointe sur le même disque que le fichier (sinon : corbeille du point de montage).
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(freedesktop, "XDG_DATA_HOME", os.fsencode(xdg))
    monkeypatch.setattr(freedesktop, "HOMETRASH_B", os.fsencode(xdg / "Trash"))
    monkeypatch.setattr(freedesktop, "HOMETRASH", str(xdg / "Trash"))
    home = tmp_path / "home"
    home.mkdir()
    (home / "notes.txt").write_text("à garder", encoding="utf-8")
    ctx = make_ctx(tmp_path)
    ctx.guard = Guard(ctx.config, home=home)  # sans corbeille simulée
    result = _execute(ctx, "delete_file", path="notes.txt")
    trash = tmp_path / "xdg" / "Trash"
    assert result.ok and not (home / "notes.txt").exists()
    assert (trash / "files" / "notes.txt").read_text(encoding="utf-8") == "à garder"
    assert (trash / "info" / "notes.txt.trashinfo").exists()


def test_windows_trash_needs_safe_backend_and_fixed_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sous Windows : sans pywin32 (send2trash sûr) ou sur un lecteur sans corbeille, refus."""
    import ctypes
    import types

    from mikata.agent import guard as guard_module
    from mikata.tools.base import GuardDenied

    target = tmp_path / "home" / "notes.txt"
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(guard_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "send2trash.win.modern", None)  # pywin32 absent
    with pytest.raises(GuardDenied, match="pywin32"):
        guard_module._check_windows_trash(target)

    monkeypatch.setitem(sys.modules, "send2trash.win.modern", types.ModuleType("modern"))
    drive_type = {"value": 2}  # 2 = amovible, 3 = disque fixe
    kernel32 = types.SimpleNamespace(GetDriveTypeW=lambda drive: drive_type["value"])
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    with pytest.raises(GuardDenied, match="pas de corbeille"):
        guard_module._check_windows_trash(target)
    drive_type["value"] = 3
    guard_module._check_windows_trash(target)  # disque fixe : autorisé


# -- aucune suppression définitive dans le code ------------------------------------------

FORBIDDEN_CALLS = re.compile(
    r"\bos\.(remove|unlink|rmdir|removedirs)\s*\(|\bshutil\.rmtree\s*\("
    r"|\.(unlink|rmdir)\s*\(|\bRemove-Item\b"
)


def test_no_permanent_deletion_anywhere_in_mikata() -> None:
    """CLAUDE.md : aucune suppression définitive nulle part (corbeille via send2trash)."""
    offenders = []
    for folder in ("mikata", "scripts"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                quoted = re.sub(r"(\"[^\"]*\"|'[^']*')", "", code)  # textes et motifs exclus
                if FORBIDDEN_CALLS.search(quoted):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")
    assert offenders == []


def test_send2trash_is_the_only_deletion_path() -> None:
    """send2trash n'est importé qu'aux trois endroits qui mettent à la corbeille : le
    garde-fou (outils), la rotation des logs et la purge des audios de détection."""
    imports = re.compile(r"^\s*(from send2trash import|import send2trash)", re.MULTILINE)
    users = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "mikata").rglob("*.py")
        if imports.search(path.read_text(encoding="utf-8"))
    )
    assert users == [
        os.path.join("mikata", "agent", "guard.py"),
        os.path.join("mikata", "logging_setup.py"),
        os.path.join("mikata", "wake", "recordings.py"),
    ]
