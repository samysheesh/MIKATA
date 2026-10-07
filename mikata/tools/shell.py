"""Outils run_powershell (filtré) et run_script (§7).

Avant toute exécution, le garde-fou lit la commande (ou le contenu du script) :
- liste noire (suppression, effacement, élévation...) : refus, rien n'est lancé ;
- liste grise (registre, services, arrêt de programme...) : « Tu confirmes ? » à voix
  haute, lancé seulement sur un oui ;
- sinon : lancé avec les droits normaux, `tools.shell_timeout_s` (30 s) au plus, sortie
  tronquée.
Les scripts autorisés sont ceux de `scripts/user/` (.ps1, .py, .bat, .cmd).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from mikata.agent.guard import AFTER_WEB, Decision, Guard, Verdict
from mikata.tools.base import (
    ActionCancelled,
    BaseTool,
    GuardDenied,
    ToolContext,
    ToolError,
    current_turn,
)
from mikata.tools.registry import register_tool

USER_SCRIPTS = "scripts/user"
SCRIPT_TYPES = (".ps1", ".py", ".bat", ".cmd")
CREATE_NO_WINDOW = 0x08000000
# Sortie en UTF-8, pour les accents (PowerShell 5 écrit sinon dans la page de code OEM).
PS_PREFIX = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
CONFIRM_MARGIN_S = 15.0


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "cp850", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class SubprocessRunner:
    """Implémente `ProcessRunner` : processus sans fenêtre, tué au-delà du délai.

    Les scripts Python écrivent en UTF-8 (sinon cp1252 sous Windows quand la sortie est
    redirigée, et les accents arrivent faux).
    """

    async def run(
        self, argv: list[str], timeout_s: float, cwd: Path | None = None
    ) -> tuple[int, str]:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            async with asyncio.timeout(timeout_s):
                output, _ = await process.communicate()
        except BaseException:  # délai dépassé ou commande annulée : on ne laisse rien tourner
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            raise
        return process.returncode or 0, _decode(output).strip()


async def _approve(ctx: ToolContext, guard: Guard, decision: Decision, what: str) -> None:
    """Refus (liste noire) ou question orale (liste grise) ; ne revient que si autorisé."""
    if decision.verdict is Verdict.DENY:
        detail = f" (« {decision.match} »)" if decision.match else ""
        raise GuardDenied(f"{decision.reason}{detail}")
    if decision.verdict is Verdict.CONFIRM:
        question = f"{what} va {decision.reason}. Tu confirmes ?"
        if not await guard.confirm(question, current_turn()):
            raise ActionCancelled(f"non lancé : l'utilisateur n'a pas confirmé ({decision.reason})")


def _report(code: int, output: str) -> str:
    return f"Code de sortie {code}.\n{output or '(aucune sortie)'}"


class PowerShellParams(BaseModel):
    command: str = Field(min_length=1, max_length=4000, description="Commande PowerShell")


@register_tool
class RunPowerShell(BaseTool[PowerShellParams]):
    name = "run_powershell"
    description = (
        "Lance une commande PowerShell (droits normaux, 30 s max) quand aucun autre outil ne "
        "convient ; jamais pour supprimer (delete_file) ; le garde-fou peut refuser ou "
        "demander confirmation."
    )
    params = PowerShellParams

    def time_limit(self) -> float:
        cfg = self.ctx.config
        return cfg.tools.shell_timeout_s + cfg.guard.confirm_timeout_s + CONFIRM_MARGIN_S

    async def run(self, args: PowerShellParams) -> str:
        guard = self.ctx.require_guard()
        turn = current_turn()
        decision = guard.check_command(args.command, after_web=bool(turn and turn.web_read))
        await _approve(self.ctx, guard, decision, "Cette commande")
        runner = self.ctx.require_shell()
        argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
        try:
            code, output = await runner.run(
                [*argv, PS_PREFIX + args.command],
                self.ctx.config.tools.shell_timeout_s,
                cwd=guard.home,
            )
        except TimeoutError as exc:
            limit = self.ctx.config.tools.shell_timeout_s
            raise ToolError(f"commande arrêtée après {limit:.0f} s") from exc
        except OSError as exc:
            raise ToolError(f"PowerShell introuvable : {exc}") from exc
        return _report(code, output)


class ScriptParams(BaseModel):
    name: str = Field(
        min_length=1, description="Nom du script de scripts/user (ex. « sauvegarde.ps1 »)"
    )
    arguments: list[str] = Field(default_factory=list, max_length=10, description="Arguments")


def script_argv(path: Path, arguments: list[str]) -> list[str]:
    kind = path.suffix.lower()
    if kind == ".ps1":
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", str(path), *arguments]  # fmt: skip
    if kind == ".py":
        return [sys.executable, str(path), *arguments]
    return ["cmd.exe", "/c", str(path), *arguments]


@register_tool
class RunScript(BaseTool[ScriptParams]):
    name = "run_script"
    description = "Lance un des scripts de l'utilisateur rangés dans scripts/user (30 s max)."
    params = ScriptParams

    def time_limit(self) -> float:
        cfg = self.ctx.config
        return cfg.tools.shell_timeout_s + cfg.guard.confirm_timeout_s + CONFIRM_MARGIN_S

    def find_script(self, name: str) -> Path:
        folder = self.ctx.config.resolve(USER_SCRIPTS)
        scripts = sorted(p for p in folder.glob("*") if p.suffix.lower() in SCRIPT_TYPES)
        wanted = name.strip().casefold()
        for path in scripts:
            if wanted in (path.name.casefold(), path.stem.casefold()):
                return path
        names = ", ".join(p.name for p in scripts) or "aucun"
        raise ToolError(f"script introuvable : {name} (scripts disponibles : {names})")

    async def run(self, args: ScriptParams) -> str:
        guard = self.ctx.require_guard()
        path = self.find_script(args.name)
        turn = current_turn()
        after_web = bool(turn and turn.web_read)
        decision = guard.check_script(path)
        for argument in args.arguments:
            if decision.verdict is not Verdict.DENY:
                arg_decision = guard.check_command(argument, after_web=after_web)
                if arg_decision.verdict is not Verdict.ALLOW:
                    decision = arg_decision
        if decision.verdict is Verdict.ALLOW and after_web and guard.cfg.confirm_after_web:
            decision = Decision(Verdict.CONFIRM, AFTER_WEB)
        await _approve(self.ctx, guard, decision, f"Le script {path.stem}")
        runner = self.ctx.require_shell()
        try:
            code, output = await runner.run(
                script_argv(path, args.arguments),
                self.ctx.config.tools.shell_timeout_s,
                cwd=path.parent,
            )
        except TimeoutError as exc:
            limit = self.ctx.config.tools.shell_timeout_s
            raise ToolError(f"script arrêté après {limit:.0f} s") from exc
        except OSError as exc:
            raise ToolError(f"impossible de lancer {path.name} : {exc}") from exc
        return _report(code, output)
