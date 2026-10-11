"""Agent v2 (phase 13a) : exécution contrôlée du code écrit par le LLM.

Le code est le corps d'une fonction asynchrone qui reçoit `sdk` ; il ne peut faire que des
appels au SDK et du calcul simple. Ceci n'est pas une sécurité absolue contre du code
malveillant : c'est un filet contre les erreurs du LLM. La vraie sécurité est le
garde-fou des outils : toute action passe par un outil.
"""

from __future__ import annotations

import ast
import asyncio
from typing import Any

# Fonctions de base exposées au code généré : rien d'autre.
ALLOWED_BUILTINS = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sorted": sorted,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "enumerate": enumerate,
    "any": any,
    "all": all,
    "zip": zip,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
}

_FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "exit",
    "quit",
}
_MAX_CODE_CHARS = 4000
_PRINT_MAX_CHARS = 2000


class CodeRejected(Exception):
    """Code refusé avant exécution (construction interdite) ; message en français."""


class _CapturedPrint:
    """`print` du code généré : écrit dans la liste, jamais sur la console."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *values: Any) -> None:
        self.lines.append(" ".join(str(value) for value in values))


def check_code(code: str) -> None:
    """Refuse les constructions interdites ; lève `CodeRejected` avec la raison."""
    if len(code) > _MAX_CODE_CHARS:
        raise CodeRejected("code trop long (plus de 4 000 caractères)")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeRejected(f"syntaxe : {exc.msg} (ligne {exc.lineno})") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeRejected("les imports sont interdits")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeRejected("global et nonlocal sont interdits")
        if isinstance(node, ast.ClassDef):
            raise CodeRejected("les classes sont interdites")
        if isinstance(node, ast.Lambda):
            raise CodeRejected("lambda est interdit")
        if isinstance(node, ast.Delete):
            raise CodeRejected("del est interdit")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CodeRejected("les attributs privés sont interdits")
        if isinstance(node, ast.Name) and (node.id in _FORBIDDEN_NAMES or node.id.startswith("_")):
            raise CodeRejected(f"nom interdit : {node.id}")


async def run_code(code: str, sdk: Any, *, timeout_s: float) -> str:
    """Exécute le corps de fonction dans un espace de noms réduit au SDK.

    Laisse remonter `SdkError`, `TimeoutError` et toute autre exception (l'exécuteur les
    attrape et les montre au LLM sous forme `type: message`).
    """
    check_code(code)
    indented = "\n".join(f"    {line}" if line.strip() else "" for line in code.splitlines())
    source = f"async def __mikata_task__(sdk):\n{indented}\n    return None"
    captured = _CapturedPrint()
    globals_dict: dict[str, Any] = {"__builtins__": {**ALLOWED_BUILTINS, "print": captured}}
    namespace: dict[str, Any] = {}
    exec(compile(source, "<sdk>", "exec"), globals_dict, namespace)
    task = namespace["__mikata_task__"]
    async with asyncio.timeout(timeout_s):
        value = await task(sdk)
    parts: list[str] = []
    if captured.lines:
        parts.append("\n".join(captured.lines)[:_PRINT_MAX_CHARS])
    if value is not None:
        parts.append(str(value))
    return "\n".join(parts).strip()
