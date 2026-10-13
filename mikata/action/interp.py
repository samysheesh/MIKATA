"""Interpréteur des scripts écrits par le modèle (docs/AGENT_CORE.md §3.4, §3.5).

Le code généré n'est jamais passé à `exec`. Il est lu par `ast` puis évalué nœud par nœud,
dans un sous-ensemble de Python :

- **rien d'autre que le SDK** : pas d'import, pas de classe, pas de fonction, pas
  d'attribut caché ; les seules fonctions sont les fonctions de base ci-dessous, les méthodes
  usuelles des chaînes, listes et dictionnaires, et ce que l'hôte expose ;
- **un budget** : nombre d'opérations, durée, taille des valeurs. Une boucle infinie
  s'arrête, et l'interpréteur rend la main à la boucle asyncio en cours de route (l'audio et
  l'interface ne gèlent pas) ;
- **la provenance** : chaque valeur porte un niveau de confiance (`Trust`), propagé par
  chaque opération. Une valeur calculée à partir d'un contenu lu (page web, fichier) reste
  marquée non fiable, y compris quand elle ne fait que **décider** d'une branche : ce qui se
  passe sous un `if` ou une boucle qui dépend d'elle est marqué de même. Les politiques
  (`mikata.action.policy`) lisent ce niveau au moment d'appeler un outil.

Méthode reprise de CaMeL (Debenedetti et al., 2025) : le flux de contrôle vient du script,
écrit à partir de la seule demande ; le contenu lu circule comme une donnée.
"""

from __future__ import annotations

import ast
import asyncio
import re
import textwrap
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from mikata.action.effects import Trust, join

MAX_STEPS = 50_000  # opérations évaluées
MAX_ITEMS = 100_000  # éléments d'une liste, tours d'une boucle
MAX_TEXT = 2_000_000  # caractères d'une chaîne
YIELD_EVERY = 200  # opérations entre deux retours à la boucle asyncio
MAX_CODE_CHARS = 6_000


class ScriptError(Exception):
    """Erreur du script, montrée au modèle pour qu'il corrige (message en français)."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"ligne {self.line} : {self.message}" if self.line else self.message


class BudgetExceeded(ScriptError):
    """Trop d'opérations, trop long ou valeur trop grosse : le script est arrêté."""


class ExpectationFailed(ScriptError):
    """`expect(...)` faux : la post-condition du script n'est pas remplie."""


class ScriptFailure(Exception):
    """Échec levé par l'hôte (outil en échec...) qu'un script peut attraper par `except`.

    `tainted` : le message dépend d'un contenu non fiable ; il ne doit pas être montré tel
    quel au modèle qui planifie."""

    def __init__(self, message: str, tainted: bool = False) -> None:
        super().__init__(message)
        self.tainted = tainted


@dataclass(slots=True)
class Val:
    """Une valeur du script et son niveau de confiance."""

    value: Any
    trust: Trust = Trust.TRUSTED


class Host(Protocol):
    """Ce que l'interpréteur demande à son hôte (`mikata.action.runtime.ScriptRuntime`)."""

    def namespaces(self) -> dict[str, set[str]]: ...

    def functions(self) -> set[str]: ...

    async def call_sdk(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val: ...

    async def call_function(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val: ...


class _Return(Exception):
    def __init__(self, value: Val) -> None:
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass(frozen=True)
class _Namespace:
    name: str


@dataclass(frozen=True)
class _SdkFn:
    name: str


@dataclass(frozen=True)
class _HostFn:
    name: str


@dataclass(frozen=True)
class _Method:
    target: Any
    name: str


@dataclass(frozen=True)
class _Lambda:
    """`lambda x: ...` : paramètres, corps et variables visibles là où il est écrit."""

    params: tuple[str, ...]
    body: ast.expr
    scopes: tuple[dict[str, Val], ...]


def _field_key(item: Any, key: Any) -> Any:
    if key is None:
        return item
    if isinstance(item, dict):
        return item.get(key)
    raise TypeError('key= attend le nom d\'un champ : sorted(lignes, key="date")')


def _sorted(items: Iterable[Any], key: Any = None, reverse: bool = False) -> list[Any]:
    return sorted(items, key=lambda item: _field_key(item, key), reverse=bool(reverse))


def _pick(chooser: Callable[..., Any]) -> Callable[..., Any]:
    def pick(*args: Any, key: Any = None, default: Any = None) -> Any:
        items = list(args[0]) if len(args) == 1 else list(args)
        if not items:
            return default
        return chooser(items, key=lambda item: _field_key(item, key))

    return pick


def _range(*args: Any) -> range:
    result = range(*args)
    if len(result) > MAX_ITEMS:
        raise BudgetExceeded(f"range trop grand (plus de {MAX_ITEMS} valeurs)")
    return result


# Fonctions de base : pures, sans accès au système. Tout le reste passe par l'hôte.
BUILTINS: dict[str, Callable[..., Any]] = {
    "len": len,
    "range": _range,
    "min": _pick(min),
    "max": _pick(max),
    "sum": sum,
    "sorted": _sorted,
    "reversed": lambda items: list(reversed(list(items))),
    "enumerate": lambda items, start=0: [list(pair) for pair in enumerate(items, start)],
    "zip": lambda *lists: [list(group) for group in zip(*lists, strict=False)],
    "any": any,
    "all": all,
    "abs": abs,
    "round": round,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "type": lambda value: type(value).__name__,  # un nom (« str », « dict »), pas la classe
    "isinstance": isinstance,
    "map": lambda function, items: [function(item) for item in items],
    "filter": lambda function, items: [item for item in items if function(item)],
}
# Méthodes permises par type : jamais `format` (il lit des attributs par une chaîne).
METHODS: dict[type, frozenset[str]] = {
    str: frozenset(
        [
            "lower",
            "upper",
            "strip",
            "lstrip",
            "rstrip",
            "split",
            "rsplit",
            "splitlines",
            "join",
            "replace",
            "startswith",
            "endswith",
            "find",
            "rfind",
            "count",
            "title",
            "capitalize",
            "casefold",
            "isdigit",
            "isalpha",
            "isalnum",
            "isspace",
            "removeprefix",
            "removesuffix",
            "partition",
        ]
    ),
    list: frozenset(
        ["append", "extend", "insert", "pop", "remove", "index", "count", "reverse", "copy"]
    ),
    dict: frozenset(["get", "keys", "values", "items", "pop", "update", "setdefault", "copy"]),
    set: frozenset(["add", "discard", "remove", "union", "intersection", "difference", "copy"]),
    tuple: frozenset(["index", "count"]),
}
_EXCEPTIONS: dict[str, tuple[type[BaseException], ...]] = {
    "Exception": (ScriptFailure, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError),
    "ToolError": (ScriptFailure,),
    "ValueError": (ValueError,),
    "KeyError": (KeyError,),
    "IndexError": (IndexError,),
    "TypeError": (TypeError,),
    "ZeroDivisionError": (ZeroDivisionError,),
}
_FORBIDDEN_STATEMENTS: dict[type, str] = {
    ast.Import: "import",
    ast.ImportFrom: "import",
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "def",
    ast.ClassDef: "class",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.Delete: "del",
    ast.With: "with",
    ast.AsyncWith: "with",
    ast.Raise: "raise",
    ast.Match: "match",
}
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
}
_UNARY: dict[type, Callable[[Any], Any]] = {
    ast.Not: lambda a: not a,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}
_COMPARE: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}


VALUE_ERRORS = (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError)
_DIGITS = re.compile(r"[0-9]+")
INTERP_FUNCTIONS = frozenset({"expect", "print", "parallel"})  # traitées par l'interpréteur


def parse_script(code: str) -> list[ast.stmt]:
    """Corps du script, ou `ScriptError` s'il est vide, trop long ou mal écrit."""
    if not code.strip():
        raise ScriptError("script vide")
    if len(code) > MAX_CODE_CHARS:
        raise ScriptError(f"script trop long (plus de {MAX_CODE_CHARS} caractères)")
    source = "async def __script__():\n" + textwrap.indent(textwrap.dedent(code), "    ")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = (exc.lineno or 1) - 1
        raise ScriptError(f"syntaxe : {exc.msg}", max(line, 1)) from exc
    function = tree.body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    return function.body


class Interpreter:
    """Exécute un script contre un hôte. Un interpréteur = une exécution."""

    def __init__(
        self,
        host: Host,
        *,
        max_steps: int = MAX_STEPS,
        timeout_s: float = 60.0,
        variables: dict[str, Val] | None = None,
    ) -> None:
        self.host = host
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.steps = 0
        self.prints: list[str] = []
        self._scopes: list[dict[str, Val]] = [dict(variables or {})]
        self._pc: list[Trust] = []
        self._deadline = 0.0
        self._at: int | None = None  # ligne en cours, pour les messages d'erreur
        self._namespaces = host.namespaces()
        self._functions = host.functions() | INTERP_FUNCTIONS

    # -- entrée ---------------------------------------------------------------------------

    async def run(self, code: str) -> Val:
        """Valeur du `return` (ou `None`). Lève `ScriptError`, ou laisse passer ce que
        l'hôte lève (refus d'une politique, annulation)."""
        body = parse_script(code)
        self._deadline = time.monotonic() + self.timeout_s
        try:
            await self._block(body)
        except _Return as done:
            return done.value
        except (_Break, _Continue):
            raise ScriptError("break ou continue en dehors d'une boucle") from None
        except VALUE_ERRORS as exc:  # non attrapée par le script : erreur du script
            raise ScriptError(f"{type(exc).__name__} : {exc}", self._at) from None
        return Val(None)

    # -- budget et provenance -------------------------------------------------------------

    async def _tick(self, node: ast.AST) -> None:
        self.steps += 1
        self._at = self._line(node) or self._at
        if self.steps > self.max_steps:
            raise BudgetExceeded(
                f"script arrêté : plus de {self.max_steps} opérations (boucle sans fin ?)",
                self._line(node),
            )
        if self.steps % YIELD_EVERY == 0:
            if time.monotonic() > self._deadline:
                raise BudgetExceeded(
                    f"script arrêté : plus de {self.timeout_s:.0f} s", self._line(node)
                )
            await asyncio.sleep(0)  # l'audio et l'interface gardent la main

    @staticmethod
    def _line(node: ast.AST) -> int | None:
        line = getattr(node, "lineno", None)
        return max(line - 1, 1) if line else None  # la ligne 1 est l'en-tête ajouté

    def _pc_trust(self) -> Trust:
        return join(*self._pc)

    def _sized(self, value: Any, node: ast.AST) -> Any:
        if isinstance(value, str) and len(value) > MAX_TEXT:
            raise BudgetExceeded("texte trop long", self._line(node))
        if isinstance(value, (list, tuple, set, dict)) and len(value) > MAX_ITEMS:
            raise BudgetExceeded("liste trop longue", self._line(node))
        if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 4096:
            raise BudgetExceeded("nombre trop grand", self._line(node))
        return value

    def _guard_method(self, method: _Method, args: list[Any], node: ast.AST) -> None:
        """Refuse avant l'appel ce qui fabriquerait un texte démesuré."""
        target = method.target
        if method.name == "replace" and isinstance(target, str) and len(args) >= 2:
            new = args[1] if isinstance(args[1], str) else ""
            if len(target) * max(len(new), 1) > MAX_TEXT:
                raise BudgetExceeded("texte trop long", self._line(node))
        if method.name == "join" and isinstance(target, str) and args:
            parts = args[0] if isinstance(args[0], (list, tuple, set)) else []
            total = sum(len(part) for part in parts if isinstance(part, str))
            if total + len(target) * len(parts) > MAX_TEXT:
                raise BudgetExceeded("texte trop long", self._line(node))

    # -- variables ------------------------------------------------------------------------

    def _lookup(self, name: str, node: ast.AST) -> Val:
        if name.startswith("_"):
            raise ScriptError(f"nom interdit : {name}", self._line(node))
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        if name in BUILTINS:
            return Val(BUILTINS[name])
        if name in self._namespaces:
            return Val(_Namespace(name))
        if name in self._functions:
            return Val(_HostFn(name))
        raise ScriptError(f"nom inconnu : {name}", self._line(node))

    def _store(self, target: ast.expr, value: Val) -> None:
        value = Val(value.value, value.trust.join(self._pc_trust()))
        if isinstance(target, ast.Name):
            if target.id.startswith("_") or target.id in self._namespaces:
                raise ScriptError(f"nom interdit : {target.id}", self._line(target))
            self._scopes[-1][target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            try:
                items = list(value.value)
            except TypeError:
                raise ScriptError("valeur non décomposable", self._line(target)) from None
            if len(items) != len(target.elts):
                raise ScriptError(
                    f"{len(target.elts)} noms pour {len(items)} valeurs", self._line(target)
                )
            for element, item in zip(target.elts, items, strict=True):
                self._store(element, Val(item, value.trust))
        else:
            raise ScriptError("affectation non permise ici", self._line(target))

    # -- instructions ---------------------------------------------------------------------

    async def _block(self, body: list[ast.stmt]) -> None:
        for statement in body:
            await self._exec(statement)

    async def _exec(self, node: ast.stmt) -> None:
        await self._tick(node)
        for kind, word in _FORBIDDEN_STATEMENTS.items():
            if isinstance(node, kind):
                raise ScriptError(f"« {word} » n'est pas permis dans un script", self._line(node))
        if isinstance(node, ast.Expr):
            await self._eval(node.value)
        elif isinstance(node, ast.Assign):
            value = await self._eval(node.value)
            for target in node.targets:
                await self._assign(target, value)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                await self._assign(node.target, await self._eval(node.value))
        elif isinstance(node, ast.AugAssign):
            await self._aug_assign(node)
        elif isinstance(node, ast.If):
            test = await self._eval(node.test)
            self._pc.append(test.trust)
            try:
                await self._block(node.body if test.value else node.orelse)
            finally:
                self._pc.pop()
        elif isinstance(node, ast.For):
            await self._for(node)
        elif isinstance(node, ast.While):
            await self._while(node)
        elif isinstance(node, ast.Return):
            value = await self._eval(node.value) if node.value is not None else Val(None)
            raise _Return(Val(value.value, value.trust.join(self._pc_trust())))
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            raise _Break
        elif isinstance(node, ast.Continue):
            raise _Continue
        elif isinstance(node, ast.Try):
            await self._try(node)
        elif isinstance(node, ast.Assert):
            test = await self._eval(node.test)
            if not test.value:
                message = (await self._eval(node.msg)).value if node.msg else "condition fausse"
                raise ExpectationFailed(str(message), self._line(node))
        else:
            raise ScriptError(
                f"« {type(node).__name__} » n'est pas permis dans un script", self._line(node)
            )

    async def _assign(self, target: ast.expr, value: Val) -> None:
        if isinstance(target, ast.Subscript):
            container = await self._eval(target.value)
            index = await self._eval(target.slice)
            if not isinstance(container.value, (list, dict)):
                raise ScriptError("on ne peut indexer qu'une liste ou un dict", self._line(target))
            try:
                container.value[index.value] = value.value
            except (IndexError, KeyError, TypeError) as exc:
                raise ScriptError(f"index invalide : {exc}", self._line(target)) from None
            # Le conteneur devient au moins aussi peu fiable que ce qu'on y met.
            container.trust = join(container.trust, value.trust, index.trust, self._pc_trust())
            self._sized(container.value, target)
        else:
            self._store(target, value)

    async def _aug_assign(self, node: ast.AugAssign) -> None:
        target = node.target
        if isinstance(target, ast.Name):
            current = self._lookup(target.id, target)
        elif isinstance(target, ast.Subscript):
            container, index = await self._eval(target.value), await self._eval(target.slice)
            current = Val(container.value[index.value], container.trust.join(index.trust))
        else:
            raise ScriptError("affectation non permise ici", self._line(node))
        operand = await self._eval(node.value)
        await self._assign(target, self._binop(type(node.op), current, operand, node))

    async def _for(self, node: ast.For) -> None:
        iterable = await self._eval(node.iter)
        items = self._iterate(iterable.value, node)
        self._pc.append(iterable.trust)
        try:
            for item in items:
                await self._tick(node)
                self._store(node.target, Val(item, iterable.trust))
                try:
                    await self._block(node.body)
                except _Continue:
                    continue
                except _Break:
                    break
            else:
                await self._block(node.orelse)
        finally:
            self._pc.pop()

    async def _while(self, node: ast.While) -> None:
        depth = len(self._pc)
        self._pc.append(Trust.TRUSTED)
        try:
            while True:
                await self._tick(node)
                test = await self._eval(node.test)
                self._pc[depth] = self._pc[depth].join(test.trust)
                if not test.value:
                    await self._block(node.orelse)
                    break
                try:
                    await self._block(node.body)
                except _Continue:
                    continue
                except _Break:
                    break
        finally:
            del self._pc[depth:]

    async def _try(self, node: ast.Try) -> None:
        try:
            await self._block(node.body)
        except (ScriptFailure, *VALUE_ERRORS) as exc:
            for handler in node.handlers:
                if self._handles(handler, exc):
                    if handler.name:
                        tainted = getattr(exc, "tainted", False)
                        self._store(
                            ast.Name(handler.name, ast.Store()),
                            Val(str(exc), Trust.UNTRUSTED if tainted else Trust.LOCAL),
                        )
                    await self._block(handler.body)
                    break
            else:
                raise
        else:
            await self._block(node.orelse)
        finally:
            await self._block(node.finalbody)

    def _handles(self, handler: ast.ExceptHandler, exc: BaseException) -> bool:
        if handler.type is None:
            return True
        names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        for name in names:
            if not isinstance(name, ast.Name) or name.id not in _EXCEPTIONS:
                raise ScriptError(
                    "except : types permis : " + ", ".join(_EXCEPTIONS), self._line(handler)
                )
            if isinstance(exc, _EXCEPTIONS[name.id]):
                return True
        return False

    def _iterate(self, value: Any, node: ast.AST) -> list[Any]:
        if isinstance(value, (dict, list, tuple, set, range, str)):
            items = list(value)
        else:
            raise ScriptError("valeur non itérable", self._line(node))
        if len(items) > MAX_ITEMS:
            raise BudgetExceeded("boucle trop longue", self._line(node))
        return items

    # -- expressions ----------------------------------------------------------------------

    async def _eval(self, node: ast.expr) -> Val:
        await self._tick(node)
        try:
            return await self._eval_node(node)
        except (AttributeError, OverflowError, RecursionError, MemoryError) as exc:
            raise ScriptError(f"{type(exc).__name__} : {exc}", self._line(node)) from None

    async def _eval_node(self, node: ast.expr) -> Val:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise ScriptError("constante non permise", self._line(node))
            return Val(node.value)
        if isinstance(node, ast.Name):
            return self._lookup(node.id, node)
        if isinstance(node, ast.Await):
            return await self._eval(node.value)
        if isinstance(node, ast.JoinedStr):
            return await self._fstring(node)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [await self._eval(element) for element in node.elts]
            values = [item.value for item in items]
            trust = join(*(item.trust for item in items))
            built: Any = (
                values
                if isinstance(node, ast.List)
                else tuple(values)
                if isinstance(node, ast.Tuple)
                else set(values)
            )
            return Val(built, trust)
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            trust = Trust.TRUSTED
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                if key_node is None:
                    raise ScriptError("** n'est pas permis", self._line(node))
                key, value = await self._eval(key_node), await self._eval(value_node)
                result[key.value] = value.value
                trust = join(trust, key.trust, value.trust)
            return Val(result, trust)
        if isinstance(node, ast.BinOp):
            left, right = await self._eval(node.left), await self._eval(node.right)
            return self._binop(type(node.op), left, right, node)
        if isinstance(node, ast.UnaryOp):
            operand = await self._eval(node.operand)
            if type(node.op) not in _UNARY:
                raise ScriptError("opérateur non permis", self._line(node))
            return Val(_UNARY[type(node.op)](operand.value), operand.trust)
        if isinstance(node, ast.BoolOp):
            trust = Trust.TRUSTED
            value: Any = None
            for part in node.values:
                current = await self._eval(part)
                trust, value = trust.join(current.trust), current.value
                if isinstance(node.op, ast.And) and not value:
                    break
                if isinstance(node.op, ast.Or) and value:
                    break
            return Val(value, trust)
        if isinstance(node, ast.Compare):
            left = await self._eval(node.left)
            trust, outcome = left.trust, True
            for op, right_node in zip(node.ops, node.comparators, strict=True):
                right = await self._eval(right_node)
                trust = trust.join(right.trust)
                if not _COMPARE[type(op)](left.value, right.value):
                    outcome = False
                    break
                left = right
            return Val(outcome, trust)
        if isinstance(node, ast.IfExp):
            test = await self._eval(node.test)
            chosen = await self._eval(node.body if test.value else node.orelse)
            return Val(chosen.value, chosen.trust.join(test.trust))
        if isinstance(node, ast.Subscript):
            container, index = await self._eval(node.value), await self._eval(node.slice)
            return Val(container.value[index.value], container.trust.join(index.trust))
        if isinstance(node, ast.Slice):
            parts = [
                (await self._eval(part)).value if part is not None else None
                for part in (node.lower, node.upper, node.step)
            ]
            return Val(slice(*parts))
        if isinstance(node, ast.Attribute):
            return await self._attribute(node)
        if isinstance(node, ast.Call):
            return await self._call(node)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return await self._comprehension(node)
        if isinstance(node, ast.Lambda):
            arguments = node.args
            if (
                arguments.vararg
                or arguments.kwarg
                or arguments.kwonlyargs
                or arguments.defaults
                or arguments.posonlyargs
            ):
                raise ScriptError("lambda : arguments simples seulement", self._line(node))
            names = tuple(argument.arg for argument in arguments.args)
            return Val(_Lambda(names, node.body, tuple(self._scopes)), self._pc_trust())
        raise ScriptError(
            f"« {type(node).__name__} » n'est pas permis dans un script", self._line(node)
        )

    def _binop(self, op: type, left: Val, right: Val, node: ast.AST) -> Val:
        if op is ast.Pow:
            if not isinstance(right.value, (int, float)) or abs(right.value) > 64:
                raise BudgetExceeded("puissance trop grande", self._line(node))
            whole = isinstance(left.value, int) and not isinstance(left.value, bool)
            if whole and left.value.bit_length() * max(abs(int(right.value)), 1) > 4096:
                raise BudgetExceeded("nombre trop grand", self._line(node))
            result = left.value**right.value
            return Val(self._sized(result, node), left.trust.join(right.trust))
        if op not in _BINOPS:
            raise ScriptError("opérateur non permis", self._line(node))
        if op is ast.Mult:
            for a, b in ((left.value, right.value), (right.value, left.value)):
                if isinstance(a, (str, list, tuple)) and isinstance(b, int):
                    limit = MAX_TEXT if isinstance(a, str) else MAX_ITEMS
                    if len(a) * max(b, 0) > limit:
                        raise BudgetExceeded("valeur trop grosse", self._line(node))
        if isinstance(left.value, type) or isinstance(right.value, type):
            # « str | None » dans un schéma d'extraction : on garde le premier type.
            return Val(left.value if isinstance(left.value, type) else right.value)
        result = _BINOPS[op](left.value, right.value)
        return Val(self._sized(result, node), left.trust.join(right.trust))

    async def _fstring(self, node: ast.JoinedStr) -> Val:
        parts: list[str] = []
        trust = Trust.TRUSTED
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(str(part.value))
                continue
            assert isinstance(part, ast.FormattedValue)
            value = await self._eval(part.value)
            trust = trust.join(value.trust)
            text: Any = value.value
            if part.conversion == ord("r"):
                text = repr(text)
            spec = ""
            if part.format_spec is not None:
                spec = str((await self._eval(part.format_spec)).value)
                if any(len(digits) > 4 for digits in _DIGITS.findall(spec)):
                    raise BudgetExceeded("largeur de format trop grande", self._line(node))
            if isinstance(text, (str, int, float, bool)) or text is None:
                parts.append(format(text, spec) if spec else str(text))
            else:
                parts.append(str(text))
        return Val(self._sized("".join(parts), node), trust)

    async def _attribute(self, node: ast.Attribute) -> Val:
        if node.attr.startswith("_"):
            raise ScriptError("les attributs cachés sont interdits", self._line(node))
        base = await self._eval(node.value)
        target = base.value
        if isinstance(target, _Namespace):
            if node.attr not in self._namespaces.get(target.name, set()):
                known = ", ".join(sorted(self._namespaces.get(target.name, set())))
                raise ScriptError(
                    f"{target.name}.{node.attr} n'existe pas ; fonctions : {known}",
                    self._line(node),
                )
            return Val(_SdkFn(f"{target.name}.{node.attr}"))
        for kind, names in METHODS.items():
            if isinstance(target, kind) and node.attr in names:
                return Val(_Method(target, node.attr), base.trust)
        if isinstance(target, dict):
            if node.attr in target:
                return Val(target[node.attr], base.trust)
            fields = ", ".join(str(key) for key in list(target)[:12])
            raise ScriptError(
                f"champ inconnu « {node.attr} » ; champs : {fields}", self._line(node)
            )
        raise ScriptError(
            f"« .{node.attr} » n'est pas permis sur {type(target).__name__}", self._line(node)
        )

    async def _call(self, node: ast.Call) -> Val:
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            raise ScriptError("* et ** ne sont pas permis dans un appel", self._line(node))
        # `parallel(a(), b())` : les appels donnés partent ensemble.
        if isinstance(node.func, ast.Name) and node.func.id == "parallel" and not node.keywords:
            results = await asyncio.gather(*(self._eval(arg) for arg in node.args))
            return Val([r.value for r in results], join(*(r.trust for r in results)))
        callee = await self._eval(node.func)
        args = [await self._eval(arg) for arg in node.args]
        kwargs = {str(kw.arg): await self._eval(kw.value) for kw in node.keywords}
        line = self._line(node) or 0
        target = callee.value
        if isinstance(target, _SdkFn):
            return await self.host.call_sdk(target.name, args, kwargs, self._pc_trust(), line)
        if isinstance(target, _HostFn):
            if target.name == "expect":
                return self._expect(args, kwargs, node)
            if target.name == "print":
                self.prints.append(" ".join(str(arg.value) for arg in args)[:2000])
                return Val(None)
            return await self.host.call_function(target.name, args, kwargs, self._pc_trust(), line)
        trust = join(callee.trust, *(a.trust for a in args), *(k.trust for k in kwargs.values()))
        raw_args = [arg.value for arg in args]
        raw_kwargs = {name: value.value for name, value in kwargs.items()}
        if isinstance(target, _Method):
            self._guard_method(target, raw_args, node)
            result = getattr(target.target, target.name)(*raw_args, **raw_kwargs)
            self._sized(target.target, node)  # extend, update... grossissent sur place
            if target.name in ("keys", "values", "items"):
                result = [list(item) if isinstance(item, tuple) else item for item in result]
            return Val(self._sized(result, node), trust)
        if isinstance(target, _Lambda):
            return await self._apply(target, args, callee.trust, node)
        if callable(target) and any(target is builtin for builtin in BUILTINS.values()):
            functional = await self._functional(target, args, kwargs, trust, node)
            if functional is not None:
                return functional
            result = target(*raw_args, **raw_kwargs)
            return Val(self._sized(result, node), trust)
        raise ScriptError("ceci ne s'appelle pas", self._line(node))

    async def _apply(self, function: _Lambda, args: list[Val], trust: Trust, node: ast.AST) -> Val:
        """Appelle un lambda : son corps est évalué ici, avec ses paramètres liés."""
        if len(args) != len(function.params):
            raise ScriptError(
                f"lambda : {len(function.params)} argument(s) attendu(s), {len(args)} donné(s)",
                self._line(node),
            )
        saved = self._scopes
        self._scopes = [*function.scopes, dict(zip(function.params, args, strict=True))]
        if len(self._scopes) > 40:
            raise BudgetExceeded("lambda trop imbriqués", self._line(node))
        try:
            result = await self._eval(function.body)
        finally:
            self._scopes = saved
        return Val(result.value, result.trust.join(trust))

    async def _functional(
        self, target: Any, args: list[Val], kwargs: dict[str, Val], trust: Trust, node: ast.AST
    ) -> Val | None:
        """`sorted`, `min`, `max`, `map`, `filter` avec un lambda : la clé de chaque élément
        est calculée par l'interpréteur, puis la fonction de base fait le reste."""
        key = kwargs.get("key")
        first = args[0].value if args else None
        if target in (BUILTINS["map"], BUILTINS["filter"]) and isinstance(first, _Lambda):
            if len(args) != 2:
                raise ScriptError(
                    "map(fonction, liste) ou filter(fonction, liste)", self._line(node)
                )
            items = self._iterate(args[1].value, node)
            results = [
                await self._apply(first, [Val(i, args[1].trust)], trust, node) for i in items
            ]
            out_trust = join(trust, *(r.trust for r in results))
            if target is BUILTINS["map"]:
                return Val([r.value for r in results], out_trust)
            kept = [item for item, r in zip(items, results, strict=True) if r.value]
            return Val(kept, out_trust)
        if key is None or not isinstance(key.value, _Lambda):
            return None
        if target not in (BUILTINS["sorted"], BUILTINS["min"], BUILTINS["max"]):
            raise ScriptError("key=lambda : seulement pour sorted, min et max", self._line(node))
        source = args[0] if len(args) == 1 else Val([a.value for a in args], trust)
        items = self._iterate(source.value, node)
        keys = [await self._apply(key.value, [Val(i, source.trust)], trust, node) for i in items]
        out_trust = join(trust, *(k.trust for k in keys))
        pairs = list(zip((k.value for k in keys), range(len(items)), items, strict=True))
        if target is BUILTINS["sorted"]:
            reverse = bool(kwargs["reverse"].value) if "reverse" in kwargs else False
            ordered = sorted(pairs, key=lambda pair: pair[0], reverse=reverse)
            return Val([item for _, _, item in ordered], out_trust)
        if not pairs:
            default = kwargs.get("default")
            return Val(default.value if default else None, out_trust)
        chooser = min if target is BUILTINS["min"] else max
        return Val(chooser(pairs, key=lambda pair: pair[0])[2], out_trust)

    def _expect(self, args: list[Val], kwargs: dict[str, Val], node: ast.AST) -> Val:
        if not args:
            raise ScriptError("expect(condition, message)", self._line(node))
        if not args[0].value:
            message = args[1].value if len(args) > 1 else kwargs.get("message", Val("")).value
            raise ExpectationFailed(str(message or "condition non remplie"), self._line(node))
        return Val(True, args[0].trust)

    async def _comprehension(self, node: Any) -> Val:
        self._scopes.append({})
        trusts: list[Trust] = []
        results: list[Any] = []

        async def walk(index: int) -> None:
            if index == len(node.generators):
                if isinstance(node, ast.DictComp):
                    key, value = await self._eval(node.key), await self._eval(node.value)
                    results.append((key.value, value.value))
                    trusts.extend((key.trust, value.trust))
                else:
                    element = await self._eval(node.elt)
                    results.append(element.value)
                    trusts.append(element.trust)
                if len(results) > MAX_ITEMS:
                    raise BudgetExceeded("liste trop longue", self._line(node))
                return
            generator = node.generators[index]
            iterable = await self._eval(generator.iter)
            trusts.append(iterable.trust)
            for item in self._iterate(iterable.value, node):
                await self._tick(node)
                self._store(generator.target, Val(item, iterable.trust))
                keep = True
                for condition in generator.ifs:
                    test = await self._eval(condition)
                    trusts.append(test.trust)
                    if not test.value:
                        keep = False
                        break
                if keep:
                    await walk(index + 1)

        try:
            await walk(0)
        finally:
            self._scopes.pop()
        trust = join(*trusts)
        if isinstance(node, ast.DictComp):
            return Val(dict(results), trust)
        if isinstance(node, ast.SetComp):
            return Val(set(results), trust)
        return Val(results, trust)


HostFunction = Callable[[list[Val], dict[str, Val], Trust, int], Awaitable[Val]]
