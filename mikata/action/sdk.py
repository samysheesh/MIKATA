"""Le SDK des scripts (docs/AGENT_CORE.md §3.3) : une façade sur le registre d'outils.

Chaque outil enregistré déclare son nom dans le SDK (`sdk = "web.search"`), son effet et la
confiance de sa sortie. Ce module en tire :

- le **catalogue** : nom du SDK -> outil, avec sa signature ;
- la **référence** donnée au modèle : des signatures Python compactes, à la place des schémas
  JSON de tous les outils ;
- la **liaison des arguments** d'un appel de script (positionnels ou nommés) vers les champs
  du modèle Pydantic de l'outil.

Le SDK n'est pas un nouvel accès au système : un appel de script finit toujours par
`ToolRegistry.execute`, donc par la validation, le garde-fou et le journal.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from mikata.action.effects import Effect, Trust


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # « str », « int », « "a"|"b" »...
    required: bool
    default: Any = None

    def render(self) -> str:
        if self.required:
            return f"{self.name}: {self.kind}"
        return f"{self.name}: {self.kind} = {self.default!r}"


@dataclass(frozen=True)
class SdkEntry:
    name: str  # « web.search »
    tool: str  # « web_search »
    params: tuple[Param, ...]
    effect: Effect
    output: Trust
    returns: str
    doc: str
    expected_s: float
    announces: bool = False  # phase 16c : l'outil s'annonce lui-même

    @property
    def namespace(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def function(self) -> str:
        return self.name.split(".", 1)[1]

    def signature(self) -> str:
        args = ", ".join(param.render() for param in self.params)
        return f"{self.name}({args}) -> {self.returns}"


class SdkBindError(Exception):
    """Appel de script mal formé (argument inconnu, en trop ou manquant)."""


def _kind(annotation: Any) -> str:
    """Nom court d'un type de champ Pydantic, lisible dans une signature."""
    origin = get_origin(annotation)
    if origin is Literal:
        return "|".join(repr(value) for value in get_args(annotation))
    if origin in (Union, types.UnionType):
        parts = [_kind(arg) for arg in get_args(annotation) if arg is not type(None)]
        return parts[0] if len(parts) == 1 else "|".join(parts)
    if origin in (list, typing.List):  # noqa: UP006
        inner = get_args(annotation)
        return f"list[{_kind(inner[0])}]" if inner else "list"
    if origin is dict:
        return "dict"
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return "dict"
        return annotation.__name__
    return "str"


def _params(model: type[BaseModel]) -> tuple[Param, ...]:
    params: list[Param] = []
    for name, info in model.model_fields.items():
        required = info.is_required()
        default = None if required or info.default is PydanticUndefined else info.default
        params.append(Param(name, _kind(info.annotation), required, default))
    # Les obligatoires d'abord : un appel positionnel suit l'ordre de la signature affichée.
    return tuple(sorted(params, key=lambda p: not p.required))


def _first_sentence(text: str, limit: int = 110) -> str:
    sentence = " ".join(text.split())
    for mark in (". ", " : ", " ; "):
        if mark in sentence:
            sentence = sentence.split(mark, 1)[0]
            break
    sentence = sentence.rstrip(".")
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


class SdkCatalog:
    """Les fonctions du SDK disponibles sur cette plateforme, tirées du registre."""

    def __init__(self, tools: Mapping[str, Any], platform: str) -> None:
        entries: dict[str, SdkEntry] = {}
        for tool_name in sorted(tools):
            tool = tools[tool_name]
            platforms = getattr(tool, "platforms", ())
            if platforms and platform in ("win32", "darwin") and platform not in platforms:
                continue
            name = getattr(tool, "sdk", "") or f"tool.{tool_name}"
            if name in entries:  # même nom sur deux plateformes : la première trouvée gagne
                continue
            entries[name] = SdkEntry(
                name=name,
                tool=tool_name,
                params=_params(tool.params),
                effect=getattr(tool, "effect", Effect.CONSEQUENTIAL),
                output=getattr(tool, "output", Trust.LOCAL),
                returns=getattr(tool, "returns", "str"),
                doc=_first_sentence(tool.description),
                expected_s=getattr(tool, "expected_s", 0.3),
                announces=bool(getattr(tool, "announces", False)),
            )
        self.entries = entries

    def namespaces(self) -> dict[str, list[SdkEntry]]:
        grouped: dict[str, list[SdkEntry]] = {}
        for entry in self.entries.values():
            grouped.setdefault(entry.namespace, []).append(entry)
        return dict(sorted(grouped.items()))

    def get(self, name: str) -> SdkEntry | None:
        return self.entries.get(name)

    def by_tool(self, tool_name: str) -> SdkEntry | None:
        return next((e for e in self.entries.values() if e.tool == tool_name), None)

    def reference(self) -> str:
        """Référence compacte pour le prompt : une ligne par fonction, groupée par espace."""
        marks = {
            Effect.READ: "",
            Effect.ACT: "",
            Effect.CONSEQUENTIAL: " [conséquent]",
            Effect.DESTRUCTIVE: " [interdit]",
        }
        lines: list[str] = []
        for entries in self.namespaces().values():
            for entry in entries:
                untrusted = " [contenu non fiable]" if entry.output is Trust.UNTRUSTED else ""
                lines.append(f"{entry.signature()}  # {entry.doc}{marks[entry.effect]}{untrusted}")
        return "\n".join(lines)

    def structured_returns(self) -> str:
        """Outils qui rendent un objet à champs dans un programme (les autres : un texte)."""
        rows = [
            f"{entry.tool}(...) -> {entry.returns}"
            for entry in self.entries.values()
            if entry.returns != "str"
        ]
        return "\n".join(sorted(rows))

    def bind(
        self, entry: SdkEntry, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Arguments d'un appel de script -> champs du modèle de l'outil."""
        names = [param.name for param in entry.params]
        if len(args) > len(names):
            raise SdkBindError(
                f"{entry.name} : {len(args)} arguments donnés, {len(names)} au plus "
                f"({entry.signature()})"
            )
        bound = dict(zip(names, args, strict=False))
        for key, value in kwargs.items():
            if key not in names:
                raise SdkBindError(
                    f"{entry.name} : argument inconnu « {key} » ({entry.signature()})"
                )
            if key in bound:
                raise SdkBindError(f"{entry.name} : argument « {key} » donné deux fois")
            bound[key] = value
        missing = [p.name for p in entry.params if p.required and p.name not in bound]
        if missing:
            raise SdkBindError(
                f"{entry.name} : argument manquant « {missing[0]} » ({entry.signature()})"
            )
        # Phase 16g : un texte attendu, un élément lu donné (contrôle, lien de
        # `ui.window`, `browser.read` ou `pick`) : son nom, ou son texte.
        kinds = {param.name: param.kind for param in entry.params}
        for key, value in bound.items():
            if kinds.get(key) == "str" and isinstance(value, dict):
                label = value.get("name") or value.get("text")
                if isinstance(label, str) and label.strip():
                    bound[key] = label
        return bound
