"""Exécution d'un script contre les vrais outils (docs/AGENT_CORE.md §3.4, §3.5).

`ScriptRuntime` est l'hôte de l'interpréteur : chaque appel du SDK passe par la politique
(`policy.decide`), puis par `ToolRegistry.execute` (validation, garde-fou, journal). Il note
tout ce qui a été fait, pour la trace, la vérification et l'apprentissage.

Trois issues qu'un script ne peut pas attraper : le refus d'une politique ou du garde-fou,
l'annulation par l'utilisateur, et le dépassement du budget.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from mikata.action.effects import Effect, Trust, join
from mikata.action.extract import ExtractionError, choose, extract
from mikata.action.interp import (
    BudgetExceeded,
    ExpectationFailed,
    Interpreter,
    ScriptError,
    ScriptFailure,
    Val,
)
from mikata.action.policy import Verdict, confirm_question, decide
from mikata.action.sdk import SdkBindError, SdkCatalog
from mikata.llm.base import ToolCallReq, Usage
from mikata.tools.base import current_turn

SKILL_TIMEOUT_S = 40.0  # une compétence appelée depuis un programme
MAX_CALLS = 60  # appels au SDK par script
MAX_PICK = 200  # options données à pick() : autant que les contrôles de uia_window (phase 17d)
# Phase 17f : types des contrôles d'interface dont une compétence retient le choix. Pas les
# contenus (lien, élément d'une liste, texte) : le bon choix peut y changer d'une fois sur
# l'autre (« le dernier opening » d'un anime).
REMEMBERED_KINDS = frozenset(
    {"bouton", "case", "menu", "onglet", "champ", "liste déroulante", "curseur"}
)
MAX_ACTIONS = 25  # dont appels qui agissent
# Phase 17f : actions d'un programme sur une cible tirée d'un contenu lu (une ligne de
# résultats, un lien, un contrôle lu à l'écran). Le 15/10, pour « lance Jet Lag de
# Tiakola », un programme a cliqué « Play » sur chacun des 18 résultats qui citaient le
# titre (une boucle sans arrêt après le premier) : la musique a été relancée 18 fois en
# 10 s. Une demande qui vise plusieurs choses lues les fait une à une, par l'outil direct,
# chacune vue par le modèle ; c'est aussi une limite de plus contre un contenu piégé.
MAX_CONTENT_ACTIONS = 1
CONTENT_ACTIONS_REASON = (
    "un programme n'agit qu'une fois sur une cible lue à l'écran (ligne, lien, contrôle "
    "lus) : la première action est faite, celle-ci ne l'est pas. Pour une seule chose "
    "demandée, c'est fini ; pour en viser une autre, utilise l'outil direct"
)
ASK_TIMEOUT_S = 8.0
REDACTED = "(détail masqué : il dépend d'un contenu lu ailleurs)"


def pick_label(item: Any) -> str:
    """Étiquette d'une option de pick() : le texte d'une valeur simple ; pour un élément
    structuré (contrôle, lien : {"name": ..., "kind": ...}), ses champs non vides."""
    if isinstance(item, dict):
        parts = [str(v).strip() for v in item.values() if isinstance(v, str | int | float)]
        return " — ".join(part for part in parts if part)
    return str(item).strip()


def _identity(item: Any) -> tuple[str, str]:
    if not isinstance(item, dict):
        return "", ""
    return str(item.get("name") or "").strip(), str(item.get("kind") or "").strip()


def choice_key(items: list[Any], index: int) -> str:
    """Phase 17f : identité stable de l'option `index` d'un pick(), pour qu'une compétence
    la retienne : « Rendre muet — bouton #1 » (nom, type, rang parmi les homonymes ; pas
    l'état, « désactivé », qui change). "" pour ce qui n'est pas un contrôle d'interface."""
    name, kind = _identity(items[index])
    if not name or kind not in REMEMBERED_KINDS:
        return ""
    nth = sum(1 for other in items[: index + 1] if _identity(other) == (name, kind))
    return f"{name} — {kind} #{nth}"


def find_choice(items: list[Any], key: str) -> int | None:
    """L'option dont l'identité (`choice_key`) est `key`, ou None si elle n'y est plus."""
    seen: dict[tuple[str, str], int] = {}
    for index, item in enumerate(items):
        name, kind = identity = _identity(item)
        seen[identity] = seen.get(identity, 0) + 1
        if name and kind in REMEMBERED_KINDS and f"{name} — {kind} #{seen[identity]}" == key:
            return index
    return None


@dataclass
class SkillChoice:
    """Phase 17f : un choix fait par pick() dans une compétence, à retenir si elle réussit
    (`recalled` : repris d'une fois précédente, à oublier si elle échoue)."""

    skill: str
    question: str
    values: dict[str, Any]
    key: str
    recalled: bool = False


class PolicyDenied(Exception):
    """Appel refusé par une politique ou par le garde-fou : le script s'arrête."""


class GuardRefused(Exception):
    """Le garde-fou d'un outil a refusé (liste noire, dossier interdit...) : le script
    s'arrête, mais le modèle peut reprendre autrement (le refus dit souvent comment)."""


class Cancelled(Exception):
    """L'utilisateur a dit non, ou « stop » : le script s'arrête."""


@dataclass
class ActionRecord:
    sdk: str
    tool: str
    args: dict[str, Any]
    effect: str
    arg_trust: str
    verdict: str
    ok: bool
    text: str
    seconds: float


@dataclass
class ScriptOutcome:
    """Résultat d'un script, tel que la boucle agent le lit."""

    ok: bool
    value: Any = None
    trust: Trust = Trust.TRUSTED
    kind: str = ""  # script | attente | outil | garde-fou | refus | annulé | budget
    error: str = ""  # message pour le modèle, expurgé si besoin
    prints: list[str] = field(default_factory=list)
    records: list[ActionRecord] = field(default_factory=list)
    read_trust: Trust = Trust.TRUSTED  # pire niveau de ce que le script a lu
    steps: int = 0
    seconds: float = 0.0

    @property
    def retryable(self) -> bool:
        """Le modèle peut-il corriger et réessayer ? Jamais après un refus ou un « non »."""
        return not self.ok and self.kind in ("script", "attente", "outil", "garde-fou")

    def text(self) -> str:
        """Valeur de retour en texte (pour la dire, ou la rendre au modèle)."""
        if self.value is None:
            return ""
        if isinstance(self.value, str):
            return self.value
        return json.dumps(self.value, ensure_ascii=False, default=str)


class ScriptRuntime:
    """Hôte de l'interpréteur pour un tour : politique, outils, extraction, parole."""

    def __init__(
        self,
        registry: Any,
        catalog: SdkCatalog,
        *,
        llm: Any = None,
        full_access: bool = False,
        turn_trust: Trust = Trust.TRUSTED,
        extract_effort: str | None = "none",
        skills: dict[str, Any] | None = None,
        jev: Any = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog
        self.llm = llm
        self.full_access = full_access
        self.turn_trust = turn_trust
        self.extract_effort = extract_effort
        self.jev = jev  # phase 15l : pick() choisit d'abord avec Jev
        self.skills = skills or {}  # phase 15e : compétences appelables, par nom d'appel
        self.skill_uses: list[tuple[str, bool, str]] = []  # (nom, réussie, erreur)
        self.skill_choices: list[SkillChoice] = []  # phase 17f : choix de pick() à retenir
        self._skill: Any = None  # la compétence en cours d'exécution
        self._skill_values: dict[str, Any] = {}
        self.records: list[ActionRecord] = []
        self.tool_calls: list[ToolCallReq] = []
        self.tool_results: list[Any] = []
        self.read_trust = Trust.TRUSTED
        self.usage: Usage | None = None
        self.llm_calls = 0
        self._calls = 0
        self._actions = 0
        self._content_actions = 0  # phase 17f : actions sur une cible lue (MAX_CONTENT_ACTIONS)

    # -- interface de l'interpréteur ------------------------------------------------------

    def namespaces(self) -> dict[str, set[str]]:
        spaces: dict[str, set[str]] = {}
        for entry in self.catalog.entries.values():
            spaces.setdefault(entry.namespace, set()).add(entry.function)
        if self.skills:  # phase 15e : skills.<nom>(...) à côté de skills.list, etc.
            spaces.setdefault("skills", set()).update(self.skills)
        return spaces

    def functions(self) -> set[str]:
        """Fonctions hors espaces de noms : extraction, parole, et chaque outil sous son nom
        d'outil (`set_volume(level=30)` vaut `audio.volume(level=30)`)."""
        host = {"extract", "pick", "say", "ask"}
        return host | {entry.tool for entry in self.catalog.entries.values()}

    async def call_sdk(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val:
        namespace, _, function = name.partition(".")
        if namespace == "skills" and function in self.skills:
            return await self._call_skill(function, args, kwargs, pc, line)
        entry = self.catalog.get(name)
        if entry is None:
            raise ScriptError(f"fonction inconnue : {name}", line)
        try:
            bound = self.catalog.bind(
                entry, tuple(a.value for a in args), {k: v.value for k, v in kwargs.items()}
            )
        except SdkBindError as exc:
            raise ScriptError(str(exc), line) from None
        arg_trust = join(pc, *(a.trust for a in args), *(v.trust for v in kwargs.values()))
        self._calls += 1
        if self._calls > MAX_CALLS:
            raise BudgetExceeded(f"plus de {MAX_CALLS} appels dans un script", line)
        if entry.effect is not Effect.READ:
            self._actions += 1
            if self._actions > MAX_ACTIONS:
                raise BudgetExceeded(f"plus de {MAX_ACTIONS} actions dans un script", line)
        decision = decide(entry, arg_trust, self.turn_trust, full_access=self.full_access)
        if decision.verdict is Verdict.DENY:
            self._record(entry, bound, arg_trust, "refusé", False, decision.reason, 0.0)
            logger.warning("Politique : {} refusé ({}).", entry.name, decision.reason)
            raise PolicyDenied(decision.reason)
        on_content = entry.effect is not Effect.READ and arg_trust is Trust.UNTRUSTED
        if on_content and self._content_actions >= MAX_CONTENT_ACTIONS:
            self._record(entry, bound, arg_trust, "refusé", False, CONTENT_ACTIONS_REASON, 0.0)
            logger.warning("Programme : {} non fait ({}).", entry.name, "une action par cible lue")
            raise ScriptFailure(CONTENT_ACTIONS_REASON)
        if decision.verdict is Verdict.CONFIRM:
            guard = self.registry.ctx.guard
            accepted = guard is not None and await guard.confirm(
                confirm_question(entry), current_turn()
            )
            if not accepted:
                self._record(entry, bound, arg_trust, "non confirmé", False, decision.reason, 0.0)
                raise Cancelled("l'utilisateur n'a pas confirmé")
        call = ToolCallReq(f"s{uuid.uuid4().hex[:8]}", entry.tool, bound)
        result = await self.registry.execute(call)
        self.tool_calls.append(call)
        self.tool_results.append(result)
        if on_content and result.ok:  # seule une action faite compte
            self._content_actions += 1
        self._record(
            entry, bound, arg_trust, decision.verdict.value, result.ok, result.content,
            result.duration_s,
        )  # fmt: skip
        if not result.ok:
            if result.status == "refusé":
                raise GuardRefused(result.content)
            if result.status == "annulé":
                raise Cancelled(result.content)
            tainted = entry.output is Trust.UNTRUSTED or arg_trust is Trust.UNTRUSTED
            raise ScriptFailure(result.content, tainted=tainted)
        trust = join(entry.output, arg_trust)
        if entry.effect is Effect.READ or entry.output is Trust.UNTRUSTED:
            self.read_trust = self.read_trust.join(entry.output)
        value = result.data if result.data is not None else result.content
        return Val(value, trust)

    async def _call_skill(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val:
        """Phase 15e : exécute une compétence (son programme, avec ses paramètres comme
        variables) dans le même hôte : mêmes politiques, même budget, même trace."""
        skill = self.skills[name]
        if self._skill is not None:
            raise ScriptError("une compétence n'en appelle pas une autre", line)
        params: list[str] = list(skill.params)
        if len(args) > len(params):
            raise ScriptError(f"skills.{name} : {len(params)} paramètre(s) au plus", line)
        values = dict(zip(params, args, strict=False))
        for key, value in kwargs.items():
            if key not in params:
                raise ScriptError(f"skills.{name} : paramètre inconnu {key}", line)
            values[key] = value
        missing = [p for p in params if p not in values]
        if missing:
            raise ScriptError(f"skills.{name} : il manque {', '.join(missing)}", line)
        variables = {k: Val(v.value, join(pc, v.trust)) for k, v in values.items()}
        interpreter = Interpreter(self, timeout_s=SKILL_TIMEOUT_S, variables=variables)
        self._skill = skill
        self._skill_values = {
            key: v.value if isinstance(v.value, str | int | float | bool) else str(v.value)
            for key, v in variables.items()
        }
        try:
            result = await interpreter.run(skill.code)
        except ScriptError as exc:
            self.skill_uses.append((name, False, str(exc)))
            raise type(exc)(f"compétence {name} : {exc}", line) from None
        except Exception as exc:
            self.skill_uses.append((name, False, str(exc)))
            raise
        finally:
            self._skill, self._skill_values = None, {}
        self.skill_uses.append((name, True, ""))
        return result

    async def call_function(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val:
        entry = self.catalog.by_tool(name)
        if entry is not None:  # un outil appelé par son nom : même chemin que le SDK
            return await self.call_sdk(entry.name, args, kwargs, pc, line)
        values = [a.value for a in args]
        named = {k: v.value for k, v in kwargs.items()}
        trust = join(pc, *(a.trust for a in args), *(v.trust for v in kwargs.values()))
        if name == "extract":
            return await self._extract(values, named, trust, line)
        if name == "pick":
            return await self._pick(values, named, trust, line)
        turn = current_turn()
        if name == "say":
            text = " ".join(str(v) for v in values).strip()
            if turn is not None and text:
                turn.speak(text)
            return Val(None)
        if name == "ask":
            question = str(values[0]) if values else ""
            answer = await turn.ask(question, ASK_TIMEOUT_S) if turn is not None else None
            return Val(answer or "", Trust.TRUSTED)  # la parole de l'utilisateur
        raise ScriptError(f"fonction inconnue : {name}", line)

    async def _pick(self, values: list[Any], named: dict[str, Any], trust: Trust, line: int) -> Val:
        """pick(liste, question) : l'élément de la liste qui répond le mieux, recopié tel
        quel. Jev d'abord (~0,4 s) ; sans lui, ou s'il ne répond pas, un appel LLM qui
        rend le numéro de l'option la plus probable (phase 16b : `extract`, qui ne devine
        jamais, répondait « rien » quand aucune option n'était certaine). Phase 17f : dans
        une compétence, le contrôle choisi la dernière fois qu'elle a réussi est repris
        sans rien demander, s'il est encore là."""
        items = values[0] if values else named.get("items")
        if isinstance(items, dict):  # phase 17b : {window, items: [...]} -> sa seule liste
            lists = [value for value in items.values() if isinstance(value, list)]
            items = lists[0] if len(lists) == 1 else items
        question = str(values[1] if len(values) > 1 else named.get("question", "")).strip()
        if not isinstance(items, list) or not question:
            raise ScriptError("pick(liste, question)", line)
        # Phase 16g : le modèle de choix voit une étiquette lisible, le script reçoit
        # l'élément lui-même (un contrôle, un lien...), pas sa représentation en texte.
        kept = [(item, pick_label(item)) for item in items]
        kept = [(item, label) for item, label in kept if label][:MAX_PICK]
        if not kept:
            raise ScriptFailure("rien à choisir : la liste est vide")
        originals, options = [item for item, _ in kept], [label for _, label in kept]
        result_trust = join(trust, Trust.LOCAL)
        recalled = self._recall(question, originals)
        if recalled is not None:
            return Val(originals[recalled], result_trust)
        if self.jev is not None:
            picked = await self.jev.pick(question, options)
            if picked is not None:
                self._note_choice(question, originals, picked[0])
                return Val(originals[picked[0]], result_trust)
        if self.llm is None:
            raise ScriptFailure("choix impossible (pas de modèle)")
        self.llm_calls += 1
        try:
            index, usage = await choose(self.llm, options, question, effort=self.extract_effort)
        except Exception as exc:  # hors ligne
            raise ScriptFailure(f"choix impossible : {exc}") from None
        if usage is not None:
            self.usage = usage if self.usage is None else self.usage + usage
        if index is None:
            raise ScriptFailure("aucune option ne répond à la question")
        self._note_choice(question, originals, index)
        return Val(originals[index], result_trust)

    def _recall(self, question: str, items: list[Any]) -> int | None:
        """Phase 17f : dans une compétence, l'option retenue pour cette question et ces
        valeurs, si elle est encore parmi `items`."""
        if self._skill is None:
            return None
        key = self._skill.recall(question, self._skill_values)
        index = find_choice(items, key) if key else None
        if index is None:
            return None
        self.skill_choices.append(
            SkillChoice(self._skill.call_name, question, dict(self._skill_values), key, True)
        )
        logger.info("Choix retenu par la compétence « {} » : « {} ».", self._skill.name, key)
        return index

    def _note_choice(self, question: str, items: list[Any], index: int) -> None:
        if self._skill is None:
            return
        key = choice_key(items, index)
        if key:
            self.skill_choices.append(
                SkillChoice(self._skill.call_name, question, dict(self._skill_values), key)
            )

    async def _extract(
        self, values: list[Any], named: dict[str, Any], trust: Trust, line: int
    ) -> Val:
        if len(values) < 2 and "fields" not in named:
            raise ScriptError("extract(donnée, champs, question)", line)
        if self.llm is None:
            raise ScriptFailure("extraction indisponible (pas de modèle)")
        data = values[0] if values else named.get("data")
        fields = values[1] if len(values) > 1 else named.get("fields")
        question = values[2] if len(values) > 2 else named.get("question", "")
        self.llm_calls += 1
        try:
            result, usage = await extract(
                self.llm, data, fields, str(question or ""), effort=self.extract_effort
            )
        except ExtractionError as exc:
            raise ScriptFailure(str(exc)) from None
        if usage is not None:
            self.usage = usage if self.usage is None else self.usage + usage
        # Ce que rend l'extraction vient du contenu : jamais mieux que « local ».
        return Val(result, join(trust, Trust.LOCAL))

    def _record(
        self,
        entry: Any,
        args: dict[str, Any],
        arg_trust: Trust,
        verdict: str,
        ok: bool,
        text: str,
        seconds: float,
    ) -> None:
        self.records.append(
            ActionRecord(
                sdk=entry.name,
                tool=entry.tool,
                args=args,
                effect=entry.effect.value,
                arg_trust=arg_trust.label,
                verdict=verdict,
                ok=ok,
                text=text[:500],
                seconds=round(seconds, 3),
            )
        )

    # -- exécution ------------------------------------------------------------------------

    async def run(
        self, code: str, *, timeout_s: float = 60.0, variables: dict[str, Val] | None = None
    ) -> ScriptOutcome:
        """Exécute `code` ; ne lève jamais (sauf annulation de la tâche asyncio)."""
        interpreter = Interpreter(self, timeout_s=timeout_s, variables=variables)
        self._content_actions = 0  # par programme
        start = time.monotonic()
        outcome = ScriptOutcome(ok=False)
        try:
            result = await interpreter.run(code)
            outcome = ScriptOutcome(ok=True, value=result.value, trust=result.trust)
        except ExpectationFailed as exc:
            outcome = ScriptOutcome(ok=False, kind="attente", error=f"attendu : {exc}")
        except BudgetExceeded as exc:
            outcome = ScriptOutcome(ok=False, kind="budget", error=str(exc))
        except ScriptError as exc:
            outcome = ScriptOutcome(ok=False, kind="script", error=str(exc))
        except ScriptFailure as exc:
            message = REDACTED if exc.tainted else str(exc)
            outcome = ScriptOutcome(ok=False, kind="outil", error=f"outil en échec : {message}")
        except PolicyDenied as exc:
            outcome = ScriptOutcome(ok=False, kind="refus", error=f"refusé : {exc}")
        except GuardRefused as exc:
            outcome = ScriptOutcome(ok=False, kind="garde-fou", error=str(exc))
        except Cancelled as exc:
            outcome = ScriptOutcome(ok=False, kind="annulé", error=str(exc))
        outcome.prints = interpreter.prints
        outcome.records = list(self.records)
        outcome.read_trust = self.read_trust
        outcome.steps = interpreter.steps
        outcome.seconds = round(time.monotonic() - start, 3)
        return outcome
