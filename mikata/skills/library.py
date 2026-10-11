"""Bibliothèque de savoir-faire (phase 12c) : les procédures que MIKATA a apprises.

Un savoir-faire est une chaîne d'appels d'outils qui a déjà réussi (« lance le dernier
opening de black clover » -> browser_open, browser_read, browser_click). La commande
suivante qui y ressemble (correspondance floue) rejoue la chaîne sans exploration.

Fichier : data/skills/skills.json — écriture atomique et chargement tolérant, comme la
mémoire longue. Un savoir-faire est une **donnée**, jamais une instruction : chaque étape
repassera par le garde-fou au rejeu, et aucune sortie d'outil n'est stockée.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DESC_MAX = 300
_TRIGGER_MAX = 300


def _fold(text: str) -> str:
    """Forme de comparaison : minuscules, sans accents ni ponctuation."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", text).strip()


# Mots-outils courants des commandes vocales : « passe le pc en silencieux » et
# « passe à la chanson d'après » ne se ressemblent pas, malgré le « passe ».
_STOP_WORDS = frozenset(
    [
        "passe", "mets", "met", "ouvre", "lance", "ferme", "fait", "faire", "cherche",
        "regarde", "dis", "donne", "affiche", "montre", "aller", "avec", "pour", "dans",
        "sur", "cette", "ce", "cet", "mon", "mes", "ton", "tes", "tout", "tous", "plus",
        "moins", "apres", "avant", "pendant", "encore", "juste", "vite", "bien", "peu",
        "beaucoup", "dernier", "derniere", "nouveau", "nouvelle", "autre", "autres",
        "petit", "petite", "grand", "grande", "voix", "ordre", "donnee", "information",
        "info", "chose", "choses",
    ]
)  # fmt: skip


def _keywords(text: str) -> set[str]:
    """Mots significatifs (≥ 4 lettres, hors mots-outils) : ce qui distingue deux commandes."""
    return {word for word in _fold(text).split() if len(word) >= 4 and word not in _STOP_WORDS}


# -- rejeu local (phase 12e) ------------------------------------------------------------


def compile_pattern(slots: str | None, params: list[str]) -> re.Pattern[str] | None:
    """Compile la phrase généralisée en regex, ou None si elle est invalide.

    « où est {query} » -> `^ où est (?P<query>.+?) $` sur du texte plié (accents ignorés).
    Un `{param}` inconnu de `params`, un nom répété, un pattern vide, ou un pattern sans
    mot littéral (il matcherait n'importe quoi) donnent None (repli sur le plan LLM).
    """
    if not slots:
        return None
    pieces: list[str] = []
    has_literal = False
    for part in re.split(r"(\{[a-z_]+\})", slots):
        if not part:
            continue
        match = re.fullmatch(r"\{([a-z_]+)\}", part)
        if match:
            name = match.group(1)
            if name not in params:
                return None
            pieces.append(f"(?P<{name}>.+?)")
        else:
            folded = _fold(part)
            if folded:
                pieces.append(re.escape(folded))
                has_literal = has_literal or any(
                    len(word) >= 3 and word not in _STOP_WORDS for word in folded.split()
                )
    if not pieces or not has_literal:
        return None
    try:
        return re.compile(r"^\s*" + "".join(pieces) + r"\s*$")
    except re.error:
        return None


def bind_slots(pattern: re.Pattern[str], phrase: str) -> dict[str, str] | None:
    """Les valeurs des paramètres extraites de la phrase parlée, ou None si elle ne
    correspond pas au pattern. Les captures sont des données, jamais des instructions."""
    match = pattern.match(_fold(phrase))
    if match is None:
        return None
    bindings = {name: value.strip() for name, value in match.groupdict().items()}
    if not bindings:
        return None
    return bindings


def _clean_step(step: dict[str, Any]) -> dict[str, Any]:
    """La forme canonique d'une étape : outil + args, et les liaisons `outputs` (12e)."""
    clean: dict[str, Any] = {"tool": str(step["tool"]), "args": dict(step.get("args") or {})}
    outputs = step.get("outputs")
    if isinstance(outputs, dict) and outputs:
        clean["outputs"] = {str(key): str(value) for key, value in outputs.items()}
    return clean


@dataclass
class Skill:
    """Une procédure apprise, rejouable."""

    name: str
    description: str
    trigger: str  # exemple de commande qui l'a créée (correspondance floue dessus)
    steps: list[dict[str, Any]]  # [{"tool": ..., "args": {...}}] avec trous {param}
    params: list[str] = field(default_factory=list)  # noms des trous (variables)
    slots: str | None = None  # phrase généralisée, ex. « où est {query} » (phase 12e)
    verify: dict[str, Any] | None = None  # {"tool": ..., "args": {...}} ou None
    failures: int = 0
    enabled: bool = True
    added: str = ""
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "steps": self.steps,
            "params": self.params,
            "slots": self.slots,
            "verify": self.verify,
            "failures": self.failures,
            "enabled": self.enabled,
            "added": self.added,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Skill | None:
        try:
            steps = [_clean_step(step) for step in raw["steps"] if isinstance(step, dict)]
            if not steps:
                return None
            verify = raw.get("verify")
            if verify is not None and not isinstance(verify, dict):
                verify = None
            slots = raw.get("slots")
            return cls(
                name=str(raw["name"]),
                description=str(raw["description"]),
                trigger=str(raw["trigger"]),
                steps=steps,
                params=[str(p) for p in raw.get("params") or []],
                slots=str(slots) if slots else None,
                verify=verify if isinstance(verify, dict) else None,
                failures=int(raw.get("failures") or 0),
                enabled=bool(raw.get("enabled", True)),
                added=str(raw.get("added") or ""),
                last_used=str(raw.get("last_used") or ""),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def plan_text(self) -> str:
        """Le plan donné au LLM quand la procédure a des paramètres à remplir."""
        lines = [f"« {self.name} » : {self.description}. Étapes connues :"]
        for step in self.steps:
            args = ", ".join(f"{key}={value!r}" for key, value in step["args"].items())
            lines.append(f"- {step['tool']}({args})")
        if self.params:
            lines.append(
                "Remplis les trous {" + "}, {".join(self.params) + "} avec ce que demande "
                "l'utilisateur."
            )
        return "\n".join(lines)


class SkillLibrary:
    def __init__(
        self,
        path: Path,
        max_skills: int,
        max_failures: int,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.path = path
        self.max_skills = max_skills
        self.max_failures = max_failures
        self._clock = clock
        self._skills: list[Skill] = self._load()

    def _load(self) -> list[Skill]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            skills = [Skill.from_dict(item) for item in data.get("skills", [])]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning("Savoir-faire illisible ({}) : {}", self.path, exc)
            return []
        return [skill for skill in skills if skill is not None][: self.max_skills]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        payload = {"skills": [skill.to_dict() for skill in self._skills]}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- lecture ----------------------------------------------------------------------

    def index(self) -> list[str]:
        """Une ligne par savoir-faire, pour le prompt système (préfixe cachable)."""
        lines = []
        for skill in self._skills:
            state = "" if skill.enabled else " (désactivé)"
            lines.append(f"{skill.name} — {skill.description}{state}")
        return lines

    def all(self) -> list[Skill]:
        return list(self._skills)

    def get(self, name: str) -> Skill | None:
        for skill in self._skills:
            if skill.name == name:
                return skill
        return None

    def match(self, command: str, cutoff: float) -> Skill | None:
        """Le savoir-faire qui correspond à la commande.

        D'abord les patterns de slots (phase 12e) : si la phrase colle exactement au
        pattern généralisé d'un savoir-faire, c'est lui — déterministe, même sans mot
        commun avec la phrase d'origine. Sinon, correspondance floue sur l'exemple et la
        description, avec un mot significatif commun (≥ 4 lettres) : sans cela, deux
        phrases courtes sans rapport se ressemblent trop, et on rejouerait une procédure
        qui n'a rien à voir.
        """
        from rapidfuzz import fuzz

        if not self._skills:
            return None
        wanted = _fold(command)
        for skill in self._skills:
            if skill.enabled and skill.slots:
                pattern = compile_pattern(skill.slots, skill.params)
                if pattern is not None and bind_slots(pattern, wanted) is not None:
                    return skill
        best_score = cutoff
        best: Skill | None = None
        for skill in self._skills:
            combined = _fold(skill.trigger) + " | " + _fold(skill.description)
            score = fuzz.WRatio(wanted, combined)
            if score >= best_score and _keywords(wanted) & _keywords(combined):
                best_score, best = score, skill
        return best

    # -- écriture ---------------------------------------------------------------------

    def learn(
        self,
        name: str,
        description: str,
        trigger: str,
        steps: list[dict[str, Any]],
        params: list[str] | None = None,
        verify: dict[str, Any] | None = None,
        slots: str | None = None,
    ) -> Skill | None:
        """Mémorise une procédure (validée), ou None si elle est déjà connue / invalide."""
        from mikata.tools.registry import registered_tools

        name = name.strip()
        description = " ".join(description.split())[:_DESC_MAX]
        trigger = " ".join(trigger.split())[:_TRIGGER_MAX]
        known = registered_tools()
        clean_steps = [_clean_step(step) for step in steps]
        if not NAME_RE.match(name):
            logger.warning("Savoir-faire refusé : nom invalide « {} ».", name)
            return None
        if not description or not trigger or not clean_steps:
            logger.warning("Savoir-faire refusé : description, exemple et étapes requis.")
            return None
        if any(step["tool"] not in known for step in clean_steps):
            unknown = [step["tool"] for step in clean_steps if step["tool"] not in known]
            logger.warning("Savoir-faire refusé : outils inconnus {}.", unknown)
            return None
        if any(_fold(skill.description) == _fold(description) for skill in self._skills):
            return None  # déjà connu
        params = list(params or [])
        slots = " ".join(slots.split()) if slots else None
        if slots is not None and compile_pattern(slots, params) is None:
            slots = None  # pattern invalide : plan LLM seulement (dégradation propre)
        skill = Skill(
            name=name,
            description=description,
            trigger=trigger,
            steps=clean_steps,
            params=params,
            slots=slots,
            verify=verify if isinstance(verify, dict) else None,
            added=self._clock().isoformat(timespec="seconds"),
        )
        if self.get(skill.name) is not None:  # nom déjà pris : on remplace la procédure
            self._skills = [s for s in self._skills if s.name != skill.name]
        self._skills.append(skill)
        if len(self._skills) > self.max_skills:
            # Le moins utilisé récemment sort (jamais utilisé : le plus ancien).
            self._skills.sort(key=lambda s: s.last_used or s.added)
            self._skills = self._skills[-(self.max_skills) :]
        self._save()
        return skill

    def touch(self, name: str) -> None:
        """Procédure rejouée avec succès : recence, réactivée, échecs à zéro."""
        skill = self.get(name)
        if skill is None:
            return
        skill.last_used = self._clock().isoformat(timespec="seconds")
        skill.failures = 0
        skill.enabled = True
        self._save()

    def record_failure(self, name: str) -> None:
        """Étape en échec au rejeu : comptée ; désactivée au-delà de `max_failures`."""
        skill = self.get(name)
        if skill is None:
            return
        skill.failures += 1
        if skill.failures >= self.max_failures:
            skill.enabled = False
            logger.warning("Savoir-faire « {} » désactivé après {} échec(s).", name, skill.failures)
        self._save()

    def forget(self, query: str) -> Skill | None:
        """Retire le savoir-faire le plus proche de `query` ; le renvoie, ou None."""
        skill = self.match(query, cutoff=60)
        if skill is None:
            return None
        self._skills = [s for s in self._skills if s.name != skill.name]
        self._save()
        return skill
