"""Préférences apprises (phase 16e, docs/AGENT_CORE.md §3.7).

Une préférence est un couple sujet → valeur (« musique → Spotify », « météo → Lyon »),
avec le nombre de fois où l'utilisateur l'a montrée. Elle ne vient que de ses propres mots :
quand il **corrige** MIKATA (« non, sur Spotify »), un petit appel au modèle ne voit que sa
demande et sa correction (`extract`), jamais ce que MIKATA a lu ou fait. Elle s'apprend donc
même quand le tour corrigé avait lu une page web (la réflexion des leçons, elle, ne tourne
pas après un contenu non fiable). Si l'utilisateur change d'avis, la nouvelle valeur
remplace l'ancienne et repart de un.

Les préférences sont données au modèle à chaque commande (préfixe stable : elles changent
rarement), avec leur nombre de confirmations ; « oublie que je préfère... » les retire.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.learn.skills import fold, words
from mikata.llm.base import Done, LLMClient, TextDelta, Usage

PREFERENCES_FILE = "data/preferences.json"
MAX_PREFERENCES = 40
MAX_CHARS = 80
MAX_PER_CORRECTION = 3

EXTRACT_SYSTEM = """\
L'utilisateur d'un assistant vocal a fait une demande, puis l'a corrigée. Si sa correction \
montre une préférence qu'il voudra sans doute à chaque fois (une appli, un site, une ville, \
une façon de faire), donne-la : un sujet court (musique, vidéo, navigateur, météo...) et sa \
valeur, avec ses mots. Si la correction ne vaut que pour cette fois (un autre titre, une \
autre personne, une erreur d'écoute), il n'y a pas de préférence.

Réponds UNIQUEMENT par un objet JSON : {"preferences": [{"sujet": "...", "valeur": "..."}]} \
(liste vide s'il n'y en a pas)."""


def parse_preferences(text: str) -> list[tuple[str, str]]:
    """(sujet, valeur) des préférences d'une réponse JSON du modèle."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return []
    try:
        raw: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    items = raw.get("preferences") if isinstance(raw, dict) else None
    found: list[tuple[str, str]] = []
    for item in (items or [])[:MAX_PER_CORRECTION] if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("sujet") and item.get("valeur"):
            found.append((str(item["sujet"]), str(item["valeur"])))
    return found


async def extract(
    llm: LLMClient, request: str, correction: str, effort: str | None = "none"
) -> tuple[list[tuple[str, str]], Usage | None]:
    """Préférences montrées par une correction, d'après les seuls mots de l'utilisateur."""
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": f"Demande : « {request} »\nCorrection : « {correction} »"},
    ]
    chunks: list[str] = []
    usage: Usage | None = None
    async for event in llm.stream(messages, [], reasoning_effort=effort):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, Done):
            usage = event.usage
    return parse_preferences("".join(chunks)), usage


@dataclass
class Preference:
    subject: str  # « musique », « navigateur », « météo »
    value: str  # « Spotify »
    seen: int = 1  # fois où l'utilisateur l'a montrée
    source: str = "correction"
    updated: str = ""

    @property
    def label(self) -> str:
        times = "une fois" if self.seen == 1 else f"{self.seen} fois"
        return f"{self.subject} : {self.value} (montré {times})"


class PreferenceStore:
    def __init__(self, path: Path, max_items: int = MAX_PREFERENCES) -> None:
        self.path = path
        self.max_items = max_items
        self.items: list[Preference] = self._load()

    def _load(self) -> list[Preference]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            logger.warning("Préférences illisibles ({}) : {}", self.path, exc)
            return []
        items = []
        for entry in raw if isinstance(raw, list) else []:
            try:
                items.append(Preference(**entry))
            except TypeError:
                continue
        return items

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_text(
            json.dumps([asdict(p) for p in self.items], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def learn(self, subject: str, value: str, source: str = "correction") -> Preference | None:
        """Note une préférence montrée par l'utilisateur ; None si elle est vide."""
        subject, value = subject.strip()[:MAX_CHARS], value.strip()[:MAX_CHARS]
        if not subject or not value:
            return None
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        key = " ".join(fold(subject).split())
        current = next((p for p in self.items if " ".join(fold(p.subject).split()) == key), None)
        if current is not None:
            if fold(current.value).split() == fold(value).split():
                current.seen += 1
            else:  # l'utilisateur a changé d'avis : la nouvelle valeur repart de un
                logger.info("Préférence « {} » : {} -> {}.", subject, current.value, value)
                current.value, current.seen = value, 1
            current.source, current.updated = source, now
            preference = current
        else:
            preference = Preference(subject, value, 1, source, now)
            self.items.append(preference)
            logger.info("Préférence apprise : {} -> {}.", subject, value)
        # Au-delà du plafond, les moins confirmées puis les plus anciennes sortent.
        self.items.sort(key=lambda p: (p.seen, p.updated), reverse=True)
        self.items = self.items[: self.max_items]
        self._save()
        return preference

    def render(self) -> str:
        """Bloc pour le prompt ("" s'il n'y a rien)."""
        return "\n".join(f"- {p.label}" for p in self.items)

    def forget(self, query: str) -> Preference | None:
        """Retire la préférence la plus proche de `query` (sujet ou valeur)."""
        wanted = words(query)
        best, best_score = None, 0
        for preference in self.items:
            score = len(wanted & words(f"{preference.subject} {preference.value}"))
            if score > best_score:
                best, best_score = preference, score
        if best is None:
            return None
        self.items.remove(best)
        self._save()
        logger.info("Préférence oubliée : {} -> {}.", best.subject, best.value)
        return best
