"""Préférences apprises (phase 16e, docs/AGENT_CORE.md §3.7).

Une préférence est un couple sujet → valeur (« musique → Spotify », « météo → Lyon »),
avec le nombre de fois où l'utilisateur l'a montrée. Elle ne vient que de ses propres mots :
la réflexion la tire d'une **correction** (« non, sur Spotify »), jamais d'un contenu lu
ailleurs (la réflexion ne tourne pas après un contenu non fiable). Si l'utilisateur change
d'avis, la nouvelle valeur remplace l'ancienne et repart de un.

Les préférences sont données au modèle à chaque commande (préfixe stable : elles changent
rarement), avec leur nombre de confirmations ; « oublie que je préfère... » les retire.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from mikata.learn.skills import fold, words

PREFERENCES_FILE = "data/preferences.json"
MAX_PREFERENCES = 40
MAX_CHARS = 80


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
