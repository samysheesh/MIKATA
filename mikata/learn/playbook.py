"""Carnet de leçons (docs/AGENT_CORE.md §3.11, niveau 1) : ce que MIKATA a appris à l'usage.

Méthode ACE (Zhang et al., 2025) : de petites entrées, chacune avec ses compteurs « utile »
et « nuisible », ajoutées et retouchées **une à une**. Le carnet n'est jamais réécrit en
bloc par un modèle : une réécriture complète finit par l'effacer.

Garde-fous (*Your Agent May Misevolve*, 2025) :

- une leçon entre dans le contexte comme une **référence**, pas comme une règle ;
- elle ne vient jamais d'un contenu lu ailleurs (page web, fichier) : seulement d'un échec
  corrigé ou d'une correction de l'utilisateur ;
- elle ne touche ni aux politiques ni au garde-fou, qui sont dans le code ;
- une leçon plus souvent nuisible qu'utile est retirée ; le fichier est lisible et
  modifiable à la main (`data/playbook.json`).
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from rapidfuzz import fuzz

PLAYBOOK_FILE = "data/playbook.json"
TEXT_MAX_CHARS = 240
SAME_LESSON_SCORE = 86  # ressemblance au-delà de laquelle deux leçons n'en font qu'une
HARMFUL_MARGIN = 2  # retirée quand nuisible > utile + marge

_STOP_WORDS = frozenset(
    [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "d",
        "l",
        "au",
        "aux",
        "et",
        "ou",
        "a",
        "à",
        "en",
        "dans",
        "sur",
        "pour",
        "par",
        "avec",
        "sans",
        "que",
        "qui",
        "quoi",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        "ce",
        "cette",
        "ces",
        "je",
        "tu",
        "il",
        "elle",
        "on",
        "nous",
        "vous",
        "est",
        "sont",
        "pas",
        "ne",
        "plus",
        "tout",
        "tous",
        "toute",
        "moi",
        "toi",
        "se",
        "y",
        "lui",
    ]
)


def _fold(text: str) -> str:
    plain = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")


def keywords(text: str) -> set[str]:
    """Mots utiles d'un texte, sans accents ni mots vides."""
    return {word for word in re.findall(r"[a-z0-9]{3,}", _fold(text)) if word not in _STOP_WORDS}


@dataclass
class Lesson:
    id: str
    scope: str  # appli ou domaine : « discord », « fichiers », « navigateur », « général »
    text: str
    source: str  # « échec corrigé » | « correction »
    helpful: int = 0
    harmful: int = 0
    created: str = ""

    @property
    def score(self) -> int:
        return self.helpful - self.harmful


class Playbook:
    """Leçons gardées sur disque ; chaque opération touche une entrée, jamais tout le carnet."""

    def __init__(self, path: Path, max_lessons: int = 60) -> None:
        self.path = path
        self.max_lessons = max_lessons
        self.lessons: list[Lesson] = self._load()

    def _load(self) -> list[Lesson]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            logger.warning("Carnet de leçons illisible ({}), ignoré : {}", self.path, exc)
            return []
        lessons: list[Lesson] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                lessons.append(Lesson(**item))
            except TypeError:
                continue
        return lessons

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        data = [asdict(lesson) for lesson in self.lessons]
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, scope: str, text: str, source: str) -> Lesson | None:
        """Ajoute une leçon, ou renforce celle qui dit déjà la même chose. None si vide."""
        text = " ".join(text.split())[:TEXT_MAX_CHARS]
        scope = " ".join(scope.split()).casefold()[:40] or "général"
        if len(text) < 12:
            return None
        for known in self.lessons:
            if fuzz.token_set_ratio(_fold(known.text), _fold(text)) >= SAME_LESSON_SCORE:
                known.helpful += 1
                self._save()
                return known
        lesson = Lesson(
            id=uuid.uuid4().hex[:8],
            scope=scope,
            text=text,
            source=source,
            created=datetime.now().isoformat(timespec="seconds"),
        )
        self.lessons.append(lesson)
        if len(self.lessons) > self.max_lessons:  # la moins utile part, pas la plus ancienne
            weakest = min(self.lessons[:-1], key=lambda item: (item.score, item.created))
            self.lessons.remove(weakest)
        self._save()
        logger.info("Leçon apprise ({}) : {}", lesson.scope, lesson.text)
        return lesson

    def select(self, request: str, context: str = "", limit: int = 6) -> list[Lesson]:
        """Leçons utiles à cette demande : celles dont le domaine ou les mots la recoupent.

        Rien n'est proposé sans recoupement : une leçon hors sujet nuit aux demandes
        courtes et directes."""
        wanted = keywords(f"{request} {context}")
        scored: list[tuple[float, Lesson]] = []
        for lesson in self.lessons:
            scope_words = keywords(lesson.scope)
            overlap = len(wanted & keywords(lesson.text))
            in_scope = bool(scope_words & wanted)
            if not in_scope and overlap < 2:
                continue
            scored.append((3.0 * in_scope + overlap + 0.2 * lesson.score, lesson))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [lesson for _, lesson in scored[:limit]]

    def mark(self, ids: list[str], helpful: bool) -> None:
        """Après un tour : les leçons présentes ont-elles aidé (réussite) ou nui (échec) ?"""
        touched = False
        for lesson in list(self.lessons):
            if lesson.id not in ids:
                continue
            touched = True
            if helpful:
                lesson.helpful += 1
            else:
                lesson.harmful += 1
                if lesson.harmful > lesson.helpful + HARMFUL_MARGIN:
                    self.lessons.remove(lesson)
                    logger.info("Leçon retirée (plus nuisible qu'utile) : {}", lesson.text)
        if touched:
            self._save()

    def forget(self, query: str) -> Lesson | None:
        """Retire la leçon dont parle `query` (demande de l'utilisateur) : au moins deux
        mots en commun avec son domaine ou son texte, ou une phrase très proche."""
        wanted = keywords(query)

        def closeness(lesson: Lesson) -> tuple[int, float]:
            shared = len(wanted & (keywords(lesson.text) | keywords(lesson.scope)))
            return shared, fuzz.token_set_ratio(_fold(lesson.text), _fold(query))

        best = max(self.lessons, key=closeness, default=None)
        if best is None:
            return None
        shared, ratio = closeness(best)
        if shared < 2 and ratio < 60:
            return None
        self.lessons.remove(best)
        self._save()
        return best


def render(lessons: list[Lesson]) -> str:
    """Bloc ajouté au contexte : des références tirées de l'expérience, pas des ordres."""
    if not lessons:
        return ""
    lines = "\n".join(f"- ({lesson.scope}) {lesson.text}" for lesson in lessons)
    return (
        "Leçons tirées de tes essais passés sur ce PC (références utiles, pas des règles ; "
        f"la demande de l'utilisateur prime) :\n{lines}"
    )
