"""Niveau de réflexion choisi à la voix (ajout phase 4).

Par défaut le LLM répond sans réflexion (`llm.reasoning_effort: none`, premier mot plus
tôt). Un mot-clé dit **en début ou en fin** de commande choisit un niveau pour cette
commande seulement : « réfléchis » -> low, « réfléchis bien » -> medium, « réfléchis à
fond » -> high (`llm.reasoning_keywords`). Il est détecté ici, par le code et non par le
LLM, puis retiré du texte envoyé.

Comparaison sans casse, accents ni ponctuation ; « réfléchis » accepte ses variantes de
transcription (réfléchi, réfléchit, réfléchie, réfléchies). Faux positifs évités : pas de
mot-clé si un sujet ou un auxiliaire l'entoure (« à quoi tu réfléchis », « j'ai bien
réfléchi », « réfléchis-tu ? »), ni s'il ne reste rien d'autre à demander.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

LEVELS = ("none", "low", "medium", "high")
_WORD_RE = re.compile(r"\w+")
_VERB_RE = re.compile(r"^reflechi(?:s|t|e|es)?$")  # forme sans accents
VERB = "reflechis"
# Mots qui font de « réfléchis » un verbe conjugué ordinaire, pas une consigne.
_SUBJECTS = {"je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "qui"}
_AUXILIARIES = {"ai", "as", "a", "avons", "avez", "ont", "avoir", "été", "t"}
_BREAK_RE = re.compile(r"[.,;:!?…]")  # fin de proposition
_LEADING_JUNK = " \t,;:.!?-–—…"
_TRAILING_JUNK = " \t,;:-–—…"


def _fold(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.casefold())
    word = "".join(c for c in word if not unicodedata.combining(c))
    return VERB if _VERB_RE.match(word) else word


@dataclass(frozen=True)
class _Token:
    raw: str  # minuscules, accents gardés (« à » n'est pas « a »)
    key: str  # forme de comparaison
    start: int
    end: int


def _tokens(text: str) -> list[_Token]:
    return [
        _Token(m.group().casefold(), _fold(m.group()), m.start(), m.end())
        for m in _WORD_RE.finditer(text)
    ]


def _joined(text: str, left: _Token, right: _Token) -> bool:
    """Vrai si aucune ponctuation ne sépare les deux mots (même proposition)."""
    return not _BREAK_RE.search(text[left.end : right.start])


def _words_before(text: str, tokens: list[_Token], index: int, count: int = 2) -> set[str]:
    """Jusqu'à `count` mots avant `tokens[index]`, sans franchir de ponctuation."""
    words: set[str] = set()
    for i in range(index - 1, max(-1, index - 1 - count), -1):
        if not _joined(text, tokens[i], tokens[i + 1]):
            break
        words.add(tokens[i].raw)
    return words


@dataclass(frozen=True)
class ReasoningRequest:
    level: str | None  # None : aucun mot-clé, niveau par défaut
    text: str  # commande sans le mot-clé
    keyword: str | None = None


def detect_reasoning(text: str, keywords: Mapping[str, Sequence[str]]) -> ReasoningRequest:
    """Cherche un mot-clé de réflexion en début ou en fin de `text` (le plus long d'abord)."""
    tokens = _tokens(text)
    candidates = sorted(
        (
            (level, [t.key for t in _tokens(keyword)], keyword)
            for level, phrases in keywords.items()
            for keyword in phrases
        ),
        key=lambda c: -len(c[1]),
    )
    for level, wanted, keyword in candidates:
        n = len(wanted)
        if n == 0 or n >= len(tokens):  # il doit rester une demande
            continue
        # « Réfléchis-tu ... ? » : question, pas consigne ; « Réfléchis bien, tu ... » : consigne.
        questioned = tokens[n].raw in _SUBJECTS and _joined(text, tokens[n - 1], tokens[n])
        if [t.key for t in tokens[:n]] == wanted and not questioned:
            rest = text[tokens[n - 1].end :].lstrip(_LEADING_JUNK)
            return ReasoningRequest(level, rest[:1].upper() + rest[1:], keyword)
        before = _words_before(text, tokens, len(tokens) - n)
        if [t.key for t in tokens[-n:]] == wanted and not before & (_SUBJECTS | _AUXILIARIES):
            head = text[: tokens[-n].start].rstrip(_TRAILING_JUNK)
            tail = text[tokens[-1].end :].strip(_TRAILING_JUNK + ".")
            return ReasoningRequest(level, f"{head} {tail}" if tail else head, keyword)
    return ReasoningRequest(None, text)


def at_least(level: str | None, minimum: str) -> str | None:
    """Niveau relevé à `minimum` s'il est plus bas ; None (choix du fournisseur) inchangé."""
    if level is None or LEVELS.index(level) >= LEVELS.index(minimum):
        return level
    return minimum
