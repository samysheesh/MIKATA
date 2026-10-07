"""Nettoyage du texte transcrit (§3 et §5).

- Retrait du mot d'activation : « Hey Jarvis, ouvre Spotify » -> « ouvre Spotify ».
  Tout ce qui précède la première occurrence part aussi : la transcription démarre au début
  du pré-roll de 2 s, qui peut contenir la fin d'une phrase dite avant le mot d'activation.
- Retrait des hallucinations classiques de Whisper sur du silence en français.
- Ajout phase 6 : reconnaissance d'une commande d'annulation (« stop », « laisse tomber »...).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Phrases que Whisper invente sur du silence ou du bruit (sous-titres de son corpus).
HALLUCINATIONS = (  # « amara.org » d'abord : son point ne doit pas couper la phrase
    r"[^.!?]*amara\.org[^.!?]*[.!?]?",
    r"sous-titr\w* (?:réalisés? )?(?:par|fait par)[^.!?]*[.!?]?",
    r"sous-titrage st'? ?\d*",
    r"merci d'avoir regardé[^.!?]*[.!?]?",
    r"abonnez-vous[^.!?]*[.!?]?",
)
_HALLUCINATION_RE = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)
_LEADING_JUNK = re.compile(r"^[\s,.;:!?…\-–—'\"«»]+")


def wake_word_pattern(words: Iterable[str]) -> re.Pattern[str]:
    """Regex tolérante : espaces, tirets ou rien entre les morceaux (« mi kata », « mi-kata »)."""
    alternatives = []
    for word in sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True):
        parts = [re.escape(p) for p in re.split(r"[\s\-]+", word)]
        alternatives.append(r"[\s\-]*".join(parts))
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def strip_wake_word(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    """(texte après la première occurrence du mot d'activation, mot trouvé ?)."""
    match = pattern.search(text)
    if match is None:
        return text.strip(), False
    return _LEADING_JUNK.sub("", text[match.end() :]).strip(), True


def clean_transcript(text: str) -> str:
    """Retire les hallucinations connues et les espaces superflus."""
    cleaned = _HALLUCINATION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# Mots de politesse ou d'hésitation autour d'une annulation (« euh stop merci »).
_CANCEL_FILLERS = frozenset({"euh", "bah", "ben", "allez", "stp", "svp", "merci", "mikata"})


def _cancel_form(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = [w for w in re.findall(r"\w+", text) if w not in _CANCEL_FILLERS]
    # « stop stop stop » -> « stop »
    return " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])


def is_cancel_command(text: str, phrases: Iterable[str]) -> bool:
    """La commande entière est une annulation (`agent.cancel_phrases`), à la casse, aux
    accents, à la ponctuation, aux répétitions et à « merci » près. « arrête la musique »
    n'en est pas une : seule la phrase complète compte."""
    form = _cancel_form(text)
    return bool(form) and form in {_cancel_form(p) for p in phrases}
