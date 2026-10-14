"""Nettoyage du texte transcrit (§3 et §5).

- Retrait du mot d'activation : « Hey Jarvis, ouvre Spotify » -> « ouvre Spotify ».
  Tout ce qui précède la première occurrence part aussi : la transcription démarre au début
  du pré-roll de 2 s, qui peut contenir la fin d'une phrase dite avant le mot d'activation.
- Retrait des hallucinations classiques de Whisper sur du silence en français.
- Ajout phase 6 : reconnaissance d'une commande d'annulation (« stop », « laisse tomber »...)
  et filtre de la fenêtre de suivi (`follow_up_rejection` : bruit, hésitation, transcription
  peu fiable).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

from mikata.stt.base import Transcript

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


# Ce que Whisper entend parfois à la place de « MIKATA » : Nikata, Nakita, Mikita, Makita...
# Retenu seulement en début de phrase, pour ne jamais couper un mot de la demande.
_HEARD_VARIANT = re.compile(r"\b[mn][iaeéy][\s\-]?[kcq]u?[aiy][\s\-]?t+[aeo]\b", re.IGNORECASE)


def _at_sentence_start(text: str, index: int) -> bool:
    before = text[:index].rstrip(" ,;:'\"«»-–—")
    return not before or before[-1] in ".!?…"


def _after(text: str, match: re.Match[str]) -> str:
    return _LEADING_JUNK.sub("", text[match.end() :]).strip()


def strip_wake_word(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    """(demande sans le mot d'activation, mot trouvé ?).

    Le pré-roll garde parfois un essai précédent (« Mikata, quelle heure ? Mikata, quelle
    heure est-il ? ») : on garde ce qui suit le **dernier** mot d'activation placé en début de
    phrase. Sinon, comme avant, ce qui suit la première occurrence."""
    canonical = list(pattern.finditer(text))
    heard = [m for m in _HEARD_VARIANT.finditer(text) if _at_sentence_start(text, m.start())]
    if not canonical and not heard:
        return text.strip(), False
    starts = [m for m in canonical if _at_sentence_start(text, m.start())] + heard
    usable = sorted((m for m in starts if _after(text, m)), key=lambda m: m.start())
    if usable:
        return _after(text, usable[-1]), True
    return _after(text, canonical[0] if canonical else heard[0]), True


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


# Fenêtre de suivi : ce qui n'est pas une phrase pour MIKATA.
FILLER_WORDS = frozenset(
    {"euh", "heu", "hum", "hmm", "mmh", "mh", "ah", "oh", "eh", "he", "bah", "ben", "hein",
     "pff", "bof", "ouh", "haha", "hahaha", "hihi", "mouais"}
)  # fmt: skip
MIN_LETTERS = 2
MIN_UNIQUE_RATIO = 0.4  # « la la la la la » : texte incohérent


def transcript_confidence(transcript: Transcript) -> float | None:
    """Confiance de 0 à 1 : probabilité moyenne des tokens × probabilité qu'il y ait de la
    parole. None si le moteur ne donne ni l'une ni l'autre."""
    if transcript.avg_logprob is None and transcript.no_speech_prob is None:
        return None
    token = math.exp(transcript.avg_logprob) if transcript.avg_logprob is not None else 1.0
    speech = 1.0 - (transcript.no_speech_prob or 0.0)
    return max(0.0, min(1.0, token * speech))


def follow_up_rejection(text: str, transcript: Transcript, min_confidence: float) -> str | None:
    """Raison d'ignorer une phrase entendue dans la fenêtre de suivi (bruit, hésitation,
    texte incohérent, transcription peu fiable), ou None si elle peut partir au LLM."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    words = re.findall(r"[^\W\d_]+", folded)
    if sum(len(w) for w in words) < MIN_LETTERS:
        return "texte trop court"
    if all(w in FILLER_WORDS for w in words):
        return "hésitation seule"
    if len(words) >= 4 and len(set(words)) / len(words) < MIN_UNIQUE_RATIO:
        return "texte incohérent (répétitions)"
    confidence = transcript_confidence(transcript)
    if confidence is not None and confidence < min_confidence:
        return f"transcription peu fiable ({describe_confidence(transcript)})"
    return None


def describe_confidence(transcript: Transcript) -> str:
    """« confiance 0.31 : no_speech 0.62, logprob -1.10 » pour le log."""
    confidence = transcript_confidence(transcript)
    if confidence is None:
        return "confiance inconnue"
    parts = []
    if transcript.no_speech_prob is not None:
        parts.append(f"no_speech {transcript.no_speech_prob:.2f}")
    if transcript.avg_logprob is not None:
        parts.append(f"logprob {transcript.avg_logprob:.2f}")
    return f"confiance {confidence:.2f} : {', '.join(parts)}"
