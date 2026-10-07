"""Texte -> parole (§8) : normalisation et découpe en fragments pour le TTS.

- `normalize_for_speech` : ceinture et bretelles si le LLM ignore le prompt (markdown,
  URL, unités, emoji).
- `FragmentSplitter` : découpe le texte qui arrive en flux (§2, technique 3). Le premier
  fragment part dès une virgule après au moins `first_chunk_min_words` mots (ou dès la
  première fin de phrase), les suivants à chaque fin de phrase. Une ponctuation n'est une
  frontière que si elle est suivie d'un espace : « 3.5 » ou « 1,2 » ne coupent pas.
"""

from __future__ import annotations

import re

MAX_FRAGMENT_WORDS = 30  # au-delà, on coupe à la dernière virgule ou au dernier espace

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_MARKS_RE = re.compile(r"(\*\*|__|\*|`|~~|^#{1,6}\s*|^\s*[-*+•]\s+|^\s*\d+\.\s+)", re.MULTILINE)
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")
_UNITS = (
    (re.compile(r"\s*°\s*C\b"), " degrés"),
    (re.compile(r"\s*°"), " degrés"),
    (re.compile(r"\s*%"), " pour cent"),
    (re.compile(r"\s*km/h\b"), " kilomètres heure"),
    (re.compile(r"\s*€"), " euros"),
    (re.compile(r"\s*\$"), " dollars"),
    (re.compile(r"\s+&\s+"), " et "),
)
_SENTENCE_END = re.compile(r"[.!?…]+[\"»)\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:]+(?=\s)")


def normalize_for_speech(text: str) -> str:
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("le lien", text)
    text = _MD_MARKS_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    for pattern, spoken in _UNITS:
        text = pattern.sub(spoken, text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> int:
    return len(text.split())


class FragmentSplitter:
    def __init__(self, first_chunk_min_words: int = 6) -> None:
        self.first_chunk_min_words = first_chunk_min_words
        self._buffer = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        """Ajoute du texte ; renvoie les fragments complets (normalisés, non vides)."""
        self._buffer += delta
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            fragment, self._buffer = self._buffer[:cut], self._buffer[cut:]
            self._push(fragment, out)
        return out

    def flush(self) -> list[str]:
        """Fin du flux : le reste part tel quel."""
        out: list[str] = []
        self._push(self._buffer, out)
        self._buffer = ""
        return out

    def _push(self, fragment: str, out: list[str]) -> None:
        spoken = normalize_for_speech(fragment)
        if spoken and re.search(r"\w", spoken):
            out.append(spoken)
            self._emitted += 1

    def _find_cut(self) -> int | None:
        buffer = self._buffer
        sentence = _SENTENCE_END.search(buffer)
        if self._emitted == 0:
            for clause in _CLAUSE_END.finditer(buffer):
                if sentence is not None and clause.start() > sentence.start():
                    break
                if _words(buffer[: clause.end()]) >= self.first_chunk_min_words:
                    return clause.end()
        if sentence is not None:
            return sentence.end()
        if _words(buffer) > MAX_FRAGMENT_WORDS:  # pas de ponctuation : on ne bloque pas le TTS
            words = list(re.finditer(r"\S+", buffer))
            head_end = words[MAX_FRAGMENT_WORDS - 1].end()  # fin du 30e mot
            clause_cuts = [m.end() for m in _CLAUSE_END.finditer(buffer[:head_end])]
            return clause_cuts[-1] if clause_cuts else head_end
        return None
