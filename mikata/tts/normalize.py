"""Texte -> parole (§8) : normalisation et découpe en fragments pour le TTS.

- `normalize_for_speech` : ceinture et bretelles si le LLM ignore le prompt (markdown,
  URL, unités, emoji). Ajout phase 6 : heures en toutes lettres (« 3 h 40 » -> « trois
  heures quarante », « 14h05 » -> « quatorze heures cinq ») et unités après un nombre, au
  bon nombre (« 1 min » -> « une minute », « 90 km/h » -> « 90 kilomètres heure »).
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
_UNITS = (  # restes sans nombre devant
    (re.compile(r"\s*°\s*C\b"), " degrés"),
    (re.compile(r"\s*°"), " degrés"),
    (re.compile(r"\s*%"), " pour cent"),
    (re.compile(r"\s*km/h\b"), " kilomètres heure"),
    (re.compile(r"\s*€"), " euros"),
    (re.compile(r"\s*\$"), " dollars"),
    (re.compile(r"\s+&\s+"), " et "),
)
# Unité après un nombre : (singulier, pluriel, féminin). Pluriel à partir de 2 (« 1,5 km »
# -> « 1,5 kilomètre »).
_UNIT_WORDS: dict[str, tuple[str, str, bool]] = {
    "km/h": ("kilomètre heure", "kilomètres heure", False),
    "km": ("kilomètre", "kilomètres", False),
    "kg": ("kilogramme", "kilogrammes", False),
    "h": ("heure", "heures", True),  # durée (« 2,5 h », « 36 h ») ; les heures sont lues avant
    "ms": ("milliseconde", "millisecondes", True),
    "min": ("minute", "minutes", True),
    "mn": ("minute", "minutes", True),
    "sec": ("seconde", "secondes", True),
    "s": ("seconde", "secondes", True),
    "°C": ("degré", "degrés", False),
    "°": ("degré", "degrés", False),
    "%": ("pour cent", "pour cent", False),
    "Go": ("gigaoctet", "gigaoctets", False),
    "Mo": ("mégaoctet", "mégaoctets", False),
    "€": ("euro", "euros", False),
    "$": ("dollar", "dollars", False),
}
_UNIT_RE = re.compile(
    r"(?<![\w.,])(-?\d+(?:[.,]\d+)?)\s*("
    + "|".join(re.escape(u) for u in sorted(_UNIT_WORDS, key=len, reverse=True))
    + r")(?![\w/])"
)
# Heures : « 3 h 40 », « 14h05 », « 9 h », « 14:05 ».
_TIME_H_RE = re.compile(r"(?<![\w.,:])([01]?\d|2[0-3])\s?[hH]\s?([0-5]\d)?(?!\w)")
_TIME_COLON_RE = re.compile(r"(?<![\w.,:])([01]?\d|2[0-3]):([0-5]\d)(?![\w:])")
_MINUS_RE = re.compile(r"(?<![\w\-])-(?=\d)")

_ONES = (
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    "onze", "douze", "treize", "quatorze", "quinze", "seize",
)  # fmt: skip
_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}


def _below_100(n: int, feminine: bool) -> str:
    one = "une" if feminine else "un"
    if n == 1:
        return one
    if n < 17:
        return _ONES[n]
    if n < 20:
        return "dix-" + _ONES[n - 10]
    tens, unit = divmod(n, 10)
    if tens in (7, 9):  # soixante-dix..., quatre-vingt-dix...
        base = "soixante" if tens == 7 else "quatre-vingt"
        rest = _below_100(10 + unit, feminine)
        return f"{base} et {rest}" if tens == 7 and unit == 1 else f"{base}-{rest}"
    if tens == 8:
        return "quatre-vingts" if unit == 0 else f"quatre-vingt-{_below_100(unit, feminine)}"
    if unit == 0:
        return _TENS[tens]
    if unit == 1:
        return f"{_TENS[tens]} et {one}"
    return f"{_TENS[tens]}-{_ONES[unit]}"


def _below_1000(n: int, feminine: bool, final: bool = True) -> str:
    hundreds, rest = divmod(n, 100)
    if not hundreds:
        return _below_100(rest, feminine)
    head = "cent" if hundreds == 1 else f"{_ONES[hundreds]} cent"
    if not rest:
        return head + ("s" if hundreds > 1 and final else "")
    return f"{head} {_below_100(rest, feminine)}"


def number_to_words(n: int, feminine: bool = False) -> str:
    """Nombre entier en toutes lettres (« 21 » -> « vingt et un », féminin « vingt et
    une »), jusqu'à 999 999 999."""
    if n < 0:
        return "moins " + number_to_words(-n, feminine)
    if n < 1000:
        return _below_1000(n, feminine)
    if n >= 1_000_000_000:
        return str(n)
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1000)
    parts = []
    if millions:
        word = _below_1000(millions, False)
        parts.append(f"{word} million" + ("s" if millions > 1 else ""))
    if thousands:
        thousands_words = _below_1000(thousands, False, final=False).removesuffix("s")
        parts.append("mille" if thousands == 1 else f"{thousands_words} mille")
    if units:
        parts.append(_below_1000(units, feminine))
    return " ".join(parts)


def _say_time(hours: str, minutes: str | None) -> str:
    h = int(hours)
    spoken = f"{number_to_words(h, feminine=True)} heure" + ("s" if h > 1 else "")
    if minutes and int(minutes):
        spoken += " " + number_to_words(int(minutes), feminine=True)
    return spoken


def _say_unit(match: re.Match[str]) -> str:
    raw, unit = match.group(1), match.group(2)
    singular, plural, feminine = _UNIT_WORDS[unit]
    value = float(raw.replace(",", "."))
    number = raw.lstrip("-")
    if number in ("1", "1,0", "1.0"):
        number = "une" if feminine else "un"
    sign = "moins " if raw.startswith("-") else ""
    return f"{sign}{number} {plural if abs(value) >= 2 else singular}"


_SENTENCE_END = re.compile(r"[.!?…]+[\"»)\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:]+(?=\s)")


def normalize_for_speech(text: str) -> str:
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("le lien", text)
    text = _MD_MARKS_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _TIME_H_RE.sub(lambda m: _say_time(m.group(1), m.group(2)), text)
    text = _TIME_COLON_RE.sub(lambda m: _say_time(m.group(1), m.group(2)), text)
    text = _UNIT_RE.sub(_say_unit, text)
    text = _MINUS_RE.sub("moins ", text)
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
