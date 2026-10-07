from __future__ import annotations

import pytest

from mikata.config import SttConfig
from mikata.stt.text import clean_transcript, strip_wake_word, wake_word_pattern

PATTERN = wake_word_pattern(SttConfig().wake_words)


@pytest.mark.parametrize(
    ("raw", "expected", "found"),
    [
        ("Hey Jarvis, ouvre Spotify.", "ouvre Spotify.", True),
        ("Et Jarvis... quelle heure est-il ?", "quelle heure est-il ?", True),
        ("MIKATA, mets le volume à 30.", "mets le volume à 30.", True),
        ("Mi kata ! Pause.", "Pause.", True),
        ("Mi-kata, stop", "stop", True),
        ("Micata ouvre Discord", "ouvre Discord", True),
        ("Donc bref. Mikata, lance Steam.", "lance Steam.", True),  # pré-roll : fin de phrase
        ("Mikata, lance Mikado", "lance Mikado", True),  # seule la 1re occurrence part
        ("MIKATA.", "", True),
        ("ouvre Discord", "ouvre Discord", False),
        ("Mikatane", "Mikatane", False),  # mot entier seulement
    ],
)
def test_strip_wake_word(raw: str, expected: str, found: bool) -> None:
    assert strip_wake_word(raw, PATTERN) == (expected, found)


def test_custom_wake_words() -> None:
    pattern = wake_word_pattern(["ok ordi", "  "])
    assert strip_wake_word("OK-ordi : allume", pattern) == ("allume", True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sous-titres réalisés par la communauté d'Amara.org", ""),
        ("Sous-titrage ST' 501", ""),
        ("Ouvre Spotify. Merci d'avoir regardé cette vidéo !", "Ouvre Spotify."),
        ("Active les sous-titres.", "Active les sous-titres."),
        ("  ouvre   Chrome ", "ouvre Chrome"),
    ],
)
def test_clean_transcript(raw: str, expected: str) -> None:
    assert clean_transcript(raw) == expected
