from __future__ import annotations

import pytest

from mikata.tts.normalize import MAX_FRAGMENT_WORDS, FragmentSplitter, normalize_for_speech


def _stream(text: str, step: int = 3, first: int = 6) -> list[str]:
    splitter = FragmentSplitter(first)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += splitter.feed(text[i : i + step])
    return out + splitter.flush()


def test_first_fragment_at_comma_after_six_words() -> None:
    text = "Demain il fera très beau sur Paris, avec 22 degrés. Profites-en bien !"
    assert _stream(text) == [
        "Demain il fera très beau sur Paris,",
        "avec 22 degrés.",
        "Profites-en bien !",
    ]


def test_short_clause_waits_for_sentence_end() -> None:
    assert _stream("Oui, bien sûr. Je regarde.") == ["Oui, bien sûr.", "Je regarde."]


def test_later_fragments_cut_by_sentence_only() -> None:
    text = "C'est fait. Ensuite, quand tu veux, on pourra parler, si tu as le temps, de ça."
    assert _stream(text) == [
        "C'est fait.",
        "Ensuite, quand tu veux, on pourra parler, si tu as le temps, de ça.",
    ]


def test_numbers_and_decimals_do_not_cut() -> None:
    assert _stream("La version 3.5 coûte 1,2 euro ce mois-ci.", step=1) == [
        "La version 3.5 coûte 1,2 euro ce mois-ci."
    ]


def test_fragment_emitted_before_stream_ends() -> None:
    splitter = FragmentSplitter(6)
    assert splitter.feed("Il est midi.") == []  # « . » en fin de tampon : on attend la suite
    assert splitter.feed(" Bon") == ["Il est midi."]
    assert splitter.flush() == ["Bon"]


def test_very_long_sentence_without_punctuation_is_cut() -> None:
    splitter = FragmentSplitter(6)
    words = " ".join(f"mot{i}" for i in range(MAX_FRAGMENT_WORDS + 5))
    out = splitter.feed(words + " ")
    assert out and len(out[0].split()) <= MAX_FRAGMENT_WORDS


def test_punctuation_only_fragments_are_dropped() -> None:
    assert _stream("... !") == []


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("**Il fait** 21°C", "Il fait 21 degrés"),
        ("Batterie à 80 %", "Batterie à 80 pour cent"),
        ("Voir https://example.com/page?x=1 pour plus", "Voir le lien pour plus"),
        ("Lis [la doc](https://x.y)", "Lis la doc"),
        ("# Titre\n- un\n- deux", "Titre un deux"),
        ("Super 😀 !", "Super !"),
        ("Tom & Jerry à 50 km/h", "Tom et Jerry à 50 kilomètres heure"),
        ("Code : ```print(1)``` fini", "Code : fini"),
    ],
)
def test_normalize_for_speech(raw: str, spoken: str) -> None:
    assert normalize_for_speech(raw) == spoken
