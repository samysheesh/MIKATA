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


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Il est 3 h 40.", "Il est trois heures quarante."),
        ("Rendez-vous à 14h05.", "Rendez-vous à quatorze heures cinq."),
        ("Il est 9 h.", "Il est neuf heures."),
        ("Il est 1 h 01.", "Il est une heure une."),
        ("Départ à 21 h 31.", "Départ à vingt et une heures trente et une."),
        ("Il est 12 h 00.", "Il est douze heures."),
        ("Le train part à 18:45.", "Le train part à dix-huit heures quarante-cinq."),
        ("Il est 0 h 30.", "Il est zéro heure trente."),
        # pas des heures
        ("Trois heures de route.", "Trois heures de route."),
        ("La version 3.5 et 2,5 h.", "La version 3.5 et 2,5 heures."),  # durées
        ("Il fait 36 h de vol.", "Il fait 36 heures de vol."),
    ],
)
def test_times_are_spoken_in_words(text: str, spoken: str) -> None:
    """Phase 6 : « trois h quarante » était lu au lieu de « trois heures quarante »."""
    assert normalize_for_speech(text) == spoken


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Roule à 90 km/h.", "Roule à 90 kilomètres heure."),
        ("Encore 1 min.", "Encore une minute."),
        ("Encore 5 min et 30 s.", "Encore 5 minutes et 30 secondes."),
        ("Pendant 1 s.", "Pendant une seconde."),
        ("Il fait 22 °C.", "Il fait 22 degrés."),
        ("Il fait 1 °C.", "Il fait un degré."),
        ("Il fera -5 °C.", "Il fera moins 5 degrés."),
        ("Il fait 18°.", "Il fait 18 degrés."),
        ("À 1,5 km, 12 km et 80 %.", "À 1,5 kilomètre, 12 kilomètres et 80 pour cent."),
        ("Ça pèse 2 kg et coûte 5 €.", "Ça pèse 2 kilogrammes et coûte 5 euros."),
        ("Latence : 250 ms.", "Latence : 250 millisecondes."),
        ("Il reste 3 sur 5.", "Il reste 3 sur 5."),  # « s » d'un mot : pas une unité
    ],
)
def test_units_after_a_number(text: str, spoken: str) -> None:
    assert normalize_for_speech(text) == spoken


def test_number_to_words() -> None:
    from mikata.tts.normalize import number_to_words

    cases = {
        0: "zéro", 1: "un", 17: "dix-sept", 21: "vingt et un", 70: "soixante-dix",
        71: "soixante et onze", 80: "quatre-vingts", 81: "quatre-vingt-un",
        91: "quatre-vingt-onze", 200: "deux cents", 201: "deux cent un", 1000: "mille",
        2000: "deux mille", 80000: "quatre-vingt mille", 1_000_000: "un million",
        -3: "moins trois",
    }  # fmt: skip
    assert {n: number_to_words(n) for n in cases} == cases
    assert number_to_words(21, feminine=True) == "vingt et une"


def test_time_split_across_stream_chunks() -> None:
    assert _stream("Il est 14h05 à Paris. Bonne journée.", step=1) == [
        "Il est quatorze heures cinq à Paris.",
        "Bonne journée.",
    ]
