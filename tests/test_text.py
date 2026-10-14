from __future__ import annotations

from typing import Any

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
        # Retours du 14/10 : ce que Whisper entend à la place de « MIKATA »...
        ("Nikata coupe mon micro sur Discord.", "coupe mon micro sur Discord.", True),
        ("Nakita, mets pause.", "mets pause.", True),
        ("Mikita ouvre Steam", "ouvre Steam", True),
        # ... et l'essai précédent gardé par le pré-roll : seule la dernière demande reste.
        (
            "Mikata, combien de fenêtres sont ouvertes ? Mikata, combien de fenêtres sont "
            "ouvertes et laquelle est devant ?",
            "combien de fenêtres sont ouvertes et laquelle est devant ?",
            True,
        ),
        (
            "Nikata coupe mon micro sur Discord. Nikata coupe mon micro sur Discord.",
            "coupe mon micro sur Discord.",
            True,
        ),
        # Un nom dans la demande n'est jamais pris pour le mot d'activation.
        ("Mikata, appelle Nikita", "appelle Nikita", True),
        ("Mikata, dis à Mikata bonjour", "dis à Mikata bonjour", True),
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Stop.", True),
        ("stop stop stop !", True),
        ("Laisse tomber, merci.", True),
        ("Euh, annule ça.", True),
        ("Tais-toi.", True),
        ("C’est bon.", True),
        ("Non.", True),
        ("arrête", True),
        # une commande qui contient le mot n'est pas une annulation
        ("Arrête la musique.", False),
        ("Stop process name notepad", False),
        ("Annule le minuteur", False),
        ("Merci.", False),
        ("", False),
    ],
)
def test_is_cancel_command(text: str, expected: bool) -> None:
    from mikata.config import AgentConfig
    from mikata.stt.text import is_cancel_command

    assert is_cancel_command(text, AgentConfig().cancel_phrases) is expected


def _transcript(no_speech: float | None = None, logprob: float | None = None) -> Any:
    from mikata.stt.base import Transcript

    return Transcript("x", 1.0, 0.1, no_speech_prob=no_speech, avg_logprob=logprob)


@pytest.mark.parametrize(
    ("text", "no_speech", "logprob", "rejected"),
    [
        ("Et à Tokyo ?", 0.05, -0.25, None),
        ("Oui.", 0.1, -0.3, None),  # réponse courte valable
        ("OK", None, None, None),  # moteur sans mesure de confiance
        ("Euh... hmm.", 0.1, -0.3, "hésitation seule"),
        ("A.", 0.1, -0.3, "texte trop court"),
        ("la la la la la la", 0.1, -0.3, "texte incohérent (répétitions)"),
        ("Tu viens ce soir ?", 0.7, -1.2, "transcription peu fiable"),
        ("Tu viens ce soir ?", 0.05, -1.5, "transcription peu fiable"),
    ],
)
def test_follow_up_rejection(
    text: str, no_speech: float | None, logprob: float | None, rejected: str | None
) -> None:
    """Phase 6 : bruit et conversations à côté dans la fenêtre de suivi."""
    from mikata.stt.text import follow_up_rejection

    reason = follow_up_rejection(text, _transcript(no_speech, logprob), min_confidence=0.4)
    if rejected is None:
        assert reason is None
    else:
        assert reason is not None and reason.startswith(rejected)


def test_transcript_confidence() -> None:
    from mikata.stt.text import describe_confidence, transcript_confidence

    assert transcript_confidence(_transcript()) is None
    assert transcript_confidence(_transcript(0.0, 0.0)) == 1.0
    assert transcript_confidence(_transcript(0.5, None)) == 0.5
    assert transcript_confidence(_transcript(None, -1.0)) == pytest.approx(0.368, abs=1e-3)
    assert describe_confidence(_transcript(0.62, -1.1)) == (
        "confiance 0.13 : no_speech 0.62, logprob -1.10"
    )
    assert describe_confidence(_transcript()) == "confiance inconnue"
