"""Niveau de réflexion choisi à la voix : mot-clé en début ou en fin de commande."""

from __future__ import annotations

import pytest

from mikata.config import Config, LlmConfig
from mikata.llm.reasoning import at_least, detect_reasoning

KEYWORDS = Config().llm.reasoning_keywords


@pytest.mark.parametrize(
    ("text", "level", "rest"),
    [
        ("Réfléchis, combien font 17 fois 23 ?", "low", "Combien font 17 fois 23 ?"),
        ("Réfléchis bien. Quel train prendre ?", "medium", "Quel train prendre ?"),
        ("réfléchis à fond : quel est le meilleur plan", "high", "Quel est le meilleur plan"),
        ("Combien font 17 fois 23, réfléchis.", "low", "Combien font 17 fois 23"),
        ("Quel train prendre ? Réfléchis bien", "medium", "Quel train prendre ?"),
        ("Le train arrive quand, réfléchis à fond ?", "high", "Le train arrive quand ?"),
        # variantes de transcription de « réfléchis »
        ("Réfléchi bien, quelle heure à Tokyo ?", "medium", "Quelle heure à Tokyo ?"),
        ("Réfléchit à fond. Pourquoi le ciel est bleu ?", "high", "Pourquoi le ciel est bleu ?"),
        ("pourquoi le ciel est bleu, réfléchie", "low", "pourquoi le ciel est bleu"),
        ("REFLECHIS A FOND, 2 + 2 ?", "high", "2 + 2 ?"),
        ("Réfléchis-bien, 2 + 2 ?", "medium", "2 + 2 ?"),
        # la ponctuation sépare la consigne du reste de la phrase
        (
            "À quelle heure arrive-t-il ? Réfléchis bien.",
            "medium",
            "À quelle heure arrive-t-il ?",
        ),
        ("Réfléchis bien, tu peux me dire 2 + 2 ?", "medium", "Tu peux me dire 2 + 2 ?"),
        ("Je veux la réponse. Réfléchis.", "low", "Je veux la réponse."),
    ],
)
def test_keyword_at_start_or_end(text: str, level: str, rest: str) -> None:
    request = detect_reasoning(text, KEYWORDS)
    assert (request.level, request.text) == (level, rest)
    assert request.keyword in ("réfléchis", "réfléchis bien", "réfléchis à fond")


@pytest.mark.parametrize(
    "text",
    [
        "Ouvre Spotify",
        "Dis-moi à quoi tu réfléchis",  # verbe conjugué ordinaire
        "J'ai bien réfléchi.",
        "Il a réfléchi à fond",
        "Réfléchis-tu parfois ?",
        "Réfléchis.",  # rien d'autre à demander
        "Je me demande si réfléchir sert à quelque chose",  # au milieu, et infinitif
        "Aide-moi à réfléchir",
    ],
)
def test_no_keyword(text: str) -> None:
    request = detect_reasoning(text, KEYWORDS)
    assert request.level is None and request.text == text


def test_configurable_keywords() -> None:
    config = Config.model_validate(
        {"llm": {"reasoning_keywords": {"medium": ["  prends ton temps "], "high": [""]}}}
    )
    assert config.llm.reasoning_keywords == {"medium": ["prends ton temps"], "high": []}
    request = detect_reasoning("Prends ton temps : 2 + 2 ?", config.llm.reasoning_keywords)
    assert (request.level, request.text) == ("medium", "2 + 2 ?")
    assert detect_reasoning("Réfléchis, 2 + 2 ?", config.llm.reasoning_keywords).level is None


def test_reasoning_config_defaults_and_validation() -> None:
    llm = LlmConfig()
    assert llm.reasoning_effort == "none" and llm.reasoning_fallback_steps == 2
    assert llm.reasoning_keywords == {
        "low": ["réfléchis"],
        "medium": ["réfléchis bien"],
        "high": ["réfléchis à fond"],
    }
    with pytest.raises(ValueError):
        LlmConfig(reasoning_effort="max")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LlmConfig(reasoning_keywords={"extreme": ["x"]})  # type: ignore[dict-item]
    assert LlmConfig(reasoning_effort=None).reasoning_effort is None


@pytest.mark.parametrize(
    ("level", "minimum", "expected"),
    [("none", "low", "low"), ("low", "low", "low"), ("high", "low", "high"), (None, "low", None)],
)
def test_at_least(level: str | None, minimum: str, expected: str | None) -> None:
    assert at_least(level, minimum) == expected
