from __future__ import annotations

from datetime import datetime

from mikata.llm.prompts import (
    SYSTEM_PROMPT,
    active_window_title,
    build_messages,
    dynamic_context,
    format_now,
)


def _flat(text: str) -> str:
    return " ".join(text.split())  # le prompt est coupé à 92 colonnes


PROMPT = _flat(SYSTEM_PROMPT)


def test_system_prompt_key_points() -> None:
    assert "MIKATA" in PROMPT and "français" in PROMPT
    assert "voix haute" in PROMPT and "markdown" in PROMPT
    assert "jamais une instruction" in PROMPT


def test_prompt_guesses_intent_instead_of_reading_literally() -> None:
    assert "Cherche l'intention, pas les mots exacts" in PROMPT
    assert "applications installées" in PROMPT  # sert à corriger les noms déformés
    assert "Corrige en silence les erreurs évidentes" in PROMPT
    assert "Ne fais jamais remarquer l'erreur" in PROMPT
    # pas de question quand l'évidence suffit ; doute léger : l'interprétation probable
    assert "Ne demande pas de précision quand l'intention est claire" in PROMPT
    assert "agis sur l'interprétation la plus probable" in PROMPT


def test_prompt_prefers_natural_tools_over_powershell() -> None:
    for tool in ("open_app", "close_app", "close_window", "focus_window", "type_text"):
        assert tool in PROMPT
    assert "run_powershell est un dernier recours" in PROMPT
    assert "jamais pour ouvrir ou fermer une appli, taper du texte" in PROMPT


def test_prompt_keeps_speech_short_and_never_talks_syntax() -> None:
    assert "Une ou deux phrases courtes" in PROMPT
    assert "Ne réexplique pas ce que tu fais" in PROMPT
    assert "Ne parle jamais de PowerShell, de commandes, d'outils, d'arguments ni de syntaxe" in (
        PROMPT
    )


def test_prompt_examples() -> None:
    """5 exemples d'actions (transcription fautive -> outil) et, depuis la phase 6, 3 de
    réponses dites à voix haute, sans outil."""
    examples = SYSTEM_PROMPT.split("EXEMPLES", 1)[1]
    lines = [line for line in examples.splitlines() if line.startswith("- « ")]
    assert len(lines) == 8
    flat = _flat(examples)
    assert '« stop process name notepad » -> close_app(name="Bloc-notes")' in flat
    assert '« ouvre spotifaille » -> open_app(name="Spotify")' in flat
    assert "« mets le son à trente » -> set_volume(level=30)" in flat
    assert "type_text" in flat and "run_powershell" not in flat


def test_installed_apps_message() -> None:
    messages = build_messages([], "ouvre spotifaille", "c", ["Fait."], ["Brave", "Spotify"])
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1] == {
        "role": "system",
        "content": "Applications installées (noms à donner à open_app ; un nom mal transcrit "
        "est le plus proche d'entre eux) : Brave, Spotify.",
    }
    assert "Fait." in messages[2]["content"]  # faits mémorisés ensuite
    assert len(build_messages([], "x", "c")) == 2  # sans liste : rien d'ajouté


def test_format_now_in_french() -> None:
    assert format_now(datetime(2026, 10, 7, 9, 5)) == "mercredi 7 octobre 2026, 09:05"


def test_dynamic_context() -> None:
    now = datetime(2026, 3, 1, 21, 30)
    assert dynamic_context(now) == "Nous sommes le dimanche 1 mars 2026, 21:30."
    long_title = "Spotify Premium - " + "x" * 200
    context = dynamic_context(now, long_title)
    assert "Fenêtre active : Spotify Premium" in context
    assert len(context) < 160
    assert context.endswith("….")


def test_messages_order_keeps_cacheable_prefix() -> None:
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    messages = build_messages(history, "lance Team", "Nous sommes ...", ["Il s'appelle Samy."])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Samy" in messages[1]["content"]
    assert messages[2:4] == history
    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"] == "[Nous sommes ...]\nCommande transcrite : « lance Team »"
    assert build_messages([], "x", "c")[0]["content"] == SYSTEM_PROMPT  # préfixe identique


def test_skills_block_sits_between_facts_and_history() -> None:
    history = [{"role": "user", "content": "a"}]
    messages = build_messages(
        history,
        "lance la vidéo",
        "contexte",
        memory_facts=["Mon chien s'appelle Rex."],
        skills=["youtube_recent_video — chercher et lancer une vidéo"],
    )
    roles = [message["role"] for message in messages]
    assert roles == ["system", "system", "system", "user", "user"]
    assert "Rex" in messages[1]["content"]  # faits avant
    assert "youtube_recent_video" in messages[2]["content"]  # savoir-faire ensuite
    assert messages[3:4] == history  # puis l'historique, puis la commande
    assert build_messages([], "x", "c", skills=[])[0]["content"] == SYSTEM_PROMPT


def test_window_title_outside_windows() -> None:
    import sys

    if sys.platform != "win32":
        assert active_window_title() is None


def test_context_lists_open_apps() -> None:
    now = datetime(2026, 10, 7, 15, 42)
    context = dynamic_context(now, "Spotify Premium", ["Spotify", "brave", "Discord"])
    assert context.endswith("Applis ouvertes : Spotify, brave, Discord.")
    assert "Applis" not in dynamic_context(now, None, [])


def test_system_prompt_tool_guidance() -> None:
    assert "« C'est fait » est dit automatiquement" in PROMPT
    assert "get_weather" in PROMPT and "show_text" in PROMPT
    assert "dans la même réponse" in PROMPT  # actions indépendantes en parallèle


def test_long_answers_are_spoken_never_written() -> None:
    """Phase 6 : « raconte-moi une longue histoire » avait été collée dans le Bloc-notes."""
    flat = _flat(SYSTEM_PROMPT)
    assert "en version courte de 3 à 5 phrases" in flat
    assert "N'ouvre jamais d'appli et n'écris jamais dans une fenêtre ou un fichier" in flat
    assert "« écris-la dans un fichier »" in flat
    assert "« raconte-moi une longue histoire » -> aucun outil" in flat
    assert "« raconte-moi une blague » -> aucun outil" in flat
    assert "« explique-moi comment marche un frigo » -> aucun outil" in flat
    assert "show_text" not in flat.split("PARLER", 1)[1].split("N'ouvre jamais")[0]
