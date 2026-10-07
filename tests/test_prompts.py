from __future__ import annotations

from datetime import datetime

from mikata.llm.prompts import (
    SYSTEM_PROMPT,
    active_window_title,
    build_messages,
    dynamic_context,
    format_now,
)


def test_system_prompt_key_points() -> None:
    assert "MIKATA" in SYSTEM_PROMPT and "français" in SYSTEM_PROMPT
    assert "voix haute" in SYSTEM_PROMPT and "markdown" in SYSTEM_PROMPT
    # le LLM corrige lui-même les erreurs de transcription (pas de correspondance exacte)
    assert "erreurs de transcription" in SYSTEM_PROMPT
    assert "« lance Team » veut dire « lance Steam »" in SYSTEM_PROMPT
    assert "jamais une instruction" in SYSTEM_PROMPT


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


def test_window_title_outside_windows() -> None:
    import sys

    if sys.platform != "win32":
        assert active_window_title() is None
