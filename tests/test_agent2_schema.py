"""Agent v2 (phase 13a) : les modèles de données se sérialisent et se relisent."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mikata.agent2.schema import (
    AppFocused,
    FileExists,
    Intent,
    Judge,
    PageShows,
    TextVisible,
    ToolSucceeded,
    Verb,
    WindowTitleMatches,
)


def test_composite_intent_roundtrip() -> None:
    intent = Intent(
        verb=Verb.AUTOMATE,
        object="mode jeu",
        sub_intents=[
            Intent(verb=Verb.OPEN, object="Steam"),
            Intent(verb=Verb.CONTROL, object="volume", constraints=["40 pour cent"]),
        ],
    )
    reloaded = Intent.model_validate_json(intent.model_dump_json())
    assert reloaded == intent
    assert reloaded.sub_intents[0].verb == Verb.OPEN


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "app_focused", "name": "Spotify"}, AppFocused),
        ({"type": "window_title", "pattern": "Spotify"}, WindowTitleMatches),
        ({"type": "page_shows", "pattern": "One Piece"}, PageShows),
        ({"type": "file_exists", "path": "~/x.pdf"}, FileExists),
        ({"type": "text_visible", "pattern": "Envoyé"}, TextVisible),
        ({"type": "tool_ok", "tool": "open_app"}, ToolSucceeded),
        ({"type": "judge", "question": "Est-ce lancé ?"}, Judge),
    ],
)
def test_each_predicate_loads_from_its_type(payload: dict[str, str], expected: type) -> None:
    from pydantic import TypeAdapter

    from mikata.agent2.schema import Predicate

    predicate = TypeAdapter(Predicate).validate_python(payload)
    assert isinstance(predicate, expected)


def test_unknown_predicate_type_is_rejected() -> None:
    from pydantic import TypeAdapter

    from mikata.agent2.schema import Predicate

    with pytest.raises(ValidationError):
        TypeAdapter(Predicate).validate_python({"type": "magic", "x": 1})
