"""Agent v2 (phase 13b) : la compréhension normalise une demande en Intention."""

from __future__ import annotations

import asyncio

from agent2_fakes import ScriptedLLM

from mikata.agent2.schema import Verb
from mikata.agent2.understand import understand

_INTENT_JSON = """\
{"verb": "play", "object": "dernier opening de One Piece",
 "object_kind": "opening d'anime", "entities": {"oeuvre": "One Piece"},
 "constraints": [], "target": null,
 "unknowns": [{"name": "titre_exact",
   "question": "quel est le titre exact du dernier opening de One Piece ?",
   "kind": "web"}],
 "sub_intents": []}"""


def _run(llm: ScriptedLLM, request: str = "lance le dernier opening de One Piece") -> tuple:
    async def scenario() -> tuple:
        return await understand(
            llm,
            request,
            context="fenêtre active : Brave",
            preferences="",
            os_name="win32",
            effort="none",
        )

    return asyncio.run(scenario())


def test_understand_produces_the_intent() -> None:
    llm = ScriptedLLM([_INTENT_JSON])
    intent, usage, calls = _run(llm)
    assert intent is not None
    assert intent.verb is Verb.PLAY
    assert intent.channel_hint is None
    assert len(intent.unknowns) == 1 and intent.unknowns[0].kind == "web"
    assert calls == 1 and usage is not None


def test_channel_hint_is_normalized() -> None:
    llm = ScriptedLLM(
        [
            '{"verb": "play", "object": "musique", "object_kind": "musique",'
            ' "entities": {}, "constraints": [], "target": null,'
            ' "channel_hint": " Spotify ", "unknowns": [], "sub_intents": []}'
        ]
    )
    intent, _, _ = _run(llm)
    assert intent is not None and intent.channel_hint == "spotify"


def test_chat_verb_clears_unknowns() -> None:
    llm = ScriptedLLM(
        [
            '{"verb": "chat", "object": "une blague", "object_kind": "",'
            ' "entities": {}, "constraints": [], "target": null,'
            ' "unknowns": [{"name": "x", "question": "quoi ?", "kind": "web"}],'
            ' "sub_intents": []}'
        ]
    )
    intent, _, _ = _run(llm, request="raconte-moi une blague")
    assert intent is not None and intent.verb is Verb.CHAT
    assert intent.unknowns == []


def test_invalid_twice_gives_none() -> None:
    llm = ScriptedLLM(["pas du json", "encore raté"])
    intent, _, calls = _run(llm)
    assert intent is None and calls == 2
