"""Phase 16e : préférences apprises des corrections, épisodes tirés des traces."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from agent_fakes import ScriptedLLM as TextLLM
from test_agent import NOW, Phrases, ScriptedLLM, Sink
from tool_fakes import make_ctx

from mikata.agent.learning import Learner
from mikata.agent.loop import Agent
from mikata.config import Config
from mikata.learn.episodes import episodes, refers_to_past, render
from mikata.learn.playbook import Playbook
from mikata.learn.preferences import PreferenceStore
from mikata.learn.reflect import parse_preferences
from mikata.learn.trace import TraceStore, TurnTrace
from mikata.llm.base import Done, TextDelta
from mikata.tools.registry import ToolRegistry

# -- préférences ------------------------------------------------------------------------------


def test_preferences_grow_with_confirmations_and_follow_a_change_of_mind(
    tmp_path: Path,
) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    store.learn("musique", "Spotify")
    store.learn("Musique", "spotify")  # même préférence, autre casse
    assert store.items[0].seen == 2 and store.items[0].label == "musique : Spotify (montré 2 fois)"
    store.learn("musique", "Deezer")  # l'utilisateur a changé d'avis
    assert store.items[0].value == "Deezer" and store.items[0].seen == 1
    store.learn("météo", "Lyon")
    again = PreferenceStore(tmp_path / "preferences.json")
    assert {p.subject for p in again.items} == {"musique", "météo"}
    assert "- météo : Lyon (montré une fois)" in again.render()
    assert again.forget("oublie ma préférence pour la musique") is not None
    assert [p.subject for p in again.items] == ["météo"]
    assert store.learn("", "x") is None


def test_preferences_are_capped(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "p.json", max_items=2)
    store.learn("a", "1")
    store.learn("b", "2")
    store.learn("b", "2")
    store.learn("c", "3")
    assert len(store.items) == 2 and store.items[0].subject == "b"  # la plus confirmée reste


def test_reflection_answer_carries_preferences() -> None:
    text = (
        'ok {"lecons": [], "preferences": '
        '[{"sujet": "musique", "valeur": "Spotify"}, {"sujet": ""}]}'
    )
    assert parse_preferences(text) == [("musique", "Spotify")]
    assert parse_preferences('{"lecons": []}') == [] and parse_preferences("rien") == []


def _trace(request: str, at: str, answer: str = "C'est fait.", tainted: bool = False) -> TurnTrace:
    return TurnTrace(
        id=request[:5], at=at, request=request, answer=answer, direct=["x"], tainted=tainted
    )


def test_preferences_are_learned_only_from_a_correction(tmp_path: Path) -> None:
    """La réflexion d'un échec corrigé dans le même tour donne des leçons, pas de
    préférence : seule une correction de l'utilisateur en donne."""
    config = Config()
    answer = json.dumps({"lecons": [], "preferences": [{"sujet": "musique", "valeur": "Spotify"}]})
    store = PreferenceStore(tmp_path / "p.json")
    learner = Learner(
        TextLLM([answer, answer]), lambda: config, None, Playbook(tmp_path / "pb.json"), None, store
    )
    before = _trace("mets de la musique", "2026-10-14T10:00:00")
    after = _trace("non, sur Spotify", "2026-10-14T10:00:20")
    asyncio.run(learner._reflect(after, None))  # échec corrigé seul : pas de préférence
    assert store.items == []
    asyncio.run(learner._reflect(after, before))
    assert [(p.subject, p.value) for p in store.items] == [("musique", "Spotify")]


def test_forget_tool_also_forgets_a_preference(tmp_path: Path) -> None:
    from mikata.tools.memory_tools import Forget, ForgetParams

    ctx = make_ctx(tmp_path)
    ctx.preferences = PreferenceStore(tmp_path / "p.json")
    ctx.preferences.learn("musique", "Spotify")
    text = asyncio.run(Forget(ctx).run(ForgetParams(fact="ma préférence pour la musique")))
    assert text == "Oublié : musique, Spotify." and ctx.preferences.items == []


# -- épisodes ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "past"),
    [
        ("remets la même vidéo qu'hier", True),
        ("qu'est-ce que tu as fait ce matin ?", True),
        ("relance la musique comme d'habitude", True),
        ("play it again", True),
        ("lance le dernier opening de Naruto", False),
        ("quel temps fait-il ?", False),
    ],
)
def test_requests_that_refer_to_the_past(command: str, past: bool) -> None:
    assert refers_to_past(command) is past


def test_episodes_rank_by_shared_words_then_recency_and_hide_untrusted_answers() -> None:
    now = datetime(2026, 10, 14, 15, 0)
    traces = [
        _trace("lance le dernier opening de Bleach", "2026-10-13T21:05:00", "C'est lancé."),
        _trace("range les fichiers texte", "2026-10-14T09:30:00"),
        _trace("résume cette page", "2026-10-14T11:00:00", "La page dit X", tainted=True),
    ]
    lines = episodes("remets l'opening d'hier", traces, now)
    assert lines[0] == "hier à 21 h 05 : « lance le dernier opening de Bleach » -> « C'est lancé. »"
    assert "aujourd'hui à 11 h 00 : « résume cette page »" in lines  # sans la réponse lue
    assert all("La page dit" not in line for line in lines)
    assert render(lines).startswith("Ce que MIKATA a fait récemment")
    assert render([]) == ""


def test_episodes_reach_the_model_only_when_the_request_refers_to_the_past(
    tmp_path: Path,
) -> None:
    traces = TraceStore(tmp_path / "traces")
    traces.save(_trace("lance le dernier opening de Bleach", "2026-10-07T09:00:00", "C'est lancé."))
    llm = ScriptedLLM([TextDelta("D'accord."), Done("stop")], [TextDelta("Ok."), Done("stop")])
    ctx = make_ctx(tmp_path)
    agent = Agent(
        llm,
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
        traces=traces,
        preferences=PreferenceStore(tmp_path / "p.json"),
    )
    agent.preferences.learn("musique", "Spotify")  # type: ignore[union-attr]

    async def scenario() -> None:
        try:
            await agent.respond("remets la même vidéo que ce matin", Sink())
            await agent.respond("lance Spotify", Sink())
        finally:
            await agent.aclose()

    asyncio.run(scenario())
    first = [m["content"] for m in llm.calls[0] if m["role"] == "system"]
    assert any("« lance le dernier opening de Bleach » -> « C'est lancé. »" in c for c in first)
    assert any("musique : Spotify" in c for c in first)  # préférences données au modèle
    second = [m["content"] for m in llm.calls[1] if m["role"] == "system"]
    assert not any("fait récemment" in c for c in second)
