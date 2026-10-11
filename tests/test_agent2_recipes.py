"""Agent v2 (phase 13d) : une demande voisine passe par la recette, jusqu'au premier échec."""

from __future__ import annotations

from pathlib import Path

from mikata.agent2.learn.recipes import RecipeStore
from mikata.agent2.learn.store import SkillMeta, SkillStore
from mikata.agent2.schema import Intent, PageShows, Unknown, Verb
from mikata.config import Agent2Config


def _intent(object_kind: str = "opening d'anime") -> Intent:
    return Intent(
        verb=Verb.PLAY,
        object="le dernier ending de Naruto",
        object_kind=object_kind,
        unknowns=[Unknown(name="titre", question="titre ?", kind="web")],
    )


def _setup(tmp_path: Path) -> tuple[RecipeStore, SkillStore, SkillMeta]:
    cfg = Agent2Config()
    store = SkillStore(tmp_path / "skills", cfg)
    meta = SkillMeta(
        description="Jouer une vidéo",
        capability="play.video",
        channel="youtube",
        params={"query": "titre"},
        done_when=PageShows(pattern="{query}"),
        status="trial",
        os=["win32"],
    )
    store.add("play_youtube_video", meta, "await sdk.browser_open('x')\nreturn 'ok'")
    recipes = RecipeStore(tmp_path / "recipes.json", store, "win32")
    return recipes, store, meta


def test_recipe_is_created_from_a_trace_and_matches_a_neighbor_intent(tmp_path: Path) -> None:
    recipes, _store, _meta = _setup(tmp_path)
    from mikata.agent2.schema import Resolved
    from mikata.agent2.trace import TraceStore

    store2 = TraceStore(tmp_path / "traces")
    trace = store2.new("lance le dernier opening de One Piece")
    trace.intent = _intent()
    trace.resolved = [Resolved(name="titre", value="x", source="web")]
    recipes.from_trace(trace, "play_youtube_video", {"query": "{titre}"})
    # Même verbe, même catégorie, objet différent : la recette s'applique.
    neighbor = _intent()
    matched = recipes.match(neighbor)
    assert matched is not None and matched.skill == "play_youtube_video"
    assert matched.skill_args == {"query": "{titre}"}


def test_a_failed_recipe_is_not_proposed_anymore(tmp_path: Path) -> None:
    recipes, _store, _meta = _setup(tmp_path)
    from mikata.agent2.learn.recipes import Recipe

    recipes.recipes["play|opening d'anime|web"] = Recipe(
        verb="play",
        object_kind="opening d'anime",
        resolve=[{"name": "titre", "kind": "web"}],
        capability="play.video",
        skill="play_youtube_video",
        skill_args={"query": "{titre}"},
        successes=1,
        failures=0,
    )
    recipes._save()
    assert recipes.match(_intent()) is not None
    recipes.record("play|opening d'anime|web", False)
    assert recipes.match(_intent()) is None


def test_key_includes_the_unknown_kinds(tmp_path: Path) -> None:
    recipes, _store, _meta = _setup(tmp_path)
    assert recipes.key(_intent()) == "play|opening d'anime|web"
