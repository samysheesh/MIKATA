"""Agent v2 (phase 13b) : les préférences montent, chutent et survivent au disque."""

from __future__ import annotations

from pathlib import Path

from mikata.agent2.preferences import PreferenceStore


def test_confirm_raises_and_caps(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    for _ in range(15):
        store.confirm("play", "musique", "spotify")
    assert store.scores("play", "musique")["spotify"] == 1.0


def test_correct_lowers_the_wrong_and_sets_the_right(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    store.confirm("play", "musique", "spotify")
    store.correct("play", "musique", "spotify", "youtube")
    scores = store.scores("play", "musique")
    assert scores["youtube"] == 0.95
    assert scores["spotify"] < 0.1


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    store = PreferenceStore(path)
    store.confirm("play", "opening d'anime", "youtube")
    reloaded = PreferenceStore(path)
    assert reloaded.scores("play", "opening d'anime") == store.scores("play", "opening d'anime")


def test_missing_or_broken_file_is_empty(tmp_path: Path) -> None:
    assert PreferenceStore(tmp_path / "absent.json").scores("play", "x") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{pas du json", encoding="utf-8")
    assert PreferenceStore(broken).scores("play", "x") == {}


def test_key_and_describe(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    assert PreferenceStore.key("play", " Opening d'anime ") == "play|opening d'anime"
    store.channels[store.key("play", "opening d'anime")] = {"youtube": 0.9}
    described = store.describe()
    assert "youtube" in described and "opening d'anime" in described
