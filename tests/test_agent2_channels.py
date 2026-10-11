"""Agent v2 (phase 13b) : capacités, disponibilité et choix du canal par score."""

from __future__ import annotations

from pathlib import Path

import pytest

from mikata.agent2.channels import available, capability_for, choose
from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.schema import Intent, Verb

INSTALLED = ["Spotify", "Discord", "Brave", "Google Chrome"]


def _intent(verb: Verb, object_kind: str, hint: str | None = None) -> Intent:
    return Intent(verb=verb, object="x", object_kind=object_kind, channel_hint=hint)


def _prefs(path: Path) -> PreferenceStore:
    return PreferenceStore(path / "preferences.json")


@pytest.mark.parametrize(
    ("verb", "kind", "expected"),
    [
        (Verb.PLAY, "opening d'anime", "play.video"),
        (Verb.PLAY, "clip", "play.video"),
        (Verb.PLAY, "musique", "play.music"),
        (Verb.PLAY, "playlist", "play.music"),
        (Verb.PLAY, "podcast", "play.podcast"),
        (Verb.SEND, "fichier", "send.file"),
        (Verb.SEND, "message", "send.message"),
        (Verb.FIND, "facture", "find.file"),
        (Verb.ORGANIZE, "documents", "organize.file"),
        (Verb.OPEN, "application", "open.app"),
        (Verb.OPEN, "site", "open.site"),
        (Verb.ASK, "météo", "ask.web"),
        (Verb.CONTROL, "volume", "control.system"),
        (Verb.SCHEDULE, "rappel", "schedule.timer"),
        (Verb.CREATE, "note", "create.document"),
        (Verb.MONITOR, "prix", "monitor.generic"),
    ],
)
def test_capability_for(verb: Verb, kind: str, expected: str) -> None:
    assert capability_for(_intent(verb, kind)) == expected


def test_hint_wins_even_with_a_strong_preference(tmp_path: Path) -> None:
    prefs = _prefs(tmp_path)
    prefs.confirm("play", "opening d'anime", "youtube")
    prefs.channels[prefs.key("play", "opening d'anime")]["youtube"] = 0.9
    channel, announce = choose(
        _intent(Verb.PLAY, "opening d'anime", hint="spotify"), prefs, INSTALLED, {}
    )
    assert channel.name == "spotify" and not announce


def test_no_preference_chooses_youtube_and_announces(tmp_path: Path) -> None:
    channel, announce = choose(
        _intent(Verb.PLAY, "opening d'anime"), _prefs(tmp_path), INSTALLED, {}
    )
    assert channel.name == "youtube" and announce


def test_strong_preference_does_not_announce(tmp_path: Path) -> None:
    prefs = _prefs(tmp_path)
    prefs.channels[prefs.key("play", "opening d'anime")] = {"youtube": 0.9}
    channel, announce = choose(_intent(Verb.PLAY, "opening d'anime"), prefs, INSTALLED, {})
    assert channel.name == "youtube" and not announce


def test_uninstalled_app_is_never_chosen(tmp_path: Path) -> None:
    channel, _ = choose(_intent(Verb.PLAY, "musique"), _prefs(tmp_path), ["Brave", "Discord"], {})
    assert channel.name == "youtube"  # Spotify absent des applis installées


def test_no_capable_channel_falls_back_to_browser(tmp_path: Path) -> None:
    channel, announce = choose(_intent(Verb.MONITOR, "prix"), _prefs(tmp_path), INSTALLED, {})
    assert channel.name == "browser" and announce


def test_available_checks_installed_apps() -> None:
    from mikata.agent2.channels import BUILTIN_CHANNELS

    spotify = next(channel for channel in BUILTIN_CHANNELS if channel.name == "spotify")
    assert available(spotify, ["Spotify"]) and not available(spotify, ["Brave"])
    web = next(channel for channel in BUILTIN_CHANNELS if channel.name == "web")
    assert available(web, [])  # un canal système est toujours disponible
