"""Cache d'état ambiant (phase 12a) : thread dédié, pause pendant les actions, contenu
jamais loggé ni écrit sur disque, lecture du salon vocal Discord."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from mikata.uia.client import UiaAccessibilityClient
from mikata.uia.controller import _discord_voice_from
from mikata.uia.state import AmbientState


def test_poller_fills_the_cache_and_summarizes() -> None:
    state = AmbientState(0.05, lambda: {"discord": "salon « Général »"})
    state.start()
    try:
        deadline = time.monotonic() + 2
        while not state.summary() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert state.summary() == "salon « Général »"
    finally:
        state.stop()


def test_poller_skips_reads_while_paused() -> None:
    hits: list[int] = []

    def snapshot() -> dict[str, str]:
        hits.append(1)
        return {"discord": "x"}

    state = AmbientState(0.02, snapshot)
    state.pause(True)
    state.start()
    try:
        time.sleep(0.15)
        assert hits == []  # en pause : aucune lecture
        state.pause(False)
        deadline = time.monotonic() + 2
        while not hits and time.monotonic() < deadline:
            time.sleep(0.02)
        assert hits
    finally:
        state.stop()


def test_snapshot_errors_are_swallowed() -> None:
    def snapshot() -> dict[str, str]:
        raise RuntimeError("Discord illisible")

    state = AmbientState(0.02, snapshot)
    state.start()
    try:
        time.sleep(0.15)
        assert state.summary() == ""  # une erreur ne casse rien et ne fuit pas
    finally:
        state.stop()


def test_stop_terminates_the_thread() -> None:
    state = AmbientState(0.02, lambda: {})
    state.start()
    thread = state._thread
    assert thread is not None and thread.is_alive()
    state.stop()
    assert not thread.is_alive() and state._thread is None
    state.stop()  # idempotent


def test_poll_s_zero_never_starts() -> None:
    state = AmbientState(0, lambda: {"discord": "x"})
    state.start()
    assert state._thread is None and state.summary() == ""


def test_summary_joins_non_empty_entries_only() -> None:
    state = AmbientState(0, lambda: {})
    state._data = {"discord": "salon « Général »", "rien": "  "}
    assert state.summary() == "salon « Général »"


def test_poller_runs_on_its_own_thread() -> None:
    seen: list[int] = []

    def snapshot() -> dict[str, str]:
        seen.append(threading.get_ident())
        return {}

    state = AmbientState(0.02, snapshot)
    state.start()
    try:
        deadline = time.monotonic() + 2
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        assert seen and seen[0] != threading.get_ident()
    finally:
        state.stop()


def _node(kind: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(ControlTypeName=kind, Name=name)


def test_discord_connected_channel_and_members_are_parsed() -> None:
    """Structure réelle constatée sur le client Discord (fenêtre française) : le salon
    connecté est un lien « <salon> (salon vocal), membre1, membre2, durée de l'appel... »."""
    nodes = [
        _node("ButtonControl", "valorant (salon vocal)"),
        _node(
            "HyperlinkControl",
            "Champions league (salon vocal), Sheeesh, T4EIIKOOO, durée de l'appel 8 h 26",
        ),
        _node("HyperlinkControl", "general (salon textuel)"),
    ]
    channel, members = _discord_voice_from(nodes)
    assert channel == "Champions league"
    assert members == ["Sheeesh", "T4EIIKOOO"]


def test_discord_connected_alone_gives_no_member() -> None:
    nodes = [_node("HyperlinkControl", "Champions league (salon vocal), durée de l'appel 2 h")]
    channel, members = _discord_voice_from(nodes)
    assert channel == "Champions league" and members == []


def test_discord_not_connected_gives_nothing() -> None:
    nodes = [
        _node("ButtonControl", "valorant (salon vocal)"),
        _node("HyperlinkControl", "general (salon textuel), Salon privé (verrouillé)"),
    ]
    channel, members = _discord_voice_from(nodes)
    assert channel is None and members == []


def test_discord_english_client_is_understood() -> None:
    nodes = [
        _node(
            "HyperlinkControl",
            "Champions league (voice channel), Sheeesh, T4EIIKOOO, duration 8 h",
        )
    ]
    channel, members = _discord_voice_from(nodes)
    assert channel == "Champions league" and members == ["Sheeesh", "T4EIIKOOO"]


def test_accessibility_client_starts_and_stops() -> None:
    """Cycle de vie du client d'accessibilité : start/stop idempotents, jamais d'erreur
    (sous Windows il s'abonne pour de vrai ; ailleurs il ne fait rien)."""
    client = UiaAccessibilityClient()
    client.start()
    client.start()
    client.stop()
    assert client._thread is None
    client.stop()
