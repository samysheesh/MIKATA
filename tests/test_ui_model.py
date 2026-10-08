"""Phase 8 : état dessiné par la mascotte, avec des événements simulés (sans Qt)."""

from __future__ import annotations

import pytest

from mikata.ui.model import STATE_LABELS, TRANSIENT_S, VISUAL_STATES, MascotModel


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


def _model(clock: Clock) -> MascotModel:
    model = MascotModel(bubble_s=8, clock=clock)
    model.handle({"type": "hello", "state": "idle", "mic": "ok"})
    return model


def test_every_state_has_a_label() -> None:
    assert set(STATE_LABELS) == set(VISUAL_STATES) and len(VISUAL_STATES) == 10


def test_without_mikata_the_mascot_is_off(clock: Clock) -> None:
    model = MascotModel(clock=clock)
    model.disconnected()
    assert model.visual() == "mic_off"
    bubble = model.bubble()
    assert bubble is not None and bubble.note == "MIKATA n'est pas lancé."
    model.handle({"type": "hello", "state": "idle"})
    assert model.visual() == "idle" and model.bubble() is None


def test_a_full_command(clock: Clock) -> None:
    """MIKATA, quelle heure est-il ? -> écoute, réflexion, parole, repos ; bulle."""
    model = _model(clock)
    seen = []
    for message, dt in (
        ({"type": "wake", "score": 0.9}, 0.0),
        ({"type": "state", "state": "listening"}, 0.2),
        ({"type": "state", "state": "transcribing"}, 1.0),
        ({"type": "heard", "id": 1, "text": "quelle heure est-il"}, 0.1),
        ({"type": "state", "state": "thinking"}, 0.1),
        ({"type": "state", "state": "speaking"}, 0.8),
        ({"type": "say", "id": 1, "text": "Il est quinze heures."}, 0.0),
        ({"type": "result", "id": 1, "outcome": "ok"}, 1.0),
        ({"type": "state", "state": "idle"}, 0.0),
    ):
        clock.t += dt
        model.handle(message)
        seen.append(model.visual())
    assert seen == [
        "wake",
        "wake",  # le sursaut dure 0,9 s même si l'écoute a commencé
        "thinking",
        "thinking",
        "thinking",
        "speaking",
        "speaking",
        "speaking",
        "idle",
    ]
    bubble = model.bubble()
    assert bubble is not None
    assert (bubble.heard, bubble.said) == ("quelle heure est-il", "Il est quinze heures.")
    clock.t += 9  # bulle_s écoulé, au repos : elle disparaît
    assert model.bubble() is None


def test_success_and_error_are_played_once(clock: Clock) -> None:
    model = _model(clock)
    model.handle({"type": "state", "state": "speaking"})
    model.handle({"type": "result", "id": 2, "outcome": "success"})
    assert model.visual() == "success"
    clock.t += TRANSIENT_S["success"] + 0.01
    assert model.visual() == "speaking"
    model.handle({"type": "error", "id": 3, "message": "serveur injoignable"})
    assert model.visual() == "error"
    assert model.bubble().note == "Erreur : serveur injoignable"  # type: ignore[union-attr]
    clock.t += TRANSIENT_S["error"] + 0.01
    assert model.visual() == "speaking"
    assert model.visual_age() == pytest.approx(0.0)


def test_follow_up_countdown(clock: Clock) -> None:
    model = _model(clock)
    model.handle({"type": "state", "state": "follow_up", "follow_up_s": 6})
    assert model.visual() == "follow_up" and model.follow_up_left() == pytest.approx(1.0)
    clock.t += 3
    assert model.follow_up_left() == pytest.approx(0.5)
    clock.t += 5
    assert model.follow_up_left() == 0.0
    model.handle({"type": "state", "state": "idle"})
    assert model.follow_up_left() is None


@pytest.mark.parametrize("status", ["paused", "busy"])
def test_microphone_off(clock: Clock, status: str) -> None:
    model = _model(clock)
    model.handle({"type": "mic", "status": status})
    assert model.visual() == "mic_off"
    model.handle({"type": "mic", "status": "ok"})
    assert model.visual() == "idle"


def test_ignored_follow_up_clears_the_bubble(clock: Clock) -> None:
    model = _model(clock)
    model.handle({"type": "state", "state": "follow_up"})
    model.handle({"type": "heard", "id": 4, "text": "tu viens manger ?"})
    model.handle({"type": "result", "id": 4, "outcome": "ignored"})
    model.handle({"type": "state", "state": "idle"})
    assert model.bubble() is None


def test_announcement_starts_a_new_bubble(clock: Clock) -> None:
    model = _model(clock)
    model.handle({"type": "heard", "id": 5, "text": "mets un minuteur"})
    model.handle({"type": "say", "id": 5, "text": "C'est fait."})
    model.handle({"type": "say", "id": None, "text": "Ton minuteur est terminé."})
    bubble = model.bubble()
    assert bubble is not None and (bubble.heard, bubble.said) == ("", "Ton minuteur est terminé.")


def test_bye_turns_it_off(clock: Clock) -> None:
    model = _model(clock)
    model.handle({"type": "state", "state": "thinking"})
    model.handle({"type": "bye"})
    assert not model.connected and model.visual() == "mic_off"
