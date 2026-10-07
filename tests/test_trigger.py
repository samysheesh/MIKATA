from __future__ import annotations

from mikata.wake.trigger import WakeTrigger


def _run(trigger: WakeTrigger, scores: list[float], t0: float = 0.0) -> list[int]:
    return [i for i, s in enumerate(scores) if trigger.update(s, t0 + i * 0.08)]


def test_needs_consecutive_frames() -> None:
    trigger = WakeTrigger(0.5, 2, 2.0)
    assert _run(trigger, [0.9, 0.1, 0.9, 0.1, 0.6, 0.7]) == [5]


def test_single_frame_mode() -> None:
    assert _run(WakeTrigger(0.5, 1, 0.0), [0.4, 0.5, 0.2]) == [1]


def test_refractory_period() -> None:
    trigger = WakeTrigger(0.5, 2, 2.0)
    # 0,08 s par trame : réfractaire jusqu'à t = 0,08 + 2 = 2,08 s (trame 26)
    hits = _run(trigger, [0.9] * 40)
    assert hits[0] == 1
    assert hits[1] >= 27


def test_suppress_after_hotkey() -> None:
    trigger = WakeTrigger(0.5, 1, 2.0)
    trigger.suppress(10.0)
    assert not trigger.update(0.99, 11.0)
    assert trigger.update(0.99, 12.5)
