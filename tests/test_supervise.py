"""Phase 7 : relance de MIKATA s'il plante (`mikata.supervise`, `python -m mikata --supervise`)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import mikata.__main__ as entry
from mikata.supervise import child_command, supervise


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _runner(
    codes: list[int], clock: Clock, lasts_s: float = 60.0
) -> tuple[list[list[str]], object]:
    calls: list[list[str]] = []

    def run(cmd: Sequence[str]) -> int:
        calls.append(list(cmd))
        clock.t += lasts_s
        return codes.pop(0)

    return calls, run


@pytest.mark.parametrize("code", [0, 2])
def test_no_restart_after_quit_or_bad_config(code: int) -> None:
    clock = Clock()
    calls, run = _runner([code], clock)
    assert supervise(["mikata"], run=run, sleep=clock.sleep, clock=clock) == code  # type: ignore[arg-type]
    assert len(calls) == 1


def test_restarts_after_a_crash() -> None:
    clock = Clock()
    calls, run = _runner([1, 3, 0], clock)
    assert supervise(["m"], run=run, sleep=clock.sleep, clock=clock, delay_s=10) == 0  # type: ignore[arg-type]
    assert len(calls) == 3 and clock.t == 3 * 60 + 2 * 10


def test_gives_up_after_too_many_crashes_in_an_hour() -> None:
    clock = Clock()
    calls, run = _runner([1] * 10, clock, lasts_s=5)
    result = supervise(  # type: ignore[arg-type]
        ["m"], run=run, sleep=clock.sleep, clock=clock, max_restarts=3, delay_s=1
    )
    assert result == 1 and len(calls) == 4  # 1 lancement + 3 relances, puis abandon


def test_old_crashes_are_forgotten() -> None:
    clock = Clock()
    calls, run = _runner([1, 1, 1, 0], clock, lasts_s=4000)  # un plantage toutes les ~67 min
    result = supervise(  # type: ignore[arg-type]
        ["m"], run=run, sleep=clock.sleep, clock=clock, max_restarts=1, window_s=3600
    )
    assert result == 0 and len(calls) == 4


def test_child_command_drops_supervise() -> None:
    assert child_command(["--supervise", "--no-tray"]) == [
        sys.executable,
        "-m",
        "mikata",
        "--no-tray",
    ]


def test_main_supervise_runs_the_supervisor(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_supervise(command: Sequence[str]) -> int:
        seen.append(list(command))
        return 0

    monkeypatch.setattr("mikata.supervise.supervise", fake_supervise)
    assert entry.main(["--config", str(config_file), "--supervise"]) == 0
    assert seen == [[sys.executable, "-m", "mikata", "--config", str(config_file)]]
    assert (config_file.parent / "data" / "logs" / "supervise.log").is_file()


def test_a_requested_restart_is_immediate_and_not_counted_as_a_crash() -> None:
    """« Redémarrer » dans l'interface (code 75) : relance tout de suite, sans attente et
    sans entamer la limite horaire de relances après plantage."""
    from mikata.supervise import RESTART_CODE

    clock = Clock()
    calls, run = _runner([RESTART_CODE] * 6 + [0], clock, lasts_s=5)
    result = supervise(  # type: ignore[arg-type]
        ["m"], run=run, sleep=clock.sleep, clock=clock, max_restarts=1, delay_s=30
    )
    assert result == 0 and len(calls) == 7  # six redémarrages, malgré max_restarts=1
    assert clock.t == 7 * 5  # aucune attente de 30 s


def test_main_relaunches_itself_on_restart_without_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(entry, "_main", lambda argv=None: entry.EXIT_RESTART)
    monkeypatch.setattr("mikata.supervise.relaunch_self", lambda argv: launched.append(list(argv)))
    monkeypatch.delenv("MIKATA_SUPERVISED", raising=False)
    assert entry.main(["--no-tray"]) == entry.EXIT_OK  # le nouveau processus prend le relais
    assert launched == [["--no-tray"]]


def test_main_leaves_the_relaunch_to_the_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(entry, "_main", lambda argv=None: entry.EXIT_RESTART)
    monkeypatch.setattr("mikata.supervise.relaunch_self", lambda argv: launched.append(list(argv)))
    monkeypatch.setenv("MIKATA_SUPERVISED", "1")
    assert entry.main(["--no-tray"]) == entry.EXIT_RESTART == 75  # le superviseur relance
    assert launched == []


def test_ui_can_ask_for_a_restart() -> None:
    from mikata.ui.protocol import UI_COMMANDS

    assert "restart" in UI_COMMANDS
