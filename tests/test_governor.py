from __future__ import annotations

import psutil
import pytest

from mikata import governor


class FakeProcess:
    def __init__(self, fail: bool = False) -> None:
        self.nice_values: list[int] = []
        self.fail = fail
        self.cpu_values = iter([0.0, 12.0])

    def nice(self, value: int) -> None:
        if self.fail:
            raise psutil.AccessDenied()
        self.nice_values.append(value)

    def cpu_percent(self, interval: float | None) -> float:
        return next(self.cpu_values)


def test_windows_below_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(governor.sys, "platform", "win32")
    monkeypatch.setattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000, raising=False)
    proc = FakeProcess()
    governor.lower_priority(proc)
    assert proc.nice_values == [0x4000]


def test_other_platforms_use_nice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(governor.sys, "platform", "linux")
    proc = FakeProcess()
    governor.lower_priority(proc)
    assert proc.nice_values == [10]


def test_failure_is_not_fatal() -> None:
    governor.lower_priority(FakeProcess(fail=True))


def test_cpu_monitor() -> None:
    monitor = governor.CpuMonitor(FakeProcess(), cpu_count=24)
    assert monitor.sample() == (12.0, 0.5)
