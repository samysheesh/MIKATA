"""Composition des listes noires par plateforme (hygiène macos_adapter).

La liste PowerShell (`BLACKLIST`) s'applique partout ; les commandes du shell natif
(`DARWIN_BLACKLIST`) ne s'ajoutent que sous macOS. Les deux listes restent des
constantes distinctes : modifier BLACKLIST (Windows) ne doit pas changer silencieusement
la sémantique macOS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tool_fakes import make_ctx

from mikata.agent import guard as guard_module
from mikata.agent.guard import BLACKLIST, DARWIN_BLACKLIST, Guard, Verdict


@pytest.fixture
def guard(tmp_path: Path) -> Guard:
    guard = make_ctx(tmp_path).guard
    assert guard is not None
    return guard


def test_blacklist_for_darwin_is_the_union() -> None:
    rules = guard_module._blacklist_for("darwin")
    assert rules == BLACKLIST + DARWIN_BLACKLIST
    assert len(rules) == len(BLACKLIST) + len(DARWIN_BLACKLIST)


@pytest.mark.parametrize("platform", ["win32", "linux", "freebsd", "other"])
def test_blacklist_for_non_darwin_is_windows_only(platform: str) -> None:
    assert guard_module._blacklist_for(platform) is BLACKLIST


# « nvram -c » : un motif de DARWIN_BLACKLIST uniquement (BLACKLIST ne contient que des
# commandes Windows/PowerShell et les alias de suppression, qui ne couvrent pas nvram).
DARWIN_ONLY_COMMAND = "nvram -c"
WINDOWS_BLACKLISTED_COMMAND = "Remove-Item notes.txt"


def test_darwin_blocks_native_shell_commands(guard: Guard, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard_module.sys, "platform", "darwin")
    decision = guard.check_command(DARWIN_ONLY_COMMAND)
    assert decision.verdict is Verdict.DENY


def test_darwin_still_blocks_windows_blacklist(
    guard: Guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard_module.sys, "platform", "darwin")
    decision = guard.check_command(WINDOWS_BLACKLISTED_COMMAND)
    assert decision.verdict is Verdict.DENY


def test_win32_does_not_block_darwin_only_patterns(
    guard: Guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard_module.sys, "platform", "win32")
    decision = guard.check_command(DARWIN_ONLY_COMMAND)
    assert decision.verdict is Verdict.ALLOW


def test_win32_blocks_windows_blacklist(guard: Guard, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard_module.sys, "platform", "win32")
    decision = guard.check_command(WINDOWS_BLACKLISTED_COMMAND)
    assert decision.verdict is Verdict.DENY
