"""Helper élevé de l'index des fichiers (phase 12d), audit de la phase 14f : un journal
qui reste illisible ne doit pas relancer un scan complet du disque en boucle."""

from __future__ import annotations

from pathlib import Path

import pytest

from mikata.files import events, helper


class Stop(Exception):
    """Sort de la boucle infinie du helper."""


def test_an_unreadable_journal_does_not_rescan_the_disk_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / events.DUMP).write_text("", encoding="utf-8")  # dump récent : pas de dump initial
    dumps: list[Path] = []
    sleeps = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] >= 10:
            raise Stop

    def unreadable(*args: object, **kwargs: object) -> tuple[list[object], int]:
        raise OSError("journal illisible")

    monkeypatch.setattr(helper, "_require_win32", lambda: None)
    monkeypatch.setattr(helper.usn, "list_ntfs_volumes", lambda: ["C:"])
    monkeypatch.setattr(helper.usn, "query_journal", lambda volume: (1, 0))
    monkeypatch.setattr(helper.usn, "read_journal", unreadable)
    monkeypatch.setattr(helper, "dump_all", lambda share: dumps.append(share) or 0)
    monkeypatch.setattr(helper.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        helper.watch(tmp_path)
    # Trois nouveaux dumps au plus, puis le volume est abandonné : les tours suivants ne
    # relisent plus rien.
    assert len(dumps) == helper._MAX_JOURNAL_FAILURES
    assert sleeps["n"] == 10


def test_a_readable_journal_resets_the_failure_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / events.DUMP).write_text("", encoding="utf-8")
    dumps: list[Path] = []
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> tuple[list[object], int]:
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # un tour sur deux échoue : jamais « de suite »
            raise OSError("journal illisible")
        return [], calls["n"]

    def fake_sleep(seconds: float) -> None:
        if calls["n"] >= 12:
            raise Stop

    monkeypatch.setattr(helper, "_require_win32", lambda: None)
    monkeypatch.setattr(helper.usn, "list_ntfs_volumes", lambda: ["C:"])
    monkeypatch.setattr(helper.usn, "query_journal", lambda volume: (1, 0))
    monkeypatch.setattr(helper.usn, "read_journal", flaky)
    monkeypatch.setattr(helper, "dump_all", lambda share: dumps.append(share) or 0)
    monkeypatch.setattr(helper.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        helper.watch(tmp_path)
    assert len(dumps) == 6  # chaque échec isolé redumpe, le volume reste suivi
