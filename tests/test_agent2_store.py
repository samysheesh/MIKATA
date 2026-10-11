"""Agent v2 (phase 13d) : le cycle de vie des savoir-faire, sans raccourci."""

from __future__ import annotations

from pathlib import Path

import pytest

from mikata.agent2.learn.store import SKILLS_DIR, SkillMeta, SkillStore, needs_approval
from mikata.agent2.sandbox import CodeRejected
from mikata.agent2.schema import PageShows
from mikata.config import Agent2Config

_CODE = "await sdk.browser_open('https://www.youtube.com')\nreturn 'ok'"


def _meta(**overrides: object) -> SkillMeta:
    fields = dict(
        description="Jouer une vidéo",
        capability="play.video",
        channel="youtube",
        params={"query": "titre"},
        done_when=PageShows(pattern="{query}"),
        os=["win32"],
    )
    fields.update(overrides)
    return SkillMeta(**fields)


def _store(tmp_path: Path, **cfg_overrides: object) -> SkillStore:
    cfg = Agent2Config(**cfg_overrides)
    return SkillStore(tmp_path / "skills", cfg)


def test_full_cycle_proposed_trial_active(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("play_video", _meta(), _CODE)
    assert store.skills["play_video"].status == "proposed"
    store.record("play_video", True)  # successes 1 -> trial
    assert store.skills["play_video"].status == "trial"
    for _ in range(2):  # successes 3 -> active
        store.record("play_video", True)
    assert store.skills["play_video"].status == "active"


def test_two_failures_demote_and_remove_from_usable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("play_video", _meta(), _CODE)
    store.record("play_video", True)
    store.record("play_video", False)
    assert store.skills["play_video"].status == "trial"  # un seul échec
    store.record("play_video", False)
    assert store.skills["play_video"].status == "demoted"
    assert store.usable("play.video", "win32") == []


def test_needs_approval_detected_and_blocks_active_until_approved(tmp_path: Path) -> None:
    code = "await sdk.trash('C:/vieux.txt')\nreturn 'ok'"
    assert needs_approval(code) and not needs_approval(_CODE)
    store = _store(tmp_path)
    store.add("nettoyage", _meta(needs_approval=True), code)
    assert store.usable("play.video", "win32") == []  # proposed, pas proposable
    for _ in range(3):
        store.record("nettoyage", True)
    assert store.skills["nettoyage"].status == "trial"  # bloqué : pas approuvé
    assert store.usable("play.video", "win32") == []  # needs_approval sans approbation
    store.skills["nettoyage"].approved = True
    store._save()
    assert store.usable("play.video", "win32") == ["nettoyage"]
    store.record("nettoyage", True)
    assert store.skills["nettoyage"].status == "active"


def test_retire_uses_the_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    store.add("play_video", _meta(), _CODE)
    trashed: list[str] = []
    monkeypatch.setattr("send2trash.send2trash", trashed.append)
    store.retire("play_video")
    assert "play_video" not in store.skills
    assert trashed and trashed[0].endswith("play_video.py")


def test_invalid_code_is_refused_at_add(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(CodeRejected):
        store.add("mauvais", _meta(), "import os\nreturn 'x'")
    assert "mauvais" not in store.skills


def test_skills_dir_constant() -> None:
    assert SKILLS_DIR == "data/agent2/skills"
