"""Phase 16f : partage des compétences (export, import proposé, activation)."""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path

import pytest
from test_skills import _skill
from tool_fakes import make_ctx

from mikata.learn.skills import SkillStore, check_shape
from mikata.llm.base import ToolCallReq
from mikata.tools.registry import ToolRegistry


def test_export_then_import_arrives_proposed_under_a_free_name(tmp_path: Path) -> None:
    mine = SkillStore(tmp_path / "a")
    skill = mine.adopt(_skill())
    skill.successes, skill.status = 5, "active"
    mine.save(skill)
    archive = mine.export(skill.call_name, tmp_path / "partage")
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["lancer-opening/SKILL.md", "lancer-opening/skill.py"]

    theirs = SkillStore(tmp_path / "b")
    theirs.adopt(_skill())  # même nom déjà pris chez l'autre
    imported = theirs.import_package(archive)
    assert imported.name == "lancer-opening-2" and imported.status == "proposed"
    assert imported.successes == 0 and imported.source == "importée de lancer-opening.zip"
    assert imported not in theirs.usable()  # rien ne s'en sert avant l'activation
    assert theirs.approve(imported.call_name).status == "trial"
    assert imported.call_name in {s.call_name for s in SkillStore(tmp_path / "b").usable()}


def test_an_existing_export_is_never_overwritten(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "a")
    skill = store.adopt(_skill())
    first = store.export(skill.call_name, tmp_path / "partage")
    before = first.read_bytes()
    second = store.export(skill.call_name, tmp_path / "partage")
    assert second.name == "lancer-opening (2).zip" and first.read_bytes() == before


def _package(tmp_path: Path, *, patterns: str, code: str, name: str = "x-y") -> Path:
    folder = tmp_path / "src" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\nmetadata:\n  mikata-status: active\n"
        f'  mikata-patterns: "{patterns}"\n---\n',
        encoding="utf-8",
    )
    (folder / "skill.py").write_text(f"def run(anime):\n    {code}\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize(
    ("patterns", "code", "reason"),
    [
        ("{anime}", "browser_open(url=anime)\n    browser_read()", "motif trop vague"),
        ("lance l'opening de {anime}", "browser_open(url=anime)", "moins de deux appels"),
        (
            "lance l'opening de {anime}",
            "browser_open(url=anime)\n    tool.inconnu()",
            "fonctions inconnues : tool.inconnu",
        ),
    ],
)
def test_a_shared_skill_is_checked_like_a_learned_one(
    tmp_path: Path, patterns: str, code: str, reason: str
) -> None:
    store = SkillStore(tmp_path / "skills")
    folder = _package(tmp_path, patterns=patterns, code=code)
    with pytest.raises(ValueError, match=reason):
        store.import_package(folder, {"browser_open", "browser_read"})
    assert store.all() == []


def test_an_unreadable_package_is_refused_cleanly(tmp_path: Path) -> None:
    folder = _package(tmp_path, patterns='a" b: "c', code="x(")
    with pytest.raises(ValueError, match="illisible"):
        SkillStore(tmp_path / "skills").import_package(folder)


def test_archive_with_an_escaping_path_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "piege.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../dehors/SKILL.md", "---\nname: dehors\n---\n")
    with pytest.raises(ValueError, match="chemin refusé"):
        SkillStore(tmp_path / "skills").import_package(archive)
    assert not (tmp_path / "dehors").exists()


def test_check_shape() -> None:
    assert check_shape(["a"], ["lance {a} vite"]) == ""
    assert "paramètres" in check_shape(["a"], ["lance le truc"])
    assert check_shape([], []) == "aucun motif"


def test_share_tools_through_the_registry(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    store.adopt(_skill())
    registry = ToolRegistry(make_ctx(tmp_path, skills=store))

    async def scenario() -> list[str]:
        out = []
        exported = await registry.execute(
            ToolCallReq("a", "export_skill", {"name": "lancer opening"})
        )
        out.append(exported.content)
        path = tmp_path / "home" / "Documents" / "MIKATA" / "compétences" / "lancer-opening.zip"
        imported = await registry.execute(ToolCallReq("b", "import_skill", {"path": str(path)}))
        out.append(imported.content)
        approved = await registry.execute(
            ToolCallReq("c", "approve_skill", {"name": "lancer-opening-2"})
        )
        out.append(approved.content)
        return out

    exported, imported, approved = asyncio.run(scenario())
    assert exported == "Compétence exportée dans compétences : lancer-opening.zip."
    assert imported.startswith("Compétence « lancer-opening-2 » importée")
    assert approved == "Compétence « lancer-opening-2 » activée, à l'essai."
    assert store.get("lancer_opening_2").status == "trial"  # type: ignore[union-attr]
