"""Outils des savoir-faire (phase 12c) : list_skills, remember_skill, forget_skill."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import make_ctx

from mikata.llm.base import ToolCallReq
from mikata.skills.library import SkillLibrary
from mikata.tools.base import ToolContext
from mikata.tools.registry import ToolRegistry, ToolResult, load_builtin_tools


def _library(tmp_path: Path) -> SkillLibrary:
    load_builtin_tools()
    library = SkillLibrary(tmp_path / "skills.json", max_skills=30, max_failures=3)
    library.learn(
        "ouvrir_exemple",
        "ouvrir le site exemple",
        "ouvre le site exemple",
        [{"tool": "open_url", "args": {"url": "https://example.com"}}],
    )
    return library


def _execute(ctx: ToolContext, tool: str, **arguments: Any) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(ctx)
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


def test_list_skills_lists_them(tmp_path: Path) -> None:
    result = _execute(make_ctx(tmp_path, skills=_library(tmp_path)), "list_skills")
    assert result.ok and "ouvrir le site exemple" in result.content


def test_list_skills_with_none_says_so(tmp_path: Path) -> None:
    load_builtin_tools()
    empty = SkillLibrary(tmp_path / "skills.json", max_skills=30, max_failures=3)
    result = _execute(make_ctx(tmp_path, skills=empty), "list_skills")
    assert result.ok and "Aucun savoir-faire" in result.content


def test_forget_skill_removes_the_closest(tmp_path: Path) -> None:
    library = _library(tmp_path)
    result = _execute(
        make_ctx(tmp_path, skills=library), "forget_skill", description="site exemple"
    )
    assert result.ok and result.content == "Oublié : ouvrir le site exemple"
    assert library.all() == []


def test_forget_skill_with_no_match_explains(tmp_path: Path) -> None:
    result = _execute(
        make_ctx(tmp_path, skills=_library(tmp_path)), "forget_skill", description="autre chose"
    )
    assert not result.ok and "aucun savoir-faire" in result.content


def test_remember_skill_uses_the_capture(tmp_path: Path) -> None:
    captured: list[str] = []

    async def capture(description: str) -> str:
        captured.append(description)
        return f"Mémorisé : {description}"

    result = _execute(
        make_ctx(tmp_path, skills=_library(tmp_path), skills_capture=capture),
        "remember_skill",
        description="régler le son du PC",
    )
    assert result.ok and captured == ["régler le son du PC"]


def test_remember_skill_without_capture_explains(tmp_path: Path) -> None:
    result = _execute(
        make_ctx(tmp_path, skills=_library(tmp_path)), "remember_skill", description="régler le son"
    )
    assert not result.ok and "aucune commande récente" in result.content


@pytest.mark.parametrize("tool", ["list_skills", "remember_skill", "forget_skill"])
def test_skills_tools_without_the_library_explain_themselves(tmp_path: Path, tool: str) -> None:
    arguments = {"description": "quelque chose"} if tool != "list_skills" else {}
    result = _execute(make_ctx(tmp_path), tool, **arguments)
    assert not result.ok and "savoir-faire" in result.content
