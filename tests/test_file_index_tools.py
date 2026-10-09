"""Outils d'index des fichiers (phase 12d) : find_path, list_folder, open_in_explorer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeDesktop, FakeIndex, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolContext
from mikata.tools.registry import ToolRegistry, ToolResult

CYBERPUNK = "D:\\Steam\\steamapps\\common\\Cyberpunk 2077.exe"


def _execute(ctx: ToolContext, tool: str, **arguments: Any) -> ToolResult:
    async def scenario() -> ToolResult:
        registry = ToolRegistry(ctx)
        try:
            return await registry.execute(ToolCallReq("1", tool, arguments))
        finally:
            await registry.aclose()

    return asyncio.run(scenario())


def test_find_path_returns_indexed_results(tmp_path: Path) -> None:
    index = FakeIndex()
    result = _execute(make_ctx(tmp_path, index=index), "find_path", name="cyberpunk")
    assert result.ok and CYBERPUNK in result.content
    assert index.queries[0][0] == "cyberpunk"  # la requête pliée est passée à l'index
    assert index.queries[0][1] == 40  # max_results * 2 avant le filtre


def test_find_path_publishes_structured_data(tmp_path: Path) -> None:
    """Phase 12e : le payload structuré alimente le rejeu local des savoir-faire."""
    result = _execute(make_ctx(tmp_path, index=FakeIndex()), "find_path", name="cyberpunk")
    assert result.ok and result.data is not None
    assert result.data["best"] == CYBERPUNK and result.data["count"] == 1
    assert result.data["paths"] == [CYBERPUNK]


def test_find_path_no_match_publishes_empty_data(tmp_path: Path) -> None:
    result = _execute(
        make_ctx(tmp_path, index=FakeIndex(results=[])), "find_path", name="cyberpunk"
    )
    assert result.ok and result.data == {"paths": [], "best": None, "count": 0}


def test_find_path_filters_by_folder(tmp_path: Path) -> None:
    index = FakeIndex(
        results=[
            {"path": CYBERPUNK, "name": "Cyberpunk 2077.exe", "is_dir": False, "mtime": 0.0},
            {
                "path": "C:\\Users\\x\\Documents\\cyberpunk notes.txt",
                "name": "cyberpunk notes.txt",
                "is_dir": False,
                "mtime": 0.0,
            },
        ]
    )
    result = _execute(
        make_ctx(tmp_path, index=index), "find_path", name="cyberpunk", folder="D:\\Steam"
    )
    assert result.ok and CYBERPUNK in result.content and "notes.txt" not in result.content


def test_find_path_when_index_building_says_so(tmp_path: Path) -> None:
    index = FakeIndex()
    index.ready_state = False
    result = _execute(make_ctx(tmp_path, index=index), "find_path", name="cyberpunk")
    assert not result.ok and "se construit" in result.content


def test_find_path_requires_three_characters(tmp_path: Path) -> None:
    result = _execute(make_ctx(tmp_path, index=FakeIndex()), "find_path", name="ab")
    assert not result.ok and "arguments invalides" in result.content


def test_find_path_with_no_match(tmp_path: Path) -> None:
    index = FakeIndex(results=[])
    result = _execute(make_ctx(tmp_path, index=index), "find_path", name="cyberpunk")
    assert result.ok and "Aucun fichier ne correspond" in result.content


def test_list_folder_lists_from_the_index(tmp_path: Path) -> None:
    index = FakeIndex(
        folder=[
            {"name": "Sous", "is_dir": True, "mtime": 0.0},
            {"name": "alpha.txt", "is_dir": False, "mtime": 0.0},
        ]
    )
    result = _execute(make_ctx(tmp_path, index=index), "list_folder", path="D:\\Docs")
    assert result.ok and "- Sous (dossier)" in result.content and "- alpha.txt" in result.content


def test_open_in_explorer_reveals_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desktop = FakeDesktop()
    monkeypatch.setattr("mikata.tools.files.os.path.isdir", lambda path: False)
    result = _execute(
        make_ctx(tmp_path, index=FakeIndex(), desktop=desktop),
        "open_in_explorer",
        path=CYBERPUNK,
    )
    assert result.ok and "Explorateur ouvert" in result.content
    assert ("reveal", (CYBERPUNK, True)) in desktop.log  # /select sur un fichier
    assert result.data == {"path": CYBERPUNK, "is_dir": False}  # phase 12e


def test_open_in_explorer_opens_a_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desktop = FakeDesktop()
    monkeypatch.setattr("mikata.tools.files.os.path.isdir", lambda path: True)
    result = _execute(
        make_ctx(tmp_path, index=FakeIndex(), desktop=desktop),
        "open_in_explorer",
        path="D:\\Steam",
    )
    assert result.ok
    assert ("reveal", ("D:\\Steam", False)) in desktop.log  # dossier : on ouvre dedans


@pytest.mark.parametrize("tool", ["find_path", "list_folder"])
def test_index_tools_without_the_service_explain_themselves(tmp_path: Path, tool: str) -> None:
    arguments = {"name": "cyberpunk"} if tool == "find_path" else {"path": "D:\\Steam"}
    result = _execute(make_ctx(tmp_path), tool, **arguments)
    assert not result.ok and "l'index des fichiers" in result.content


def test_open_in_explorer_does_not_need_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ouvrir l'Explorateur marche même sans index : le chemin vient du LLM."""
    monkeypatch.setattr("mikata.tools.files.os.path.isdir", lambda path: True)
    desktop = FakeDesktop()
    result = _execute(make_ctx(tmp_path, desktop=desktop), "open_in_explorer", path="D:\\Steam")
    assert result.ok
    assert ("reveal", ("D:\\Steam", False)) in desktop.log


def test_open_in_explorer_is_logged_ok(
    tmp_path: Path, caplog_loguru, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mikata.tools.files.os.path.isdir", lambda path: True)
    _execute(
        make_ctx(tmp_path, index=FakeIndex(), desktop=FakeDesktop()),
        "open_in_explorer",
        path="D:\\Steam",
    )
    assert any("OK | open_in_explorer(" in line for line in caplog_loguru)
