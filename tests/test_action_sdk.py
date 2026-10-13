"""Phase 14b (docs/AGENT_CORE.md §3.3) : chaque outil déclare son effet, la confiance de sa
sortie et son nom dans le SDK ; le catalogue en tire une référence compacte et lie les
arguments d'un appel de script ; les outils de lecture publient un résultat structuré."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from tool_fakes import FakeBrowser, FakeDesktop, make_ctx

from mikata.action.effects import Effect, Trust, join
from mikata.action.sdk import SdkBindError, SdkCatalog
from mikata.llm.base import ToolCallReq
from mikata.tools.base import BaseTool
from mikata.tools.browser import parse_page
from mikata.tools.registry import (
    ToolRegistry,
    function_schema,
    load_builtin_tools,
    register_tool,
    registered_tools,
)


def _builtin() -> dict[str, type[Any]]:
    load_builtin_tools()
    return {
        name: cls
        for name, cls in registered_tools().items()
        if cls.__module__.startswith("mikata.tools.")
    }


def test_every_builtin_tool_declares_its_place_in_the_action_space() -> None:
    tools = _builtin()
    assert len(tools) >= 56
    for name, cls in tools.items():
        assert "sdk" in vars(cls) and "effect" in vars(cls), f"{name} : sdk et effect obligatoires"
        assert isinstance(cls.effect, Effect) and isinstance(cls.output, Trust), name
        assert cls.effect is not Effect.DESTRUCTIVE, name  # aucun outil destructif n'existe


def test_effects_match_what_the_tools_do() -> None:
    tools = _builtin()
    effect = {name: cls.effect for name, cls in tools.items()}
    assert effect["web_search"] is Effect.READ and effect["read_file"] is Effect.READ
    assert effect["set_volume"] is Effect.ACT and effect["open_app"] is Effect.ACT
    for name in ("run_powershell", "run_shell", "write_file", "move_file", "delete_file",
                 "type_text", "press_keys", "shutdown", "remember"):  # fmt: skip
        assert effect[name] is Effect.CONSEQUENTIAL, name
    # Ce qui rapporte un contenu écrit par quelqu'un d'autre est marqué non fiable.
    for name in ("web_search", "fetch_page", "browser_read", "read_file", "clipboard_read",
                 "uia_read", "uia_window", "run_powershell"):  # fmt: skip
        assert tools[name].output is Trust.UNTRUSTED, name
    assert tools["set_volume"].output is Trust.LOCAL
    # Un outil lent le dit : l'accusé de réception s'appuie dessus.
    assert tools["browser_open"].expected_s >= 2 and tools["set_volume"].expected_s < 1


def test_trust_join_keeps_the_least_trusted() -> None:
    assert join() is Trust.TRUSTED
    assert join(Trust.TRUSTED, Trust.LOCAL) is Trust.LOCAL
    assert join(Trust.LOCAL, Trust.UNTRUSTED, Trust.TRUSTED) is Trust.UNTRUSTED
    assert Trust.UNTRUSTED.label == "non fiable"


def test_an_undeclared_tool_is_treated_as_consequential() -> None:
    class _P(BaseModel):
        x: int = 0

    class _Quiet(BaseTool[_P]):
        name = "quiet_test_tool"
        description = "Outil de test sans déclaration."
        params = _P

        async def run(self, args: _P) -> str:
            return "ok"

    assert _Quiet.effect is Effect.CONSEQUENTIAL and _Quiet.sdk == ""
    catalog = SdkCatalog({"quiet_test_tool": _Quiet}, "win32")
    assert next(iter(catalog.entries)) == "tool.quiet_test_tool"

    class _Bad(_Quiet):
        name = "bad_sdk_tool"
        sdk = "PasUnNom"

    with pytest.raises(TypeError, match="espace"):
        register_tool(_Bad)


def test_catalog_is_unique_per_platform_and_smaller_than_the_schemas() -> None:
    tools = _builtin()
    windows = SdkCatalog(tools, "win32")
    mac = SdkCatalog(tools, "darwin")
    assert windows.get("shell.run").tool == "run_powershell"
    assert mac.get("shell.run").tool == "run_shell"
    assert len(windows.entries) == len(tools) - 1  # l'autre shell n'est pas proposé
    assert windows.by_tool("web_search").name == "web.search"
    reference = windows.reference()
    assert "web.search(query: str) -> {answer, results: [{title, snippet, url}]}" in reference
    assert "audio.volume(level: int = None, change: int = None) -> str" in reference
    assert "[conséquent]" in reference and "[contenu non fiable]" in reference
    schemas = json.dumps([function_schema(cls) for cls in tools.values()], ensure_ascii=False)
    assert len(reference) < len(schemas) / 2  # la référence coûte moins de la moitié


def test_bind_maps_script_arguments_to_tool_fields() -> None:
    catalog = SdkCatalog(_builtin(), "win32")
    write = catalog.get("files.write")
    assert catalog.bind(write, ("a.txt", "bonjour"), {"mode": "append"}) == {
        "path": "a.txt",
        "content": "bonjour",
        "mode": "append",
    }
    # Les champs obligatoires passent d'abord, même déclarés après un champ facultatif.
    click = catalog.get("ui.click")
    assert catalog.bind(click, ("Lire",), {"window": "Spotify"}) == {
        "control": "Lire",
        "window": "Spotify",
    }
    with pytest.raises(SdkBindError, match="argument manquant « content »"):
        catalog.bind(write, ("a.txt",), {})
    with pytest.raises(SdkBindError, match="argument inconnu « chemin »"):
        catalog.bind(write, (), {"chemin": "a.txt", "content": "x"})
    with pytest.raises(SdkBindError, match="donné deux fois"):
        catalog.bind(write, ("a.txt", "x"), {"path": "b.txt"})
    with pytest.raises(SdkBindError, match="1 au plus"):
        catalog.bind(catalog.get("apps.open"), ("a", "b"), {})


def test_parse_page_splits_what_the_browser_read() -> None:
    page = parse_page(
        "Titre : Mairie\nURL : https://mairie.fr\nTexte :\nOuvert de 8 h à 17 h.\nFermé le "
        "samedi.\nLiens : Accueil | Contact\nBoutons : Rechercher"
    )
    assert page == {
        "title": "Mairie",
        "url": "https://mairie.fr",
        "text": "Ouvert de 8 h à 17 h.\nFermé le samedi.",
        "links": ["Accueil", "Contact"],
        "buttons": ["Rechercher"],
    }
    assert parse_page("")["links"] == []


def test_read_tools_publish_structured_data(tmp_path: Path) -> None:
    browser = FakeBrowser(page="Titre : T\nURL : https://x\nTexte :\nBonjour.\nLiens : A | B")

    async def scenario() -> dict[str, Any]:
        registry = ToolRegistry(make_ctx(tmp_path, desktop=FakeDesktop(), browser=browser))
        home = tmp_path / "home"
        (home / "note.txt").write_text("contenu", encoding="utf-8")
        try:
            return {
                "page": await registry.execute(ToolCallReq("1", "browser_read", {})),
                "windows": await registry.execute(ToolCallReq("2", "list_windows", {})),
                "file": await registry.execute(
                    ToolCallReq("3", "read_file", {"path": str(home / "note.txt")})
                ),
                "volume": await registry.execute(ToolCallReq("4", "list_skills", {})),
            }
        finally:
            await registry.aclose()

    results = asyncio.run(scenario())
    assert results["page"].data["links"] == ["A", "B"] and results["page"].data["title"] == "T"
    assert results["windows"].data["windows"][0] == {"app": "Spotify", "title": "Spotify Premium"}
    assert results["file"].data["text"] == "contenu"
    assert results["volume"].data is None  # pas de résultat structuré : le script reçoit le texte


class _Fields(BaseModel):
    """Champs variés, pour le rendu des signatures."""

    name: str = Field(min_length=1)
    tags: list[str] | None = None
    mode: str = "a"


def test_signature_rendering_handles_optional_and_list_fields() -> None:
    class _Tool(BaseTool[_Fields]):
        name = "sig_test_tool"
        description = "Outil de test. Deuxième phrase ignorée."
        params = _Fields
        sdk = "demo.run"
        effect = Effect.READ

        async def run(self, args: _Fields) -> str:
            return "ok"

    entry = SdkCatalog({"sig_test_tool": _Tool}, "linux").get("demo.run")
    expected = "demo.run(name: str, tags: list[str] = None, mode: str = 'a') -> str"
    assert entry.signature() == expected
    assert entry.doc == "Outil de test"
