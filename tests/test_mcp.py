"""Phase 16f : outils externes par le Model Context Protocol (vrai sous-processus)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import make_ctx

from mikata.action.effects import Effect, Trust
from mikata.action.runtime import ScriptRuntime
from mikata.action.sdk import SdkCatalog
from mikata.config import McpConfig, McpServerConfig
from mikata.llm.base import ToolCallReq
from mikata.mcp.bridge import McpBridge, params_model, safe_name
from mikata.tools.registry import ToolRegistry

SERVER = Path(__file__).with_name("mcp_fake_server.py")


def _config(**changes: Any) -> McpConfig:
    server = McpServerConfig(
        name="Notes Test",
        command=sys.executable,
        args=[str(SERVER)],
        env={"CLE_TEST": "${CLE_TEST}"},
    )
    return McpConfig(enabled=True, servers=[server], **changes)


def test_tools_of_a_server_become_mikata_tools(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("CLE_TEST=valeur-du-env\n", encoding="utf-8")
    registry = ToolRegistry(make_ctx(tmp_path))

    async def scenario() -> tuple[list[str], Any, Any, Any, Any]:
        bridge = McpBridge(_config(), tmp_path / ".env")
        try:
            added = await bridge.start(registry)
            hour = await registry.execute(ToolCallReq("a", "mcp_notes_test_heure", {}))
            note = await registry.execute(
                ToolCallReq("b", "mcp_notes_test_ajouter_note", {"texte": "lait", "urgent": True})
            )
            refused = await registry.execute(
                ToolCallReq("c", "mcp_notes_test_ajouter_note", {"texte": "échec"})
            )
            invalid = await registry.execute(ToolCallReq("d", "mcp_notes_test_ajouter_note", {}))
            return added, hour, note, refused, invalid
        finally:
            await bridge.close()

    added, hour, note, refused, invalid = asyncio.run(scenario())
    # Deux pages de tools/list, une ligne parasite et une demande d'échantillonnage ignorées.
    assert added == ["mcp_notes_test_heure", "mcp_notes_test_ajouter_note"]
    assert hour.ok and hour.content == "Il est 12 h (secret : valeur-du-env)."  # « ${CLE_TEST} »
    assert note.ok and note.content == "Note ajoutée : lait (urgent=True)."
    assert not refused.ok and "Note refusée." in refused.content
    assert not invalid.ok  # paramètre obligatoire manquant : refusé avant le serveur


def test_effects_and_trust_follow_the_server_hints(tmp_path: Path) -> None:
    registry = ToolRegistry(make_ctx(tmp_path))

    async def scenario() -> SdkCatalog:
        bridge = McpBridge(_config())
        try:
            await bridge.start(registry)
            return SdkCatalog(registry.tools, "win32")
        finally:
            await bridge.close()

    catalog = asyncio.run(scenario())
    read = catalog.get("mcp.notes_test_heure")
    write = catalog.get("mcp.notes_test_ajouter_note")
    assert read is not None and read.effect is Effect.READ
    assert write is not None and write.effect is Effect.CONSEQUENTIAL  # « oui » après lecture
    assert read.output is Trust.UNTRUSTED and write.output is Trust.UNTRUSTED


def test_a_script_can_call_an_mcp_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(make_ctx(tmp_path))

    async def scenario() -> Any:
        bridge = McpBridge(_config())
        try:
            await bridge.start(registry)
            runtime = ScriptRuntime(registry, SdkCatalog(registry.tools, "win32"))
            return await runtime.run("h = mcp.notes_test_heure()\nreturn h")
        finally:
            await bridge.close()

    outcome = asyncio.run(scenario())
    assert outcome.ok and outcome.value.startswith("Il est 12 h")
    assert outcome.trust is Trust.UNTRUSTED  # ce qu'un serveur externe rend


def test_agent_sees_mcp_tools_after_refresh(tmp_path: Path) -> None:
    """Les serveurs démarrent après l'Agent : le modèle reçoit leurs outils et les
    programmes peuvent les appeler une fois le catalogue reconstruit."""
    from test_agent import NOW, Phrases, ScriptedLLM

    from mikata.agent.loop import Agent

    ctx = make_ctx(tmp_path)
    agent = Agent(
        ScriptedLLM(),
        ctx.config,
        phrases=Phrases(),
        window_title=lambda: None,
        clock=lambda: NOW,
        tools=ToolRegistry(ctx),
        long_memory=ctx.memory,
    )
    assert agent.tools is not None and agent._actions is not None

    async def scenario() -> set[str]:
        bridge = McpBridge(_config())
        try:
            await bridge.start(agent.tools)  # type: ignore[arg-type]
            assert "mcp.notes_test_heure" not in agent._actions.action_names()  # type: ignore[union-attr]
            agent.refresh_tools()
            return agent._actions.action_names()  # type: ignore[union-attr]
        finally:
            await bridge.close()
            await agent.aclose()

    names = asyncio.run(scenario())
    assert {"mcp.notes_test_heure", "mcp_notes_test_ajouter_note"} <= names
    schemas = {s["function"]["name"] for s in agent.tools.schemas()}
    assert "mcp_notes_test_ajouter_note" in schemas


def test_a_missing_server_is_reported_not_fatal(tmp_path: Path) -> None:
    registry = ToolRegistry(make_ctx(tmp_path))
    broken = McpServerConfig(name="absent", command="programme-qui-n-existe-pas-mikata")
    config = McpConfig(enabled=True, servers=[broken, *_config().servers])

    async def scenario() -> list[str]:
        bridge = McpBridge(config)
        try:
            return await bridge.start(registry)
        finally:
            await bridge.close()

    assert asyncio.run(scenario()) == ["mcp_notes_test_heure", "mcp_notes_test_ajouter_note"]


def test_schema_to_params_and_names() -> None:
    model = params_model(
        "x",
        {
            "properties": {
                "texte": {"type": "string"},
                "n": {"type": "integer"},
                "drôle de nom": {"type": "string"},
            },
            "required": ["texte"],
        },
    )
    assert set(model.model_fields) == {"texte", "n"}  # « drôle de nom » : refusé par les API
    assert model(texte="a").model_dump(exclude_none=True) == {"texte": "a"}
    # Noms dangereux pour Pydantic ou Python : alias, le vrai nom reste visible et envoyé.
    risky = params_model(
        "y",
        {
            "properties": {"from": {"type": "string"}, "model_config": {"type": "string"}},
            "required": ["from"],
        },
    )
    assert set(risky.model_fields) == {"arg_from", "arg_model_config"}
    assert set(risky.model_json_schema()["properties"]) == {"from", "model_config"}
    args = risky.model_validate({"from": "a", "model_config": "b"})
    assert args.model_dump(by_alias=True, exclude_none=True) == {"from": "a", "model_config": "b"}
    assert safe_name("Mon Serveur-Notes !") == "mon_serveur_notes"
    assert len(f"mcp_{safe_name('s' * 50, 20)}_{safe_name('t' * 90)}") <= 64


def test_a_tool_with_an_unusable_schema_is_skipped_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mikata.mcp import bridge as bridge_module
    from mikata.mcp.client import McpServer, McpTool

    original, build = McpServer.list_tools, bridge_module.params_model

    async def with_a_broken_tool(self: McpServer) -> list[McpTool]:
        return [McpTool("casse", "", {}), *await original(self)]

    def refuse(name: str, schema: dict[str, Any]) -> Any:
        if name == "casse":
            raise ValueError("schéma inutilisable")
        return build(name, schema)

    monkeypatch.setattr(McpServer, "list_tools", with_a_broken_tool)
    monkeypatch.setattr(bridge_module, "params_model", refuse)
    registry = ToolRegistry(make_ctx(tmp_path))

    async def scenario() -> list[str]:
        bridge = McpBridge(_config())
        try:
            return await bridge.start(registry)
        finally:
            await bridge.close()

    assert asyncio.run(scenario()) == ["mcp_notes_test_heure", "mcp_notes_test_ajouter_note"]
