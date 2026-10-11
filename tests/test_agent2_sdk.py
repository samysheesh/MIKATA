"""Agent v2 (phase 13a) : le SDK appelle les outils existants, compte et enregistre."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from agent2_fakes import make_registry
from tool_fakes import FakeDesktop

from mikata.agent2.sdk import ActionLimitReached, Sdk, SdkError, sdk_reference
from mikata.tools.registry import ToolRegistry


def _sdk(
    tmp_path: Path, max_actions: int = 25, **ctx_overrides: object
) -> tuple[Sdk, ToolRegistry]:
    registry = make_registry(tmp_path, **ctx_overrides)
    sdk = Sdk(registry, max_actions=max_actions, observe_lines=40)
    return sdk, registry


def test_open_app_calls_the_tool_and_records(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    sdk, _ = _sdk(tmp_path, desktop=desktop)

    async def scenario() -> None:
        result = await sdk.open_app("Spotify")
        assert "ouvert" in result
        assert ("launch", "Spotify") in desktop.log  # l'outil passe par desktop.launch
        assert len(sdk._recorder.actions) == 1
        action = sdk._recorder.actions[0]
        assert action.tool == "open_app" and action.ok and action.args == {"name": "Spotify"}

    asyncio.run(scenario())


def test_failing_tool_raises_sdkerror(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    desktop.fail.add("launch")  # open_app échoue au lancement de l'appli
    sdk, _ = _sdk(tmp_path, desktop=desktop)

    async def scenario() -> None:
        with pytest.raises(SdkError):
            await sdk.open_app("Introuvable")

    asyncio.run(scenario())


def test_unknown_tool_raises_sdkerror(tmp_path: Path) -> None:
    sdk, _ = _sdk(tmp_path)

    async def scenario() -> None:
        with pytest.raises(SdkError, match="outil inconnu"):
            await sdk.tool("n_existe_pas", x=1)

    asyncio.run(scenario())


def test_action_limit_is_enforced(tmp_path: Path) -> None:
    sdk, _ = _sdk(tmp_path, max_actions=2)

    async def scenario() -> None:
        await sdk.tool("get_time")
        await sdk.tool("get_time")
        with pytest.raises(ActionLimitReached):
            await sdk.tool("get_time")

    asyncio.run(scenario())


def test_observe_is_free_and_never_raises(tmp_path: Path) -> None:
    sdk, _ = _sdk(tmp_path)

    async def scenario() -> None:
        observation = await sdk.observe()
        assert not sdk._recorder.actions  # lecture seule : aucune action comptée
        assert observation.active_app == ""  # services absents -> champs vides

    asyncio.run(scenario())


def test_observe_reads_the_active_window(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    sdk, _ = _sdk(tmp_path, desktop=desktop)

    async def scenario() -> None:
        observation = await sdk.observe()
        # FakeDesktop.windows renvoie « Spotify : Spotify Premium » en premier.
        assert "spotify" in observation.active_app.casefold()
        assert "spotify premium" in observation.title.casefold()

    asyncio.run(scenario())


def test_sdk_reference_lists_every_public_method() -> None:
    reference = sdk_reference()
    import inspect

    for name, _member in inspect.getmembers(Sdk, predicate=inspect.iscoroutinefunction):
        if name.startswith("_") or name == "tool":
            continue
        assert f"{name}(" in reference, f"sdk_reference() ne mentionne pas {name}"
    assert len(reference.splitlines()) < 60
