from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from tool_fakes import FakeDesktop, make_config, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools import registry as registry_module
from mikata.tools.base import BaseTool, NoParams, Tool, ToolError
from mikata.tools.registry import (
    FAMILIES,
    ToolRegistry,
    function_schema,
    register_tool,
    registered_tools,
)

PHASE_4_TOOLS = {
    # system
    "get_time",
    "system_status",
    "set_timer",
    "set_reminder",
    "lock",
    # apps
    "open_app",
    "close_app",
    "focus_window",
    "list_windows",
    # web (get_weather : ajout phase 4)
    "open_url",
    "web_search",
    "fetch_page",
    "get_weather",
    # media
    "media_control",
    "set_volume",
    "mute",
    "set_output_device",
    # ui
    "show_text",
    "clipboard_read",
    "clipboard_write",
    # memory
    "remember",
    "forget",
}
PHASE_5_TOOLS = {
    "close_window",  # une seule fenêtre (la fenêtre active par défaut)
    # system : reportés de la phase 4 (délai d'annulation)
    "shutdown",
    "restart",
    "sleep",
    # input
    "type_text",
    "press_keys",
    "mouse",
    # files
    "find_files",
    "read_file",
    "write_file",
    "move_file",
    "delete_file",
    # shell
    "run_powershell",
    "run_script",
}
PHASE_6_TOOLS = {"set_history_duration"}  # durée de l'historique, à la voix
PHASE_12A_TOOLS = {  # pilotage fin des applications (arbre d'accessibilité Windows)
    "uia_window",
    "uia_read",
    "uia_click",
    "uia_set_text",
    "uia_state",
}
TERMINAL = {
    "set_timer",
    "set_reminder",
    "lock",
    "open_app",
    "close_app",
    "focus_window",
    "open_url",
    "media_control",
    "set_volume",
    "mute",
    "set_output_device",
    "show_text",
    "clipboard_write",
    "remember",
    "forget",
    "shutdown",
    "restart",
    "sleep",
    "type_text",
    "press_keys",
    "mouse",
    "close_window",
    "uia_click",
    "uia_set_text",
}


def _registry(tmp_path: Path, **kwargs: object) -> ToolRegistry:
    return ToolRegistry(make_ctx(tmp_path, **kwargs))  # type: ignore[arg-type]


def test_catalogue_and_terminal_tools(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    every = PHASE_4_TOOLS | PHASE_5_TOOLS | PHASE_6_TOOLS | PHASE_12A_TOOLS
    assert set(registry.tools) - {"echo_test"} == every
    assert {n for n in registry.tools if registry.is_terminal(n)} - {"echo_test"} == TERMINAL
    assert FAMILIES == (
        "system", "apps", "web", "media", "ui", "memory_tools", "input", "files", "shell", "uia"
    )  # fmt: skip
    for tool in registry.tools.values():
        assert isinstance(tool, Tool)  # respecte l'interface de §9


def test_schemas_are_stable_sorted_and_lean(tmp_path: Path) -> None:
    every = PHASE_4_TOOLS | PHASE_5_TOOLS | PHASE_6_TOOLS | PHASE_12A_TOOLS
    registry = ToolRegistry(make_ctx(tmp_path), include=every)
    schemas = registry.schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == sorted(names)
    # identiques d'un appel à l'autre (cache de prompt)
    assert schemas == ToolRegistry(make_ctx(tmp_path), include=every).schemas()
    for schema in schemas:
        assert schema["type"] == "function"
        params = schema["function"]["parameters"]
        assert params["type"] == "object" and "properties" in params
        assert "description" not in params
        assert set(params.get("required", [])) <= set(params["properties"])
        assert all("title" not in prop for prop in params["properties"].values())
        assert schema["function"]["description"].endswith(".")
    show = next(s for s in schemas if s["function"]["name"] == "show_text")
    assert set(show["function"]["parameters"]["properties"]) == {"title", "text"}
    volume = next(s for s in schemas if s["function"]["name"] == "set_volume")
    assert volume["function"]["parameters"]["properties"]["level"]["anyOf"][0]["maximum"] == 100


def test_register_tool_checks_shape() -> None:
    class Missing:
        pass

    with pytest.raises(TypeError, match="name"):
        register_tool(Missing)

    class NotAsync(BaseTool[NoParams]):
        name = "pas_async"
        description = "Test."
        params = NoParams

        def run(self, args: NoParams) -> str:  # type: ignore[override]
            return ""

    with pytest.raises(TypeError, match="coroutine"):
        register_tool(NotAsync)

    class Duplicate(BaseTool[NoParams]):
        name = "get_time"
        description = "Doublon."
        params = NoParams

        async def run(self, args: NoParams) -> str:
            return ""

    from mikata.tools.registry import load_builtin_tools

    load_builtin_tools()
    with pytest.raises(ValueError, match="deux fois"):
        register_tool(Duplicate)
    assert "pas_async" not in registered_tools()


class EchoParams(BaseModel):
    text: str = Field(min_length=1)
    count: int = Field(default=1, ge=1, le=3)


class _Echo(BaseTool[EchoParams]):
    name = "echo_test"
    description = "Répète le texte."
    params = EchoParams
    terminal = True

    async def run(self, args: EchoParams) -> str:
        if args.text == "erreur":
            raise ToolError("ça n'a pas marché")
        if args.text == "bug":
            raise RuntimeError("boum")
        if args.text == "lent":
            await asyncio.sleep(5)
        return args.text * args.count


@pytest.fixture(autouse=True)
def _echo_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """L'outil de test n'existe que le temps d'un test de ce module."""
    registry_module.load_builtin_tools()
    monkeypatch.setattr(registry_module, "_REGISTRY", dict(registry_module._REGISTRY))
    register_tool(_Echo)


def _execute(tmp_path: Path, name: str, arguments: dict[str, object], **cfg: object) -> object:
    config = make_config(tmp_path, tools={"timeout_s": 0.05, "result_max_chars": 200, **cfg})
    registry = ToolRegistry(make_ctx(tmp_path, config=config), include={"echo_test"})
    return asyncio.run(registry.execute(ToolCallReq("c1", name, arguments)))


def test_execute_success_and_terminal(tmp_path: Path) -> None:
    result = _execute(tmp_path, "echo_test", {"text": "ok", "count": 2})
    assert (result.ok, result.content, result.terminal, result.call_id) == (  # type: ignore[attr-defined]
        True,
        "okok",
        True,
        "c1",
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, "arguments invalides pour echo_test (text : Field required)"),
        ({"text": "x", "count": 9}, "count : Input should be less than or equal to 3"),
        ({"text": "erreur"}, "ça n'a pas marché"),
        ({"text": "bug"}, "erreur inattendue dans echo_test : boum"),
        ({"text": "lent"}, "echo_test n'a pas répondu en 0 s"),
    ],
)
def test_execute_failures_become_messages(
    tmp_path: Path, arguments: dict[str, object], expected: str
) -> None:
    result = _execute(tmp_path, "echo_test", arguments)
    assert result.ok is False  # type: ignore[attr-defined]
    assert expected in result.content  # type: ignore[attr-defined]


def test_unknown_tool_and_truncation(tmp_path: Path) -> None:
    result = _execute(tmp_path, "inconnu", {})
    assert not result.ok and result.content == "outil inconnu : inconnu"  # type: ignore[attr-defined]
    assert not result.terminal  # type: ignore[attr-defined]
    long = _execute(tmp_path, "echo_test", {"text": "a" * 150, "count": 3})
    assert long.ok and len(long.content) == 200 + len(" […tronqué]")  # type: ignore[attr-defined]


def test_function_schema_of_tool_without_params(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    schema = function_schema(registry.tools["get_time"])
    assert schema["function"]["parameters"] == {"properties": {}, "type": "object"}


def test_warm_builds_app_index_once_and_never_raises(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    registry = _registry(tmp_path, desktop=desktop)
    asyncio.run(registry.warm())
    assert desktop.log == [("list_apps", False)]

    class Broken(FakeDesktop):
        def list_apps(self, refresh: bool = False) -> list:  # type: ignore[type-arg]
            raise OSError("pas de menu Démarrer")

    asyncio.run(_registry(tmp_path, desktop=Broken()).warm())  # pas d'exception
    asyncio.run(_registry(tmp_path).warm())  # sans bureau : rien à faire


def test_aclose_cancels_timers_and_closes_http(tmp_path: Path) -> None:
    class Http:
        closed = False

        async def aclose(self) -> None:
            Http.closed = True

    async def scenario() -> bool:
        ctx = make_ctx(tmp_path, http=Http())
        registry = ToolRegistry(ctx)
        ctx.spawn(asyncio.sleep(60), "minuteur")
        task = next(iter(ctx.tasks))
        await registry.ctx.run_blocking(lambda: None)
        await registry.aclose()
        return task.cancelled()

    assert asyncio.run(scenario()) and Http.closed


def test_missing_services_explain_themselves(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    result = asyncio.run(registry.execute(ToolCallReq("1", "open_app", {"name": "Spotify"})))
    assert not result.ok and "bureau Windows" in result.content
    result = asyncio.run(registry.execute(ToolCallReq("2", "set_volume", {"level": 30})))
    assert not result.ok and "volume" in result.content
