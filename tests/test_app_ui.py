"""Phase 8 : l'orchestrateur et l'interface (événements publiés, pause, réglages à chaud,
micro occupé, menu de l'icône). L'interface est simulée ; MIKATA ne dépend jamais d'elle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import edit_config
from test_app import RecordingTray
from test_app_conversation import ContentLLM, _app
from test_app_listening import FakeSTT, _run_until

from mikata import app as app_module
from mikata.agent.loop import TurnResult
from mikata.app import App, State, changed_settings, is_live_setting, ui_outcome
from mikata.config import load_config
from mikata.tools.registry import ToolResult
from mikata.ui.bridge import UiBridge
from mikata.ui.protocol import encode


class FakeBridge:
    """Canal simulé : garde ce que MIKATA publie."""

    def __init__(self, connected: bool = True, fail: bool = False) -> None:
        self.messages: list[dict[str, Any]] = []
        self.connected = connected
        self.fail = fail
        self.started = self.closed = False

    async def start(self) -> None:
        if self.fail:
            raise OSError("port refusé")
        self.started = True

    def publish(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    async def close(self) -> None:
        self.closed = True

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind]


def _with_ui(app: App, bridge: Any, launches: list[list[str]] | None = None) -> App:
    app._ui = bridge
    if launches is not None:
        app._ui_launcher = lambda argv: launches.append(list(argv))
    return app


def test_a_command_is_published_to_the_interface(config_file: Path) -> None:
    app, _, _, _ = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, quelle heure est-il ?"),
        llm=ContentLLM(lambda c: "Il est quinze heures."),
    )
    bridge, launches = FakeBridge(), []
    _with_ui(app, bridge, launches)
    _run_until(app, lambda: bool(bridge.of("result")) and app.state is State.IDLE)

    assert bridge.started and bridge.closed
    assert launches == [[]]  # ui.mascot : la mascotte est lancée avec MIKATA
    states = [m["state"] for m in bridge.of("state")]
    assert states == ["idle", "listening", "transcribing", "thinking", "speaking", "idle"]
    assert bridge.of("wake")[0]["score"] == 0.9
    (heard,) = bridge.of("heard")
    assert heard["text"] == "quelle heure est-il ?"
    assert [m["text"] for m in bridge.of("say")] == ["Il est quinze heures."]
    assert bridge.of("result") == [{"type": "result", "id": heard["id"], "outcome": "ok"}]
    kinds = [m["type"] for m in bridge.messages]
    assert kinds.index("wake") < kinds.index("heard") < kinds.index("say") < kinds.index("result")


def test_mikata_runs_when_the_interface_channel_fails(config_file: Path) -> None:
    app, _, _, tts = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, bonjour."),
        llm=ContentLLM(lambda c: "Bonjour."),
    )
    launches: list[list[str]] = []
    _with_ui(app, FakeBridge(fail=True), launches)
    _run_until(app, lambda: bool(tts.texts) and app.state is State.IDLE)
    assert tts.texts == ["Bonjour."] and launches == [] and app._ui is None


def test_ui_outcome() -> None:
    ok = ToolResult("1", "open_app", True, "ok", True, 0.1)
    failed = ToolResult("2", "open_app", False, "introuvable", True, 0.1)
    assert ui_outcome(TurnResult(), cancelled=True) == "cancelled"
    assert ui_outcome(TurnResult(outcome="server"), cancelled=False) == "error"
    assert ui_outcome(TurnResult(outcome="ignored"), cancelled=False) == "ignored"
    assert ui_outcome(TurnResult(tool_results=[ok]), cancelled=False) == "success"
    assert ui_outcome(TurnResult(shortcut=True), cancelled=False) == "success"
    assert ui_outcome(TurnResult(tool_results=[ok, failed]), cancelled=False) == "ok"
    assert ui_outcome(TurnResult(answer="Il est midi."), cancelled=False) == "ok"


def test_pause_ignores_the_wake_word(config_file: Path) -> None:
    app, _, _, _ = _app(
        config_file,
        detections={31: 0.9},
        speech=set(range(25, 38)),
        stt=FakeSTT("Hey Jarvis, bonjour."),
        llm=ContentLLM(lambda c: "Bonjour."),
        frames=80,
    )
    bridge = FakeBridge()
    _with_ui(app, bridge)
    app._on_ui_command({"type": "pause", "paused": True})  # depuis le menu de la mascotte
    audio = app._audio
    _run_until(app, lambda: audio._i >= 80)  # type: ignore[union-attr]
    assert app.paused and app.detections == 0 and not bridge.of("heard")
    assert bridge.of("mic") == [{"type": "mic", "status": "paused"}]
    app.ui_action("pause")  # « Pause / reprise » de l'icône
    assert not app.paused and bridge.of("mic")[-1]["status"] == "ok"


def test_tray_actions_open_or_launch_the_interface(config_file: Path) -> None:
    app = App(load_config(config_file))
    launches: list[list[str]] = []
    bridge = FakeBridge(connected=False)
    _with_ui(app, bridge, launches)
    app.ui_action("settings")
    app.ui_action("mascot")
    assert launches == [["--settings"], []]  # pas d'interface : elle est lancée
    bridge.connected = True
    app.ui_action("settings")
    app.ui_action("mascot")
    app.ui_action("inconnu")
    assert [m["type"] for m in bridge.messages] == ["open_settings", "show_mascot"]
    assert len(launches) == 2


def test_tray_menu_is_offered_only_with_the_interface(config_file: Path) -> None:
    class Tray(RecordingTray):
        def start(self, on_quit: Any, on_action: Any = None) -> None:
            super().start(on_quit)
            self.on_action = on_action

    with_ui, without = Tray(), Tray()
    for tray, bridge in ((with_ui, FakeBridge()), (without, None)):
        app = App(load_config(config_file), tray=tray, ui=bridge)  # type: ignore[arg-type]
        asyncio.run(asyncio.wait_for(app.run(), 5))
    assert callable(with_ui.on_action) and without.on_action is None


def test_reload_config_applies_live_settings(config_file: Path) -> None:
    app, _, _, _ = _app(
        config_file,
        detections={},
        speech=set(),
        stt=FakeSTT(""),
        llm=ContentLLM(lambda c: ""),
    )
    bridge = FakeBridge()
    _with_ui(app, bridge)
    edit_config(
        config_file,
        wake={"threshold": 0.7, "refractory_s": 3.0},
        agent={"follow_up_s": 4, "history_persist_h": 48},
        llm={"reasoning_effort": "low"},
        audio={"input_device": "Autre micro"},
    )
    asyncio.run(app.reload_config())
    (reply,) = bridge.of("config")
    assert reply["error"] is None and reply["restart"] == ["audio.input_device"]
    assert sorted(reply["applied"]) == [
        "agent.follow_up_s",
        "agent.history_persist_h",
        "llm.reasoning_effort",
        "wake.refractory_s",
        "wake.threshold",
    ]
    assert app._trigger.threshold == 0.7 and app._trigger.refractory_s == 3.0
    assert app.config.agent.follow_up_s == 4
    assert app._agent is not None and app._agent.config.llm.reasoning_effort == "low"
    assert app._agent.memory.reset_after_s == 48 * 3600


def test_invalid_config_is_reported_not_applied(config_file: Path) -> None:
    app = App(load_config(config_file))
    bridge = FakeBridge()
    _with_ui(app, bridge)
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace("threshold: 0.5", "threshold: 3", 1),
        encoding="utf-8",
    )
    asyncio.run(app.reload_config())
    (reply,) = bridge.of("config")
    assert reply["error"] and reply["applied"] == []
    assert app.config.wake.threshold == 0.5


def test_live_settings_are_real_config_keys() -> None:
    from mikata.config import Config

    config = Config()
    for key in app_module.LIVE_SETTINGS:
        section, _, name = key.partition(".")
        assert hasattr(getattr(config, section), name), key
    assert is_live_setting("wake.threshold") and not is_live_setting("audio.input_device")
    other = config.model_copy(update={"ui": config.ui.model_copy(update={"bubble_s": 3})})
    assert changed_settings(config, other) == ["ui.bubble_s"]


def test_silent_microphone_is_reported_busy(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MIC_SILENT_S", 0.05)
    app, _, _, _ = _app(
        config_file, detections={}, speech=set(), stt=FakeSTT(""), llm=ContentLLM(lambda c: ""),
        frames=3,
    )  # fmt: skip
    bridge = FakeBridge()
    _with_ui(app, bridge)
    _run_until(app, lambda: bool(bridge.of("mic")), timeout=3)
    assert bridge.of("mic")[0]["status"] == "busy"


def test_real_channel_end_to_end(config_file: Path) -> None:
    """Interface simulée sur le vrai canal TCP : hello, état, pause, au revoir."""
    config = load_config(config_file)
    channel = config.resolve("data/ui_channel.json")
    app = App(config)
    app._ui = UiBridge(channel, app._on_ui_command, app._ui_snapshot)

    async def scenario() -> list[dict[str, Any]]:
        task = asyncio.create_task(app.run())
        for _ in range(200):
            if channel.exists():
                break
            await asyncio.sleep(0.01)
        info = json.loads(channel.read_text(encoding="utf-8"))
        reader, writer = await asyncio.open_connection("127.0.0.1", info["port"])
        writer.write(encode({"type": "hello", "token": info["token"]}))
        writer.write(encode({"type": "pause", "paused": True}))
        await writer.drain()
        received = [json.loads(await asyncio.wait_for(reader.readline(), 2)) for _ in range(2)]
        app.request_shutdown("test")
        await asyncio.wait_for(task, 5)
        received.append(json.loads(await asyncio.wait_for(reader.readline(), 2)))
        writer.close()
        return received

    hello, mic, bye = asyncio.run(scenario())
    assert hello == {
        "type": "hello",
        "version": 1,
        "state": "idle",
        "follow_up_s": 0,
        "mic": "ok",
        "tasks": [
            {
                "id": "browser",
                "label": "Navigateur de travail — lecture en arrière-plan",
                "running": False,
                "visible": False,
                "tabs": [],
            }
        ],  # phase 13f : onglet « En cours »
    }
    assert mic == {"type": "mic", "status": "paused"} and bye == {"type": "bye"}
    assert app.paused


def test_restart_request_from_the_interface_stops_mikata_for_a_restart(config_file: Path) -> None:
    app, _, _, _ = _app(
        config_file,
        detections={},
        speech=set(),
        stt=FakeSTT(""),
        llm=ContentLLM(lambda c: ""),
        frames=10,
    )
    bridge = FakeBridge()
    _with_ui(app, bridge)
    stops: list[str] = []
    app.request_shutdown = stops.append  # type: ignore[method-assign]  # pas de boucle ici
    assert not app.restart_requested
    app._on_ui_command({"type": "restart"})
    assert app.restart_requested and len(stops) == 1  # __main__ sortira avec le code 75


def test_reload_config_switches_the_audio_output_live(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 15b : changer la sortie dans les réglages s'applique sans redémarrer."""
    import mikata.__main__ as main_module

    app = App(load_config(config_file))
    bridge = FakeBridge()
    _with_ui(app, bridge)
    switched: list[str] = []

    class Player:
        sample_rate = 48_000

        def switch(self, device: str, loop: object) -> None:
            switched.append(device)

        def play_nowait(self, pcm: object) -> None: ...

    app._player = Player()  # type: ignore[assignment]
    monkeypatch.setattr(main_module, "find_output_device", lambda c: c.audio.output_device)
    edit_config(config_file, audio={"output_device": "Haut-parleurs"})
    asyncio.run(app.reload_config())
    (reply,) = bridge.of("config")
    assert reply["error"] is None and reply["applied"] == ["audio.output_device"]
    assert reply["restart"] == [] and switched == ["Haut-parleurs"]
    assert "wake" in app._earcons and app._earcons.sample_rate == 48_000  # type: ignore[attr-defined]
