"""Phase 8 : canal local MIKATA <-> interface (protocole, serveur côté MIKATA)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from mikata.ui.bridge import QUEUE_SIZE, UiBridge, ui_command
from mikata.ui.protocol import (
    MIKATA_EVENTS,
    UI_COMMANDS,
    Channel,
    ProtocolError,
    decode,
    encode,
    read_channel,
    write_channel,
)


def test_encode_decode_round_trip() -> None:
    line = encode({"type": "heard", "id": 3, "text": "quelle heure est-il à Tokyo ?"})
    assert line.endswith(b"\n") and b"Tokyo" in line
    assert decode(line, MIKATA_EVENTS) == {
        "type": "heard",
        "id": 3,
        "text": "quelle heure est-il à Tokyo ?",
    }
    long = decode(encode({"type": "say", "text": "a" * 5000}), MIKATA_EVENTS)
    assert len(long["text"]) == 2000  # texte tronqué, pas de ligne géante
    with pytest.raises(ProtocolError):
        decode(b"pas du json", MIKATA_EVENTS)
    with pytest.raises(ProtocolError):
        decode(encode({"type": "quit_everything"}), UI_COMMANDS)  # type inconnu de ce côté


def test_channel_file(tmp_path: Path) -> None:
    channel = Channel.new(54321)
    assert len(channel.token) >= 30
    path = tmp_path / "data" / "ui_channel.json"
    write_channel(path, channel)
    assert read_channel(path) == channel
    path.write_text("{", encoding="utf-8")
    assert read_channel(path) is None
    assert read_channel(tmp_path / "absent.json") is None


def test_tasks_messages_round_trip() -> None:
    """Phase 13f : l'onglet « En cours » liste et ferme l'arrière-plan."""
    line = encode({"type": "tasks", "tasks": [{"id": "browser", "label": "x", "running": True}]})
    decoded = decode(line, MIKATA_EVENTS)
    assert decoded["tasks"] == [{"id": "browser", "label": "x", "running": True}]
    decode(encode({"type": "list_tasks"}), UI_COMMANDS)
    assert decode(encode({"type": "close_task", "id": "browser"}), UI_COMMANDS)["id"] == "browser"
    with pytest.raises(ProtocolError):
        decode(encode({"type": "close_task", "id": "browser"}), MIKATA_EVENTS)


async def _connect(bridge: UiBridge, token: str | None = None) -> tuple[Any, Any]:
    assert bridge.channel is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", bridge.channel.port)
    writer.write(encode({"type": "hello", "token": token or bridge.channel.token}))
    await writer.drain()
    return reader, writer


async def _read(reader: Any) -> dict[str, Any]:
    return json.loads(await asyncio.wait_for(reader.readline(), 2))


def test_bridge_sends_events_and_receives_commands(tmp_path: Path) -> None:
    commands: list[dict[str, Any]] = []

    async def scenario() -> None:
        bridge = UiBridge(tmp_path / "ch.json", commands.append, lambda: {"state": "idle"})
        bridge.publish({"type": "state", "state": "listening"})  # aucune interface : rien
        channel = await bridge.start()
        assert read_channel(tmp_path / "ch.json") == channel
        reader, writer = await _connect(bridge)
        hello = await _read(reader)
        assert hello["type"] == "hello" and hello["state"] == "idle" and hello["version"] == 1
        assert bridge.connected
        bridge.publish({"type": "wake", "score": 0.9})
        bridge.publish({"type": "heard", "id": 1, "text": "ouvre Spotify"})
        assert (await _read(reader))["type"] == "wake"
        assert (await _read(reader))["text"] == "ouvre Spotify"
        writer.write(encode({"type": "pause", "paused": True}))
        writer.write(b"n'importe quoi\n")  # ignoré, la connexion reste ouverte
        writer.write(encode({"type": "reload_config"}))
        await writer.drain()
        for _ in range(50):
            if len(commands) == 2:
                break
            await asyncio.sleep(0.01)
        await bridge.close()
        assert (await _read(reader))["type"] == "bye"
        writer.close()

    asyncio.run(scenario())
    assert [c["type"] for c in commands] == ["pause", "reload_config"]


def test_bridge_rejects_a_wrong_token(tmp_path: Path) -> None:
    async def scenario() -> bytes:
        bridge = UiBridge(tmp_path / "ch.json", lambda c: None)
        await bridge.start()
        reader, writer = await _connect(bridge, token="mauvais")
        data = await asyncio.wait_for(reader.read(), 2)  # fermée sans rien envoyer
        assert not bridge.connected
        await bridge.close()
        writer.close()
        return data

    assert asyncio.run(scenario()) == b""


def test_a_stuck_interface_never_blocks_mikata(tmp_path: Path) -> None:
    """publish ne fait qu'ajouter à une file bornée : une interface qui ne lit rien ne
    ralentit pas MIKATA, elle perd les plus vieux messages."""

    async def scenario() -> None:
        bridge = UiBridge(tmp_path / "ch.json", lambda c: None)
        await bridge.start()
        reader, writer = await _connect(bridge)
        await _read(reader)
        loop = asyncio.get_running_loop()
        start = loop.time()
        for i in range(10_000):
            bridge.publish({"type": "say", "id": i, "text": "x" * 100})
        assert loop.time() - start < 0.5
        client = next(iter(bridge._clients))
        assert len(client.queue) <= QUEUE_SIZE
        await bridge.close()
        writer.close()

    asyncio.run(scenario())


def test_ui_command_uses_the_same_interpreter() -> None:
    import sys

    assert ui_command(["--settings"]) == [sys.executable, "-m", "mikata.ui", "--settings"]
