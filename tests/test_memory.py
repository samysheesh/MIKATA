from __future__ import annotations

import pytest

from mikata.agent.memory import ShortTermMemory, estimate_tokens, is_reset_command


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_history_as_messages() -> None:
    memory = ShortTermMemory(clock=Clock())
    memory.add("quelle heure est-il", "Il est midi.")
    assert memory.messages() == [
        {"role": "user", "content": "quelle heure est-il"},
        {"role": "assistant", "content": "Il est midi."},
    ]
    assert len(memory) == 1


def test_oldest_exchanges_dropped_first() -> None:
    memory = ShortTermMemory(max_tokens=100, clock=Clock())
    for i in range(10):
        memory.add(f"question {i} " + "x" * 60, f"réponse {i}")
    assert memory.tokens <= 100
    contents = [m["content"] for m in memory.messages()]
    assert contents[-1] == "réponse 9"
    assert not any("question 0 " in c for c in contents)


def test_reset_after_inactivity() -> None:
    clock = Clock()
    memory = ShortTermMemory(reset_after_s=600, clock=clock)
    memory.add("a", "b")
    clock.t = 599
    assert len(memory.messages()) == 2
    memory.touch()
    clock.t = 599 + 601
    assert memory.messages() == []


def test_clear() -> None:
    memory = ShortTermMemory(clock=Clock())
    memory.add("a", "b")
    memory.clear()
    assert memory.messages() == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nouvelle conversation", True),
        ("Nouvelle conversation.", True),
        ("  nouvelle   conversation ! ", True),
        ("lance une nouvelle conversation Discord", False),
        ("nouveau", False),
    ],
)
def test_reset_command(text: str, expected: bool) -> None:
    assert is_reset_command(text) is expected


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 4
    assert 25 <= estimate_tokens("x" * 100) <= 35


def test_exchange_keeps_tool_steps() -> None:
    memory = ShortTermMemory(max_tokens=6000)
    steps = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function",
         "function": {"name": "open_app", "arguments": '{"name": "Spotify"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "Spotify ouvert."},
    ]  # fmt: skip
    memory.add("ouvre Spotify", "C'est fait.", steps)
    assert [m["role"] for m in memory.messages()] == ["user", "assistant", "tool", "assistant"]
    assert memory.messages()[1]["tool_calls"][0]["id"] == "a"
    assert memory.tokens > ShortTermMemory().tokens + 20  # les étapes comptent dans le plafond
