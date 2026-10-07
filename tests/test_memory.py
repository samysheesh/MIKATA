from __future__ import annotations

from pathlib import Path

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


def test_history_persists_across_restarts(tmp_path: Path) -> None:
    """Phase 6 (agent.history_persist) : l'historique survit au redémarrage."""
    path = tmp_path / "data" / "history.json"
    clock = Clock()
    memory = ShortTermMemory(reset_after_s=24 * 3600, clock=clock, path=path)
    steps = [{"role": "tool", "tool_call_id": "a", "content": "ok"}]
    memory.add("ouvre Spotify", "C'est fait.", steps)
    memory.add("quelle heure est-il", "Il est midi.")
    clock.t += 3 * 3600  # trois heures plus tard, après un redémarrage
    reloaded = ShortTermMemory(reset_after_s=24 * 3600, clock=clock, path=path)
    assert reloaded.messages() == memory.messages() and len(reloaded) == 2
    reloaded.clear()  # « nouvelle conversation » : le fichier est vidé (pas supprimé)
    assert path.is_file()
    assert len(ShortTermMemory(reset_after_s=24 * 3600, clock=clock, path=path)) == 0


def test_persisted_history_expires(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    clock = Clock()
    ShortTermMemory(reset_after_s=3600, clock=clock, path=path).add("bonjour", "Salut.")
    clock.t += 2 * 3600
    assert len(ShortTermMemory(reset_after_s=3600, clock=clock, path=path)) == 0


def test_unreadable_history_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{pas du json", encoding="utf-8")
    memory = ShortTermMemory(path=path)
    assert len(memory) == 0
    memory.add("bonjour", "Salut.")  # réécrit proprement
    assert len(ShortTermMemory(path=path)) == 1


def test_agent_memory_follows_config(tmp_path: Path) -> None:
    from conftest import write_test_config

    from mikata.agent.loop import short_term_memory
    from mikata.config import load_config

    # Phase 6 : gardé sur disque 7 jours par défaut, résumé au-delà du plafond.
    default = short_term_memory(load_config(write_test_config(tmp_path / "a.yaml")))
    assert default.path == tmp_path / "data" / "history.json"
    assert default.reset_after_s == 7 * 24 * 3600 and default.summarize
    in_ram = load_config(
        write_test_config(
            tmp_path / "b.yaml", agent={"history_persist": False, "history_summarize": False}
        )
    )
    memory = short_term_memory(in_ram)
    assert memory.path is None and memory.reset_after_s == 600 and not memory.summarize


def test_summary_replaces_dropped_exchanges_and_survives_restart(tmp_path: Path) -> None:
    """Phase 6 : plafond atteint -> les plus anciens échanges attendent leur résumé."""
    path = tmp_path / "history.json"
    clock = Clock()
    memory = ShortTermMemory(max_tokens=100, clock=clock, path=path, summarize=True)
    for i in range(5):  # ~21 tokens par échange : le premier sort de l'historique
        memory.add(f"question numéro {i} assez longue pour peser", f"Réponse {i}.")
    assert memory.pending == 1 and len(memory) == 4 and memory.tokens <= 100
    generation, previous, exchanges = memory.summary_request()
    assert previous == "" and exchanges[0] == ("question numéro 0 assez longue pour peser",
                                                "Réponse 0.")  # fmt: skip
    assert memory.apply_summary(generation, len(exchanges), "L'utilisateur pose des questions.")
    assert memory.pending == 0
    assert memory.messages()[0] == {
        "role": "system",
        "content": "Résumé de la conversation plus ancienne : L'utilisateur pose des questions.",
    }
    reloaded = ShortTermMemory(max_tokens=100, clock=clock, path=path, summarize=True)
    assert reloaded.summary == memory.summary and reloaded.messages() == memory.messages()


def test_without_summary_oldest_exchanges_are_dropped() -> None:
    memory = ShortTermMemory(max_tokens=60, clock=Clock())
    for i in range(5):
        memory.add(f"question numéro {i} assez longue pour peser", f"Réponse {i}.")
    assert memory.pending == 0 and memory.summary == "" and memory.tokens <= 60


def test_new_conversation_clears_everything_and_drops_stale_summaries(tmp_path: Path) -> None:
    memory = ShortTermMemory(max_tokens=60, clock=Clock(), path=tmp_path / "h.json", summarize=True)
    for i in range(5):
        memory.add(f"question numéro {i} assez longue pour peser", f"Réponse {i}.")
    generation, _, exchanges = memory.summary_request()
    memory.clear()  # « nouvelle conversation » pendant que le résumé est calculé
    assert not memory.apply_summary(generation, len(exchanges), "Résumé périmé.")
    assert (memory.summary, memory.pending, len(memory)) == ("", 0, 0)
    assert ShortTermMemory(path=tmp_path / "h.json").messages() == []


def test_duration_set_by_voice_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    clock = Clock()
    memory = ShortTermMemory(reset_after_s=7 * 24 * 3600, clock=clock, path=path)
    memory.add("bonjour", "Salut.")
    memory.set_keep(30 * 24 * 3600)  # « garde la conversation un mois »
    memory.clear()  # ne change pas la durée
    memory.add("tu te souviens ?", "Oui.")
    clock.t += 20 * 24 * 3600  # 20 jours plus tard, après un redémarrage
    reloaded = ShortTermMemory(reset_after_s=7 * 24 * 3600, clock=clock, path=path)
    assert reloaded.reset_after_s == 30 * 24 * 3600 and len(reloaded) == 1


def test_backlog_of_unsummarized_exchanges_is_bounded() -> None:
    """Résumés impossibles (hors ligne) : l'arriéré ne dépasse pas le plafond de tokens."""
    memory = ShortTermMemory(max_tokens=60, clock=Clock(), summarize=True)
    for i in range(20):
        memory.add(f"question numéro {i} assez longue pour peser", f"Réponse {i}.")
    _, _, exchanges = memory.summary_request()
    assert 1 <= len(exchanges) <= 3 and exchanges[-1][0].startswith("question numéro 1")
