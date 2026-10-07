from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import numpy as np

from mikata.agent.loop import Agent
from mikata.config import Config
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq
from mikata.llm.client import LlmError


class ScriptedLLM:
    """Chaque appel rejoue le script suivant ; un élément Exception est levé à cet endroit."""

    def __init__(self, *scripts: list[Any], delay: float = 0.0) -> None:
        self.scripts = list(scripts)
        self.calls: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]]] = []
        self.efforts: list[str | None] = []
        self.delay = delay
        self.prewarmed = 0

    async def prewarm(self) -> None:
        self.prewarmed += 1

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append([dict(m) for m in messages])
        self.tools.append(tools)
        self.efforts.append(reasoning_effort)
        for item in self.scripts.pop(0):
            await asyncio.sleep(self.delay)
            if isinstance(item, Exception):
                raise item
            yield item


class Sink:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.audio: list[tuple[Any, int]] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        self.audio.append((pcm, sample_rate))


class Phrases:
    sample_rate = 22_050

    def get(self, key: str) -> Any:
        return np.full(10, len(key), dtype=np.float32)


NOW = datetime(2026, 10, 7, 15, 42)


def _agent(llm: ScriptedLLM, config: Config | None = None, phrases: Any = None) -> Agent:
    return Agent(
        llm, config or Config(), phrases=phrases, window_title=lambda: "Steam", clock=lambda: NOW
    )


def _run(agent: Agent, command: str, sink: Sink) -> Any:
    return asyncio.run(agent.respond(command, sink))


def test_streamed_answer_goes_to_speech_by_fragments() -> None:
    llm = ScriptedLLM(
        [
            TextDelta("D'accord, je lance Steam"),
            TextDelta(" tout de suite, voilà. Bon"),
            TextDelta(" jeu !"),
            Done("stop"),
        ]
    )
    agent, sink = _agent(llm), Sink()
    result = _run(agent, "lance Team", sink)
    assert sink.said == ["D'accord, je lance Steam tout de suite,", "voilà.", "Bon jeu !"]
    assert result.answer == "D'accord, je lance Steam tout de suite, voilà. Bon jeu !"
    assert result.outcome == "ok" and result.attempts == 1 and result.first_token_at is not None
    last = llm.calls[0][-1]["content"]
    assert "Commande transcrite : « lance Team »" in last  # texte brut : le LLM corrige
    assert "mercredi 7 octobre 2026, 15:42" in last and "Fenêtre active : Steam" in last
    assert agent.memory.messages()[-2:] == [
        {"role": "user", "content": "lance Team"},
        {"role": "assistant", "content": result.answer},
    ]


def test_history_is_sent_on_next_turn() -> None:
    llm = ScriptedLLM([TextDelta("Il est midi."), Done("stop")], [TextDelta("Oui."), Done("stop")])
    agent = _agent(llm)
    _run(agent, "quelle heure", Sink())
    _run(agent, "tu es sûr", Sink())
    roles = [m["role"] for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[1][1]["content"] == "quelle heure"


def test_one_immediate_retry_then_success(caplog_loguru: list[str]) -> None:
    llm = ScriptedLLM([LlmError("pas de réponse en 8 s")], [TextDelta("Me revoilà."), Done("stop")])
    sink = Sink()
    result = _run(_agent(llm), "salut", sink)
    assert result.attempts == 2 and result.outcome == "ok"
    assert sink.said == ["Me revoilà."]
    logs = "\n".join(caplog_loguru)
    assert "Appel LLM en échec (1/2) : pas de réponse en 8 s" in logs
    assert "Appel LLM réussi à la tentative 2/2." in logs  # la relance se voit dans le log


def test_server_unreachable_plays_cached_phrase() -> None:
    llm = ScriptedLLM([LlmError("a")], [LlmError("b")])
    sink = Sink()
    result = _run(_agent(llm, phrases=Phrases()), "salut", sink)
    assert result.outcome == "server" and result.attempts == 2
    assert sink.said == [] and len(sink.audio) == 1 and sink.audio[0][1] == 22_050


def test_server_phrase_without_cache_is_synthesized() -> None:
    sink = Sink()
    _run(_agent(ScriptedLLM([LlmError("a")], [LlmError("b")])), "salut", sink)
    assert sink.said == ["Je n'arrive pas à joindre le serveur."]


def test_failure_after_first_words_keeps_them() -> None:
    llm = ScriptedLLM([TextDelta("Il fait beau. Et"), LlmError("coupé")])
    sink = Sink()
    result = _run(_agent(llm), "météo", sink)
    assert result.attempts == 1  # pas de retry : on ne répète pas la réponse
    assert sink.said == ["Il fait beau.", "Et"]


def test_turn_timeout() -> None:
    config = Config()
    config = config.model_copy(
        update={"llm": config.llm.model_copy(update={"turn_timeout_s": 0.05})}
    )
    llm = ScriptedLLM([TextDelta("Euh"), TextDelta(" ..."), Done("stop")], delay=0.04)
    sink = Sink()
    result = _run(_agent(llm, config), "question", sink)
    assert result.outcome == "timeout"
    assert sink.said[-1] == "Désolé, ça prend trop de temps."


def test_new_conversation_command() -> None:
    llm = ScriptedLLM([TextDelta("A."), Done("stop")])
    agent, sink = _agent(llm), Sink()
    _run(agent, "bonjour", Sink())
    result = _run(agent, "Nouvelle conversation.", sink)
    assert result.outcome == "reset"
    assert agent.memory.messages() == []
    assert sink.said == ["D'accord, nouvelle conversation."]
    assert len(llm.calls) == 1  # pas d'appel LLM


def test_tool_calls_without_registry_are_not_run() -> None:
    llm = ScriptedLLM([ToolCallReq("1", "open_app", {"name": "Steam"}), Done("tool_calls")])
    agent, sink = _agent(llm), Sink()
    result = _run(agent, "lance Steam", sink)
    assert result.answer == "" and len(result.tool_calls) == 1 and result.outcome == "limit"
    assert sink.said == ["Désolé, quelque chose s'est mal passé."]
    assert agent.memory.messages() == []


def test_parallel_turns_do_not_block_each_other() -> None:
    # 2 tours de 2 morceaux à 0,1 s : ~0,2 s en parallèle, ~0,4 s l'un après l'autre.
    llm = ScriptedLLM(*[[TextDelta(f"Réponse {i}."), Done("stop")] for i in range(2)], delay=0.1)
    agent = _agent(llm)

    async def scenario() -> float:
        start = asyncio.get_running_loop().time()
        await asyncio.gather(*(agent.respond(f"commande {i}", Sink()) for i in range(2)))
        return asyncio.get_running_loop().time() - start

    assert asyncio.run(scenario()) < 0.3
    assert len(agent.memory) == 2


def test_prewarm_delegates() -> None:
    llm = ScriptedLLM()
    asyncio.run(_agent(llm).prewarm())
    assert llm.prewarmed == 1


# -- phase 6 : fenêtre de suivi jugée par le LLM, résumé de l'historique --------------------


def _follow_up(agent: Agent, command: str, sink: Sink) -> Any:
    return asyncio.run(agent.respond(command, sink, follow_up=True))


def test_follow_up_marker_means_not_for_mikata() -> None:
    """Le LLM juge la phrase hors sujet : rien n'est dit, fait, ni gardé."""
    llm = ScriptedLLM(
        [TextDelta("[IGN"), TextDelta("ORER]"), ToolCallReq("c1", "open_app", {}), Done("stop")]
    )
    agent = _agent(llm)
    sink = Sink()
    result = _follow_up(agent, "tu viens ce soir chez Paul", sink)
    assert result.outcome == "ignored" and result.answer == ""
    assert sink.said == [] and sink.audio == [] and len(agent.memory) == 0
    note = llm.calls[0][-1]["content"]
    assert "Fenêtre de suivi" in note and "[IGNORER]" in note


def test_follow_up_never_asks_to_repeat() -> None:
    llm = ScriptedLLM([TextDelta("Désolé, j'ai pas bien entendu. Tu peux répéter ?"), Done("stop")])
    sink = Sink()
    result = _follow_up(_agent(llm), "bruit de fond", sink)
    assert result.outcome == "ignored" and sink.said == []


def test_follow_up_answer_is_spoken_normally() -> None:
    llm = ScriptedLLM(
        [TextDelta("À Tokyo, il est "), TextDelta("vingt-trois heures."), Done("stop")]
    )
    sink = Sink()
    result = _follow_up(_agent(llm), "et à Tokyo", sink)
    assert result.outcome == "ok" and sink.said == ["À Tokyo, il est vingt-trois heures."]


def test_marker_outside_follow_up_is_not_special() -> None:
    llm = ScriptedLLM([TextDelta("[IGNORER] est un mot."), Done("stop")])
    sink = Sink()
    result = _run(_agent(llm), "lis ce mot", sink)
    assert result.outcome == "ok" and sink.said == ["[IGNORER] est un mot."]
    assert "Fenêtre de suivi" not in llm.calls[0][-1]["content"]


class SummaryLLM:
    """Répond « Réponse n », et résume quand on lui donne le prompt de résumé."""

    def __init__(self) -> None:
        self.summaries: list[str] = []
        self.answers = 0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        from mikata.agent.loop import SUMMARY_PROMPT

        if messages[0]["content"] == SUMMARY_PROMPT:
            self.summaries.append(messages[1]["content"])
            yield TextDelta(f"Résumé numéro {len(self.summaries)} : l'utilisateur aime le jazz.")
        else:
            self.answers += 1
            yield TextDelta(f"Réponse {self.answers}.")
        yield Done("stop")


def test_oldest_exchanges_are_summarized_not_dropped() -> None:
    from mikata.agent.memory import ShortTermMemory

    llm = SummaryLLM()
    memory = ShortTermMemory(max_tokens=60, summarize=True)
    agent = Agent(llm, Config(), memory, window_title=lambda: None, clock=lambda: NOW)

    async def scenario() -> None:
        for i in range(6):
            await agent.respond(f"j'aime le jazz, phrase numéro {i} un peu longue", Sink())
            if agent._summary_task is not None:
                await agent._summary_task

    asyncio.run(scenario())
    assert llm.summaries and "Utilisateur : j'aime le jazz, phrase numéro 0" in llm.summaries[0]
    assert memory.summary.startswith("Résumé numéro") and memory.pending == 0
    history = memory.messages()
    assert history[0]["role"] == "system" and "l'utilisateur aime le jazz" in history[0]["content"]
    assert memory.tokens <= 60 + 40  # plafond (+ le dernier échange, toujours gardé)
