"""Phase 6b : arrêt vocal sans mot d'activation (`stt.voice_stop.VoiceStopSpotter`)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import write_test_config

from mikata.config import AgentConfig, load_config
from mikata.stt.base import Transcript
from mikata.stt.voice_stop import StopHeard, VoiceStopSpotter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHRASES = AgentConfig().voice_stop_phrases

# Valeur des trames -> mots entendus. Une phrase = suite de trames de valeurs différentes.
WORDS = {
    1: "Arrête",
    2: "de dire stop.",
    3: "Stop !",
    4: "Non merci.",
    5: "Tais-toi.",
    6: "Ça suffit !",
    7: "Arrête la musique.",
    8: "Chut.",
}


class WordsSTT:
    """Texte des mots présents dans l'audio, dans l'ordre (sans répéter un mot qui dure)."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls.append(audio)
        await asyncio.sleep(0.001)
        values = [int(v) for v in audio[::1280]]
        words = [
            WORDS[v] for i, v in enumerate(values) if v in WORDS and (i == 0 or v != values[i - 1])
        ]
        return Transcript(text=" ".join(words), duration_s=len(audio) / 16_000, latency_s=0.04)


def _frame(value: int) -> np.ndarray:
    return np.full(1280, value, np.int16)


async def _feed(spotter: VoiceStopSpotter, values: list[int], t0: float = 100.0) -> None:
    """Une trame toutes les 80 ms (horodatage simulé) ; la boucle tourne entre deux trames."""
    for i, value in enumerate(values):
        spotter.add(_frame(value), value in WORDS, t0 + i * 0.08)
        await asyncio.sleep(0.002)
    await asyncio.sleep(0.02)  # verdicts en attente


def _run(values: list[int], **kwargs: Any) -> tuple[list[StopHeard], WordsSTT, VoiceStopSpotter]:
    heard: list[StopHeard] = []
    stt = WordsSTT()

    async def main() -> VoiceStopSpotter:
        spotter = VoiceStopSpotter(stt, PHRASES, heard.append, **kwargs)
        await _feed(spotter, values)
        return spotter

    return heard, stt, asyncio.run(main())


@pytest.mark.parametrize("word", [3, 5, 6, 8])
def test_stop_phrase_is_heard(word: int) -> None:
    heard, stt, _ = _run([0, 0, word, word, 0, 0, 0])
    assert len(heard) == 1
    assert heard[0].text == WORDS[word]
    assert heard[0].last_speech_at == pytest.approx(100.24)  # 4e trame : fin de parole
    assert heard[0].ended_at == pytest.approx(100.40)  # 160 ms de silence (2 trames)
    assert len(stt.calls) == 1


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 2, 2, 2, 0, 0],  # « arrête de dire stop »
        [1, 1, 0, 2, 2, 0, 0],  # « arrête... de dire stop » : courte pause au milieu
        [4, 4, 0, 0],  # « non merci » : annule après « MIKATA », mais pas un arrêt vocal
        [7, 7, 7, 0, 0],  # « arrête la musique »
    ],
)
def test_longer_or_other_phrase_does_not_stop(values: list[int]) -> None:
    heard, _, _ = _run(values)
    assert heard == []


def test_pause_inside_a_phrase_discards_the_early_transcription() -> None:
    """Le texte de « arrête » (transcrit dès la 1re trame de silence) est jeté : la phrase
    reprend, seul le texte complet compte."""
    heard, stt, _ = _run([1, 1, 0, 2, 2, 0, 0])
    assert heard == [] and len(stt.calls) == 2  # « arrête », puis la phrase entière


def test_long_speech_is_not_even_transcribed() -> None:
    heard, stt, spotter = _run([3] * 30 + [0, 0, 0], max_s=2.0)  # 2,4 s de parole
    assert heard == [] and stt.calls == [] and spotter.transcriptions == 0


def test_pre_roll_keeps_the_start_of_the_word() -> None:
    _, stt, _ = _run([9, 9, 9, 9, 9, 0, 3, 0, 0])  # 9 : son pas encore reconnu comme voix
    audio = stt.calls[0][::1280].tolist()
    assert audio == [9, 9, 9, 0, 3, 0]  # 4 trames de pré-roll (320 ms), la parole, 1 silence


def test_two_stops_in_a_row() -> None:
    heard, _, _ = _run([3, 0, 0, 0, 5, 0, 0])
    assert [h.text for h in heard] == ["Stop !", "Tais-toi."]


def test_reset_forgets_the_pending_phrase() -> None:
    heard: list[StopHeard] = []

    async def main() -> None:
        spotter = VoiceStopSpotter(WordsSTT(), PHRASES, heard.append)
        spotter.add(_frame(3), True, 1.0)
        spotter.add(_frame(0), False, 1.08)  # transcription lancée
        spotter.add(_frame(0), False, 1.16)  # fin : verdict en attente
        spotter.reset()  # la réponse s'est terminée entre-temps
        await asyncio.sleep(0.02)
        assert not spotter.in_phrase

    asyncio.run(main())
    assert heard == []


def test_longer_silence_setting() -> None:
    heard, _, _ = _run([3, 0, 0, 0], silence_ms=320)  # 4 trames de silence attendues
    assert heard == []
    heard, _, _ = _run([3, 0, 0, 0, 0], silence_ms=320)
    assert len(heard) == 1


def test_transcription_error_is_not_a_stop() -> None:
    class BrokenSTT:
        async def transcribe(self, audio: np.ndarray) -> Transcript:
            raise RuntimeError("GPU occupé")

    heard: list[StopHeard] = []

    async def main() -> None:
        await _feed(VoiceStopSpotter(BrokenSTT(), PHRASES, heard.append), [3, 0, 0])

    asyncio.run(main())
    assert heard == []


# -- configuration -------------------------------------------------------------------------


def test_voice_stop_phrases_default_and_cancel_phrases_unchanged() -> None:
    agent = AgentConfig()
    assert agent.voice_stop is True
    assert agent.voice_stop_phrases == [
        "stop", "arrête", "arrête-toi", "tais-toi", "ça suffit", "chut",
    ]  # fmt: skip
    assert "non" in agent.cancel_phrases and "non merci" in agent.cancel_phrases
    assert len(agent.cancel_phrases) == 21  # liste de la phase 6, inchangée


def test_config_yaml_sets_voice_stop() -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    agent = load_config(PROJECT_ROOT / "config.yaml", env_file=PROJECT_ROOT / "absent.env").agent
    assert raw["agent"]["voice_stop_phrases"] == AgentConfig().voice_stop_phrases
    assert agent.voice_stop and agent.voice_stop_silence_ms == 160
    assert agent.voice_stop_max_s == 2.0


def test_voice_stop_phrases_are_editable(tmp_path: Path) -> None:
    path = write_test_config(
        tmp_path / "config.yaml", agent={"voice_stop_phrases": ["silence", "stop"]}
    )
    assert load_config(path).agent.voice_stop_phrases == ["silence", "stop"]
    with pytest.raises(Exception, match="voice_stop_phrases"):
        load_config(write_test_config(tmp_path / "vide.yaml", agent={"voice_stop_phrases": []}))
    with pytest.raises(Exception, match="voice_stop_silence_ms"):
        load_config(write_test_config(tmp_path / "x.yaml", agent={"voice_stop_silence_ms": 40}))
