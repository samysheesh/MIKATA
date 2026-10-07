"""Arrêt vocal sans mot d'activation pendant que MIKATA parle (ajout phase 6b).

Pendant qu'une réponse est jouée et que personne n'a le micro, l'orchestrateur passe chaque
trame et son verdict VAD à `VoiceStopSpotter`, qui découpe ta parole en phrases :
- une phrase plus longue que `agent.voice_stop_max_s` n'est même pas transcrite (ce n'est
  pas un « stop ») ;
- une phrase courte est transcrite dès sa première trame de silence (en avance, comme la
  transcription spéculative) ; si tu reprends la parole, ce résultat est jeté ;
- après `agent.voice_stop_silence_ms` de silence, si le texte **entier** est une phrase de
  `agent.voice_stop_phrases` (« stop », « tais-toi »...), `on_stop` est rappelé et
  l'orchestrateur coupe le son. « Arrête de dire stop » ou « non merci » ne coupent rien :
  toute autre parole est ignorée.

Avec un casque, le micro n'entend presque pas la voix de MIKATA : pas d'annulation d'écho.
Vit dans la boucle asyncio, comme `CommandRecorder` : `add` ne bloque jamais.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger

from mikata.stt.base import STT, Transcript
from mikata.stt.speculative import frames_for
from mikata.stt.text import clean_transcript, is_cancel_command

PRE_ROLL_MS = 320  # trames gardées avant la première trame de voix (« s » de « stop »)


@dataclass
class StopHeard:
    """Phrase d'arrêt reconnue. Horodatages : time.monotonic (capture des trames)."""

    text: str
    last_speech_at: float  # dernière trame de parole
    ended_at: float  # trame de silence qui a confirmé la fin de la phrase
    transcript: Transcript


class VoiceStopSpotter:
    def __init__(
        self,
        stt: STT,
        phrases: Sequence[str],
        on_stop: Callable[[StopHeard], None],
        *,
        silence_ms: int = 160,
        max_s: float = 2.0,
    ) -> None:
        self._stt = stt
        self._phrases = list(phrases)
        self._on_stop = on_stop
        self._end_frames = frames_for(silence_ms)
        self._max_frames = frames_for(max_s * 1000)
        self._recent: deque[np.ndarray] = deque(maxlen=frames_for(PRE_ROLL_MS))
        self._chunks: list[np.ndarray] | None = None  # phrase en cours (None : aucune)
        self._frames = 0  # trames depuis le début de la phrase
        self._silent = 0
        self._too_long = False
        self._last_speech_at = 0.0
        self._task: asyncio.Task[Transcript] | None = None  # transcription de la phrase
        self._decisions: set[asyncio.Task[None]] = set()
        self.transcriptions = 0

    @property
    def in_phrase(self) -> bool:
        return self._chunks is not None

    def add(self, frame: np.ndarray, is_speech: bool, captured_at: float) -> None:
        if self._chunks is None:
            if is_speech:
                self._chunks = [*self._recent, frame]
                self._recent.clear()
                self._frames, self._silent, self._too_long = 1, 0, False
                self._last_speech_at = captured_at
            else:
                self._recent.append(frame)
            return
        self._frames += 1
        if not self._too_long:
            self._chunks.append(frame)
        if is_speech:
            self._silent = 0
            self._last_speech_at = captured_at
            self._drop_transcription()  # la phrase continue : texte périmé
            if self._frames > self._max_frames and not self._too_long:
                self._too_long = True  # trop longue pour un « stop » : plus rien à garder
                self._chunks = []
            return
        self._silent += 1
        if self._silent == 1 and not self._too_long:
            self._task = asyncio.create_task(self._stt.transcribe(np.concatenate(self._chunks)))
            self.transcriptions += 1
        if self._silent >= self._end_frames:
            self._end_phrase(captured_at)

    def _end_phrase(self, ended_at: float) -> None:
        task, self._task = self._task, None
        if self._too_long:
            logger.debug("Arrêt vocal : parole de plus de {} trames ignorée.", self._max_frames)
        elif task is not None:
            decision = asyncio.create_task(self._decide(task, self._last_speech_at, ended_at))
            self._decisions.add(decision)
            decision.add_done_callback(self._decisions.discard)
        self._chunks = None

    async def _decide(
        self, task: asyncio.Task[Transcript], last_speech_at: float, ended_at: float
    ) -> None:
        try:
            transcript = await task
        except Exception as exc:
            logger.warning("Arrêt vocal : transcription impossible ({}).", exc)
            return
        text = clean_transcript(transcript.text)
        if not is_cancel_command(text, self._phrases):
            if text:
                logger.debug("Arrêt vocal : « {} » ignoré (pas une phrase d'arrêt).", text)
            return
        self._on_stop(StopHeard(text, last_speech_at, ended_at, transcript))

    def _drop_transcription(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def reset(self) -> None:
        """Oublie la phrase en cours et les verdicts en attente (plus rien à couper, ou le
        micro sert à une commande)."""
        self._drop_transcription()
        for decision in self._decisions:
            decision.cancel()
        self._decisions.clear()
        self._chunks = None
        self._recent.clear()
