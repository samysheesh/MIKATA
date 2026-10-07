"""Enregistrement d'une commande, fin de parole et transcription spéculative (§2, §3, §5).

L'enregistrement démarre au **début du pré-roll** (les 2 s d'avant la détection) : on peut
dire « MIKATA, ouvre Spotify » d'une traite, le début de la commande est déjà dans le tampon
quand le bip part. Le mot d'activation est ensuite retiré du texte.

Fin de parole (VAD sur les trames reçues après la détection) :
- parole puis `end_silence_ms` de silence -> fin ; dès `speculative_ms` de silence, Whisper
  transcrit en avance. Si la parole reprend, ce résultat est jeté ; sinon il sert tel quel.
- aucune parole après la détection pendant `end_silence_ms` : la commande a peut-être été
  dite d'une traite et finie avant le bip (« MIKATA, pause »). On transcrit le pré-roll :
  s'il reste du texte après le mot d'activation, c'est la commande ; sinon on attend
  (« MIKATA… [bip] ouvre Spotify ») jusqu'à `no_speech_timeout_s`. Si la transcription du
  pré-roll n'est pas finie à ce moment-là (GPU très chargé), on attend son verdict au lieu
  de perdre la commande.
- `max_utterance_s` atteint : fin forcée.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from enum import StrEnum

import numpy as np
from loguru import logger

from mikata.audio.base import FRAME_MS, SAMPLE_RATE
from mikata.config import VadConfig
from mikata.stt.base import STT, Transcript
from mikata.stt.text import clean_transcript, strip_wake_word


def frames_for(ms: float) -> int:
    """Nombre de trames de 80 ms le plus proche (au moins 1) : 250 ms -> 3, 500 ms -> 6."""
    return max(1, round(ms / FRAME_MS))


class EndReason(StrEnum):
    SILENCE = "silence"
    MAX_LENGTH = "durée maximale"
    PRE_ROLL = "commande dite d'une traite"
    NO_SPEECH = "aucune parole"


class CommandRecorder:
    """Une commande en cours : audio, décision de fin, transcriptions en avance.

    Vit dans la boucle asyncio : `add` est appelé pour chaque trame, `on_end` est rappelé
    une seule fois (depuis `add` ou à la fin de la transcription du pré-roll).
    """

    def __init__(
        self,
        stt: STT,
        pre_roll: np.ndarray,
        cfg: VadConfig,
        wake_pattern: re.Pattern[str],
        on_end: Callable[[EndReason], None],
        check_pre_roll: bool = True,
    ) -> None:
        self._stt = stt
        self._chunks: list[np.ndarray] = [pre_roll]
        self._samples = len(pre_roll)
        self._pre_roll_samples = len(pre_roll)
        self._wake_pattern = wake_pattern
        self._on_end = on_end
        self._speculative_frames = frames_for(cfg.speculative_ms)
        self._end_frames = frames_for(cfg.end_silence_ms)
        self._no_speech_frames = frames_for(cfg.no_speech_timeout_s * 1000)
        self._max_frames = frames_for(cfg.max_utterance_s * 1000)

        self._frames = 0
        self._silent_frames = 0
        self._speech_seen = False
        self._check_pre_roll = check_pre_roll
        self._timed_out = False  # délai sans parole atteint, verdict du pré-roll attendu
        self._pre_roll_task: asyncio.Task[Transcript] | None = None
        self._spec_task: asyncio.Task[Transcript] | None = None
        self._spec_valid = False  # aucune parole depuis le lancement du spéculatif
        self.reason: EndReason | None = None
        self.speculative_runs = 0
        self.used_speculative = False

    @property
    def finished(self) -> bool:
        return self.reason is not None

    @property
    def audio(self) -> np.ndarray:
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0]

    @property
    def duration_s(self) -> float:
        return self._samples / SAMPLE_RATE

    # -- trames -------------------------------------------------------------------------

    def add(self, frame: np.ndarray, is_speech: bool) -> None:
        if self.finished or self._timed_out:
            return
        self._chunks.append(frame)
        self._samples += len(frame)
        self._frames += 1
        if is_speech:
            self._speech_seen = True
            self._silent_frames = 0
            self._spec_valid = False  # la parole reprend : le spéculatif est périmé
        else:
            self._silent_frames += 1

        if self._frames >= self._max_frames:
            self._finish(EndReason.MAX_LENGTH)
        elif self._speech_seen:
            if self._silent_frames == self._speculative_frames:
                self._speculate()
            if self._silent_frames >= self._end_frames:
                self._finish(EndReason.SILENCE)
        else:
            if self._check_pre_roll and self._frames == self._end_frames:
                self._pre_roll_task = asyncio.create_task(
                    self._stt.transcribe(self.audio[: self._pre_roll_samples])
                )
                self._pre_roll_task.add_done_callback(self._pre_roll_done)
            if self._frames >= self._no_speech_frames:
                if self._pre_roll_task is not None and not self._pre_roll_task.done():
                    self._timed_out = True  # `_pre_roll_done` tranchera
                else:
                    self._finish(EndReason.NO_SPEECH)

    def _speculate(self) -> None:
        if self._spec_task is not None and not self._spec_task.done():
            return  # un calcul GPU à la fois ; le suivant attendra la fin de parole
        self._spec_task = asyncio.create_task(self._stt.transcribe(self.audio))
        self._spec_valid = True
        self.speculative_runs += 1

    def _pre_roll_done(self, task: asyncio.Task[Transcript]) -> None:
        if self.finished or task.cancelled():
            return
        if task.exception() is not None:
            logger.warning("Transcription du pré-roll impossible : {}", task.exception())
            if self._timed_out:
                self._finish(EndReason.NO_SPEECH)
            return
        text, found = strip_wake_word(clean_transcript(task.result().text), self._wake_pattern)
        logger.debug("Pré-roll : « {} » (mot d'activation trouvé : {})", text, found)
        if found and text and not self._speech_seen:
            self._finish(EndReason.PRE_ROLL)
        elif self._timed_out:
            self._finish(EndReason.NO_SPEECH)

    def _finish(self, reason: EndReason) -> None:
        if self.finished:
            return
        self.reason = reason
        if reason is not EndReason.PRE_ROLL and self._pre_roll_task is not None:
            self._pre_roll_task.cancel()
        self._on_end(reason)

    # -- résultat -----------------------------------------------------------------------

    async def transcribe(self) -> Transcript:
        """Transcription finale (après `on_end`), en réutilisant le spéculatif s'il est valide."""
        if self.reason is EndReason.PRE_ROLL and self._pre_roll_task is not None:
            self.used_speculative = True
            return self._pre_roll_task.result()
        if self._spec_task is not None and self._spec_valid:
            try:
                result = await self._spec_task
            except Exception as exc:
                logger.warning("Transcription spéculative en échec, nouvel essai : {}", exc)
            else:
                self.used_speculative = True
                return result
        return await self._stt.transcribe(self.audio)

    def cancel(self) -> None:
        for task in (self._spec_task, self._pre_roll_task):
            if task is not None:
                task.cancel()
