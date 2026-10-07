"""Orchestrateur et machine à états (docs/ARCHITECTURE.md §3).

Phase 1 : capture continue → tampon circulaire → mot d'activation (executor dédié)
→ bip + journal de détection. Le raccourci clavier a le même effet.
Phase 2 : écoute depuis le début du pré-roll (VAD Silero) → transcription (Whisper,
spéculative) → texte sans le mot d'activation.
Phase 3 : réflexion (LLM en streaming) → parole (TTS par fragments) ; son de réflexion.
Phase 4 : outils (état ACTING), annonces des minuteurs et rappels (`announce`).
Phase 5 : écoute courte sans mot d'activation (`listen_once`) pour les confirmations
orales du garde-fou et le « stop » avant extinction.
Phase 6 :
- **barge-in** : « MIKATA » (ou le raccourci) pendant qu'une réponse est jouée coupe le son
  de **cette** réponse tout de suite (`SpeechStream.cancel`) et lance l'écoute. Les autres
  commandes continuent (décision de la phase 3) ; leurs réponses attendent que tu aies fini
  de parler (`Speaker.hold`).
- **annulation** : une commande qui n'est qu'un « stop », « annule », « laisse tomber »...
  (`agent.cancel_phrases`) n'est pas envoyée au LLM. Elle annule la commande dont la
  réponse vient d'être coupée si elle tourne encore, sinon la dernière commande en cours ;
  en fenêtre de suivi, elle la ferme. Sans rien à annuler ni réponse coupée, « stop » part
  au LLM comme une commande normale (pause de la musique).
- **fenêtre de suivi** : après une réponse, si rien d'autre n'est en cours, le micro écoute
  `agent.follow_up_s` secondes sans mot d'activation (état FOLLOW_UP) ; ce qui est dit
  devient une nouvelle commande, avec l'historique. Sinon : bip de fin. Dans le bruit ou
  une conversation à côté : une transcription peu fiable (`agent.follow_up_min_confidence`)
  est ignorée, puis le LLM juge si la phrase s'adresse à MIKATA
  (`agent.follow_up_use_llm_gate`). Phrase ignorée : bip de fin, aucune réponse.
Phase 6b : **arrêt vocal** sans mot d'activation. Pendant qu'une réponse est jouée et que
personne n'a le micro, le VAD tourne et `stt.voice_stop.VoiceStopSpotter` transcrit les
phrases courtes ; « stop », « tais-toi »... (`agent.voice_stop_phrases`, phrase entière)
coupent le son de cette réponse et arrêtent sa commande si elle travaille encore, comme
« MIKATA, stop ». Toute autre parole est ignorée. Latence de coupure dans le log.

Commandes en parallèle : chaque commande a son propre état (`_Request.state`). Il n'y a
qu'un micro, donc une seule commande **écoute** à la fois (`_listening`) ; dès que sa fin de
parole est détectée, le micro est libre et « MIKATA » peut lancer la suivante pendant que la
première transcrit, réfléchit, agit ou parle. Rien n'est refusé ni mis en file, sauf la
parole elle-même : deux réponses ne se superposent pas (`audio.player.Speaker`).
L'état affiché (icône) est le plus « actif » des états en cours.
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.audio.capture import CapturedFrame
from mikata.audio.ringbuffer import RingBuffer
from mikata.config import Config
from mikata.events import BusEvent, EventBus, Shutdown, WakeDetected
from mikata.stt.base import STT
from mikata.stt.speculative import CommandRecorder, EndReason
from mikata.stt.text import (
    clean_transcript,
    describe_confidence,
    follow_up_rejection,
    is_cancel_command,
    strip_wake_word,
    wake_word_pattern,
)
from mikata.stt.voice_stop import StopHeard, VoiceStopSpotter
from mikata.vad.base import VAD
from mikata.wake.base import WakeWordDetector
from mikata.wake.recordings import (
    NegativeCollector,
    estimate_tail_silence_s,
    purge_older_than,
    save_detection,
)
from mikata.wake.trigger import WakeTrigger

if TYPE_CHECKING:
    from mikata.agent.loop import Agent, TurnResult
    from mikata.audio.player import Speaker, SpeechStream

WAKE_LOG_S = 2.0
HOTKEY_PRE_ROLL_S = 0.3  # raccourci : pas de mot d'activation à retirer, peu de pré-roll
# Le tampon garde le pré-roll + une marge : entre la détection (thread audio) et le début de
# l'écoute (orchestrateur), quelques trames peuvent arriver ; on veut le pré-roll complet
# **avant** la détection, plus tout ce qui a suivi.
RING_MARGIN_S = 1.0
FOLLOW_UP_PRE_ROLL_S = 0.3  # suivi : tu peux commencer à parler pile à la fin de la réponse
FOLLOW_UP_SOURCE = "suivi"
# Réponse orale d'une commande dont la réponse a été coupée : « non » à une confirmation,
# « stop » à un compte à rebours (garde-fou : dans le doute, rien n'est fait).
INTERRUPTED_ANSWER = "stop"
VOICE_STOP_TARGET_MS = 300  # arrêt vocal : coupure visée après la fin de parole


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"


# Priorité d'affichage quand plusieurs commandes sont en cours (la plus haute gagne).
DISPLAY_PRIORITY = (
    State.LISTENING,
    State.SPEAKING,
    State.ACTING,
    State.THINKING,
    State.TRANSCRIBING,
    State.FOLLOW_UP,
)


class StatusIcon(Protocol):
    """Icône de la zone de notification (implémentée par `ui.tray.TrayIcon`)."""

    def start(self, on_quit: Callable[[], None]) -> None: ...
    def set_state(self, state: State) -> None: ...
    def stop(self) -> None: ...


class AudioSource(Protocol):
    """Capture micro (implémentée par `audio.capture.AudioCapture`)."""

    dropped_frames: int
    overflows: int

    def start(self, loop: asyncio.AbstractEventLoop) -> None: ...
    async def read(self) -> CapturedFrame: ...
    def stop(self) -> None: ...


class SoundOutput(Protocol):
    """Sortie audio (implémentée par `audio.player.AudioPlayer`)."""

    @property
    def output_latency_s(self) -> float: ...
    def start(self, loop: asyncio.AbstractEventLoop) -> None: ...
    def play_nowait(self, pcm: np.ndarray) -> None: ...
    def close(self) -> None: ...


class HotkeySource(Protocol):
    """Raccourci clavier global (implémenté par `ui.hotkeys.HotkeyListener`)."""

    def start(self, on_trigger: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...


class CpuSampler(Protocol):
    def sample(self) -> tuple[float, float]: ...


class SpeechToText(STT, Protocol):
    """STT chargé au démarrage (implémenté par `stt.whisper_local.WhisperSTT`)."""

    def load(self) -> None: ...
    def close(self) -> None: ...


class ResettableVAD(VAD, Protocol):
    def reset(self) -> None: ...


@dataclass(eq=False)
class _Request:
    """Une commande, de la détection à la fin de la réponse. Horodatages : time.monotonic."""

    id: int
    source: str
    wake_at: float  # bip
    wake_frame_at: float  # capture de la trame de détection
    recorder: CommandRecorder | None = None
    state: State = State.LISTENING
    end_at: float = 0.0  # fin de parole détectée
    reason: EndReason | None = None
    stream: SpeechStream | None = None
    marks: dict[str, float] = field(default_factory=dict)
    # Ajouts phase 6
    task: asyncio.Task[None] | None = None  # traitement (transcription -> réponse)
    barged: _Request | None = None  # commande dont la réponse a été coupée par celle-ci
    follow_up: bool = False  # réponse jouée jusqu'au bout : fenêtre de suivi possible

    @property
    def last_syllable_at(self) -> float:
        """Dernière trame de parole ; à défaut (commande finie avant le bip), la détection."""
        if self.recorder is not None and self.recorder.last_speech_at is not None:
            return self.recorder.last_speech_at
        return self.wake_frame_at


class App:
    def __init__(
        self,
        config: Config,
        tray: StatusIcon | None = None,
        bus: EventBus | None = None,
        *,
        audio: AudioSource | None = None,
        detector: WakeWordDetector | None = None,
        player: SoundOutput | None = None,
        earcons: Mapping[str, np.ndarray] | None = None,
        hotkeys: HotkeySource | None = None,
        cpu: CpuSampler | None = None,
        vad: ResettableVAD | None = None,
        stt: SpeechToText | None = None,
        agent: Agent | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self.config = config
        self.bus = bus or EventBus()
        self._tray = tray
        self._audio = audio
        self._detector = detector
        self._player = player
        self._earcons = earcons or {}
        self._hotkeys = hotkeys
        self._cpu = cpu
        self._vad = vad
        self._stt = stt
        self._agent = agent
        self._speaker = speaker
        self.state = State.IDLE
        self._wake_pattern = wake_word_pattern(config.stt.wake_words)
        self._listening: _Request | None = None  # la commande qui a le micro
        self._active: dict[int, _Request] = {}  # commandes en cours (y compris l'écoute)
        self._ids = itertools.count(1)
        self._request_tasks: set[asyncio.Task[None]] = set()
        # Dernières commandes reconnues et réponses (bornées : MIKATA tourne h24).
        self.commands: deque[str] = deque(maxlen=100)
        self.answers: deque[str] = deque(maxlen=100)

        self.ring = RingBuffer.for_seconds(config.audio.pre_roll_s + RING_MARGIN_S)
        self._written = 0  # échantillons écrits dans le tampon depuis le démarrage
        self._wake_written = 0  # valeur de `_written` à la dernière détection
        wake = config.wake
        self._trigger = WakeTrigger(wake.threshold, wake.consecutive_frames, wake.refractory_s)
        self._collector = NegativeCollector() if wake.collect_negatives else None
        self._wake_logs_dir = config.resolve("data/wake_logs")
        self._negatives_dir = config.resolve("data/negatives")
        self._wake_executor: Executor | None = None
        self._io_executor: Executor | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._wake_audio: np.ndarray | None = None
        self.detections = 0
        # Arrêt vocal (phase 6b) : actif seulement pendant qu'une réponse est jouée.
        self._voice_stop: VoiceStopSpotter | None = None
        self._voice_stop_armed = False
        agent_cfg = config.agent
        if agent_cfg.voice_stop and stt is not None and vad is not None and speaker is not None:
            self._voice_stop = VoiceStopSpotter(
                stt,
                agent_cfg.voice_stop_phrases,
                self._on_voice_stop,
                silence_ms=agent_cfg.voice_stop_silence_ms,
                max_s=agent_cfg.voice_stop_max_s,
            )

    # -- cycle de vie ---------------------------------------------------------------------

    def request_shutdown(self, reason: str) -> None:
        """Demande l'arrêt ; appelable depuis n'importe quel thread."""
        self.bus.publish_threadsafe(Shutdown(reason=reason))

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            logger.debug("État : {} -> {}", self.state, state)
        self.state = state
        if self._tray is not None:
            self._tray.set_state(state)

    def _refresh_state(self) -> None:
        """État affiché = le plus prioritaire des états des commandes en cours."""
        states = {request.state for request in self._active.values()}
        state = next((s for s in DISPLAY_PRIORITY if s in states), State.IDLE)
        if state is not self.state:
            self._set_state(state)

    def _set_listening(self, request: _Request | None) -> None:
        """Commande qui a le micro. Tant que quelqu'un parle, aucune nouvelle phrase de
        réponse ne commence (`Speaker.hold`)."""
        if (request is None) == (self._listening is None):
            self._listening = request
            return
        self._listening = request
        if self._speaker is not None:
            if request is None:
                self._speaker.release()
            else:
                self._speaker.hold()

    def _set_request_state(self, request: _Request, state: State) -> None:
        if request.state is not state:
            logger.debug("Commande #{} : {} -> {}", request.id, request.state, state)
            request.state = state
        self._refresh_state()

    async def run(self) -> None:
        """Boucle principale : rend la main après un `Shutdown` ou une annulation (Ctrl+C)."""
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        self._wake_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-wake")
        self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-io")
        try:
            if self._tray is not None:
                self._tray.start(on_quit=lambda: self.request_shutdown("menu Quitter"))
            if self._player is not None:
                self._player.start(loop)
            if self._stt is not None:
                logger.info("Chargement de Whisper ({})...", self.config.stt.model)
                await asyncio.to_thread(self._stt.load)
            if self._speaker is not None:
                logger.info("Chargement de la voix ({})...", self.config.tts.backend)
                await asyncio.to_thread(self._speaker.load)
            if self._agent is not None:
                self._spawn(self._agent.prewarm(), "préchauffage LLM")
            if self._audio is not None:
                self._audio.start(loop)
                self._tasks.append(asyncio.create_task(self._audio_loop(), name="audio"))
            if self._hotkeys is not None:
                self._start_hotkeys()
            if self._cpu is not None and self.config.logging.stats_interval_s > 0:
                self._tasks.append(asyncio.create_task(self._stats_loop(), name="stats"))
            self._submit_io(self._purge_wake_logs)
            if self._collector is not None:
                logger.warning(
                    "Mode collect_negatives actif : extraits dans {}", self._negatives_dir
                )

            self._set_state(State.IDLE)
            logger.info("MIKATA démarré, en attente.")
            while True:
                event = await self.bus.get()
                if isinstance(event, Shutdown):
                    logger.info("Arrêt demandé ({}).", event.reason or "sans motif")
                    break
                try:
                    await self._handle(event)
                except Exception:  # un événement raté ne doit jamais arrêter l'assistant
                    logger.exception("Erreur pendant le traitement de {}.", type(event).__name__)
        finally:
            await self._shutdown()

    def _start_hotkeys(self) -> None:
        assert self._hotkeys is not None

        def on_trigger() -> None:  # thread pynput
            self.bus.publish_threadsafe(WakeDetected(score=1.0, source="hotkey"))

        try:
            self._hotkeys.start(on_trigger)
        except Exception as exc:  # le raccourci est un secours : jamais bloquant
            logger.warning("Raccourci clavier indisponible : {}", exc)
            self._hotkeys = None

    async def _shutdown(self) -> None:
        if self._voice_stop is not None:
            self._voice_stop.reset()
        for request in self._active.values():
            if request.recorder is not None:
                request.recorder.cancel()
            if request.stream is not None:
                request.stream.cancel()
        tasks = [*self._tasks, *self._request_tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._request_tasks.clear()
        for name, aclose in (
            ("voix", self._speaker.aclose if self._speaker else None),
            ("LLM", self._agent.aclose if self._agent else None),
        ):
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.exception("Erreur à l'arrêt ({}).", name)
        for name, stop in (
            ("raccourci", self._hotkeys.stop if self._hotkeys else None),
            ("micro", self._audio.stop if self._audio else None),
            ("sortie", self._player.close if self._player else None),
            ("whisper", self._stt.close if self._stt else None),
            ("icône", self._tray.stop if self._tray else None),
        ):
            if stop is None:
                continue
            try:
                stop()
            except Exception:
                logger.exception("Erreur à l'arrêt ({}).", name)
        if self._wake_executor is not None:
            self._wake_executor.shutdown(wait=True, cancel_futures=True)
        if self._io_executor is not None:  # les écritures en attente vont jusqu'au bout
            self._io_executor.shutdown(wait=True)
        logger.info("MIKATA arrêté proprement.")

    # -- audio et mot d'activation ----------------------------------------------------------

    async def _audio_loop(self) -> None:
        assert self._audio is not None
        loop = asyncio.get_running_loop()
        half_threshold = self._trigger.threshold / 2
        while True:
            captured = await self._audio.read()
            self.ring.write(captured.samples)
            self._written += len(captured.samples)
            # Pas d'await entre l'écriture dans le tampon et cette lecture : une trame est soit
            # dans le pré-roll d'une commande, soit ajoutée à son enregistrement, jamais les deux.
            request = self._listening
            if self._collector is not None:
                clip = self._collector.add(captured.samples)
                if clip is not None:
                    self._submit_io(self._save_negative, clip, datetime.now())
            recorder = request.recorder if request is not None else None
            if recorder is not None and not recorder.finished and self._vad is not None:
                speech = await loop.run_in_executor(
                    self._wake_executor, self._vad.is_speech, captured.samples
                )
                recorder.add(captured.samples, speech, captured.captured_at)
                if speech and request is not None and request.state is State.FOLLOW_UP:
                    self._set_request_state(request, State.LISTENING)  # tu parles
            if self._voice_stop is not None:
                await self._spot_voice_stop(request is None, captured)
            if self._detector is None:
                continue
            score = await loop.run_in_executor(
                self._wake_executor, self._detector.process, captured.samples
            )
            if score >= half_threshold:
                logger.debug("Score mot d'activation : {:.3f}", score)
            if self._trigger.update(score, captured.captured_at):
                self._wake_audio = self.ring.read_last(round(WAKE_LOG_S * SAMPLE_RATE))
                self._wake_written = self._written
                self.bus.publish(WakeDetected(score=score, ts=captured.captured_at))

    async def _spot_voice_stop(self, mic_free: bool, captured: CapturedFrame) -> None:
        """Arrêt vocal : VAD + découpe des phrases tant qu'une réponse est jouée et que le
        micro ne sert à aucune commande (sinon c'est l'écoute qui entend « stop »)."""
        assert self._voice_stop is not None and self._vad is not None
        if not mic_free or self._playing_stream() is None:
            if self._voice_stop_armed:
                self._voice_stop_armed = False
                self._voice_stop.reset()
            return
        loop = asyncio.get_running_loop()
        if not self._voice_stop_armed:
            self._voice_stop_armed = True
            if self._wake_executor is not None:  # même thread que is_speech : ordre garanti
                self._wake_executor.submit(self._vad.reset)
        speech = await loop.run_in_executor(
            self._wake_executor, self._vad.is_speech, captured.samples
        )
        self._voice_stop.add(captured.samples, speech, captured.captured_at)

    def _on_voice_stop(self, heard: StopHeard) -> None:
        """« stop » dit sans « MIKATA » pendant une réponse : coupe le son de la réponse en
        train d'être jouée, et arrête sa commande si elle travaille encore (comme « MIKATA,
        stop »). Les autres commandes continuent."""
        stream = self._playing_stream()
        if stream is None or self._listening is not None:
            logger.info("Arrêt vocal : « {} » entendu, plus rien à couper.", heard.text)
            return
        stream.cancel()  # AudioPlayer.stop() : silence dès le bloc audio suivant
        cut_at = time.monotonic()
        owner = next((r for r in self._active.values() if r.stream is stream), None)
        stopped = ""
        if (
            owner is not None
            and "answered" not in owner.marks
            and owner.task is not None
            and not owner.task.done()
        ):
            owner.task.cancel()  # plus d'outil ni d'appel LLM pour une réponse coupée
            stopped = " ; commande arrêtée"
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        total_ms = (cut_at - heard.last_speech_at) * 1000 + out_ms
        label = f"#{owner.id}" if owner is not None else stream.label or "annonce"
        log = logger.info if total_ms < VOICE_STOP_TARGET_MS else logger.warning
        log(
            "Arrêt vocal : « {} » -> réponse {} coupée ≈ {:.0f} ms après la fin de parole "
            "(silence {:.0f} + attente transcription {:.0f} + sortie {:.0f} ms ; Whisper "
            "{:.0f} ms ; objectif < {} ms){}.",
            heard.text,
            label,
            total_ms,
            (heard.ended_at - heard.last_speech_at) * 1000,
            (cut_at - heard.ended_at) * 1000,
            out_ms,
            heard.transcript.latency_s * 1000,
            VOICE_STOP_TARGET_MS,
            stopped,
        )

    def _playing_stream(self) -> SpeechStream | None:
        """Réponse en train d'être jouée (premier son parti, pas coupée), ou None."""
        stream = self._speaker.current if self._speaker is not None else None
        if stream is None or stream.first_sound_at is None or stream.cancelled:
            return None
        return stream

    async def _handle(self, event: BusEvent) -> None:
        if isinstance(event, WakeDetected):
            self._on_wake(event)
        else:
            logger.debug("Événement non géré : {}", type(event).__name__)

    def _on_wake(self, event: WakeDetected) -> None:
        now = time.monotonic()
        listening = self._listening
        if listening is not None:  # le micro sert déjà à une commande
            if listening.source == FOLLOW_UP_SOURCE and "wake" not in listening.marks:
                # « MIKATA » dit dans la fenêtre de suivi : elle écoute déjà, le mot sera
                # retiré du texte ; le bip confirme qu'on t'entend.
                listening.marks["wake"] = now
                self._play("wake")
                self._set_request_state(listening, State.LISTENING)
            logger.debug("Mot d'activation ignoré pendant une écoute ({}).", event.source)
            return
        barged = self._barge_in(event)
        self._play("wake")
        self.detections += 1
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        pipeline_ms = (now - event.ts) * 1000
        if self._agent is not None:  # TLS prête quand la commande partira (§2, technique 2)
            self._spawn(self._agent.prewarm(), "préchauffage LLM")
        if self._active:
            logger.info("Nouvelle commande pendant {} autre(s) en cours.", len(self._active))

        if event.source == "hotkey":
            self._trigger.suppress(now)
            logger.info("Raccourci clavier : bip lancé (+ sortie {:.0f} ms).", out_ms)
            self._start_listening(event, now, barged)
            return

        if self._detector is not None and self._wake_executor is not None:
            self._wake_executor.submit(self._detector.reset)
        audio, self._wake_audio = self._wake_audio, None
        tail_s = estimate_tail_silence_s(audio) if audio is not None else None
        if tail_s is None:
            logger.info(
                "Mot d'activation détecté (score {:.2f}) : bip {:.0f} ms après la trame "
                "(+ sortie {:.0f} ms).",
                event.score,
                pipeline_ms,
                out_ms,
            )
        else:
            logger.info(
                "Mot d'activation détecté (score {:.2f}) : bip ≈ {:.0f} ms après la fin du mot "
                "(détection {:.0f} + traitement {:.0f} + sortie {:.0f} ms).",
                event.score,
                tail_s * 1000 + pipeline_ms + out_ms,
                tail_s * 1000,
                pipeline_ms,
                out_ms,
            )
        if audio is not None:
            self._submit_io(self._save_wake_log, audio, event.score, datetime.now())
        self._start_listening(event, now, barged)

    def _barge_in(self, event: WakeDetected) -> _Request | None:
        """Coupe la réponse **en train d'être jouée** (et seulement elle) ; renvoie sa
        commande. Une réponse pas encore commencée n'est pas touchée : elle attendra la fin
        de l'écoute."""
        stream = self._playing_stream()
        if stream is None:
            return None
        stream.cancel()  # AudioPlayer.stop() : silence dès le bloc audio suivant
        cut_at = time.monotonic()
        owner = next((r for r in self._active.values() if r.stream is stream), None)
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        tail_s = None
        if event.source != "hotkey" and self._wake_audio is not None:
            tail_s = estimate_tail_silence_s(self._wake_audio)
        label = f"#{owner.id}" if owner is not None else stream.label or "annonce"
        if tail_s is None:
            logger.info(
                "Barge-in : réponse {} coupée {:.0f} ms après la trame (+ sortie {:.0f} ms).",
                label,
                (cut_at - event.ts) * 1000,
                out_ms,
            )
        else:
            logger.info(
                "Barge-in : réponse {} coupée ≈ {:.0f} ms après la fin du mot (détection {:.0f} "
                "+ traitement {:.0f} + sortie {:.0f} ms).",
                label,
                tail_s * 1000 + (cut_at - event.ts) * 1000 + out_ms,
                tail_s * 1000,
                (cut_at - event.ts) * 1000,
                out_ms,
            )
        return owner

    # -- écoute (une seule à la fois : il n'y a qu'un micro) -------------------------------

    @property
    def _can_listen(self) -> bool:
        return self._stt is not None and self._vad is not None and self._audio is not None

    async def listen_once(self, timeout_s: float) -> str:
        """Écoute une réponse courte sans mot d'activation (« oui », « stop »...).

        Attend que le micro soit libre, bipe, écoute jusqu'à la fin de la phrase (ou
        `timeout_s` sans parole), puis transcrit. Renvoie "" si rien n'a été dit.
        """
        if not self._can_listen:
            return ""
        assert self._stt is not None and self._vad is not None
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout_s
        while self._listening is not None:  # une autre commande a le micro
            if time.monotonic() >= deadline:
                return ""
            await asyncio.sleep(0.05)
        now = time.monotonic()
        self._play("wake")
        if self._wake_executor is not None:
            self._wake_executor.submit(self._vad.reset)
        ended: asyncio.Future[EndReason] = loop.create_future()

        def on_end(reason: EndReason) -> None:
            if not ended.done():
                ended.set_result(reason)

        request = _Request(id=next(self._ids), source="réponse", wake_at=now, wake_frame_at=now)
        vad = self.config.vad.model_copy(update={"no_speech_timeout_s": max(0.5, deadline - now)})
        pre_roll = self.ring.read_last(round(HOTKEY_PRE_ROLL_S * SAMPLE_RATE))
        request.recorder = CommandRecorder(
            self._stt, pre_roll, vad, self._wake_pattern, on_end=on_end, check_pre_roll=False
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._set_request_state(request, State.LISTENING)
        try:
            if await ended is EndReason.NO_SPEECH:
                logger.info("Réponse orale : rien entendu en {:.0f} s.", timeout_s)
                return ""
            if self._listening is request:
                self._set_listening(None)
            self._set_request_state(request, State.TRANSCRIBING)
            transcript = await request.recorder.transcribe()
            text, _ = strip_wake_word(clean_transcript(transcript.text), self._wake_pattern)
            logger.info("Réponse orale : « {} »", text)
            return text
        finally:
            self._finish(request)

    def _start_listening(
        self, event: WakeDetected, now: float, barged: _Request | None = None
    ) -> None:
        if self._stt is None or self._vad is None or self._audio is None:
            return
        hotkey = event.source == "hotkey"
        if hotkey:
            samples = round(HOTKEY_PRE_ROLL_S * SAMPLE_RATE)
        else:  # pré-roll complet avant la trame de détection + les trames arrivées depuis
            samples = round(self.config.audio.pre_roll_s * SAMPLE_RATE)
            samples += self._written - self._wake_written
        pre_roll = self.ring.read_last(min(samples, self.ring.capacity))
        if self._wake_executor is not None:  # même thread que is_speech : ordre garanti
            self._wake_executor.submit(self._vad.reset)
        request = _Request(
            id=next(self._ids),
            source=event.source,
            wake_at=now,
            wake_frame_at=event.ts,
            barged=barged,
        )
        request.recorder = CommandRecorder(
            self._stt,
            pre_roll,
            self.config.vad,
            self._wake_pattern,
            on_end=functools.partial(self._on_recorder_end, request),
            check_pre_roll=not hotkey,
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._set_request_state(request, State.LISTENING)
        logger.debug(
            "Commande #{} : écoute, {:.1f} s de pré-roll.", request.id, len(pre_roll) / SAMPLE_RATE
        )

    def _on_recorder_end(self, request: _Request, reason: EndReason) -> None:
        request.end_at = time.monotonic()
        request.reason = reason
        if self._listening is request:
            self._set_listening(None)  # micro libre : une autre commande peut commencer
        if reason is EndReason.NO_SPEECH:
            if request.source == FOLLOW_UP_SOURCE:
                logger.info("Fenêtre de suivi terminée : rien de dit.")
            else:
                logger.info("Écoute terminée : {}.", reason)
            self._play("end")
            self._finish(request)
            return
        self._set_request_state(request, State.TRANSCRIBING)
        request.task = self._spawn(self._process(request), f"commande #{request.id}")

    def _start_follow_up(self) -> None:
        """Fenêtre de suivi : écoute `agent.follow_up_s` sans mot d'activation, seulement si
        plus rien n'est en cours (sinon la dernière réponse à finir l'ouvrira)."""
        seconds = self.config.agent.follow_up_s
        if seconds <= 0 or not self._can_listen or self._listening is not None or self._active:
            return
        if self._speaker is not None and self._speaker.busy:
            return
        assert self._stt is not None and self._vad is not None
        now = time.monotonic()
        if self._wake_executor is not None:
            self._wake_executor.submit(self._vad.reset)
        request = _Request(
            id=next(self._ids),
            source=FOLLOW_UP_SOURCE,
            wake_at=now,
            wake_frame_at=now,
            state=State.FOLLOW_UP,
        )
        request.recorder = CommandRecorder(
            self._stt,
            self.ring.read_last(round(FOLLOW_UP_PRE_ROLL_S * SAMPLE_RATE)),
            self.config.vad.model_copy(update={"no_speech_timeout_s": seconds}),
            self._wake_pattern,
            on_end=functools.partial(self._on_recorder_end, request),
            check_pre_roll=False,
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._refresh_state()
        logger.debug("Commande #{} : fenêtre de suivi ({:.0f} s).", request.id, seconds)

    # -- traitement d'une commande (plusieurs en parallèle) --------------------------------

    def _spawn(self, coro: object, name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        self._request_tasks.add(task)
        task.add_done_callback(self._request_tasks.discard)
        return task

    async def _process(self, request: _Request) -> None:
        try:
            text = await self._transcribe(request)
            if text and self._cancel_requested(request, text):
                return
            if text and self._agent is not None and self._speaker is not None:
                await self._respond(request, text)
        except asyncio.CancelledError:
            if request.stream is not None:
                request.stream.cancel()  # plus un mot de cette réponse
            raise
        except Exception:
            logger.exception("Commande #{} : erreur.", request.id)
            self._play("error")
        finally:
            if request.stream is not None:
                request.stream.close()
            self._finish(request)
        if request.follow_up:
            self._start_follow_up()

    def _cancel_requested(self, request: _Request, text: str) -> bool:
        """« stop », « annule »... : True si la commande est une annulation, traitée ici
        (sans LLM). Cible : la commande dont la réponse vient d'être coupée si elle tourne
        encore, sinon la dernière commande en cours."""
        if not is_cancel_command(text, self.config.agent.cancel_phrases):
            return False
        running = [
            r
            for r in self._active.values()
            if r is not request and r.task is not None and not r.task.done()
        ]
        barged = request.barged
        target = barged if barged is not None and barged in running else None
        if target is None and running:
            target = max(running, key=lambda r: r.id)
        if target is not None:
            assert target.task is not None
            target.task.cancel()
            logger.info(
                "Commande #{} : « {} » -> commande #{} annulée.", request.id, text, target.id
            )
        elif barged is not None or request.source == FOLLOW_UP_SOURCE:
            logger.info("Commande #{} : « {} » -> rien d'autre à annuler.", request.id, text)
        else:
            return False  # rien à annuler : « stop » est une vraie commande (musique...)
        self._play("end")
        return True

    async def _transcribe(self, request: _Request) -> str:
        assert request.recorder is not None
        recorder = request.recorder
        transcript = await recorder.transcribe()
        request.marks["stt_ready"] = time.monotonic()
        raw = transcript.text
        text, found = strip_wake_word(clean_transcript(raw), self._wake_pattern)
        logger.info(
            "Mesures : fin_parole +{:.0f} ms après le bip, stt_prêt +{:.0f} ms après "
            "fin_parole ({}), Whisper {:.2f} s pour {:.1f} s d'audio, fin : {}.",
            (request.end_at - request.wake_at) * 1000,
            (request.marks["stt_ready"] - request.end_at) * 1000,
            "spéculatif" if recorder.used_speculative else "direct",
            transcript.latency_s,
            transcript.duration_s,
            request.reason,
        )
        if request.source != "hotkey" and not found:
            logger.debug("Mot d'activation absent du texte : rien de retiré.")
        if not text:
            logger.info("Rien compris (texte brut : « {} »).", raw)
            self._play("end")
            return ""
        if request.source == FOLLOW_UP_SOURCE:
            min_confidence = self.config.agent.follow_up_min_confidence
            rejection = follow_up_rejection(text, transcript, min_confidence)
            if rejection is not None:
                logger.info("Fenêtre de suivi : « {} » ignorée ({}).", text, rejection)
                self._play("end")
                return ""
            logger.info(
                "Fenêtre de suivi : « {} » retenue ({}).", text, describe_confidence(transcript)
            )
        logger.success("Commande #{} : « {} » (texte brut : « {} »)", request.id, text, raw)
        self.commands.append(text)
        return text

    async def _respond(self, request: _Request, text: str) -> None:
        assert self._agent is not None and self._speaker is not None
        self._set_request_state(request, State.THINKING)
        stream = self._speaker.open(label=f"#{request.id}")
        request.stream = stream
        stream.on_first_sound = functools.partial(self._on_first_sound, request)
        gated = request.source == FOLLOW_UP_SOURCE and self.config.agent.follow_up_use_llm_gate
        if not gated:  # suivi jugé par le LLM : pas de son tant qu'on ne sait pas si on répond
            self._spawn(self._thinking_sound(request), f"réflexion #{request.id}")
        result = await self._agent.respond(
            text,
            stream,
            label=f"#{request.id}",
            on_phase=functools.partial(self._on_agent_phase, request),
            listen=functools.partial(self._listen_for, stream) if self._can_listen else None,
            follow_up=gated,
        )
        request.marks["answered"] = time.monotonic()  # le LLM a fini ; reste à tout dire
        stream.close()
        await stream.wait()
        request.marks["done"] = time.monotonic()
        if result.outcome == "ignored":  # pas pour MIKATA : bip de fin, retour au repos
            self._play("end")
            return
        request.follow_up = not stream.cancelled
        if result.answer:
            self.answers.append(result.answer)
        self._log_response(request, result)

    async def _listen_for(self, stream: SpeechStream, timeout_s: float) -> str:
        """Réponse orale pour une commande. Si sa réponse a été coupée (barge-in), la
        question ou l'annonce n'a pas été entendue : réponse « stop », qui refuse une
        confirmation et annule une extinction (jamais l'inverse)."""
        if stream.cancelled:
            logger.info("Réponse coupée avant l'écoute : vaut « {} ».", INTERRUPTED_ANSWER)
            return INTERRUPTED_ANSWER
        return await self.listen_once(timeout_s)

    def _on_first_sound(self, request: _Request) -> None:
        request.marks["first_sound"] = time.monotonic()
        self._set_request_state(request, State.SPEAKING)

    def _on_agent_phase(self, request: _Request, phase: str) -> None:
        """Outils en cours (ACTING), puis retour au LLM (THINKING, ou SPEAKING s'il a déjà
        parlé)."""
        if phase == "acting":
            self._set_request_state(request, State.ACTING)
        elif request.stream is not None and request.stream.first_sound_at is not None:
            self._set_request_state(request, State.SPEAKING)
        else:
            self._set_request_state(request, State.THINKING)

    async def announce(self, text: str) -> None:
        """Annonce hors commande (minuteur, rappel) : son « timer » puis phrase dite, à la
        suite des réponses en cours."""
        self._play("timer")
        if self._speaker is None:
            return
        stream = self._speaker.open(label="annonce")
        stream.say(text)
        stream.close()
        await stream.wait()

    async def _thinking_sound(self, request: _Request) -> None:
        """Petit son si rien n'est joué `thinking_sound_after_s` après la fin de parole (§2)."""
        delay = request.end_at + self.config.agent.thinking_sound_after_s - time.monotonic()
        await asyncio.sleep(max(0.0, delay))
        stream = request.stream
        if request.id in self._active and stream is not None and stream.first_sound_at is None:
            self._play("thinking")

    def _log_response(self, request: _Request, result: TurnResult) -> None:
        def since_end(t: float | None) -> str:
            return "—" if t is None else f"+{(t - request.end_at) * 1000:.0f} ms"

        stream = request.stream
        first_sound = stream.first_sound_at if stream is not None else None
        after_syllable = (
            f"{(first_sound - request.last_syllable_at) * 1000:.0f} ms"
            if first_sound is not None
            else "—"
        )
        logger.info(
            "Réponse #{} : premier son ≈ {} après la dernière syllabe, réflexion {} (depuis "
            "fin_parole : stt_prêt {}, llm_premier_token {}, premier_son {}, total {}) ; "
            "chemin : {} ; issue : {}.",
            request.id,
            after_syllable,
            result.effort_label,
            since_end(request.marks.get("stt_ready")),
            since_end(result.first_token_at),
            since_end(first_sound),
            since_end(request.marks.get("done")),
            result.path,
            result.outcome,
        )
        if result.answer:
            logger.info("Réponse #{} : « {} »", request.id, result.answer)

    def _finish(self, request: _Request) -> None:
        if self._listening is request:
            self._set_listening(None)
        if request.recorder is not None:
            request.recorder.cancel()
        self._active.pop(request.id, None)
        self._refresh_state()

    def _play(self, earcon: str) -> None:
        if self._player is not None and earcon in self._earcons:
            self._player.play_nowait(self._earcons[earcon])

    # -- écritures disque (thread mikata-io) -----------------------------------------------

    def _submit_io(self, fn: Callable[..., None], *args: object) -> None:
        if self._io_executor is not None:
            self._io_executor.submit(self._guarded, fn, *args)

    @staticmethod
    def _guarded(fn: Callable[..., None], *args: object) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("Erreur d'écriture sur le disque.")

    def _save_wake_log(self, audio: np.ndarray, score: float, now: datetime) -> None:
        path = save_detection(self._wake_logs_dir, audio, score, now)
        logger.debug("Audio de détection : {}", path)
        self._purge_wake_logs()

    def _purge_wake_logs(self) -> None:
        count = purge_older_than(self._wake_logs_dir, self.config.wake.log_days)
        if count:
            logger.info("{} ancien(s) audio(s) de détection mis à la corbeille.", count)

    def _save_negative(self, clip: np.ndarray, now: datetime) -> None:
        from mikata.audio.wavfile import write_wav

        write_wav(NegativeCollector.path_for(self._negatives_dir, now), clip, SAMPLE_RATE)

    # -- statistiques -----------------------------------------------------------------------

    async def _stats_loop(self) -> None:
        assert self._cpu is not None
        interval = self.config.logging.stats_interval_s
        while True:
            await asyncio.sleep(interval)
            one_core, total = self._cpu.sample()
            dropped = self._audio.dropped_frames if self._audio else 0
            overflows = self._audio.overflows if self._audio else 0
            logger.info(
                "CPU du processus : {:.1f} % d'un cœur ({:.2f} % du PC) ; "
                "détections : {} ; trames perdues : {} ; débordements micro : {}",
                one_core,
                total,
                self.detections,
                dropped,
                overflows,
            )
