"""Orchestrateur et machine à états (docs/ARCHITECTURE.md §3).

Phase 1 : capture continue → tampon circulaire → mot d'activation (executor dédié)
→ bip + journal de détection. Le raccourci clavier a le même effet. Les transitions
IDLE → LISTENING → ... arrivent avec les phases 2 à 6.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum
from typing import Protocol

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.audio.capture import CapturedFrame
from mikata.audio.ringbuffer import RingBuffer
from mikata.config import Config
from mikata.events import BusEvent, EventBus, Shutdown, WakeDetected
from mikata.wake.base import WakeWordDetector
from mikata.wake.recordings import (
    NegativeCollector,
    estimate_tail_silence_s,
    purge_older_than,
    save_detection,
)
from mikata.wake.trigger import WakeTrigger

WAKE_LOG_S = 2.0


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"


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
        self.state = State.IDLE

        self.ring = RingBuffer.for_seconds(config.audio.pre_roll_s)
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
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for name, stop in (
            ("raccourci", self._hotkeys.stop if self._hotkeys else None),
            ("micro", self._audio.stop if self._audio else None),
            ("sortie", self._player.close if self._player else None),
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
            if self._collector is not None:
                clip = self._collector.add(captured.samples)
                if clip is not None:
                    self._submit_io(self._save_negative, clip, datetime.now())
            if self._detector is None:
                continue
            score = await loop.run_in_executor(
                self._wake_executor, self._detector.process, captured.samples
            )
            if score >= half_threshold:
                logger.debug("Score mot d'activation : {:.3f}", score)
            if self._trigger.update(score, captured.captured_at):
                self._wake_audio = self.ring.read_last(round(WAKE_LOG_S * SAMPLE_RATE))
                self.bus.publish(WakeDetected(score=score, ts=captured.captured_at))

    async def _handle(self, event: BusEvent) -> None:
        if isinstance(event, WakeDetected):
            self._on_wake(event)
        else:
            logger.debug("Événement non géré en phase 1 : {}", type(event).__name__)

    def _on_wake(self, event: WakeDetected) -> None:
        now = time.monotonic()
        if self._player is not None and "wake" in self._earcons:
            self._player.play_nowait(self._earcons["wake"])
        self.detections += 1
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        pipeline_ms = (now - event.ts) * 1000

        if event.source == "hotkey":
            self._trigger.suppress(now)
            logger.info("Raccourci clavier : bip lancé (+ sortie {:.0f} ms).", out_ms)
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
