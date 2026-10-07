"""Capture micro WASAPI via sounddevice (§3).

Chaîne : callback PortAudio (copie du bloc + horodatage, rien d'autre)
→ file thread-safe → thread `mikata-audio` (mono, 16 kHz, int16, trames de 80 ms)
→ file asyncio lue par l'orchestrateur.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from mikata.audio.base import FRAME_SAMPLES, SAMPLE_RATE, Frame
from mikata.audio.devices import DeviceInfo
from mikata.audio.wavfile import float_to_int16

BLOCK_S = 0.02  # blocs PortAudio de 20 ms : faible gigue, coût négligeable
MAX_QUEUED_FRAMES = 250  # 20 s ; au-delà on jette les plus anciennes (consommateur bloqué)
RESAMPLE_QUALITY = "LQ"  # ~6 ms de retard à 48 -> 16 kHz (HQ : ~21 ms), suffisant pour la voix


@dataclass(frozen=True, eq=False)
class CapturedFrame:
    samples: Frame  # int16 mono 16 kHz, 1 280 échantillons
    captured_at: float  # time.monotonic() à l'arrivée du dernier bloc de la trame


class FrameAssembler:
    """Blocs float32 (n, canaux) au taux natif → trames int16 mono 16 kHz de 80 ms."""

    def __init__(self, in_rate: int, channels: int, channel: int | None = None) -> None:
        if channel is not None and not 1 <= channel <= channels:
            raise ValueError(f"canal {channel} absent (le périphérique en a {channels})")
        self._channel = channel
        self._resampler: Any = None
        if in_rate != SAMPLE_RATE:
            import soxr

            self._resampler = soxr.ResampleStream(
                in_rate, SAMPLE_RATE, 1, dtype="float32", quality=RESAMPLE_QUALITY
            )
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[Frame]:
        if block.ndim == 1:
            mono = block
        elif self._channel is not None:
            mono = block[:, self._channel - 1]
        else:
            mono = block.mean(axis=1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
        self._pending = np.concatenate((self._pending, mono))
        count = len(self._pending) // FRAME_SAMPLES
        frames = [
            float_to_int16(self._pending[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
            for i in range(count)
        ]
        self._pending = self._pending[count * FRAME_SAMPLES :]
        return frames


class AudioCapture:
    def __init__(self, device: DeviceInfo, channel: int | None = None) -> None:
        self.device = device
        self._channel = channel
        self._channels = min(device.max_input_channels, 8)
        self._rate = int(device.default_samplerate)  # WASAPI partagé : taux du mixeur Windows
        self._raw: queue.SimpleQueue[tuple[np.ndarray, float, bool]] = queue.SimpleQueue()
        self._frames: asyncio.Queue[CapturedFrame] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.dropped_frames = 0
        self.overflows = 0

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Thread PortAudio : copie + horodatage uniquement (jamais de calcul ici).
        self._raw.put((indata.copy(), time.monotonic(), bool(status)))

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        self._loop = loop
        assembler = FrameAssembler(self._rate, self._channels, self._channel)
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._process, args=(assembler,), name="mikata-audio", daemon=True
        )
        self._thread.start()
        self._stream = sd.InputStream(
            device=self.device.index,
            samplerate=self._rate,
            channels=self._channels,
            dtype="float32",
            blocksize=round(self._rate * BLOCK_S),
            latency="low",
            callback=self._callback,
            finished_callback=self._finished,
        )
        self._stream.start()
        canal = f"canal {self._channel}" if self._channel else "moyenne des canaux"
        logger.info(
            "Micro : « {} » ({}, {} Hz, {} canal(aux), {}).",
            self.device.name,
            self.device.host_api,
            self._rate,
            self._channels,
            canal,
        )

    def _finished(self) -> None:
        if not self._stopping.is_set():
            logger.error("Le flux micro s'est arrêté (périphérique débranché ?).")

    def _process(self, assembler: FrameAssembler) -> None:
        while not self._stopping.is_set():
            try:
                block, captured_at, flagged = self._raw.get(timeout=0.2)
            except queue.Empty:
                continue
            if flagged:
                self.overflows += 1
            try:
                frames = assembler.push(block)
            except Exception:
                logger.exception("Erreur de traitement audio.")
                continue
            for frame in frames:
                self._deliver(CapturedFrame(frame, captured_at))

    def _deliver(self, frame: CapturedFrame) -> None:
        assert self._loop is not None
        try:
            self._loop.call_soon_threadsafe(self._enqueue, frame)
        except RuntimeError:  # boucle fermée : arrêt en cours
            self._stopping.set()

    def _enqueue(self, frame: CapturedFrame) -> None:
        if self._frames.qsize() >= MAX_QUEUED_FRAMES:
            self._frames.get_nowait()
            self.dropped_frames += 1
        self._frames.put_nowait(frame)

    async def read(self) -> CapturedFrame:
        return await self._frames.get()

    def clear(self) -> int:
        """Oublie les trames en attente (ex. : l'audio d'avant un appui sur Entrée)."""
        count = 0
        while not self._frames.empty():
            self._frames.get_nowait()
            count += 1
        return count

    def stop(self) -> None:
        """Ferme le flux et attend le thread de traitement. Idempotent."""
        self._stopping.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Erreur à la fermeture du flux micro.")
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
