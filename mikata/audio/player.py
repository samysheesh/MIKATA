"""Lecture dans le casque (§8) : flux de sortie WASAPI ouvert en permanence.

Garder le flux ouvert évite les 20 à 50 ms d'ouverture du périphérique à chaque son :
le bip part au bloc suivant. Le callback ne fait que recopier des échantillons déjà prêts.
La phase 3 ajoutera la file de phrases du TTS au-dessus de `play()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

import numpy as np
from loguru import logger

from mikata.audio.devices import DeviceInfo


class _Playback:
    __slots__ = ("done", "loop", "pcm", "pos")

    def __init__(
        self,
        pcm: np.ndarray,
        loop: asyncio.AbstractEventLoop | None,
        done: asyncio.Future[None] | None,
    ) -> None:
        self.pcm = pcm
        self.pos = 0
        self.loop = loop
        self.done = done

    def finish(self) -> None:
        if self.loop is None or self.done is None:
            return
        with contextlib.suppress(RuntimeError):  # boucle fermée
            self.loop.call_soon_threadsafe(_resolve, self.done)


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class AudioPlayer:
    """Implémente l'interface `Player` (tts/base.py) + `play_nowait` pour les sons courts."""

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device
        self.sample_rate = int(device.default_samplerate)
        self._channels = min(device.max_output_channels, 2)
        self._current: _Playback | None = None
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    @property
    def output_latency_s(self) -> float:
        """Latence de sortie annoncée par PortAudio (0 tant que le flux est fermé)."""
        return float(self._stream.latency) if self._stream is not None else 0.0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        self._loop = loop
        self._stream = sd.OutputStream(
            device=self.device.index,
            samplerate=self.sample_rate,
            channels=self._channels,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Sortie : « {} » ({}, {} Hz, latence {:.0f} ms).",
            self.device.name,
            self.device.host_api,
            self.sample_rate,
            self.output_latency_s * 1000,
        )

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Thread PortAudio : recopie d'échantillons préparés, rien d'autre.
        current = self._current
        if current is None:
            outdata.fill(0)
            return
        pos = current.pos
        chunk = current.pcm[pos : pos + frames]
        n = len(chunk)
        outdata[:n] = chunk[:, None]
        outdata[n:] = 0
        current.pos = pos + n
        if current.pos >= len(current.pcm):
            if self._current is current:
                self._current = None
            current.finish()

    def _replace(self, playback: _Playback | None) -> None:
        with self._lock:
            previous, self._current = self._current, playback
        if previous is not None:
            previous.finish()

    def play_nowait(self, pcm: np.ndarray) -> None:
        """Joue immédiatement `pcm` (float32 mono à `sample_rate`), coupe le son en cours."""
        self._replace(_Playback(np.ascontiguousarray(pcm, dtype=np.float32), None, None))

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Joue `pcm` et attend la fin (ou un `stop()`)."""
        if sample_rate != self.sample_rate:
            import soxr

            pcm = soxr.resample(pcm.astype(np.float32), sample_rate, self.sample_rate)
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        self._replace(_Playback(np.ascontiguousarray(pcm, dtype=np.float32), loop, done))
        await done

    def stop(self) -> None:
        """Coupe immédiatement le son en cours (barge-in)."""
        self._replace(None)

    def close(self) -> None:
        self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Erreur à la fermeture du flux de sortie.")
