"""Lecture dans le casque (§8).

`AudioPlayer` : flux de sortie ouvert en permanence via sounddevice (WASAPI sous
Windows, CoreAudio sous macOS ; le son part au bloc suivant), avec deux canaux mixés :
- **parole** (`play` / `stop`) : une phrase à la fois, coupable immédiatement (barge-in) ;
- **effets** (`play_nowait`) : bips et sons d'interface, joués **par-dessus** la parole sans
  l'interrompre (avec des commandes en parallèle, le bip d'une nouvelle commande ne doit pas
  couper la réponse de la précédente).
Le callback ne fait que recopier et additionner des échantillons déjà prêts.

`Speaker` : file de phrases (§8). Chaque commande ouvre un `SpeechStream` ; la phrase N+1 est
synthétisée pendant que N est jouée. Les flux sont joués dans l'ordre d'ouverture : deux voix
ne se superposent jamais, mais une commande n'attend jamais l'autre pour réfléchir.

Ajouts phase 6 : `Speaker.hold()` / `release()` : pendant que l'utilisateur parle (écoute
d'une commande, d'une réponse ou fenêtre de suivi), aucune nouvelle phrase ne commence ; la
phrase en cours n'est pas coupée (c'est le rôle du barge-in, `SpeechStream.cancel`).
`Speaker.busy` : une réponse est en cours ou en attente.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from loguru import logger

from mikata.audio.devices import DeviceInfo
from mikata.tts.base import TTS


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
    """Implémente l'interface `Player` (tts/base.py) + `play_nowait` pour les effets."""

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device
        self.sample_rate = int(device.default_samplerate)
        self._channels = min(device.max_output_channels, 2)
        self._speech: _Playback | None = None
        self._effect: _Playback | None = None
        self._stream: Any = None

    @property
    def output_latency_s(self) -> float:
        """Latence de sortie annoncée par PortAudio (0 tant que le flux est fermé)."""
        return float(self._stream.latency) if self._stream is not None else 0.0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

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

    @staticmethod
    def _advance(playback: _Playback, frames: int) -> np.ndarray:
        chunk = playback.pcm[playback.pos : playback.pos + frames]
        playback.pos += len(chunk)
        return chunk

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Thread PortAudio : recopie (et addition des deux canaux), rien d'autre.
        outdata.fill(0)
        speech, effect = self._speech, self._effect
        if speech is not None:
            chunk = self._advance(speech, frames)
            outdata[: len(chunk)] = chunk[:, None]
            if speech.pos >= len(speech.pcm):
                if self._speech is speech:
                    self._speech = None
                speech.finish()
        if effect is not None:
            chunk = self._advance(effect, frames)
            outdata[: len(chunk)] += chunk[:, None]
            if effect.pos >= len(effect.pcm) and self._effect is effect:
                self._effect = None

    def play_nowait(self, pcm: np.ndarray) -> None:
        """Effet sonore (float32 mono à `sample_rate`), mixé par-dessus la parole."""
        self._effect = _Playback(np.ascontiguousarray(pcm, dtype=np.float32), None, None)

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Joue une phrase et attend sa fin (ou un `stop()`). Coupe la phrase en cours."""
        if sample_rate != self.sample_rate:
            import soxr

            pcm = soxr.resample(pcm.astype(np.float32), sample_rate, self.sample_rate)
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        previous, self._speech = (
            self._speech,
            _Playback(np.ascontiguousarray(pcm, dtype=np.float32), loop, done),
        )
        if previous is not None:
            previous.finish()
        await done

    def stop(self) -> None:
        """Coupe immédiatement la parole en cours (barge-in) ; les effets continuent."""
        previous, self._speech = self._speech, None
        if previous is not None:
            previous.finish()

    def close(self) -> None:
        self.stop()
        self._effect = None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Erreur à la fermeture du flux de sortie.")


# -- file de phrases ----------------------------------------------------------------------


class SpeechPlayer(Protocol):
    @property
    def output_latency_s(self) -> float: ...
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None: ...
    def stop(self) -> None: ...


class PhraseSource(Protocol):
    sample_rate: int

    def lookup(self, text: str) -> np.ndarray | None: ...


_END = object()


class SpeechStream:
    """Les phrases d'une commande, dans l'ordre. `say` ne bloque jamais."""

    def __init__(self, speaker: Speaker, label: str) -> None:
        self._speaker = speaker
        self.label = label
        self._items: asyncio.Queue[Any] = asyncio.Queue()
        self._done = asyncio.Event()
        self._tasks: list[asyncio.Task[np.ndarray]] = []
        self.cancelled = False
        self.closed = False
        self.fragments = 0
        self.first_sound_at: float | None = None
        self.on_first_sound: Callable[[], None] | None = None
        self.on_say: Callable[[str], None] | None = None  # ajout phase 8 : bulle de la mascotte

    def say(self, text: str) -> None:
        """Ajoute une phrase : synthèse lancée tout de suite, lecture à son tour."""
        if self.closed or self.cancelled or not text.strip():
            return
        if self.on_say is not None:
            self.on_say(text)
        cached = self._speaker.cache.lookup(text) if self._speaker.cache else None
        if cached is not None:
            self.say_audio(cached, self._speaker.cache.sample_rate)  # type: ignore[union-attr]
            return
        task = asyncio.create_task(self._speaker.tts.synthesize(text))
        self._tasks.append(task)
        self._items.put_nowait((task, self._speaker.tts))
        self.fragments += 1

    def say_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Ajoute un audio déjà prêt (phrase en cache)."""
        if self.closed or self.cancelled:
            return
        self._items.put_nowait((pcm, sample_rate))
        self.fragments += 1

    def close(self) -> None:
        """Plus aucune phrase à venir."""
        if not self.closed:
            self.closed = True
            self._items.put_nowait(_END)

    def cancel(self) -> None:
        """Abandonne les phrases restantes et coupe celle en cours si c'est la nôtre."""
        if self.cancelled:
            return
        self.cancelled = True
        for task in self._tasks:
            task.cancel()
        if self._speaker.current is self:
            self._speaker.player.stop()
        self.close()

    async def wait(self) -> None:
        await self._done.wait()

    async def drain(self) -> None:
        """Attend que tout ce qui a été dit jusqu'ici soit joué (ajout phase 5 : poser une
        question avant d'écouter la réponse)."""
        if self.closed or self.cancelled:
            await self.wait()
            return
        marker = asyncio.Event()
        self._items.put_nowait(marker)
        await marker.wait()


class Speaker:
    def __init__(self, tts: TTS, player: SpeechPlayer, cache: PhraseSource | None = None) -> None:
        self.tts = tts
        self.player = player
        self.cache = cache
        self.current: SpeechStream | None = None
        self._streams: asyncio.Queue[SpeechStream] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._open: asyncio.Event | None = None  # créé dans la boucle, au premier besoin
        self._holds = 0

    def _gate(self) -> asyncio.Event:
        if self._open is None:
            self._open = asyncio.Event()
            if not self._holds:
                self._open.set()
        return self._open

    def hold(self) -> None:
        """Aucune nouvelle phrase ne commence jusqu'au `release()` correspondant."""
        self._holds += 1
        self._gate().clear()

    def release(self) -> None:
        self._holds = max(0, self._holds - 1)
        if not self._holds:
            self._gate().set()

    @property
    def held(self) -> bool:
        return self._holds > 0

    @property
    def busy(self) -> bool:
        """Une réponse est jouée, ou attend son tour."""
        return self.current is not None or not self._streams.empty()

    def load(self) -> None:
        """Charge le moteur TTS et le cache de phrases (bloquant, au démarrage)."""
        load = getattr(self.tts, "load", None)
        if load is not None:
            load()
        warm = getattr(self.cache, "warm", None)
        if warm is not None:
            warm()

    def open(self, label: str = "") -> SpeechStream:
        stream = SpeechStream(self, label)
        self._streams.put_nowait(stream)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="speaker")
        return stream

    async def _run(self) -> None:
        while True:
            stream = await self._streams.get()
            self.current = stream
            try:
                await self._play_stream(stream)
            finally:
                self.current = None
                stream._done.set()

    async def _play_stream(self, stream: SpeechStream) -> None:
        while True:
            item = await stream._items.get()
            if item is _END:
                return
            if isinstance(item, asyncio.Event):  # SpeechStream.drain : tout ce qui précède est joué
                item.set()
                continue
            if stream.cancelled:
                continue
            try:
                if isinstance(item[0], asyncio.Task):
                    task, tts = item
                    pcm, rate = await task, tts.sample_rate
                else:
                    pcm, rate = item
            except asyncio.CancelledError:
                if stream.cancelled:
                    continue
                raise
            except Exception:
                logger.exception("Synthèse vocale impossible pour un fragment.")
                continue
            if self.held:  # l'utilisateur parle : la phrase attend qu'il ait fini
                await self._gate().wait()
                if stream.cancelled:
                    continue
            if stream.first_sound_at is None:
                stream.first_sound_at = time.monotonic() + self.player.output_latency_s
                if stream.on_first_sound is not None:
                    stream.on_first_sound()
            await self.player.play(pcm, rate)

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.player.stop()
        close = getattr(self.tts, "close", None)
        if close is not None:
            close()
