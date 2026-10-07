"""Interfaces de synthèse et de lecture audio (docs/ARCHITECTURE.md §8 et §9)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mikata.config import ConfigError


@runtime_checkable
class TTS(Protocol):
    sample_rate: int

    async def synthesize(self, text: str) -> np.ndarray:
        """PCM float32 mono à `sample_rate`."""
        ...


@runtime_checkable
class Player(Protocol):
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None: ...

    def stop(self) -> None:
        """Arrêt immédiat (barge-in)."""
        ...


KEEP_START_MS = 10  # Piper commence par ~160 ms de silence : autant de latence perçue
KEEP_END_MS = 200  # garde une pause naturelle entre deux fragments


def trim_silence(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Retire le silence de début (sauf `KEEP_START_MS`) et plafonne celui de fin."""
    if len(pcm) == 0:
        return pcm
    threshold = max(0.005, 0.02 * float(np.abs(pcm).max()))
    loud = np.nonzero(np.abs(pcm) > threshold)[0]
    if len(loud) == 0:
        return pcm[:0]
    start = max(0, int(loud[0]) - KEEP_START_MS * sample_rate // 1000)
    end = min(len(pcm), int(loud[-1]) + 1 + KEEP_END_MS * sample_rate // 1000)
    return pcm[start:end]


class TtsModelMissingError(ConfigError):
    """Modèle de voix introuvable."""


class LocalTTS:
    """Socle des moteurs locaux : executor dédié à 1 worker (§3), chargement explicite.

    Les sous-classes implémentent `load()` et `_synthesize(text) -> float32 mono`.
    """

    name = "tts"
    sample_rate = 22_050

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-tts")

    def load(self) -> None:
        raise NotImplementedError

    def _synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def synthesize_sync(self, text: str) -> np.ndarray:
        """Synthèse bloquante (hors de la boucle asyncio), silences de bord retirés."""
        pcm = np.ascontiguousarray(self._synthesize(text), dtype=np.float32)
        return trim_silence(pcm, self.sample_rate)

    async def synthesize(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.synthesize_sync, text)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def require_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise TtsModelMissingError(
            "Modèle(s) de voix introuvable(s) :\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nLance : uv run python scripts/download_models.py"
        )


def cpu_session(model: Path, threads: int) -> Any:
    """Session ONNX sur CPU limitée à `threads` threads (§10 : 4 max pour le TTS)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )


def create_tts(config: Any, backend: str | None = None) -> tuple[LocalTTS, str]:
    """Moteur TTS de la config (ou `backend`), non chargé, et son identifiant de voix
    (clé du cache de phrases : moteur + voix + vitesse)."""
    tts_cfg = config.tts
    backend = backend or tts_cfg.backend
    if backend == "piper":
        from mikata.tts.piper_tts import PiperTTS

        voice_id = f"piper|{tts_cfg.voice.name}|{tts_cfg.speed}"
        return PiperTTS(config.resolve(tts_cfg.voice), tts_cfg.speed, tts_cfg.cpu_threads), voice_id
    if backend == "kokoro":
        from mikata.tts.kokoro_tts import KokoroTTS

        voice_id = f"kokoro|{tts_cfg.kokoro_model.name}|{tts_cfg.kokoro_voice}|{tts_cfg.speed}"
        tts = KokoroTTS(
            config.resolve(tts_cfg.kokoro_model),
            config.resolve(tts_cfg.kokoro_voices),
            tts_cfg.kokoro_voice,
            tts_cfg.speed,
            tts_cfg.cpu_threads,
        )
        return tts, voice_id
    raise ValueError(f"moteur TTS inconnu : {backend}")
