"""Transcription Parakeet v3 via `parakeet-mlx` sur Apple Silicon (macOS).

Mêmes signatures que `whisper_local.WhisperSTT` (interface `STT`, §5 et §9) : un seul
calcul à la fois via un executor dédié à 1 worker, la boucle asyncio n'est jamais bloquée.

Modèle : `mlx-community/parakeet-tdt-0.6b-v3`. Les poids sont récupérés au premier usage
par la bibliothèque depuis le HuggingFace Hub ; ils atterrissent dans le cache local
`~/.cache/huggingface/hub` (variable `HF_HOME` pour le déplacer).

Parakeet v3 détecte la langue tout seul (25 langues européennes, dont le français) : on ne
force **jamais** `language`, contrairement au backend Whisper (`language="fr"`). Parakeet
n'accepte pas non plus de prompt initial ; les propriétés `prompt` et l'argument `vocabulary`
existent uniquement pour respecter la surface publique de `WhisperSTT`.

Import paresseux : `parakeet_mlx` n'est importé que dans `load()` / `transcribe_sync()`
(dépendance macOS Apple Silicon).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.config import StartupError, SttConfig
from mikata.stt.base import Transcript

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WARMUP_S = 1.0


class SttUnavailableError(StartupError):
    """Parakeet MLX ne peut pas tourner (paquet absent, modèle illisible)."""


def _import_parakeet() -> Any:
    """Import paresseux de `parakeet_mlx` (dépendance macOS) avec erreur claire."""
    try:
        import parakeet_mlx
    except ImportError as exc:
        raise SttUnavailableError(
            "Parakeet MLX indisponible : installe l'extra darwin "
            "(« uv sync --extra darwin », ou « pip install parakeet-mlx »)."
        ) from exc
    return parakeet_mlx


class ParakeetMlxSTT:
    """Implémente l'interface `STT` sur Apple Silicon via `parakeet-mlx`. Un executor
    dédié à 1 worker : une seule transcription à la fois, la boucle asyncio n'est jamais
    bloquée."""

    def __init__(
        self,
        cfg: SttConfig,
        download_root: Path,
        vocabulary: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        """`download_root` et `vocabulary` sont conservés pour respecter la signature de
        `WhisperSTT` : Parakeet récupère ses poids via le cache HuggingFace et n'utilise
        ni prompt initial ni vocabulaire d'applications."""
        self.cfg = cfg
        self.download_root = download_root
        self.vocabulary = vocabulary
        self._model: Any = None
        self._ready = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-stt")

    @property
    def prompt(self) -> str:
        """Parakeet v3 ne prend pas de prompt initial : la propriété existe pour respecter
        la surface publique de `WhisperSTT`."""
        return self.cfg.initial_prompt

    @property
    def loaded(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Charge le modèle (téléchargement au premier usage) et le préchauffe (bloquant)."""
        parakeet_mlx = _import_parakeet()
        start = time.perf_counter()
        self._model = parakeet_mlx.from_pretrained(DEFAULT_MODEL)
        self.transcribe_sync(np.zeros(round(WARMUP_S * SAMPLE_RATE), dtype=np.int16))
        self._ready = True
        logger.info(
            "Parakeet v3 prêt via MLX ({}) en {:.1f} s.",
            DEFAULT_MODEL,
            time.perf_counter() - start,
        )

    def transcribe_sync(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        """Transcription bloquante (int16 ou float32 mono 16 kHz) ; hors de la boucle
        asyncio. `prompt` est ignoré (Parakeet n'accepte pas de prompt initial) et n'existe
        que pour respecter la signature de `WhisperSTT`."""
        if self._model is None:
            raise RuntimeError("Parakeet MLX n'est pas chargé")
        import mlx.core as mx

        start = time.perf_counter()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        samples = mx.array(np.ascontiguousarray(pcm, dtype=np.float32))
        result = self._model.transcribe(samples)  # langue auto-détectée (v3)
        text = str(getattr(result, "text", "")).strip()
        return Transcript(
            text=text,
            duration_s=len(audio) / SAMPLE_RATE,
            latency_s=time.perf_counter() - start,
        )

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        loop = asyncio.get_running_loop()
        prompt = self.prompt  # lu dans la boucle asyncio, pas dans le thread STT
        return await loop.run_in_executor(self._executor, self.transcribe_sync, audio, prompt)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._model = None
        self._ready = False
