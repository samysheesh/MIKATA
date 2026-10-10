"""Transcription Whisper via `mlx-whisper` sur Apple Silicon (macOS, option B).

Mêmes signatures que `whisper_local.WhisperSTT` (interface `STT`, §5 et §9) : un seul
calcul à la fois via un executor dédié, langue forcée à « fr », prompt initial
`stt.initial_prompt` complété des noms d'applications (`stt.prompt_apps`).

Modèle : `large-v3-turbo` porté en MLX (`mlx-community/whisper-large-v3-turbo`).
Import paresseux : `mlx_whisper` n'est importé que dans `load()` (dépendance macOS).
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

# Modèle faster-whisper (config) -> dépôt MLX équivalent.
MLX_MODEL_BY_NAME: dict[str, str] = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

WARMUP_S = 1.0


class SttUnavailableError(StartupError):
    """Whisper MLX ne peut pas tourner (paquet absent, modèle illisible)."""


class MlxWhisperSTT:
    """Implémente l'interface `STT` sur Apple Silicon. Un executor dédié à 1 worker :
    une seule transcription à la fois, la boucle asyncio n'est jamais bloquée."""

    def __init__(
        self,
        cfg: SttConfig,
        download_root: Path,
        vocabulary: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        """`vocabulary` : noms des applications installées, lus à chaque transcription
        (liste en mémoire, jamais bloquante) si `stt.prompt_apps`."""
        from mikata.stt.whisper_local import build_prompt

        self.cfg = cfg
        self.download_root = download_root
        self.vocabulary = vocabulary
        self._build_prompt = build_prompt
        self._model_path: str = MLX_MODEL_BY_NAME.get(
            cfg.model, f"mlx-community/whisper-{cfg.model}"
        )
        self._ready = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-stt")
        self._prompt_key: tuple[str, ...] | None = None
        self._prompt = cfg.initial_prompt

    @property
    def prompt(self) -> str:
        """Prompt initial de la prochaine transcription (recalculé si les applis changent)."""
        if not self.cfg.prompt_apps or self.vocabulary is None:
            return self.cfg.initial_prompt
        try:
            names = tuple(self.vocabulary())
        except Exception as exc:  # vocabulaire facultatif
            logger.debug("Noms d'applications indisponibles pour Whisper : {!r}", exc)
            return self._prompt
        if names != self._prompt_key:
            self._prompt_key = names
            self._prompt = self._build_prompt(
                self.cfg.initial_prompt, names, self.cfg.prompt_max_chars
            )
            logger.debug("Prompt Whisper ({} caractères) : {}", len(self._prompt), self._prompt)
        return self._prompt

    @property
    def loaded(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Vérifie `mlx_whisper` et préchauffe (bloquant, plusieurs secondes)."""
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise SttUnavailableError(
                "Whisper MLX indisponible : installe « mlx-whisper » "
                "(macOS Apple Silicon uniquement)."
            ) from exc
        start = time.perf_counter()
        self.transcribe_sync(np.zeros(round(WARMUP_S * SAMPLE_RATE), dtype=np.int16))
        self._ready = True
        logger.info(
            "Whisper {} prêt via MLX ({}) en {:.1f} s.",
            self.cfg.model,
            self._model_path,
            time.perf_counter() - start,
        )

    def transcribe_sync(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        """Transcription bloquante (int16 ou float32 mono 16 kHz) ; hors de la boucle asyncio.
        `prompt` : prompt initial (défaut : `stt.initial_prompt`)."""
        import mlx_whisper

        start = time.perf_counter()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        result: dict[str, Any] = mlx_whisper.transcribe(  # type: ignore[attr-defined]
            np.ascontiguousarray(pcm, dtype=np.float32),
            path_or_hf_repo=self._model_path,
            language="fr",
            initial_prompt=(self.cfg.initial_prompt if prompt is None else prompt) or None,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = str(result.get("text", "")).strip()
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
        self._ready = False
