"""Transcription Parakeet v3 via `sherpa-onnx` sur Linux (CPU).

Mêmes signatures que `whisper_local.WhisperSTT` (interface `STT`, §5 et §9) : un seul
calcul à la fois via un executor dédié à 1 worker, la boucle asyncio n'est jamais bloquée.

Modèle prébuilt : `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`. Les quatre fichiers
(`encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`) doivent être
présents sous `download_root/parakeet` ; ce module ne les télécharge pas (le script
`scripts/download_models.py` s'en charge). Le décodage tourne sur CPU (`provider="cpu"`).

Parakeet v3 détecte la langue tout seul (25 langues européennes, dont le français) : on ne
force **jamais** `language`. Parakeet ne prend pas de prompt initial ; les propriétés
`prompt` et l'argument `vocabulary` existent uniquement pour respecter la surface publique
de `WhisperSTT`.

Import paresseux : `sherpa_onnx` n'est importé que dans `load()` / `transcribe_sync()`.
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

ASSET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
MODEL_DIR = "parakeet"
WARMUP_S = 1.0


class SttUnavailableError(StartupError):
    """Parakeet sherpa-onnx ne peut pas tourner (paquet ou fichiers du modèle absents)."""


def _import_sherpa() -> Any:
    """Import paresseux de `sherpa_onnx` (dépendance Linux) avec erreur claire."""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise SttUnavailableError(
            "Parakeet sherpa-onnx indisponible : installe l'extra linux "
            "(« uv sync --extra linux », ou « pip install sherpa-onnx »)."
        ) from exc
    return sherpa_onnx


def _model_paths(download_root: Path) -> dict[str, str]:
    """Chemins des quatre assets sous `download_root/parakeet` ; erreur claire s'ils manquent."""
    directory = download_root / MODEL_DIR
    missing = [name for name in ASSET_FILES if not (directory / name).is_file()]
    if missing:
        raise SttUnavailableError(
            f"Fichiers du modèle Parakeet absents dans {directory} : {', '.join(missing)}.\n"
            "Lance « python scripts/download_models.py » pour les récupérer."
        )
    return {
        "encoder": str(directory / "encoder.int8.onnx"),
        "decoder": str(directory / "decoder.int8.onnx"),
        "joiner": str(directory / "joiner.int8.onnx"),
        "tokens": str(directory / "tokens.txt"),
    }


class ParakeetSherpaSTT:
    """Implémente l'interface `STT` sur Linux via `sherpa-onnx` (CPU). Un executor dédié à
    1 worker : une seule transcription à la fois, la boucle asyncio n'est jamais bloquée."""

    def __init__(
        self,
        cfg: SttConfig,
        download_root: Path,
        vocabulary: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        """`vocabulary` est conservé pour respecter la signature de `WhisperSTT` : Parakeet
        n'utilise ni prompt initial ni vocabulaire d'applications."""
        self.cfg = cfg
        self.download_root = download_root
        self.vocabulary = vocabulary
        self._recognizer: Any = None
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
        """Vérifie les assets, construit le recognizer et préchauffe (bloquant)."""
        sherpa_onnx = _import_sherpa()
        paths = _model_paths(self.download_root)
        start = time.perf_counter()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            tokens=paths["tokens"],
            model_type="nemo_transducer",
            provider="cpu",
        )
        self.transcribe_sync(np.zeros(round(WARMUP_S * SAMPLE_RATE), dtype=np.int16))
        self._ready = True
        logger.info(
            "Parakeet v3 prêt via sherpa-onnx ({} / CPU) en {:.1f} s.",
            self.download_root / MODEL_DIR,
            time.perf_counter() - start,
        )

    def transcribe_sync(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        """Transcription bloquante (int16 ou float32 mono 16 kHz) ; hors de la boucle
        asyncio. `prompt` est ignoré (Parakeet n'accepte pas de prompt initial) et n'existe
        que pour respecter la signature de `WhisperSTT`."""
        if self._recognizer is None:
            raise RuntimeError("Parakeet sherpa-onnx n'est pas chargé")
        start = time.perf_counter()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(pcm, dtype=np.float32))
        self._recognizer.decode_stream(stream)  # langue auto-détectée (v3)
        text = str(stream.result.text).strip()
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
        self._recognizer = None
        self._ready = False
