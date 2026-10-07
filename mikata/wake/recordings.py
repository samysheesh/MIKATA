"""Enregistrements liés au mot d'activation (§4).

- `data/wake_logs/` : 2 s d'audio à chaque détection, nom = date + score (purge après N jours).
- `data/negatives/` : mode `collect_negatives`, extraits de 10 s de la vie courante.
La purge envoie à la corbeille (send2trash), jamais de suppression définitive.
Les fonctions d'écriture sont appelées dans un executor, jamais dans la boucle asyncio.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.audio.wavfile import write_wav

NEGATIVE_CLIP_S = 10.0


def detection_path(directory: Path, score: float, now: datetime) -> Path:
    return directory / f"{now:%Y%m%d-%H%M%S-%f}_score{score:.2f}.wav"


def save_detection(directory: Path, audio: np.ndarray, score: float, now: datetime) -> Path:
    path = detection_path(directory, score, now)
    write_wav(path, audio, SAMPLE_RATE)
    return path


def purge_older_than(
    directory: Path,
    days: float,
    now: float | None = None,
    trash: Callable[[str], None] | None = None,
) -> int:
    """Envoie à la corbeille les .wav de `directory` (récursif) plus vieux que `days` jours."""
    if not directory.is_dir():
        return 0
    if trash is None:
        from send2trash import send2trash as trash
    limit = (time.time() if now is None else now) - days * 86_400
    count = 0
    for path in directory.rglob("*.wav"):
        try:
            if path.stat().st_mtime < limit:
                trash(str(path))
                count += 1
        except OSError as exc:
            logger.warning("Purge impossible pour {} : {}", path, exc)
    return count


class NegativeCollector:
    """Accumule les trames et rend un extrait complet toutes les 10 s."""

    def __init__(self, clip_s: float = NEGATIVE_CLIP_S, sample_rate: int = SAMPLE_RATE) -> None:
        self._clip_samples = round(clip_s * sample_rate)
        self._parts: list[np.ndarray] = []
        self._count = 0

    def add(self, frame: np.ndarray) -> np.ndarray | None:
        self._parts.append(frame)
        self._count += len(frame)
        if self._count < self._clip_samples:
            return None
        clip = np.concatenate(self._parts)
        self._parts = [clip[self._clip_samples :]] if len(clip) > self._clip_samples else []
        self._count = len(clip) - self._clip_samples
        return clip[: self._clip_samples]

    @staticmethod
    def path_for(directory: Path, now: datetime) -> Path:
        return directory / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}.wav"


def estimate_tail_silence_s(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float | None:
    """Durée entre la fin de la voix et la fin de `audio` (estimation par énergie).

    Sert à mesurer le délai « fin du mot → détection ». None si aucune voix nette.
    """
    win = sample_rate // 100  # fenêtres de 10 ms
    n = len(audio) // win
    if n < 10:
        return None
    x = audio[: n * win].astype(np.float32).reshape(n, win)
    rms = np.sqrt((x**2).mean(axis=1))
    floor = float(np.percentile(rms, 20))
    peak = float(rms.max())
    if peak < max(4 * floor, 300.0):  # pas de voix nettement au-dessus du bruit de fond
        return None
    threshold = max(4 * floor, 0.02 * peak)  # -34 dB : garde les consonnes finales (« s »)
    last = int(np.nonzero(rms >= threshold)[0][-1])
    return (n - 1 - last) * win / sample_rate
