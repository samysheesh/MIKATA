"""Lecture / écriture de WAV PCM 16 bits mono (bibliothèque standard uniquement)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def float_to_int16(pcm: np.ndarray) -> np.ndarray:
    return np.clip(np.round(pcm * 32768.0), -32768, 32767).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Écrit `samples` (int16, ou float32 dans [-1, 1]) en WAV mono 16 bits.

    Écriture dans un fichier temporaire puis renommage : jamais de WAV tronqué.
    """
    data = samples if samples.dtype == np.int16 else float_to_int16(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.ascontiguousarray(data).tobytes())
    tmp.replace(path)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Lit un WAV PCM 16 bits ; renvoie (float32 mono dans [-1, 1], fréquence)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path} : seul le PCM 16 bits est pris en charge")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    pcm = raw.reshape(-1, channels).mean(axis=1) if channels > 1 else raw
    return (pcm / 32768.0).astype(np.float32), rate
