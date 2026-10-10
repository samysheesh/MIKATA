"""Sons d'interface (§8) : préchargés en mémoire au taux de la sortie, volume appliqué.

Les fichiers `assets/sounds/<nom>.wav` peuvent être remplacés par les tiens ; s'il en
manque un, le son par défaut est synthétisé. `DEFAULT_RATE` n'est que le taux de
synthèse de repli : le rééchantillonnage soxr adapte chaque son au taux réel du
périphérique (WASAPI comme CoreAudio).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
from loguru import logger

from mikata.audio.wavfile import read_wav

EARCON_NAMES = ("wake", "end", "thinking", "error", "timer", "miss")  # timer : phase 4, miss : 9
DEFAULT_RATE = 48_000


def _tone(freqs: list[tuple[float, float]], duration_s: float, rate: int) -> np.ndarray:
    """Glissando de fréquences (f_début, f_fin) superposées, enveloppe douce."""
    n = round(duration_s * rate)
    t = np.arange(n) / rate
    out = np.zeros(n, dtype=np.float64)
    for f0, f1 in freqs:
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration_s))
        out += np.sin(phase)
    out /= len(freqs)
    fade = min(n // 4, round(0.01 * rate))
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    envelope *= np.exp(-2.5 * t / duration_s)
    return (0.8 * out * envelope).astype(np.float32)


def synthesize(name: str, rate: int = DEFAULT_RATE) -> np.ndarray:
    if name == "wake":  # bip montant court (~120 ms)
        return _tone([(880, 1320)], 0.12, rate)
    if name == "end":  # bip descendant
        return _tone([(1320, 660)], 0.15, rate)
    if name == "thinking":  # petit son discret
        return 0.5 * _tone([(660, 660), (990, 990)], 0.08, rate)
    if name == "error":  # deux notes graves
        low = _tone([(330, 300)], 0.12, rate)
        gap = np.zeros(round(0.04 * rate), dtype=np.float32)
        return np.concatenate((low, gap, low))
    if name == "timer":  # trois notes claires (minuteur, rappel)
        note = _tone([(1046, 1046), (1568, 1568)], 0.18, rate)
        gap = np.zeros(round(0.08 * rate), dtype=np.float32)
        return np.concatenate((note, gap, note, gap, note))
    if name == "miss":  # deux tics aigus brefs : raté du mot d'activation noté (phase 9)
        tick = 0.6 * _tone([(2093, 2093)], 0.04, rate)
        gap = np.zeros(round(0.05 * rate), dtype=np.float32)
        return np.concatenate((tick, gap, tick))
    raise KeyError(name)


class Earcons(Mapping[str, np.ndarray]):
    def __init__(self, sounds_dir: Path, sample_rate: int, volume: float) -> None:
        self.sample_rate = sample_rate
        self._sounds: dict[str, np.ndarray] = {}
        for name in EARCON_NAMES:
            path = sounds_dir / f"{name}.wav"
            if path.is_file():
                pcm, rate = read_wav(path)
            else:
                logger.warning("Son {} absent, son par défaut utilisé.", path)
                pcm, rate = synthesize(name), DEFAULT_RATE
            if rate != sample_rate:
                import soxr

                pcm = soxr.resample(pcm, rate, sample_rate)
            self._sounds[name] = np.ascontiguousarray(pcm * volume, dtype=np.float32)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._sounds[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sounds)

    def __len__(self) -> int:
        return len(self._sounds)
