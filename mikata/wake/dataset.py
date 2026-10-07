"""Données du mot d'activation définitif (phase 7, §4) : tes enregistrements, ce qui sert à
l'entraînement et ce qui est mis de côté pour le benchmark.

    data/wakeword/
      positives/train/<condition>/NNN.wav   tes « MIKATA » : entraînement + vérificateur
      positives/eval/<condition>/NNN.wav    30 « MIKATA » jamais vus à l'entraînement
      hard_negatives/<mot>/NNN.wav          tes mots proches (mikado, Nikita...) : négatifs
      eval_negatives.txt                    extraits de data/negatives figés pour le benchmark
    data/negatives/AAAA-MM-JJ/HHMMSS.wav    ta vie courante (mode wake.collect_negatives)

Le jeu d'évaluation (positifs `eval` + ~3 h de négatifs listés dans eval_negatives.txt) n'entre
jamais dans l'entraînement ni dans le vérificateur : c'est lui qui dit si le modèle tient
ses promesses (< 0,2 faux déclenchement par heure, < 5 % de ratés).
"""

from __future__ import annotations

import math
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mikata.audio.base import SAMPLE_RATE

EVAL_NEGATIVE_HOURS = 3.0
NEGATIVE_CLIP_S = 10.0


@dataclass(frozen=True)
class Condition:
    """Une façon de dire « MIKATA » : consigne, nombre de prises (entraînement, évaluation)."""

    key: str
    instruction: str
    train: int
    eval: int
    detector: bool = True  # False : sert au vérificateur et au benchmark, pas au détecteur


# Ce que tu dois enregistrer : 100 prises pour l'entraînement, 30 pour l'évaluation (un autre
# jour). « phrase » n'entraîne pas le détecteur : il apprend à réagir juste après le mot, pas
# à la fin de la commande ; elle sert au vérificateur et au benchmark (usage réel).
POSITIVE_PLAN: tuple[Condition, ...] = (
    Condition("normal", "voix normale, assis à ta place habituelle (~30 cm du micro)", 20, 6),
    Condition("bas", "voix basse, presque chuchotée (comme la nuit)", 10, 3),
    Condition("fort", "voix forte, comme pour appeler quelqu'un dans la pièce", 10, 3),
    Condition("fatigue", "voix fatiguée, lente, un peu traînante", 10, 3),
    Condition("sourire", "en souriant, vite, avec énergie", 10, 3),
    Condition("loin", "debout à environ 2 m du micro", 10, 3),
    Condition("cote", "tête tournée sur le côté, ou dos au micro", 10, 3),
    Condition("bruit", "en tapant au clavier, ou avec un ventilateur ou une vidéo en fond", 10, 3),
    Condition(
        "phrase",
        "d'une traite avec une commande : « MIKATA, ouvre Spotify », « MIKATA, quelle heure "
        "est-il »... (varie la commande)",
        10,
        3,
        detector=False,
    ),
)

# Mots proches dits par toi (3 prises chacun) : négatifs difficiles pour le détecteur et
# pour le vérificateur.
HARD_NEGATIVES: tuple[str, ...] = (
    "mikado",
    "Nikita",
    "Akira",
    "il m'a quitté",
    "mi-carré",
    "Mika",
    "ma carte",
    "mi casa",
    "Takata",
    "chiquita",
    "Michaela",
    "Nakata",
)
HARD_NEGATIVE_TAKES = 3


def slug(text: str) -> str:
    """Nom de dossier sûr : « il m'a quitté » -> « il_m_a_quitte »."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", folded).strip("_")


@dataclass(frozen=True)
class WakewordData:
    """Chemins des données du mot d'activation (racine : data/wakeword)."""

    root: Path

    def positives(self, split: str, condition: str | None = None) -> Path:
        base = self.root / "positives" / split
        return base / condition if condition else base

    def hard_negatives(self, word: str | None = None) -> Path:
        base = self.root / "hard_negatives"
        return base / slug(word) if word else base

    @property
    def eval_list(self) -> Path:
        return self.root / "eval_negatives.txt"


def list_wavs(directory: Path) -> list[Path]:
    """.wav de `directory` (récursif), triés ; [] si le dossier n'existe pas."""
    return sorted(directory.rglob("*.wav")) if directory.is_dir() else []


def next_take(directory: Path) -> Path:
    """Prochain fichier libre : 001.wav, 002.wav..."""
    existing = {p.stem for p in directory.glob("*.wav")} if directory.is_dir() else set()
    n = 1
    while f"{n:03d}" in existing:
        n += 1
    return directory / f"{n:03d}.wav"


def detector_positives(data: WakewordData) -> list[Path]:
    """Positifs d'entraînement du détecteur (sans la condition « phrase »)."""
    keys = [c.key for c in POSITIVE_PLAN if c.detector]
    return [p for key in keys for p in list_wavs(data.positives("train", key))]


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def total_hours(paths: list[Path]) -> float:
    total = 0.0
    for path in paths:
        try:
            total += wav_duration_s(path)
        except (OSError, EOFError, wave.Error):
            continue
    return total / 3600


def freeze_eval_negatives(
    negatives_dir: Path, eval_list: Path, hours: float = EVAL_NEGATIVE_HOURS
) -> list[Path]:
    """Extraits de `negatives_dir` réservés au benchmark (~`hours` heures).

    Choisis une fois pour toutes, répartis sur toute la collecte (tous les jours, toutes les
    situations), puis lus dans `eval_list` : ils ne passent jamais dans l'entraînement. Tant
    que la collecte n'a pas le double de `hours`, rien n'est figé (liste vide).
    """
    if eval_list.is_file():
        names = [line.strip() for line in eval_list.read_text(encoding="utf-8").splitlines()]
        return [negatives_dir / n for n in names if n and (negatives_dir / n).is_file()]
    clips = list_wavs(negatives_dir)
    needed = math.ceil(hours * 3600 / NEGATIVE_CLIP_S)
    if len(clips) < 2 * needed:
        return []
    picks = sorted({round(i) for i in np.linspace(0, len(clips) - 1, needed)})
    chosen = [clips[i] for i in picks]
    eval_list.parent.mkdir(parents=True, exist_ok=True)
    eval_list.write_text(
        "".join(f"{p.relative_to(negatives_dir).as_posix()}\n" for p in chosen), encoding="utf-8"
    )
    return chosen


def training_negatives(negatives_dir: Path, eval_list: Path) -> list[Path]:
    """Extraits de la vie courante pour l'entraînement : tout sauf ceux du benchmark."""
    reserved = set(freeze_eval_negatives(negatives_dir, eval_list))
    return [p for p in list_wavs(negatives_dir) if p not in reserved]


def user_speech_clips(stt_eval_dir: Path) -> list[Path]:
    """Tes phrases enregistrées pour eval_stt.py (phase 2), sans celles qui contiennent
    « MIKATA » : de ta voix qui ne dit pas le mot, pour le vérificateur."""
    clips = []
    for wav in list_wavs(stt_eval_dir):
        txt = wav.with_suffix(".txt")
        text = txt.read_text(encoding="utf-8") if txt.is_file() else ""
        if txt.is_file() and "mikata" not in slug(text).replace("_", ""):
            clips.append(wav)
    return clips


# -- traitement du signal --------------------------------------------------------------------


def _rms_10ms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    win = sample_rate // 100
    n = len(audio) // win
    x = audio[: n * win].astype(np.float32).reshape(n, win)
    return np.sqrt((x**2).mean(axis=1))


def voice_bounds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[int, int] | None:
    """(début, fin) de la voix en échantillons, par énergie ; None si aucune voix nette."""
    rms = _rms_10ms(audio, sample_rate)
    if len(rms) < 10:
        return None
    floor = float(np.percentile(rms, 5))  # 5 % : une prise peut être presque toute en voix
    peak = float(rms.max())
    if peak < max(4 * floor, 300.0):
        return None
    threshold = max(4 * floor, 0.02 * peak)  # -34 dB : garde les consonnes (« k », « t »)
    voiced = np.nonzero(rms >= threshold)[0]
    win = sample_rate // 100
    return int(voiced[0]) * win, (int(voiced[-1]) + 1) * win


def trim_to_voice(
    audio: np.ndarray, margin_s: float = 0.15, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Coupe le silence autour de la voix (marge `margin_s`) ; inchangé si aucune voix."""
    bounds = voice_bounds(audio, sample_rate)
    if bounds is None:
        return audio
    margin = round(margin_s * sample_rate)
    return audio[max(0, bounds[0] - margin) : min(len(audio), bounds[1] + margin)]


def noise_floor(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Niveau (écart-type) des 10 % de fenêtres de 10 ms les plus calmes."""
    rms = _rms_10ms(audio, sample_rate)
    return float(np.percentile(rms, 10)) if len(rms) else 0.0


def pad_with_floor_noise(
    audio: np.ndarray,
    before_s: float,
    after_s: float,
    rng: np.random.Generator,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Ajoute du bruit au niveau du fond de la prise avant et après : le détecteur a besoin
    d'environ 1 s d'audio avant le mot pour donner un score."""
    level = max(noise_floor(audio, sample_rate), 1.0)

    def noise(seconds: float) -> np.ndarray:
        n = round(seconds * sample_rate)
        return np.clip(rng.normal(0, level, n), -32768, 32767).astype(np.int16)

    return np.concatenate((noise(before_s), audio.astype(np.int16), noise(after_s)))
