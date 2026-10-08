"""Benchmark du mot d'activation (faux déclenchements/h, ratés, délai) — phase 7 (§4).

Jeu d'évaluation figé, jamais vu à l'entraînement :
- tes 30 « MIKATA » de data/wakeword/positives/eval, chacun placé dans 1,5 s de vie courante
  avant et 1 s après (comme en vrai : le micro tourne en continu) ;
- ~2 h de ta vie courante (data/negatives, liste figée dans data/wakeword/eval_negatives.txt),
  passées d'un bloc dans le détecteur comme en vrai.

Le détecteur ne tourne qu'une fois ; on rejoue ensuite les règles de déclenchement de MIKATA
(`WakeTrigger` : seuil, trames consécutives, pause après détection) pour chaque réglage, avec
et sans vérificateur. Résultat : la courbe seuil -> (faux déclenchements par heure, ratés,
délai médian), et le meilleur réglage qui tient les cibles (< 0,2 FD/h et < 5 % de ratés),
prêt à copier dans config.yaml.

    uv run python scripts/eval_wakeword.py                       # modèle de config.yaml
    uv run python scripts/eval_wakeword.py --model models/mikata.onnx --verifier models/v2.pkl
    uv run python scripts/eval_wakeword.py --no-verifier
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mikata.wake.dataset import (
    WakewordData,
    freeze_eval_negatives,
    list_wavs,
    noise_floor,
    voice_bounds,
)
from mikata.wake.trigger import WakeTrigger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "wakeword"
NEGATIVES_DIR = PROJECT_ROOT / "data" / "negatives"
FRAME = 1280
FRAME_S = 0.08
SAMPLE_RATE = 16_000
LEAD_S, TAIL_S = 1.5, 1.0
TARGET_FA_PER_HOUR = 0.2
TARGET_MISS_RATE = 0.05
MAX_DELAY_MS = 150  # §4 : détection moins de 150 ms après la fin du mot
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.1, 1.0, 0.05))
VERIFIER_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


class Scorer(Protocol):
    """`mikata.wake.oww.OpenWakeWordDetector` (ou un faux dans les tests)."""

    @property
    def has_verifier(self) -> bool: ...
    def scores(
        self, frame: np.ndarray, gate: float | None = None
    ) -> tuple[float, float | None]: ...
    def reset(self) -> None: ...


@dataclass
class FrameScores:
    """Score du modèle et probabilité « ta voix » (NaN si non calculée) pour chaque trame."""

    scores: np.ndarray
    voices: np.ndarray

    @property
    def hours(self) -> float:
        return len(self.scores) * FRAME_S / 3600


def score_stream(detector: Scorer, audio: np.ndarray, gate: float) -> FrameScores:
    """Passe `audio` (int16 16 kHz) trame par trame dans le détecteur, comme MIKATA."""
    n = len(audio) // FRAME
    scores = np.zeros(n, dtype=np.float32)
    voices = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        score, voice = detector.scores(audio[i * FRAME : (i + 1) * FRAME], gate)
        scores[i] = score
        if voice is not None:
            voices[i] = voice
    return FrameScores(scores, voices)


def effective(fs: FrameScores, verifier_threshold: float | None) -> np.ndarray:
    """Score vu par `WakeTrigger` : 0 quand le vérificateur refuse la trame (comme
    `OpenWakeWordDetector.process`)."""
    if verifier_threshold is None:
        return fs.scores
    refused = ~np.isnan(fs.voices) & (fs.voices < verifier_threshold)
    return np.where(refused, 0.0, fs.scores)


def triggers(
    scores: np.ndarray, threshold: float, consecutive: int, refractory_s: float
) -> list[int]:
    """Indices des trames où MIKATA déclencherait (mêmes règles que l'application)."""
    trigger = WakeTrigger(threshold, consecutive, refractory_s)
    return [i for i, score in enumerate(scores) if trigger.update(float(score), i * FRAME_S)]


@dataclass
class Positive:
    """Un « MIKATA » d'évaluation placé dans de la vie courante."""

    name: str
    frames: FrameScores
    word_start: int  # trame où la voix commence
    word_end_s: float  # fin de la voix (s depuis le début du flux)


@dataclass
class Setting:
    threshold: float
    consecutive: int
    verifier_threshold: float | None


@dataclass
class Result:
    setting: Setting
    false_per_hour: float
    miss_rate: float
    median_delay_ms: float | None
    missed: list[str] = field(default_factory=list)

    @property
    def meets_targets(self) -> bool:
        return self.false_per_hour < TARGET_FA_PER_HOUR and self.miss_rate < TARGET_MISS_RATE


def evaluate(
    negatives: FrameScores, positives: Sequence[Positive], setting: Setting, refractory_s: float
) -> Result:
    neg = effective(negatives, setting.verifier_threshold)
    false = len(triggers(neg, setting.threshold, setting.consecutive, refractory_s))
    delays, missed = [], []
    for p in positives:
        scores = effective(p.frames, setting.verifier_threshold)
        hits = [
            i
            for i in triggers(scores, setting.threshold, setting.consecutive, refractory_s)
            if i >= p.word_start
        ]
        if hits:
            delays.append(((hits[0] + 1) * FRAME_S - p.word_end_s) * 1000)
        else:
            missed.append(p.name)
    return Result(
        setting=setting,
        false_per_hour=false / negatives.hours if negatives.hours else float("inf"),
        miss_rate=len(missed) / len(positives) if positives else 1.0,
        median_delay_ms=statistics.median(delays) if delays else None,
        missed=missed,
    )


def sweep(
    negatives: FrameScores,
    positives: Sequence[Positive],
    refractory_s: float,
    thresholds: Iterable[float] = THRESHOLDS,
    consecutive: Iterable[int] = (1, 2),
    verifier_thresholds: Iterable[float | None] = (None,),
) -> list[Result]:
    return [
        evaluate(negatives, positives, Setting(t, n, v), refractory_s)
        for v in verifier_thresholds
        for n in consecutive
        for t in thresholds
    ]


def recommend(results: Sequence[Result]) -> Result | None:
    """Réglage conseillé parmi ceux qui tiennent les cibles (None si aucun) :
    1. le moins de faux déclenchements, puis le moins de ratés (décision §1 : le moins de
       faux positifs possible, puis la détection rapide) ;
    2. délai médian ≤ 150 ms (§4) si possible, 1 trame plutôt que 2 (plus rapide) ;
    3. au **milieu** des seuils restants (vérificateur, puis modèle) : une marge des deux
       côtés, plutôt qu'un réglage au bord qui raterait un jour de fatigue."""
    good = [r for r in results if r.meets_targets]
    if not good:
        return None
    fewest_false = min(r.false_per_hour for r in good)
    good = [r for r in good if r.false_per_hour == fewest_false]
    fewest_missed = min(r.miss_rate for r in good)
    good = [r for r in good if r.miss_rate == fewest_missed]
    fast = [r for r in good if r.median_delay_ms is not None and r.median_delay_ms <= MAX_DELAY_MS]
    good = fast or good
    good = [r for r in good if r.setting.consecutive == min(x.setting.consecutive for x in good)]
    vthrs = sorted(
        {r.setting.verifier_threshold for r in good}, key=lambda v: -1 if v is None else v
    )
    vthr = vthrs[(len(vthrs) - 1) // 2]
    good = [r for r in good if r.setting.verifier_threshold == vthr]
    good.sort(key=lambda r: r.setting.threshold)
    return good[(len(good) - 1) // 2]


def closest(results: Sequence[Result]) -> Result | None:
    """Sans réglage qui tient les cibles : le meilleur compromis (ratés + FD/h normalisés)."""
    if not results:
        return None
    return min(
        results,
        key=lambda r: r.miss_rate / TARGET_MISS_RATE + r.false_per_hour / TARGET_FA_PER_HOUR,
    )


# -- données -------------------------------------------------------------------------------


def read_int16(path: Path) -> np.ndarray | None:
    from mikata.audio.wavfile import float_to_int16, read_wav

    pcm, rate = read_wav(path)
    if rate != SAMPLE_RATE:
        print(f"{path} : {rate} Hz, 16 kHz attendu ; ignoré", file=sys.stderr)
        return None
    return float_to_int16(pcm)


BACKGROUND_CLIPS = 60  # 10 min de vie courante suffisent pour entourer les positifs


class Background:
    """Morceaux de vie courante pour entourer les positifs ; à défaut, du bruit au niveau du
    fond de la prise."""

    def __init__(self, clips: Sequence[np.ndarray], rng: np.random.Generator) -> None:
        picked = list(clips[:BACKGROUND_CLIPS])
        self.audio = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int16)
        self.rng = rng

    def take(self, seconds: float, like: np.ndarray) -> np.ndarray:
        n = round(seconds * SAMPLE_RATE)
        if len(self.audio) < n:
            level = max(noise_floor(like), 1.0)
            return np.clip(self.rng.normal(0, level, n), -32768, 32767).astype(np.int16)
        start = int(self.rng.integers(0, len(self.audio) - n + 1))
        return self.audio[start : start + n]


def score_positive(
    detector: Scorer, name: str, clip: np.ndarray, background: Background, gate: float
) -> Positive | None:
    bounds = voice_bounds(clip)
    if bounds is None:
        print(f"{name} : pas de voix nette, ignoré", file=sys.stderr)
        return None
    lead, tail = background.take(LEAD_S, clip), background.take(TAIL_S, clip)
    detector.reset()
    frames = score_stream(detector, np.concatenate((lead, clip, tail)), gate)
    start = len(lead) + bounds[0]
    return Positive(name, frames, start // FRAME, (len(lead) + bounds[1]) / SAMPLE_RATE)


# -- rapport -------------------------------------------------------------------------------


def _cell(r: Result | None) -> str:
    if r is None:
        return f"{'':>24}"
    delay = "—" if r.median_delay_ms is None else f"{round(r.median_delay_ms)} ms"
    star = "*" if r.meets_targets else " "
    return f"{r.false_per_hour:>7.2f} {r.miss_rate * 100:>5.1f} % {delay:>7}{star}"


def format_report(
    results: Sequence[Result], hours: float, n_positives: int, every: float = 0.1
) -> str:
    """Courbe seuil -> (FD/h, ratés, délai médian) : une ligne par seuil multiple de `every`
    (plus celle du réglage conseillé), 1 et 2 trames consécutives côte à côte."""
    best = recommend(results)
    lines = [
        f"Jeu d'évaluation : {n_positives} « MIKATA », {hours:.2f} h de vie courante.",
        f"Cibles : < {TARGET_FA_PER_HOUR} faux déclenchement (FD) par heure, "
        f"< {TARGET_MISS_RATE * 100:.0f} % de ratés ; * = cibles tenues.",
    ]
    groups: dict[float | None, list[Result]] = {}
    for r in results:
        groups.setdefault(r.setting.verifier_threshold, []).append(r)
    for vthr, group in groups.items():
        title = "sans vérificateur" if vthr is None else f"avec vérificateur (voix >= {vthr})"
        cells = {(r.setting.threshold, r.setting.consecutive): r for r in group}
        frames = sorted({r.setting.consecutive for r in group})
        header = "".join(
            f"  {f'--- {n} trame' + ('s' if n > 1 else '') + ' ---':^24}" for n in frames
        )
        lines += [
            f"\n{title} :",
            f"  seuil{header}",
            "       " + "".join(f"  {'FD/h':>7} {'ratés':>7} {'délai':>7} " for _ in frames),
        ]
        for t in sorted({r.setting.threshold for r in group}):
            on_grid = abs(t / every - round(t / every)) < 1e-6
            if not on_grid and (best is None or best.setting.threshold != t):
                continue
            lines.append(f"  {t:>5.2f}" + "".join(f"  {_cell(cells.get((t, n)))}" for n in frames))
    if best is not None:
        lines += [
            "\nRéglage conseillé, au milieu de la plage qui tient les cibles (à copier dans "
            "config.yaml, section wake) :",
            *config_lines(best),
        ]
    else:
        near = closest(results)
        lines.append("\nAucun réglage ne tient les deux cibles.")
        if near is not None:
            lines += [
                f"Meilleur compromis : FD/h {near.false_per_hour:.2f}, ratés "
                f"{near.miss_rate * 100:.1f} %.",
                *config_lines(near),
                "Pistes : le vérificateur de ta voix (train_verifier.py) s'il n'est pas activé ;"
                " plus de vie courante (collect_negatives) et d'autres prises « MIKATA », puis"
                " réentraînement (README, phase 7).",
            ]
            if near.missed:
                lines.append(f"Ratés : {', '.join(near.missed[:10])}")
    return "\n".join(lines)


def config_lines(r: Result) -> list[str]:
    lines = [
        f"  threshold: {r.setting.threshold:.2f}",
        f"  consecutive_frames: {r.setting.consecutive}",
    ]
    if r.setting.verifier_threshold is None:
        lines.append("  verifier: null")
    else:
        lines.append(f"  verifier_threshold: {r.setting.verifier_threshold:.2f}")
    return lines


# -- programme -----------------------------------------------------------------------------


def run(
    detector: Scorer,
    positives: Sequence[tuple[str, np.ndarray]],
    negatives: Sequence[np.ndarray],
    refractory_s: float,
    thresholds: Sequence[float] = THRESHOLDS,
    seed: int = 0,
) -> tuple[list[Result], FrameScores, list[Positive]]:
    gate = min(thresholds)
    print(f"Vie courante : {sum(len(c) for c in negatives) / SAMPLE_RATE / 3600:.2f} h...")
    detector.reset()
    parts = []  # extrait par extrait, sans remise à zéro : un seul flux continu, comme en vrai
    for i, clip in enumerate(negatives):
        parts.append(score_stream(detector, clip, gate))
        if len(negatives) >= 10 and (i + 1) % (len(negatives) // 10) == 0:
            print(f"  {(i + 1) * 100 // len(negatives)} %")
    neg = FrameScores(
        np.concatenate([p.scores for p in parts]) if parts else np.zeros(0, np.float32),
        np.concatenate([p.voices for p in parts]) if parts else np.zeros(0, np.float32),
    )
    background = Background(negatives, np.random.default_rng(seed))
    scored = [score_positive(detector, name, clip, background, gate) for name, clip in positives]
    pos = [p for p in scored if p is not None]
    vthrs: list[float | None] = [None]
    if detector.has_verifier:
        vthrs += list(VERIFIER_THRESHOLDS)
    return sweep(neg, pos, refractory_s, thresholds, (1, 2), vthrs), neg, pos


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    parser.add_argument("--model", type=Path, default=None, help="défaut : wake.model")
    parser.add_argument("--verifier", type=Path, default=None, help="défaut : wake.verifier")
    parser.add_argument("--no-verifier", action="store_true", help="modèle seul")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="tes prises")
    parser.add_argument("--positives", type=Path, default=None, help="défaut : prises « eval »")
    parser.add_argument("--negatives", type=Path, default=NEGATIVES_DIR, help="vie courante")
    parser.add_argument(
        "--all-negatives",
        action="store_true",
        help="tous les .wav de --negatives au lieu de la liste figée (autre dossier de test)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from mikata.config import load_config
    from mikata.wake.oww import OpenWakeWordDetector

    args = parse_args(argv)
    config = load_config(args.config)
    data = WakewordData(args.root)
    model = args.model or config.resolve(config.wake.model)
    verifier = None
    if not args.no_verifier:
        verifier = args.verifier or (
            config.resolve(config.wake.verifier) if config.wake.verifier else None
        )
    positive_paths = list_wavs(args.positives or data.positives("eval"))
    if args.all_negatives:
        negative_paths = list_wavs(args.negatives)
    else:
        negative_paths = freeze_eval_negatives(args.negatives, data.eval_list)
    if not positive_paths or not negative_paths:
        print(
            f"Jeu d'évaluation incomplet : {len(positive_paths)} « MIKATA », "
            f"{len(negative_paths)} extraits de vie courante. Voir "
            "record_wakeword_samples.py plan."
        )
        return 1
    detector: Any = OpenWakeWordDetector(model, verifier, threshold=min(THRESHOLDS))
    print(f"Modèle : {model.name}" + (f" + vérificateur {verifier.name}" if verifier else ""))
    root = args.positives or data.positives("eval")
    positives = [
        (str(p.relative_to(root)), a) for p in positive_paths if (a := read_int16(p)) is not None
    ]
    negatives = [a for p in negative_paths if (a := read_int16(p)) is not None]
    results, neg, pos = run(detector, positives, negatives, config.wake.refractory_s)
    print("\n" + format_report(results, neg.hours, len(pos)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
