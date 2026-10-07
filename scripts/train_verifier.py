"""Vérificateur de ta voix (§4, étage 2) — phase 7.

Entraîné sur ton PC en une ou deux minutes, sans GPU, après mikata.onnx :
- positifs : tes « MIKATA » d'entraînement (data/wakeword/positives/train), tels que le
  modèle les voit au moment où il les détecte ;
- négatifs : ta voix qui dit autre chose (mots proches de data/wakeword/hard_negatives, et
  tes phrases de eval_stt.py dans data/stt_eval, sauf celles qui contiennent « MIKATA »).
Les prises « eval » ne sont jamais utilisées : elles servent au benchmark.

    uv run python scripts/train_verifier.py                 # -> models/mikata_verifier.pkl
    uv run python scripts/train_verifier.py --model models/mikata.onnx --output models/x.pkl

Puis : scripts/eval_wakeword.py pour choisir wake.verifier_threshold.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mikata.wake.dataset import (
    WakewordData,
    list_wavs,
    pad_with_floor_noise,
    user_speech_clips,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "wakeword"
STT_EVAL_DIR = PROJECT_ROOT / "data" / "stt_eval"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mikata.onnx"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "mikata_verifier.pkl"
LEAD_S, TAIL_S = 1.5, 0.5  # le modèle a besoin d'audio avant le mot pour donner un score
MIN_POSITIVES = 20


def load_padded(paths: Sequence[Path], rng: np.random.Generator) -> list[np.ndarray]:
    """Prises en int16 16 kHz, précédées et suivies de bruit au niveau de leur fond."""
    from mikata.audio.wavfile import float_to_int16, read_wav

    clips = []
    for path in paths:
        pcm, rate = read_wav(path)
        if rate != 16_000:
            print(f"{path} : {rate} Hz, 16 kHz attendu ; ignoré", file=sys.stderr)
            continue
        clips.append(pad_with_floor_noise(float_to_int16(pcm), LEAD_S, TAIL_S, rng))
    return clips


def reference_clips(data: WakewordData, stt_eval: Path) -> tuple[list[Path], list[Path]]:
    """(tes « MIKATA » d'entraînement, ta voix sans « MIKATA »)."""
    positives = list_wavs(data.positives("train"))
    negatives = list_wavs(data.hard_negatives()) + user_speech_clips(stt_eval)
    return positives, negatives


def train(
    model: Path,
    output: Path,
    positives: list[np.ndarray],
    negatives: list[np.ndarray],
    trainer: Callable[..., Any] | None = None,
) -> None:
    """`openwakeword.train_custom_verifier` (régression logistique sur les features du modèle,
    là où il dépasse 0,5 pour les positifs ; toutes les trames pour les négatifs)."""
    if trainer is None:
        from openwakeword.custom_verifier_model import train_custom_verifier as trainer
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer(
        positive_reference_clips=positives,
        negative_reference_clips=negatives,
        output_path=str(output),
        model_name=str(model),
        inference_framework="onnx",
        melspec_model_path=str(model.parent / "melspectrogram.onnx"),
        embedding_model_path=str(model.parent / "embedding_model.onnx"),
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="modèle entraîné")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="tes prises")
    parser.add_argument("--stt-eval", type=Path, default=STT_EVAL_DIR, help="tes phrases")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"Modèle introuvable : {args.model} (entraînement sur Colab : README, phase 7).")
        return 1
    positive_paths, negative_paths = reference_clips(WakewordData(args.root), args.stt_eval)
    if len(positive_paths) < MIN_POSITIVES or not negative_paths:
        print(
            f"Pas assez de prises : {len(positive_paths)} « MIKATA » (au moins {MIN_POSITIVES}) "
            f"et {len(negative_paths)} phrases sans « MIKATA ». Lance "
            "record_wakeword_samples.py record --set train puis --set hard."
        )
        return 1
    rng = np.random.default_rng(0)
    print(f"{len(positive_paths)} « MIKATA », {len(negative_paths)} phrases sans « MIKATA ».")
    try:
        train(
            args.model,
            args.output,
            load_padded(positive_paths, rng),
            load_padded(negative_paths, rng),
        )
    except ValueError as exc:  # aucun positif détecté par le modèle
        print(
            f"Échec : {exc}\nLe modèle ne détecte aucune de tes prises au-dessus de 0,5 : "
            "vérifie le modèle avec eval_wakeword.py --no-verifier, ou réentraîne-le."
        )
        return 1
    print(f"Vérificateur : {args.output}")
    print("Étape suivante : uv run python scripts/eval_wakeword.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
