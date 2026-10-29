"""Enregistrement de tes « MIKATA » pour l'entraînement — phase 7 (§4).

Ce qu'il faut enregistrer (détail : `plan`) :
- 100 « MIKATA » pour l'entraînement, dans 9 situations (voix normale, basse, forte,
  fatiguée, en souriant, loin du micro, tête tournée, avec du bruit, dans une phrase) ;
- 36 mots proches dits par toi (mikado, Nikita, Akira, « il m'a quitté »... 3 fois chacun) ;
- un autre jour, 30 « MIKATA » pour l'évaluation, jamais utilisés à l'entraînement ;
- plusieurs heures de vie courante avec `wake.collect_negatives: true` (au moins 4 h :
  2 h figées pour l'évaluation, le reste pour l'entraînement).

Usage :

    uv run python scripts/record_wakeword_samples.py plan            # où tu en es
    uv run python scripts/record_wakeword_samples.py record --set train
    uv run python scripts/record_wakeword_samples.py record --set hard
    uv run python scripts/record_wakeword_samples.py record --set eval   # un autre jour
    uv run python scripts/record_wakeword_samples.py pack            # archive pour Colab

Chaque prise : Entrée, puis tu dis la phrase ; l'enregistrement s'arrête tout seul quand tu
te tais. Ensuite : Entrée = suivante, r = refaire (l'ancienne part à la corbeille), q =
quitter. Relancer reprend là où tu t'étais arrêté. Fichiers : data/wakeword/ (16 kHz mono).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mikata.wake.dataset import (
    HARD_NEGATIVE_TAKES,
    HARD_NEGATIVES,
    POSITIVE_PLAN,
    WakewordData,
    detector_positives,
    freeze_eval_negatives,
    list_wavs,
    next_take,
    total_hours,
    training_negatives,
    voice_bounds,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "wakeword"
NEGATIVES_DIR = PROJECT_ROOT / "data" / "negatives"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_wakeword.py"
PACK_NAME = "mikata_wakeword_colab.zip"
MIN_NEGATIVE_HOURS = 4.0  # 2 h pour l'évaluation + au moins 2 h pour l'entraînement
PACK_NEGATIVE_HOURS = 4.0  # négatifs envoyés à Colab (taille de l'archive : ~115 Mo / h)
WORD_MAX_S = 2.0  # « MIKATA » seul : au-delà, il y a autre chose dans la prise
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Take:
    """Une prise à enregistrer : consigne affichée et dossier de destination."""

    label: str
    say: str
    directory: Path
    target: int
    word: bool  # True : « MIKATA » seul (durée contrôlée)


def takes_for(data: WakewordData, split: str) -> list[Take]:
    if split == "hard":
        return [
            Take(
                f"mot proche : {w}", f"« {w} »", data.hard_negatives(w), HARD_NEGATIVE_TAKES, False
            )
            for w in HARD_NEGATIVES
        ]
    takes = []
    for c in POSITIVE_PLAN:
        target = c.train if split == "train" else c.eval
        say = "« MIKATA, ... » (une commande)" if not c.detector else "« MIKATA »"
        takes.append(
            Take(
                f"{c.key} : {c.instruction}", say, data.positives(split, c.key), target, c.detector
            )
        )
    return takes


def remaining(take: Take) -> int:
    return max(0, take.target - len(list_wavs(take.directory)))


def plan_report(data: WakewordData, negatives_dir: Path) -> str:
    """Ce qui est enregistré, ce qui manque, et la prochaine étape."""
    lines = [
        "Positifs « MIKATA » (fait / à faire) :",
        f"  {'situation':<10} {'entraînement':>13} {'évaluation':>11}",
    ]
    for c in POSITIVE_PLAN:
        done_train = len(list_wavs(data.positives("train", c.key)))
        done_eval = len(list_wavs(data.positives("eval", c.key)))
        counts = f"{done_train:>6} / {c.train:<5} {done_eval:>4} / {c.eval:<4}"
        lines.append(f"  {c.key:<10} {counts}  {c.instruction}")
    hard = sum(len(list_wavs(data.hard_negatives(w))) for w in HARD_NEGATIVES)
    lines.append(f"\nMots proches : {hard} / {len(HARD_NEGATIVES) * HARD_NEGATIVE_TAKES}")
    clips = list_wavs(negatives_dir)
    hours = total_hours(clips)
    lines.append(
        f"Vie courante (data/negatives) : {hours:.1f} h sur {MIN_NEGATIVE_HOURS:.0f} h conseillées"
    )
    frozen = (
        freeze_eval_negatives(negatives_dir, data.eval_list) if hours >= MIN_NEGATIVE_HOURS else []
    )
    if frozen:
        lines.append(
            f"  dont {total_hours(frozen):.1f} h figées pour l'évaluation ({data.eval_list.name})"
        )
    missing = {
        split: sum(remaining(t) for t in takes_for(data, split))
        for split in ("train", "hard", "eval")
    }
    lines.append("\nProchaine étape :")
    if hours < MIN_NEGATIVE_HOURS:
        lines.append(
            "  - collecte de la vie courante : wake.collect_negatives: true, puis utilise le PC"
        )
    for split, label in (
        ("train", "record --set train"),
        ("hard", "record --set hard"),
        ("eval", "record --set eval (un autre jour)"),
    ):
        if missing[split]:
            lines.append(f"  - {label} : {missing[split]} prise(s)")
    if not any(missing.values()) and hours >= MIN_NEGATIVE_HOURS:
        lines.append("  - tout est prêt : pack, puis entraînement sur Colab (docs/WAKE_WORD.md)")
    return "\n".join(lines)


# -- enregistrement ------------------------------------------------------------------------


class _Frames(Protocol):
    async def read(self) -> Any: ...


class _Vad(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


async def record_take(
    capture: _Frames,
    vad: _Vad,
    max_s: float = 4.0,
    end_silence_ms: int = 500,
    start_timeout_s: float = 4.0,
) -> np.ndarray | None:
    """De l'appui sur Entrée jusqu'à `end_silence_ms` de silence après la voix. Le début
    (avant que tu parles) est gardé : le détecteur a besoin d'audio avant le mot."""
    vad.reset()
    frames: list[np.ndarray] = []
    silent, speech_seen = 0, False
    end_frames = max(1, round(end_silence_ms / 80))
    while len(frames) * 0.08 < max_s:
        frame = (await capture.read()).samples
        frames.append(frame)
        if vad.is_speech(frame):
            speech_seen, silent = True, 0
        else:
            silent += 1
        if speech_seen and silent >= end_frames:
            break
        if not speech_seen and len(frames) * 0.08 >= start_timeout_s:
            return None
    return np.concatenate(frames) if speech_seen else None


def check_take(audio: np.ndarray, word: bool) -> str | None:
    """Problème évident dans une prise (à refaire), ou None."""
    bounds = voice_bounds(audio)
    if bounds is None:
        return "voix trop faible ou absente"
    voiced_s = (bounds[1] - bounds[0]) / SAMPLE_RATE
    if word and voiced_s > WORD_MAX_S:
        return f"{voiced_s:.1f} s de voix : dis seulement « MIKATA »"
    if voiced_s < 0.2:
        return "trop court"
    if int(np.abs(audio).max()) >= 32000:
        return "saturé : éloigne-toi un peu ou baisse le gain du micro"
    return None


async def _record(args: argparse.Namespace) -> int:
    import sounddevice as sd
    from send2trash import send2trash

    from mikata.audio.capture import AudioCapture
    from mikata.audio.devices import collect_devices, default_device_index, find_device
    from mikata.audio.wavfile import write_wav
    from mikata.config import load_config
    from mikata.vad.silero import SileroVAD

    config = load_config(args.config)
    data = WakewordData(args.root)
    mic = find_device(
        collect_devices(sd), config.audio.input_device, "input", default_device_index(sd, "input")
    )
    vad = SileroVAD(config.resolve(config.vad.model), config.vad.threshold)
    capture = AudioCapture(mic, channel=config.audio.input_channel)
    capture.start(asyncio.get_running_loop())
    takes = [t for t in takes_for(data, args.set) if args.only in (None, t.directory.name)]
    print(f"Micro : {mic.name}. {sum(remaining(t) for t in takes)} prise(s) à faire.")
    try:
        for take in takes:
            if remaining(take) == 0:
                continue
            print(f"\n=== {take.label}")
            while remaining(take) > 0:
                done = take.target - remaining(take)
                answer = await asyncio.to_thread(
                    input,
                    f"[{done + 1}/{take.target}] Entrée, puis dis {take.say} (q = quitter) : ",
                )
                if answer.strip().lower() == "q":
                    return 0
                capture.clear()  # oublie l'audio d'avant l'appui
                audio = await record_take(capture, vad, max_s=4.0 if take.word else 6.0)
                if audio is None:
                    print("   rien entendu, on recommence")
                    continue
                problem = check_take(audio, take.word)
                path = next_take(take.directory)
                write_wav(path, audio, SAMPLE_RATE)
                note = f" — attention : {problem}" if problem else ""
                print(f"   {path.relative_to(args.root)} ({len(audio) / SAMPLE_RATE:.1f} s){note}")
                again = await asyncio.to_thread(input, "   Entrée = garder, r = refaire : ")
                if again.strip().lower() == "r":
                    send2trash(str(path))  # jamais de suppression définitive
    finally:
        capture.stop()
    print("\n" + plan_report(data, args.negatives))
    return 0


# -- archive pour Colab --------------------------------------------------------------------


def pack(
    data: WakewordData,
    negatives_dir: Path,
    output: Path,
    negative_hours: float = PACK_NEGATIVE_HOURS,
    train_script: Path = TRAIN_SCRIPT,
) -> dict[str, int]:
    """Archive pour l'entraînement : script, positifs du détecteur, mots proches, négatifs
    d'entraînement (jamais ceux de l'évaluation). Renvoie le nombre de fichiers par dossier."""
    positives = detector_positives(data)
    hard = list_wavs(data.hard_negatives())
    negatives = training_negatives(negatives_dir, data.eval_list)
    rng = np.random.default_rng(0)
    max_clips = round(negative_hours * 3600 / 10)
    if len(negatives) > max_clips:  # pris au hasard sur toute la collecte
        picks = sorted(rng.choice(len(negatives), max_clips, replace=False))
        negatives = [negatives[i] for i in picks]
    counts = {"positives": len(positives), "hard_negatives": len(hard), "negatives": len(negatives)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(train_script, "train_wakeword.py")
        for folder, files, base in (
            ("positives", positives, data.positives("train")),
            ("hard_negatives", hard, data.hard_negatives()),
            ("negatives", negatives, negatives_dir),
        ):
            for path in files:
                name = path.relative_to(base).as_posix().replace("/", "__")
                archive.write(path, f"data/{folder}/{name}")
        manifest = {"created": datetime.now().isoformat(timespec="seconds"), **counts}
        archive.writestr("data/manifest.json", json.dumps(manifest, indent=2))
    return counts


def _pack(args: argparse.Namespace) -> int:
    data = WakewordData(args.root)
    if not detector_positives(data):
        print("Aucun « MIKATA » enregistré : lance d'abord « record --set train ».")
        return 1
    if not freeze_eval_negatives(args.negatives, data.eval_list):
        print(
            f"Pas assez de vie courante dans {args.negatives} pour mettre 2 h de côté "
            f"(il en faut au moins {MIN_NEGATIVE_HOURS:.0f} h) : l'archive est faite sans "
            "négatifs à toi, et le benchmark ne sera pas possible."
        )
    output = args.root / PACK_NAME
    counts = pack(data, args.negatives, output, args.negative_hours)
    size_mb = output.stat().st_size / 1e6
    print(
        f"{output} ({size_mb:.0f} Mo) : {counts['positives']} « MIKATA », "
        f"{counts['hard_negatives']} mots proches, {counts['negatives']} extraits de vie courante."
    )
    print("Étape suivante : entraînement sur Colab (docs/WAKE_WORD.md).")
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dossier des prises")
    parser.add_argument("--negatives", type=Path, default=NEGATIVES_DIR, help="vie courante")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="ce qui est enregistré et ce qui manque")
    record = sub.add_parser("record", help="enregistrer des prises")
    record.add_argument("--set", choices=("train", "eval", "hard"), default="train")
    record.add_argument("--only", default=None, help="une seule situation (ex. : loin)")
    packer = sub.add_parser("pack", help="archive pour l'entraînement sur Colab")
    packer.add_argument("--negative-hours", type=float, default=PACK_NEGATIVE_HOURS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        print(plan_report(WakewordData(args.root), args.negatives))
        return 0
    if args.command == "record":
        return asyncio.run(_record(args))
    return _pack(args)


if __name__ == "__main__":
    sys.exit(main())
