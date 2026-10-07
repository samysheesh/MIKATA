"""Benchmark STT (§5) : WER et latence de Whisper sur tes vraies commandes.

1. Enregistrer le jeu de test (une fois) : chaque phrase de `scripts/stt_phrases_fr.txt`
   s'affiche, tu appuies sur Entrée et tu la lis ; l'enregistrement s'arrête tout seul
   quand tu te tais (Silero VAD). Résultat : data/stt_eval/NN.wav + NN.txt (texte attendu,
   modifiable à la main). Les phrases déjà enregistrées sont sautées.

       uv run python scripts/eval_stt.py record

2. Mesurer, en comparant des réglages :

       uv run python scripts/eval_stt.py run --compute-type float16 int8_float16 --beam-size 1 2

   Pour chaque combinaison : WER (taux d'erreur sur les mots, après normalisation : casse,
   ponctuation), latence médiane et p95 par commande, pires phrases.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "stt_eval"
DEFAULT_PHRASES = PROJECT_ROOT / "scripts" / "stt_phrases_fr.txt"
RECORD_MAX_S = 8.0
RECORD_END_SILENCE_MS = 700
RECORD_START_TIMEOUT_S = 5.0


# -- texte et WER --------------------------------------------------------------------------


def normalize(text: str) -> list[str]:
    """Mots en minuscules, sans ponctuation ; « l'heure » -> « l heure »."""
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(substitutions + suppressions + insertions, nombre de mots de la référence)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # suppression
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_word != hyp_word),  # substitution
                )
            )
        previous = current
    return previous[-1], len(ref)


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> float:
    """WER du corpus : somme des erreurs / somme des mots de référence."""
    errors = words = 0
    for reference, hypothesis in pairs:
        e, n = word_errors(reference, hypothesis)
        errors += e
        words += n
    return errors / words if words else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


# -- jeu de test ---------------------------------------------------------------------------


def load_phrases(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def load_dataset(directory: Path) -> list[tuple[Path, str]]:
    """Paires (NN.wav, texte de NN.txt), triées ; un .wav sans .txt est ignoré."""
    items = []
    for wav in sorted(directory.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.is_file():
            items.append((wav, txt.read_text(encoding="utf-8").strip()))
    return items


# -- enregistrement ------------------------------------------------------------------------


class _Frames(Protocol):
    async def read(self) -> Any: ...


class _Vad(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


async def record_one(
    capture: _Frames,
    vad: _Vad,
    max_s: float = RECORD_MAX_S,
    end_silence_ms: int = RECORD_END_SILENCE_MS,
    start_timeout_s: float = RECORD_START_TIMEOUT_S,
) -> np.ndarray | None:
    """Enregistre une phrase : de maintenant à `end_silence_ms` de silence après la voix.

    None si personne n'a parlé dans les `start_timeout_s` premières secondes.
    """
    vad.reset()
    frames: list[np.ndarray] = []
    silent = 0
    speech_seen = False
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


async def _record(args: argparse.Namespace) -> int:
    import sounddevice as sd

    from mikata.audio.capture import AudioCapture
    from mikata.audio.devices import collect_devices, default_device_index, find_device
    from mikata.audio.wavfile import write_wav
    from mikata.config import load_config
    from mikata.vad.silero import SileroVAD

    config = load_config(args.config)
    mic = find_device(
        collect_devices(sd), config.audio.input_device, "input", default_device_index(sd, "input")
    )
    vad = SileroVAD(config.resolve(config.vad.model), config.vad.threshold)
    phrases = load_phrases(args.phrases)
    args.dir.mkdir(parents=True, exist_ok=True)
    capture = AudioCapture(mic, channel=config.audio.input_channel)
    capture.start(asyncio.get_running_loop())
    print(f"Micro : {mic.name}. {len(phrases)} phrases, dossier {args.dir}")
    try:
        for number, phrase in enumerate(phrases, start=1):
            wav = args.dir / f"{number:02d}.wav"
            if wav.exists():
                continue
            await asyncio.to_thread(input, f"\n[{number:02d}] « {phrase} » — Entrée puis lis : ")
            capture.clear()  # oublie l'audio d'avant l'appui
            audio = await record_one(capture, vad)
            if audio is None:
                print("   rien entendu, phrase sautée (relance pour la refaire)")
                continue
            write_wav(wav, audio, 16_000)
            wav.with_suffix(".txt").write_text(phrase + "\n", encoding="utf-8")
            print(f"   enregistré ({len(audio) / 16_000:.1f} s)")
    finally:
        capture.stop()
    print(f"\nJeu de test : {len(load_dataset(args.dir))} phrases dans {args.dir}")
    return 0


# -- mesure --------------------------------------------------------------------------------


@dataclass
class RunResult:
    compute_type: str
    beam_size: int
    wer: float
    latency_median_s: float
    latency_p95_s: float
    load_s: float
    worst: list[tuple[float, str, str]]  # (erreurs / mots, référence, hypothèse)


class _SyncSTT(Protocol):
    def load(self) -> None: ...
    def transcribe_sync(self, audio: np.ndarray) -> Any: ...
    def close(self) -> None: ...


def evaluate(
    stt: _SyncSTT, dataset: Sequence[tuple[np.ndarray, str]], compute_type: str, beam_size: int
) -> RunResult:
    from mikata.stt.text import clean_transcript

    start = time.perf_counter()
    stt.load()
    load_s = time.perf_counter() - start
    pairs, latencies, worst = [], [], []
    try:
        for audio, reference in dataset:
            result = stt.transcribe_sync(audio)
            hypothesis = clean_transcript(result.text)
            pairs.append((reference, hypothesis))
            latencies.append(result.latency_s)
            errors, words = word_errors(reference, hypothesis)
            if errors:
                worst.append((errors / max(words, 1), reference, hypothesis))
    finally:
        stt.close()
    worst.sort(key=lambda item: item[0], reverse=True)
    return RunResult(
        compute_type=compute_type,
        beam_size=beam_size,
        wer=corpus_wer(pairs),
        latency_median_s=statistics.median(latencies) if latencies else 0.0,
        latency_p95_s=percentile(latencies, 95),
        load_s=load_s,
        worst=worst[:5],
    )


def format_results(results: Sequence[RunResult], count: int) -> str:
    lines = [
        f"{count} commandes",
        f"{'compute_type':<14} {'beam':>4} {'WER':>7} {'méd. (ms)':>10} {'p95 (ms)':>9}",
    ]
    for r in results:
        lines.append(
            f"{r.compute_type:<14} {r.beam_size:>4} {r.wer * 100:>6.1f}% "
            f"{r.latency_median_s * 1000:>10.0f} {r.latency_p95_s * 1000:>9.0f}"
        )
    for r in results:
        if r.worst:
            lines.append(f"\nPires phrases ({r.compute_type}, beam {r.beam_size}) :")
            lines += [
                f"  {e * 100:4.0f}%  « {ref} » -> " + (f"« {hyp} »" if hyp else "(rien)")
                for e, ref, hyp in r.worst
            ]
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    from mikata.audio.wavfile import float_to_int16, read_wav
    from mikata.config import load_config
    from mikata.stt.whisper_local import WhisperSTT

    items = load_dataset(args.dir)
    if not items:
        print(f"Aucune paire .wav/.txt dans {args.dir} : lance d'abord « record ».")
        return 1
    dataset = []
    for wav, reference in items:
        pcm, rate = read_wav(wav)
        if rate != 16_000:
            print(f"{wav} : {rate} Hz, 16 kHz attendu", file=sys.stderr)
            return 1
        dataset.append((float_to_int16(pcm), reference))

    config = load_config(args.config)
    results = []
    for compute_type, beam_size in itertools.product(args.compute_type, args.beam_size):
        stt_cfg = config.stt.model_copy(
            update={
                "model": args.model or config.stt.model,
                "compute_type": compute_type,
                "beam_size": beam_size,
            }
        )
        print(f"{stt_cfg.model} {compute_type} beam {beam_size}...")
        stt = WhisperSTT(stt_cfg, download_root=config.resolve("models/whisper"))
        results.append(evaluate(stt, dataset, compute_type, beam_size))
    print("\n" + format_results(results, len(dataset)))
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dossier du jeu de test")
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="enregistrer le jeu de test")
    record.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES)
    run = sub.add_parser("run", help="mesurer WER et latence")
    run.add_argument("--model", default=None, help="modèle Whisper (défaut : config.yaml)")
    run.add_argument("--compute-type", nargs="+", default=["float16"])
    run.add_argument("--beam-size", nargs="+", type=int, default=[1])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from loguru import logger

    logger.remove()  # sortie lisible : seulement les messages du script
    logger.add(sys.stderr, level="WARNING")
    args = parse_args(argv)
    if args.command == "record":
        return asyncio.run(_record(args))
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
