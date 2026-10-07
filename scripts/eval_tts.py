"""Benchmark TTS (§8) : Piper ou Kokoro ?

Usage : uv run python scripts/eval_tts.py [--backends piper kokoro] [--runs 5] [--play]

Pour chaque moteur dont les modèles sont présents : temps de chargement, latence de synthèse
d'un fragment de 8 mots (le premier fragment d'une réponse : critère < 200 ms), facteur temps
réel sur des réponses typiques. Les audios sont écrits dans data/tts_eval/ pour les écouter
(`--play` les joue dans le casque de config.yaml). Le choix final se fait à l'oreille, avec
ces chiffres : `tts.backend` dans config.yaml.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "tts_eval"
FIRST_FRAGMENT = "D'accord, je lance la musique sur Spotify maintenant."  # 8 mots
SENTENCES = (
    FIRST_FRAGMENT,
    "Il est quinze heures quarante-deux.",
    "Il fait 18 degrés à Paris, avec quelques nuages et un vent léger.",
    "Je n'ai pas trouvé de fichier qui s'appelle rapport de stage dans tes documents.",
    "Ta carte graphique est à 54 degrés et le processeur tourne à 12 pour cent.",
)
TARGET_MS = 200.0


@dataclass
class Result:
    backend: str
    load_s: float
    first_fragment_ms: float
    rtf: float
    sample_rate: int
    files: list[Path]


def measure(tts: Any, name: str, runs: int, out_dir: Path) -> Result:
    from mikata.audio.wavfile import write_wav
    from mikata.tts.normalize import normalize_for_speech

    start = time.perf_counter()
    tts.load()
    load_s = time.perf_counter() - start
    tts.synthesize_sync("Bonjour.")  # préchauffage, comme au démarrage de MIKATA
    timings = []
    for _ in range(runs):
        t = time.perf_counter()
        tts.synthesize_sync(normalize_for_speech(FIRST_FRAGMENT))
        timings.append(time.perf_counter() - t)
    synth_s = audio_s = 0.0
    files = []
    for number, sentence in enumerate(SENTENCES, start=1):
        t = time.perf_counter()
        pcm = tts.synthesize_sync(normalize_for_speech(sentence))
        synth_s += time.perf_counter() - t
        audio_s += len(pcm) / tts.sample_rate
        path = out_dir / f"{name}_{number:02d}.wav"
        write_wav(path, pcm, tts.sample_rate)
        files.append(path)
    tts.close()
    return Result(
        backend=name,
        load_s=load_s,
        first_fragment_ms=statistics.median(timings) * 1000,
        rtf=synth_s / audio_s if audio_s else 0.0,
        sample_rate=tts.sample_rate,
        files=files,
    )


def format_results(results: Sequence[Result]) -> str:
    lines = [
        f"{'moteur':<8} {'chargement':>11} {'1er fragment':>13} {'temps réel':>11}  Hz",
    ]
    for r in results:
        flag = "OK" if r.first_fragment_ms < TARGET_MS else f"> {TARGET_MS:.0f} ms"
        lines.append(
            f"{r.backend:<8} {r.load_s:>9.1f} s {r.first_fragment_ms:>8.0f} ms {flag:>4}"
            f" {r.rtf:>9.2f}x  {r.sample_rate}"
        )
    lines.append(
        "\n1er fragment : synthèse d'une phrase de 8 mots (médiane). Temps réel : temps de "
        "calcul / durée de l'audio (< 1 = plus rapide que la parole)."
    )
    for r in results:
        lines.append(f"Écoute {r.backend} : {r.files[0]} ... {r.files[-1].name}")
    return "\n".join(lines)


def play(files: Sequence[Path], config: Any) -> None:
    import numpy as np
    import sounddevice as sd
    import soxr

    from mikata.audio.devices import collect_devices, default_device_index, find_device
    from mikata.audio.wavfile import read_wav

    device = find_device(
        collect_devices(sd),
        config.audio.output_device,
        "output",
        default_device_index(sd, "output"),
    )
    for path in files:
        pcm, rate = read_wav(path)
        print(f"  ♪ {path.name}")
        target = int(device.default_samplerate)  # WASAPI partagé : taux du périphérique imposé
        if rate != target:
            pcm = soxr.resample(pcm.astype(np.float32), rate, target)
        sd.play(pcm, target, device=device.index, blocking=True)


def main(argv: Sequence[str] | None = None) -> int:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    parser.add_argument("--backends", nargs="+", default=["piper", "kokoro"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--play", action="store_true", help="joue les audios dans le casque")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    from mikata.config import ConfigError, load_config
    from mikata.tts.base import create_tts

    config = load_config(args.config)
    results = []
    for backend in args.backends:
        tts, _ = create_tts(config, backend)
        print(f"{backend} : chargement et mesures...")
        try:
            results.append(measure(tts, backend, args.runs, args.out))
        except ConfigError as exc:
            print(f"[!!] {backend} : {exc}", file=sys.stderr)
    if not results:
        return 1
    print("\n" + format_results(results))
    if args.play:
        for r in results:
            print(f"\n{r.backend} :")
            play(r.files, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
