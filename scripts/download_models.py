"""Télécharge les modèles dans models/ (phases 1 à 3).

Usage : uv run python scripts/download_models.py [--no-whisper]

- openWakeWord (§4) : hey_jarvis_v0.1.onnx -> models/hey_jarvis.onnx (développement),
  melspectrogram.onnx et embedding_model.onnx (features communs à tous les modèles)
- Silero VAD v5 (§5) : models/silero_vad.onnx
- Voix (§8) : Piper fr_FR-siwis-medium (.onnx + .onnx.json, ~63 Mo) et Kokoro-82M en ONNX
  fp32 + fichier des voix (~350 Mo) ; les deux, pour pouvoir les comparer (eval_tts.py)
- Whisper (§5) : le modèle `stt.model` de config.yaml (large-v3-turbo, ~1,6 Go) dans
  models/whisper/, là où MIKATA le cherche au démarrage

Les fichiers déjà présents sont conservés (jamais écrasés).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

OWW_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODELS = {
    "hey_jarvis.onnx": f"{OWW_URL}/hey_jarvis_v0.1.onnx",
    "melspectrogram.onnx": f"{OWW_URL}/melspectrogram.onnx",
    "embedding_model.onnx": f"{OWW_URL}/embedding_model.onnx",
    "silero_vad.onnx": f"{SILERO_URL}/silero_vad.onnx",
    "fr_FR-siwis-medium.onnx": f"{PIPER_URL}/fr_FR-siwis-medium.onnx",
    "fr_FR-siwis-medium.onnx.json": f"{PIPER_URL}/fr_FR-siwis-medium.onnx.json",
    "kokoro-v1.0.onnx": f"{KOKORO_URL}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{KOKORO_URL}/voices-v1.0.bin",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "models"
WHISPER_SUBDIR = "whisper"


def _print_progress(name: str, done: int, total: int) -> None:
    """Barre de progression sur une ligne (rien si sortie non interactive)."""
    if not sys.stdout.isatty():
        return
    width = 30
    if total > 0:
        frac = min(done / total, 1.0)
        filled = int(width * frac)
        bar = "█" * filled + "─" * (width - filled)
        line = f"\r{name} [{bar}] {100 * frac:5.1f}% {done / 1e6:.1f}/{total / 1e6:.1f} Mo"
    else:
        line = f"\r{name} … {done / 1e6:.1f} Mo téléchargés"
    print(line, end="", flush=True)


def download(url: str, dest: Path) -> int:
    """Télécharge vers `dest` via un fichier temporaire ; renvoie la taille en octets."""
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        length = response.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else 0
        size = 0
        last = 0.0
        _print_progress(dest.name, 0, total)
        while chunk := response.read(1 << 16):
            out.write(chunk)
            size += len(chunk)
            now = time.monotonic()
            if now - last >= 0.1:  # ~10 Hz, pas de spam stdout
                _print_progress(dest.name, size, total)
                last = now
        _print_progress(dest.name, size, total or size)
    if sys.stdout.isatty():
        print()  # termine la ligne de progression
    if size == 0:
        raise OSError(f"fichier vide reçu depuis {url}")
    tmp.replace(dest)
    return size


def download_whisper(model: str, cache_dir: Path) -> Path:
    """Télécharge (ou retrouve) le modèle faster-whisper dans `cache_dir`."""
    from faster_whisper.utils import download_model

    return Path(download_model(model, cache_dir=str(cache_dir)))


def whisper_model_name(config_path: Path | None) -> str:
    try:
        from mikata.config import ConfigError, load_config

        return load_config(config_path).stt.model
    except ConfigError as exc:
        print(f"[!!] config.yaml illisible, modèle Whisper par défaut : {exc}", file=sys.stderr)
        return "large-v3-turbo"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dossier cible")
    parser.add_argument("--config", type=Path, default=None, help="chemin de config.yaml")
    parser.add_argument(
        "--no-whisper", action="store_true", help="ne pas télécharger le modèle Whisper"
    )
    args = parser.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for local_name, url in MODELS.items():
        dest = args.dir / local_name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[OK] {dest} (déjà présent)")
            continue
        try:
            size = download(url, dest)
        except OSError as exc:
            print(f"[!!] {local_name} : échec du téléchargement : {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"[OK] {dest} ({size / 1e6:.1f} Mo)")

    if not args.no_whisper:
        model = whisper_model_name(args.config)
        print(f"Whisper {model} : vérification / téléchargement (~1,6 Go la première fois)...")
        try:
            path = download_whisper(model, args.dir / WHISPER_SUBDIR)
        except Exception as exc:  # réseau, disque, nom de modèle inconnu
            print(f"[!!] Whisper {model} : échec du téléchargement : {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"[OK] Whisper {model} : {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
