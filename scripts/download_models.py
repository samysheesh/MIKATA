"""Télécharge les modèles openWakeWord dans models/ (phase 1, §4).

Usage : uv run python scripts/download_models.py

- hey_jarvis_v0.1.onnx  -> models/hey_jarvis.onnx (modèle de développement)
- melspectrogram.onnx, embedding_model.onnx (features, communs à tous les modèles)

Les fichiers déjà présents sont conservés (jamais écrasés).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

RELEASE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MODELS = {
    "hey_jarvis.onnx": "hey_jarvis_v0.1.onnx",
    "melspectrogram.onnx": "melspectrogram.onnx",
    "embedding_model.onnx": "embedding_model.onnx",
}
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"


def download(url: str, dest: Path) -> int:
    """Télécharge vers `dest` via un fichier temporaire ; renvoie la taille en octets."""
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
        size = 0
        while chunk := response.read(1 << 16):
            out.write(chunk)
            size += len(chunk)
    if size == 0:
        raise OSError(f"fichier vide reçu depuis {url}")
    tmp.replace(dest)
    return size


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="dossier cible")
    args = parser.parse_args(argv)
    args.dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for local_name, remote_name in MODELS.items():
        dest = args.dir / local_name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[OK] {dest} (déjà présent)")
            continue
        try:
            size = download(f"{RELEASE_URL}/{remote_name}", dest)
        except OSError as exc:
            print(f"[!!] {local_name} : échec du téléchargement : {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"[OK] {dest} ({size / 1e6:.1f} Mo)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
