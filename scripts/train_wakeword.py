"""Entraînement de models/mikata.onnx (openWakeWord) sur Colab ou un PC Linux avec GPU — phase 7.

Ce script tourne **hors du projet** : `record_wakeword_samples.py pack` le met dans
data/wakeword/mikata_wakeword_colab.zip avec tes données. Sur Google Colab (Exécution >
Modifier le type d'exécution > GPU T4), après avoir déposé l'archive dans /content :

    !unzip -q -o /content/mikata_wakeword_colab.zip -d /content/mikata
    !python /content/mikata/train_wakeword.py setup    # ~15 min : environnement + données
    !python /content/mikata/train_wakeword.py train    # ~1 h 30 à 3 h sur un T4

Résultat : /content/mikata/out/mikata.onnx, à copier dans models/ sur ton PC.
Relancer `train` reprend où il s'était arrêté (chaque étape déjà faite est sautée).

Données d'entraînement (§4) :
- positifs : tes « MIKATA » (recopiés `--user-copies` fois : chaque copie reçoit un bruit,
  un écho et un égaliseur différents), plus des milliers de « Mikata » synthétiques dits par
  des centaines de voix Piper, françaises (accent français) et anglaises ;
- négatifs difficiles : tes mots proches (mikado, Nikita...) et les mêmes en voix synthétiques ;
- négatifs généraux : ~2 000 h de features publiques d'openWakeWord (ACAV100M), plus ta vie
  courante (data/negatives), qui sert aussi de bruit de fond mélangé aux positifs.
Le jeu d'évaluation (tes 30 prises « eval » et 2 h de vie courante) n'est pas dans l'archive.

Dépendances : installées par `setup` dans un environnement Python 3.11 à part (versions
épinglées, compatibles avec openwakeword 0.6.0), jamais dans le projet MIKATA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Colab fixe MPLBACKEND sur un backend de Jupyter absent de l'environnement 3.11 isolé :
# matplotlib plante au chargement (openwakeword -> torchmetrics). On impose un backend sans écran.
os.environ["MPLBACKEND"] = "agg"

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
VENV_PYTHON = "3.11"

# Versions qui marchent ensemble avec openwakeword 0.6.0 (son notebook d'entraînement), Python 3.11.
TORCH = ["torch==2.1.2", "torchaudio==2.1.2"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
PACKAGES = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "scikit-learn==1.3.2",
    "pyyaml==6.0.1",
    "tqdm==4.66.4",
    "requests==2.32.3",
    # Features (melspectrogramme + embedding) sur CPU : onnxruntime-gpu exigerait un CUDA et un
    # cuDNN précis ; le GPU sert à l'augmentation et à l'entraînement (PyTorch).
    "onnxruntime==1.17.1",
    "soundfile==0.12.1",  # lecture des .wav par torchaudio
    "speechbrain==0.5.16",
    "audiomentations==0.33.0",
    "torch-audiomentations==0.11.0",
    "acoustics==0.2.6",
    "mutagen==1.47.0",
    "pronouncing==0.2.0",
    "torchinfo==1.8.0",
    "torchmetrics==1.2.0",
    "pathvalidate==3.2.0",
    "setuptools==69.5.1",  # pronouncing a besoin de pkg_resources (retiré des setuptools récents)
    "onnx==1.15.0",  # export du modèle (torch.onnx.export)
]
NO_DEPS = ["openwakeword==0.6.0", "piper-tts==1.8.0"]  # sans tflite-runtime ni onnxruntime CPU

OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
FEATURES_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
ACAV_FILE = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"  # ~17 Go
VALIDATION_FILE = "validation_set_features.npy"  # ~11 h, mesure des faux déclenchements
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
RIR_URL = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"  # réponses de pièces (MIT)

# Voix Piper : (chemin dans piper-voices, langue). mls : 125 locuteurs français ; libritts_r :
# 904 locuteurs anglais ; vctk : 109 locuteurs anglais (accents britanniques variés).
VOICES = (
    ("fr/fr_FR/mls/medium/fr_FR-mls-medium", "fr"),
    ("fr/fr_FR/upmc/medium/fr_FR-upmc-medium", "fr"),
    ("fr/fr_FR/siwis/medium/fr_FR-siwis-medium", "fr"),
    ("fr/fr_FR/tom/medium/fr_FR-tom-medium", "fr"),
    ("fr/fr_FR/gilles/low/fr_FR-gilles-low", "fr"),
    ("en/en_US/libritts_r/medium/en_US-libritts_r-medium", "en"),
    ("en/en_GB/vctk/medium/en_GB-vctk-medium", "en"),
)
# Variantes de prononciation (§4) : mikata, mi kata, mee kah tah.
POSITIVE_TEXTS = {
    "fr": ("Mikata", "Mikata !", "Mikata ?", "Mikata.", "Mi-kata", "Mikata,"),
    "en": ("mikata", "Mikata!", "mee kah tah", "mi kata", "Mee-kah-tah."),
}
# Négatifs difficiles : mots proches (les 12 que tu enregistres, plus d'autres).
NEGATIVE_TEXTS = {
    "fr": (
        "mikado", "Nikita", "Akira", "il m'a quitté", "mi-carré", "Mika", "ma carte", "mi casa",
        "Takata", "chiquita", "Michaela", "Nakata", "Micah", "Mickaël", "mi-temps", "ma cata",
        "Hakata", "Sakura", "Nikata", "Mi kado", "Kata", "Mikasa", "Pikachu", "il m'a cassé",
        "ni gâté", "mi-carême", "Makita", "un mikado", "Mikael", "Tikka",
    ),
    "en": ("mikado", "Nikita", "Akira", "Makita", "Mikasa", "Michael", "Takata", "Micah",
           "my kettle", "me cutter"),
}  # fmt: skip


@dataclass
class Settings:
    workdir: Path
    data: Path
    n_positive: int = 20_000
    n_negative: int = 10_000
    n_validation: int = 2_000
    user_copies: int = 30
    steps: int = 50_000
    seed: int = 0

    @property
    def venv(self) -> Path:
        return self.workdir / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def downloads(self) -> Path:
        return self.workdir / "downloads"

    @property
    def model_dir(self) -> Path:
        return self.workdir / "oww" / "mikata"


def log(message: str) -> None:
    print(f"[mikata] {message}", flush=True)


# -- installation ----------------------------------------------------------------------------


def run(cmd: Sequence[str | Path], **kwargs: Any) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def download(url: str, dest: Path) -> Path:
    """Télécharge `url` vers `dest` (sauté si déjà là)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"téléchargement {url}")
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 22)
    tmp.replace(dest)
    return dest


def setup(s: Settings) -> None:
    s.workdir.mkdir(parents=True, exist_ok=True)
    if not s.python.is_file():
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        run([sys.executable, "-m", "uv", "venv", "--python", VENV_PYTHON, s.venv])
    uv = [sys.executable, "-m", "uv", "pip", "install", "--python", s.python]
    run([*uv, *TORCH, "--extra-index-url", TORCH_INDEX])
    run([*uv, *PACKAGES])
    run([*uv, "--no-deps", *NO_DEPS])
    where = "import openwakeword, os; print(os.path.dirname(openwakeword.__file__))"
    package = subprocess.run([s.python, "-c", where], check=True, capture_output=True, text=True)
    resources = Path(package.stdout.strip()) / "resources" / "models"
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        download(f"{OWW_RELEASE}/{name}", resources / name)
    for path, _ in VOICES:
        for suffix in (".onnx", ".onnx.json"):
            download(
                f"{PIPER_URL}/{path}{suffix}", s.downloads / "voices" / (Path(path).name + suffix)
            )
    download(f"{FEATURES_URL}/{VALIDATION_FILE}", s.downloads / VALIDATION_FILE)
    download(f"{FEATURES_URL}/{ACAV_FILE}", s.downloads / ACAV_FILE)
    try:
        download(RIR_URL, s.downloads / "mit_rirs.zip")
    except OSError as exc:  # facultatif : sans réponses de pièces, pas d'écho simulé
        log(f"réponses de pièces indisponibles ({exc}) : entraînement sans écho simulé")
    log("installation terminée ; lance maintenant : train")


# -- préparation des données (dans l'environnement Python 3.11) ------------------------------


def read_wav(path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
        channels, width = w.getnchannels(), w.getsampwidth()
    if width == 1:  # 8 bits non signés
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.int32) - 128) << 8
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.int32)
    elif width == 3:  # 24 bits : on garde les 16 bits de poids fort
        raw = np.frombuffer(frames[: len(frames) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        audio = raw[:, 1].astype(np.int32) | (raw[:, 2].astype(np.int8).astype(np.int32) << 8)
    elif width == 4:  # 32 bits entiers
        audio = np.frombuffer(frames[: len(frames) // 4 * 4], dtype=np.int32) >> 16
    else:
        raise wave.Error(f"largeur d'échantillon non gérée : {width} octets")
    if channels > 1:
        audio = audio[: len(audio) // channels * channels].reshape(-1, channels).mean(axis=1)
    return np.clip(audio, -32768, 32767).astype(np.int16), rate


def write_wav(path: Path, audio: Any, rate: int = SAMPLE_RATE) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(audio, dtype=np.int16).tobytes())


def to_16k(audio: Any, rate: int) -> Any:
    """Rééchantillonne en 16 kHz int16 (22 050 Hz pour la plupart des voix Piper)."""
    import numpy as np
    from scipy.signal import resample_poly

    if rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.int16)
    g = np.gcd(SAMPLE_RATE, rate)
    out = resample_poly(np.asarray(audio, dtype=np.float32), SAMPLE_RATE // g, rate // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


def trim(audio: Any, margin_s: float = 0.1) -> Any:
    """Silence autour de la voix retiré (énergie par fenêtres de 10 ms)."""
    import numpy as np

    win = SAMPLE_RATE // 100
    n = len(audio) // win
    if n < 5:
        return audio
    rms = np.sqrt((audio[: n * win].astype(np.float32).reshape(n, win) ** 2).mean(axis=1))
    floor, peak = float(np.percentile(rms, 5)), float(rms.max())
    if peak < max(4 * floor, 300.0):
        return audio
    voiced = np.nonzero(rms >= max(4 * floor, 0.02 * peak))[0]
    margin = round(margin_s * SAMPLE_RATE)
    return audio[
        max(0, voiced[0] * win - margin) : min(len(audio), (voiced[-1] + 1) * win + margin)
    ]


def count_wavs(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.wav")) if directory.is_dir() else 0


def copy_user_clips(sources: Iterable[Path], dest: Path, copies: int, prefix: str) -> int:
    """Tes prises, coupées autour de la voix, recopiées `copies` fois (l'augmentation donne à
    chaque copie un bruit et un écho différents)."""
    n = 0
    for source in sorted(sources):
        audio, rate = read_wav(source)
        clip = trim(to_16k(audio, rate))
        for k in range(copies):
            write_wav(dest / f"{prefix}_{source.stem}_{k:02d}.wav", clip)
            n += 1
    return n


@dataclass(frozen=True)
class TtsJob:
    voice: str  # nom du fichier .onnx
    speaker: int | None
    text: str
    length_scale: float
    noise_scale: float
    noise_w: float


def tts_jobs(
    voices: Sequence[tuple[str, str, int]],
    texts: dict[str, Sequence[str]],
    count: int,
    rng: random.Random,
) -> list[TtsJob]:
    """`count` synthèses : voix et locuteur au hasard (moitié français, moitié anglais),
    texte, débit et timbre variés. `voices` : (nom, langue, nombre de locuteurs)."""
    by_lang: dict[str, list[tuple[str, int]]] = {}
    for name, lang, speakers in voices:
        if lang in texts:
            by_lang.setdefault(lang, []).append((name, speakers))
    langs = sorted(by_lang)
    jobs = []
    for i in range(count):
        lang = langs[i % len(langs)]
        name, speakers = rng.choice(by_lang[lang])
        jobs.append(
            TtsJob(
                voice=name,
                speaker=rng.randrange(speakers) if speakers > 1 else None,
                text=rng.choice(list(texts[lang])),
                length_scale=rng.uniform(0.75, 1.35),
                noise_scale=rng.uniform(0.4, 0.9),
                noise_w=rng.uniform(0.5, 1.0),
            )
        )
    return jobs


def synthesize(jobs: Sequence[TtsJob], voices_dir: Path, dest: Path, prefix: str) -> None:
    """Synthèse Piper (CPU), 16 kHz, silence retiré. Reprend où elle en était."""
    import numpy as np
    from piper import PiperVoice, SynthesisConfig

    cache: dict[str, Any] = {}
    for i, job in enumerate(jobs):
        path = dest / f"{prefix}_{i:06d}.wav"
        if path.exists():
            continue
        voice = cache.get(job.voice)
        if voice is None:
            voice = cache[job.voice] = PiperVoice.load(voices_dir / job.voice)
        config = SynthesisConfig(
            speaker_id=job.speaker,
            length_scale=job.length_scale,
            noise_scale=job.noise_scale,
            noise_w_scale=job.noise_w,
        )
        chunks = list(voice.synthesize(job.text, config))
        if not chunks:
            continue
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        write_wav(path, trim(to_16k(audio, chunks[0].sample_rate)))
        if i % 1000 == 0:
            log(f"{prefix} : {i}/{len(jobs)}")


def voice_inventory(voices_dir: Path) -> list[tuple[str, str, int]]:
    """(fichier .onnx, langue, nombre de locuteurs) des voix téléchargées."""
    found = []
    for path, lang in VOICES:
        model = voices_dir / (Path(path).name + ".onnx")
        config = model.with_name(model.name + ".json")
        if model.is_file() and config.is_file():
            speakers = int(json.loads(config.read_text(encoding="utf-8")).get("num_speakers", 1))
            found.append((model.name, lang, speakers))
    return found


def negative_windows(embeddings: Any, size: int = 16) -> Any:
    """Features (trames, 96) -> fenêtres (n, 16, 96) sans recouvrement, comme ACAV100M."""
    import numpy as np

    n = embeddings.shape[0] // size
    return np.asarray(embeddings[: n * size], dtype=np.float16).reshape(n, size, -1)


def user_negative_features(clips: Sequence[Path], output: Path) -> int:
    """Features openWakeWord de ta vie courante, pour l'entraînement (négatifs)."""
    import numpy as np
    from openwakeword.utils import AudioFeatures

    if output.is_file():
        return int(np.load(output, mmap_mode="r").shape[0])
    features = AudioFeatures(device="cpu", ncpu=os.cpu_count() or 1)
    windows = []
    for path in clips:
        audio, rate = read_wav(path)
        audio = to_16k(audio, rate)
        if len(audio) >= SAMPLE_RATE * 2:
            windows.append(negative_windows(features._get_embeddings(audio)))
    stacked = np.concatenate(windows) if windows else np.zeros((0, 16, 96), dtype=np.float16)
    np.save(output, stacked)
    return int(stacked.shape[0])


def extract_rirs(archive: Path, dest: Path) -> list[Path]:
    """Réponses de pièces du MIT (32 kHz) -> 16 kHz."""
    import zipfile

    # Marqueur d'extraction complète : HORS de `dest`, car openwakeword lit tout ce que `dest`
    # contient comme un son.
    done = dest.parent / ".rirs16k_extrait"
    if done.is_file():
        return sorted(dest.glob("*.wav"))
    if not archive.is_file():
        return []
    # Les fichiers d'origine vont HORS de `dest` : openwakeword lit tout ce que contient `dest`
    # et plante sur un sous-dossier.
    raw_dir = dest.parent / "rirs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.lower().endswith(".wav"):
                raw = raw_dir / Path(name).name
                raw.write_bytes(z.read(name))
                try:
                    audio, rate = read_wav(raw)
                except (wave.Error, EOFError, ValueError):  # fichier illisible : on l'ignore
                    log(f"réponse de pièce ignorée (format non géré) : {raw.name}")
                    continue
                write_wav(dest / raw.name, to_16k(audio, rate))
    done.write_text("ok", encoding="utf-8")
    log(f"réponses de pièces : {count_wavs(dest)} fichiers")
    return sorted(dest.glob("*.wav"))


def training_config(
    s: Settings,
    background: Sequence[Path],
    rir_dirs: Sequence[Path],
    user_features: Path | None,
    stub_dir: Path,
) -> dict[str, Any]:
    """Configuration de openwakeword/train.py (clés de son examples/custom_model.yml)."""
    features = {"ACAV100M_sample": str(s.downloads / ACAV_FILE)}
    per_class = {"ACAV100M_sample": 1024, "adversarial_negative": 64, "positive": 64}
    if user_features is not None:
        features["mikata_vie_courante"] = str(user_features)
        per_class["mikata_vie_courante"] = 256
    return {
        "model_name": "mikata",
        "target_phrase": ["mikata"],
        "custom_negative_phrases": [],
        "n_samples": s.n_positive,
        "n_samples_val": s.n_validation,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(stub_dir),
        "output_dir": str(s.model_dir.parent),
        "rir_paths": [str(d) for d in rir_dirs],
        "background_paths": [str(d) for d in background],
        "background_paths_duplication_rate": [1] * len(background),
        "false_positive_validation_data_path": str(s.downloads / VALIDATION_FILE),
        "augmentation_rounds": 1,
        "feature_data_files": features,
        "batch_n_per_class": per_class,
        "model_type": "dnn",
        "layer_size": 32,
        "steps": s.steps,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }


STUB = '''"""Remplace piper-sample-generator : les clips sont déjà générés par train_wakeword.py."""


def generate_samples(*args, **kwargs):
    raise RuntimeError("clips déjà préparés : ne pas utiliser --generate_clips")
'''


def fill(dest: Path, target: int, make: Callable[[int], None]) -> None:
    have = count_wavs(dest)
    if have < target:
        make(target - have)
    log(f"{dest.name} : {count_wavs(dest)} clips")


def prepare(s: Settings) -> Path:
    """Clips et features pour openwakeword/train.py ; renvoie le fichier de configuration."""
    import yaml

    rng = random.Random(s.seed)
    data, out = s.data, s.model_dir
    voices_dir = s.downloads / "voices"
    voices = voice_inventory(voices_dir)
    if not voices:
        raise SystemExit("Aucune voix Piper : lance d'abord « setup ».")
    user_pos = sorted((data / "positives").glob("*.wav"))
    user_neg = sorted((data / "hard_negatives").glob("*.wav"))
    log(f"tes prises : {len(user_pos)} « MIKATA », {len(user_neg)} mots proches")

    pos_train, pos_test = out / "positive_train", out / "positive_test"
    neg_train, neg_test = out / "negative_train", out / "negative_test"
    if not any(pos_train.glob("user_*.wav")):
        copy_user_clips(user_pos, pos_train, s.user_copies, "user")
    if not any(neg_train.glob("user_*.wav")):
        copy_user_clips(user_neg, neg_train, s.user_copies, "user")
    for dest, texts, total, prefix in (
        (pos_train, POSITIVE_TEXTS, s.n_positive, "tts"),
        (pos_test, POSITIVE_TEXTS, s.n_validation, "tts"),
        (neg_train, NEGATIVE_TEXTS, s.n_negative, "tts"),
        (neg_test, NEGATIVE_TEXTS, s.n_validation, "tts"),
    ):
        user = sum(1 for _ in dest.glob("user_*.wav"))
        jobs = tts_jobs(voices, texts, max(0, total - user), random.Random(rng.random()))
        fill(dest, total, lambda _n, j=jobs, d=dest, p=prefix: synthesize(j, voices_dir, d, p))

    background = [data / "negatives"] if count_wavs(data / "negatives") else []
    user_features = None
    if background:
        user_features = out.parent / "mikata_vie_courante_features.npy"
        n = user_negative_features(sorted((data / "negatives").glob("*.wav")), user_features)
        log(f"vie courante : {n} fenêtres de 1,28 s")
    rirs = s.downloads / "rirs16k"
    rir_dirs = [rirs] if extract_rirs(s.downloads / "mit_rirs.zip", rirs) else []
    stub_dir = s.workdir / "stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "generate_samples.py").write_text(STUB, encoding="utf-8")
    if not background:
        log("aucune vie courante dans l'archive : pas de bruit de fond mélangé aux positifs")
    config = training_config(s, background, rir_dirs, user_features, stub_dir)
    config["n_samples"] = count_wavs(pos_train)
    config["n_samples_val"] = count_wavs(pos_test)
    path = s.workdir / "mikata.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


FEATURE_FILES = (
    "positive_features_train.npy",
    "positive_features_test.npy",
    "negative_features_train.npy",
    "negative_features_test.npy",
)


def incomplete_features(model_dir: Path) -> bool:
    """Vrai si un essai interrompu a laissé une partie seulement des caractéristiques :
    openwakeword saute alors leur calcul dès que le premier fichier existe, puis plante."""
    present = [(model_dir / name).is_file() for name in FEATURE_FILES]
    return any(present) and not all(present)


REQUIRED_MODULES = (
    "torch", "torchinfo", "torchmetrics", "scipy", "yaml", "tqdm", "openwakeword", "onnx",
    "pkg_resources", "matplotlib", "pronouncing", "speechbrain", "audiomentations",
    "torch_audiomentations", "acoustics", "mutagen", "soundfile", "pathvalidate",
)  # fmt: skip


def missing_modules(names: Iterable[str] = REQUIRED_MODULES) -> list[str]:
    """Modules dont openwakeword/train.py a besoin et qui manquent à l'environnement : on le
    sait tout de suite plutôt qu'après 30 minutes d'entraînement."""
    import importlib.util

    return [name for name in names if importlib.util.find_spec(name) is None]


def train(s: Settings) -> int:
    if Path(sys.prefix).resolve() != s.venv.resolve():  # relance dans l'environnement 3.11
        if not s.python.is_file():
            raise SystemExit("Environnement absent : lance d'abord « setup ».")
        return subprocess.run([str(s.python), __file__, *sys.argv[1:]]).returncode
    missing = missing_modules()
    if missing:
        raise SystemExit(
            "Modules manquants dans l'environnement d'entraînement : "
            + ", ".join(missing)
            + ". Relance « setup » ou installe-les avec uv pip install --python <venv>/bin/python."
        )
    config = prepare(s)
    exported = s.model_dir.parent / "mikata.onnx"
    command = [sys.executable, "-m", "openwakeword.train", "--training_config", str(config),
               "--augment_clips", "--train_model"]  # fmt: skip
    if incomplete_features(s.model_dir):
        log("caractéristiques d'un essai interrompu : elles sont recalculées")
        command.append("--overwrite")
    result = subprocess.run(command)
    # La conversion finale en .tflite échoue sans TensorFlow : seul le .onnx compte (§4).
    if not exported.is_file():
        log(f"échec de l'entraînement (code {result.returncode}) : voir les messages ci-dessus")
        return 1
    final = s.workdir / "out" / "mikata.onnx"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, final)
    log(f"modèle prêt : {final} ({final.stat().st_size // 1024} Ko) -> models/mikata.onnx")
    return 0


def parse_args(argv: Sequence[str] | None) -> tuple[str, Settings]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("setup", "train"))
    parser.add_argument("--workdir", type=Path, default=HERE, help="dossier de travail")
    parser.add_argument("--data", type=Path, default=HERE / "data", help="données de l'archive")
    parser.add_argument("--positives", type=int, default=Settings.n_positive)
    parser.add_argument("--negatives", type=int, default=Settings.n_negative)
    parser.add_argument("--validation", type=int, default=Settings.n_validation)
    parser.add_argument("--user-copies", type=int, default=Settings.user_copies)
    parser.add_argument("--steps", type=int, default=Settings.steps)
    args = parser.parse_args(argv)
    settings = Settings(
        workdir=args.workdir.resolve(),
        data=args.data.resolve(),
        n_positive=args.positives,
        n_negative=args.negatives,
        n_validation=args.validation,
        user_copies=args.user_copies,
        steps=args.steps,
    )
    return args.command, settings


def main(argv: Sequence[str] | None = None) -> int:
    command, settings = parse_args(argv)
    if command == "setup":
        setup(settings)
        return 0
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return train(settings)


if __name__ == "__main__":
    sys.exit(main())
