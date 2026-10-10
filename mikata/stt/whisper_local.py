"""faster-whisper `large-v3-turbo` sur la RTX 3080, chargé h24 (§5).

Piège Windows n°1 : CTranslate2 a besoin des DLL CUDA 12 cuBLAS et cuDNN 9. Elles viennent
des paquets pip `nvidia-cublas-cu12` et `nvidia-cudnn-cu12` (dossiers `nvidia/*/bin`), qu'on
ajoute au chemin de recherche des DLL **avant** d'importer faster-whisper.
Aucune bascule CPU automatique (décision §1) : sans GPU, erreur claire au démarrage.
L'amorçage des DLL CUDA et la vérification du GPU via ctranslate2 ne concernent que
Windows (`sys.platform == "win32"`) : sous macOS, `stt.device = "mlx"` sélectionne le
backend MLX (`mikata/stt/mlx_whisper.py`, Apple Silicon), avec repli faster-whisper CPU
si `mlx-whisper` est absent.

Ajout phase 6 : le prompt initial de Whisper (`stt.initial_prompt`, vocabulaire fixe :
« MIKATA », Bloc-notes, minuteur...) est complété par les noms des applications installées
(`stt.prompt_apps`), dans la limite de `stt.prompt_max_chars` : Whisper écrit plus souvent
« Spotify » ou « Counter-Strike 2 » que « spot if aï ». Effet mesurable avec
`scripts/eval_stt.py run --prompt none config apps`.

Phase 6b : la langue reste forcée (`language="fr"`, jamais de détection automatique), et
`stt.initial_prompt` finit par des amorces de suites courtes (« Et à Tokyo ? Et demain ? ») :
sur deux mots, Whisper lisait « Yeah, Tokyo ». Les noms d'applications passent donc
**avant** `stt.initial_prompt` : la fin du prompt est ce que Whisper lit juste avant ta
phrase, et ce qu'il garde si le prompt est trop long (il coupe le début).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.config import StartupError, SttConfig
from mikata.stt.base import Transcript

CUDA_PACKAGES = ("cublas", "cudnn")
WARMUP_S = 1.0
APP_NAME_MAX_WORDS = 4  # « Microsoft Flight Simulator 2024 » ; plus long : rarement dit tel quel
_APP_NAME_SKIP = re.compile(r"[\\/:]|https?|\.exe\b|\(x86\)|\bx64\b", re.IGNORECASE)


def build_prompt(base: str, app_names: Sequence[str] = (), max_chars: int = 500) -> str:
    """Prompt initial de Whisper : des noms d'applications tant qu'ils tiennent dans
    `max_chars` (les noms courts d'abord : ce sont ceux qu'on dit), puis `base`.

    `base` est à la fin : c'est le texte le plus proche de ta phrase (amorces « Et à
    Tokyo ? »), et Whisper ne garde que les ~220 derniers tokens de prompt.
    """
    prompt = base.strip()
    seen = {name.casefold() for name in re.split(r"[,.;]\s*", prompt) if name}
    names = []
    for name in app_names:
        name = " ".join(name.split())
        key = name.casefold()
        if key in seen or not name or _APP_NAME_SKIP.search(name):
            continue
        if len(name.split()) > APP_NAME_MAX_WORDS:
            continue
        seen.add(key)
        names.append(name)
    names.sort(key=lambda n: (len(n.split()), len(n), n.casefold()))
    picked: list[str] = []
    size = len(prompt) + len(". Applications : ")
    for name in names:
        if size + len(name) + 2 > max_chars:
            break
        picked.append(name)
        size += len(name) + 2
    if not picked:
        return prompt
    picked.sort(key=str.casefold)
    apps = f"Applications : {', '.join(picked)}."
    return f"{apps} {prompt}" if prompt else apps


class SttUnavailableError(StartupError):
    """Whisper ne peut pas tourner (GPU absent, DLL CUDA manquantes, modèle illisible)."""


def cuda_dll_directories() -> list[Path]:
    """Dossiers `nvidia/<paquet>/bin` installés par pip (Windows)."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    dirs = []
    for root in spec.submodule_search_locations:
        for package in CUDA_PACKAGES:
            candidate = Path(root) / package / "bin"
            if candidate.is_dir():
                dirs.append(candidate)
    return dirs


def add_cuda_dll_directories() -> list[Path]:
    """Rend les DLL CUDA des paquets pip trouvables. À appeler avant `import faster_whisper`.

    `os.add_dll_directory` sert aux DLL chargées par Python ; cuDNN charge ensuite ses
    sous-DLL avec `LoadLibrary`, qui ne regarde que le PATH : on ajoute donc aussi au PATH.
    """
    if sys.platform != "win32":
        return []
    dirs = cuda_dll_directories()
    for directory in dirs:
        os.add_dll_directory(str(directory))  # type: ignore[attr-defined,unused-ignore]
    if dirs:
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
    return dirs


def _explain_cuda_error(exc: Exception) -> SttUnavailableError:
    message = str(exc)
    lowered = message.lower()
    if any(name in lowered for name in ("cublas", "cudnn", ".dll", "cannot be loaded")):
        hint = (
            "DLL CUDA introuvables (cuBLAS / cuDNN 9). Lance « uv sync » pour installer "
            "nvidia-cublas-cu12 et nvidia-cudnn-cu12."
        )
    elif "out of memory" in lowered:
        hint = "Mémoire GPU insuffisante : essaie stt.compute_type = int8_float16."
    else:
        hint = "Vérifie le pilote NVIDIA (nvidia-smi doit afficher la RTX 3080)."
    return SttUnavailableError(f"Whisper ne peut pas utiliser le GPU : {message}\n{hint}")


def gpu_memory_used_mb() -> float | None:
    """Mémoire GPU utilisée (tous processus), via NVML ; None si indisponible."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return float(info.used) / 1024**2
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _reliability(segments: Sequence[Any]) -> tuple[float | None, float | None]:
    """(no_speech_prob le plus haut, avg_logprob moyen pondéré par la longueur du texte)."""
    no_speech = [
        s.no_speech_prob for s in segments if getattr(s, "no_speech_prob", None) is not None
    ]
    weighted = [
        (s.avg_logprob, max(1, len(s.text.strip())))
        for s in segments
        if getattr(s, "avg_logprob", None) is not None
    ]
    total = sum(weight for _, weight in weighted)
    logprob = sum(value * weight for value, weight in weighted) / total if total else None
    return (max(no_speech) if no_speech else None), logprob


class WhisperSTT:
    """Implémente l'interface `STT`. Un executor dédié à 1 worker (§3) : un seul calcul GPU
    à la fois, la boucle asyncio n'est jamais bloquée."""

    def __init__(
        self,
        cfg: SttConfig,
        download_root: Path,
        vocabulary: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        """`vocabulary` (ajout phase 6) : noms des applications installées, lus à chaque
        transcription (liste en mémoire, jamais bloquante) si `stt.prompt_apps`."""
        self.cfg = cfg
        self.download_root = download_root
        self.vocabulary = vocabulary
        self._model: Any = None
        self._delegate: Any = None  # MlxWhisperSTT (macOS, stt.device = "mlx"), import paresseux
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-stt")
        self._prompt_key: tuple[str, ...] | None = None
        self._prompt = cfg.initial_prompt

    @property
    def prompt(self) -> str:
        """Prompt initial de la prochaine transcription (recalculé si les applis changent)."""
        if not self.cfg.prompt_apps or self.vocabulary is None:
            return self.cfg.initial_prompt
        try:
            names = tuple(self.vocabulary())
        except Exception as exc:  # vocabulaire facultatif
            logger.debug("Noms d'applications indisponibles pour Whisper : {!r}", exc)
            return self._prompt
        if names != self._prompt_key:
            self._prompt_key = names
            self._prompt = build_prompt(self.cfg.initial_prompt, names, self.cfg.prompt_max_chars)
            logger.debug("Prompt Whisper ({} caractères) : {}", len(self._prompt), self._prompt)
        return self._prompt

    @property
    def loaded(self) -> bool:
        if self._delegate is not None:
            return bool(self._delegate.loaded)
        return self._model is not None

    def _load_mlx(self) -> None:
        """Backend MLX (macOS) ; repli faster-whisper CPU si `mlx-whisper` est absent."""
        try:
            from mikata.stt.mlx_whisper import MlxWhisperSTT

            delegate = MlxWhisperSTT(self.cfg, self.download_root, self.vocabulary)
            delegate.load()
        except Exception as exc:
            logger.warning(
                "Backend MLX indisponible ({}), repli faster-whisper CPU.",
                exc,
            )
            delegate = None
        if delegate is not None:
            self._delegate = delegate
            return
        compute = self.cfg.compute_type if self.cfg.compute_type != "float16" else "int8"
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.cfg.model,
                device="cpu",
                compute_type=compute,
                download_root=str(self.download_root),
            )
        except RuntimeError as exc:
            raise SttUnavailableError(f"Whisper CPU de repli impossible : {exc}") from exc
        logger.info("Whisper {} prêt sur CPU (repli macOS, {}).", self.cfg.model, compute)

    def load(self) -> None:
        """Charge le modèle et le préchauffe (bloquant, plusieurs secondes)."""
        cfg = self.cfg
        device = getattr(cfg, "device", "cuda")
        if device == "mlx":
            self._load_mlx()
            return
        if device == "cuda" and sys.platform == "win32":
            dirs = add_cuda_dll_directories()
            if dirs:
                logger.debug("DLL CUDA : {}", ", ".join(map(str, dirs)))
        import ctranslate2
        from faster_whisper import WhisperModel

        if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            raise SttUnavailableError(
                "Aucun GPU NVIDIA utilisable par Whisper (stt.device = cuda).\n"
                "Vérifie le pilote NVIDIA : « nvidia-smi » doit afficher la RTX 3080. "
                "Pas de bascule CPU automatique (décision de l'architecture)."
            )

        before = gpu_memory_used_mb() if cfg.device == "cuda" else None
        start = time.perf_counter()
        kwargs = {
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "download_root": str(self.download_root),
        }
        try:
            try:
                self._model = WhisperModel(cfg.model, local_files_only=True, **kwargs)
            except FileNotFoundError:  # huggingface_hub.LocalEntryNotFoundError : pas en cache
                logger.warning(
                    "Modèle Whisper {} absent de {} : téléchargement (~1,6 Go)...",
                    cfg.model,
                    self.download_root,
                )
                self._model = WhisperModel(cfg.model, **kwargs)
            self.transcribe_sync(np.zeros(round(WARMUP_S * SAMPLE_RATE), dtype=np.int16))
        except SttUnavailableError:
            raise
        except RuntimeError as exc:
            raise _explain_cuda_error(exc) from exc

        after = gpu_memory_used_mb() if cfg.device == "cuda" else None
        vram = f", VRAM +{after - before:.0f} Mo" if before is not None and after else ""
        logger.info(
            "Whisper {} prêt sur {} ({}) en {:.1f} s{}.",
            cfg.model,
            cfg.device,
            cfg.compute_type,
            time.perf_counter() - start,
            vram,
        )

    def transcribe_sync(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        """Transcription bloquante (int16 ou float32 mono 16 kHz) ; hors de la boucle asyncio.
        `prompt` : prompt initial (défaut : `stt.initial_prompt`)."""
        if self._delegate is not None:
            return self._delegate.transcribe_sync(audio, prompt)
        if self._model is None:
            raise RuntimeError("Whisper n'est pas chargé")
        start = time.perf_counter()
        pcm = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        segments, _info = self._model.transcribe(
            pcm,
            language="fr",
            beam_size=self.cfg.beam_size,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # le découpage est déjà fait par Silero
            initial_prompt=(self.cfg.initial_prompt if prompt is None else prompt) or None,
        )
        decoded = list(segments)  # décodage ici
        text = "".join(segment.text for segment in decoded).strip()
        no_speech, logprob = _reliability(decoded)
        return Transcript(
            text=text,
            duration_s=len(audio) / SAMPLE_RATE,
            latency_s=time.perf_counter() - start,
            no_speech_prob=no_speech,
            avg_logprob=logprob,
        )

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        loop = asyncio.get_running_loop()
        prompt = self.prompt  # lu dans la boucle asyncio, pas dans le thread STT
        return await loop.run_in_executor(self._executor, self.transcribe_sync, audio, prompt)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        delegate, self._delegate = self._delegate, None
        if delegate is not None:
            delegate.close()
        self._model = None
