"""Cache audio des phrases fréquentes (§2, technique 5).

Les phrases sont synthétisées au démarrage (ou relues depuis `data/tts_cache/`, clé =
moteur + voix + vitesse + texte) et gardées en mémoire : « C'est fait » ou « Je n'arrive pas
à joindre le serveur » partent sans attendre le TTS, même hors ligne.
"""

from __future__ import annotations

import hashlib
import re
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from mikata.audio.wavfile import read_wav, write_wav

PHRASES: dict[str, str] = {
    "done": "C'est fait.",
    "looking": "Je regarde.",
    "server": "Je n'arrive pas à joindre le serveur.",
    "timeout": "Désolé, ça prend trop de temps.",
    "error": "Désolé, quelque chose s'est mal passé.",
    "reset": "D'accord, nouvelle conversation.",
    "voice_fallback": "Je passe sur ma voix de base.",  # phase 11 : ElevenLabs indisponible
    # Phase 15c : phrases d'attente (réglage agent.narration « bavard »), selon l'étape.
    "wait_thinking": "Je réfléchis, un instant.",
    "wait_browser": "Je suis sur la page, une seconde.",
    "wait_search": "Je cherche sur internet.",
    "wait_files": "Je regarde dans tes fichiers.",
    "wait_app": "Je regarde dans l'application.",
    "wait_work": "Je m'en occupe, un instant.",
    "wait_long": "Ça prend un peu plus de temps, je continue.",
    # Phase 17f : l'action n'a rien changé dans la fenêtre (vérifié pendant la réponse).
    "no_effect": "Attends, ça n'a pas pris.",
    "fixed": "Voilà, c'est bon.",
    "still_no_effect": "Je ne vois toujours aucun changement, regarde de ton côté.",
}
FALLBACK_KEY = "voice_fallback"


class _SyncTTS(Protocol):
    name: str
    sample_rate: int

    def synthesize_sync(self, text: str) -> np.ndarray: ...


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


class PhraseCache:
    def __init__(self, tts: _SyncTTS, cache_dir: Path, voice_id: str) -> None:
        self._tts = tts
        self._dir = cache_dir
        self._voice_id = voice_id
        self._audio: dict[str, np.ndarray] = {}
        self._by_text: dict[str, str] = {_norm(text): key for key, text in PHRASES.items()}

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def _path(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self._voice_id}|{text}".encode()).hexdigest()[:16]
        return self._dir / f"{self._tts.name}-{digest}.wav"

    def warm(self) -> None:
        """Charge ou synthétise toutes les phrases (bloquant : thread TTS ou démarrage)."""
        for key, text in PHRASES.items():
            path = self._path(text)
            pcm: np.ndarray | None = None
            if path.is_file():
                try:
                    pcm, rate = read_wav(path)
                    if rate != self._tts.sample_rate:
                        pcm = None
                except (OSError, ValueError, EOFError, wave.Error):  # fichier abîmé
                    pcm = None
            if pcm is None:
                pcm = self._tts.synthesize_sync(text)
                try:
                    write_wav(path, pcm, self._tts.sample_rate)
                except OSError as exc:
                    logger.warning("Cache de phrases non écrit ({}) : {}", path, exc)
            self._audio[key] = np.ascontiguousarray(pcm, dtype=np.float32)

    def get(self, key: str) -> np.ndarray | None:
        return self._audio.get(key)

    def lookup(self, text: str) -> np.ndarray | None:
        """Audio en cache si `text` est exactement une phrase fréquente."""
        key = self._by_text.get(_norm(text))
        return self._audio.get(key) if key else None

    @staticmethod
    def text(key: str) -> str:
        return PHRASES[key]
