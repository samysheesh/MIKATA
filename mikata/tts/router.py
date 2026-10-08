"""Routeur de voix (ajout phase 11) : la voix du profil en cours, et Kokoro en secours.

- Profil = skin de la mascotte + voix (`ui.profiles` de config.yaml). Le skin choisi dans
  les réglages donne la voix ; un skin sans profil parle avec `tts.backend`. Changement à
  chaud (`apply`), entre deux fragments : la phrase en cours n'est pas coupée.
- Tout sort à `ROUTER_RATE` (24 kHz) : la file de phrases lit la fréquence après la
  synthèse, elle doit rester la même quel que soit le moteur (Piper est rééchantillonné).
- ElevenLabs indisponible (crédit, plafond, réseau, erreur d'API) : le fragment est dit par
  Kokoro, précédé une seule fois de « Je passe sur ma voix de base. », et l'interface est
  prévenue (`on_event("fallback", raison)`). Dès qu'ElevenLabs sert de nouveau (panne finie,
  renouvellement du crédit), retour sur sa voix (`on_event("resume", ...)`).
- Phrases fréquentes (« C'est fait. ») : une série par voix, gardées en mémoire ; pour
  ElevenLabs, synthétisées une seule fois puis relues de son cache disque.
- Une seule synthèse à la fois (executor à 1 worker, §3), comme avant.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mikata.tts.base import TtsModelMissingError
from mikata.tts.cache import FALLBACK_KEY, PHRASES, PhraseCache, _norm
from mikata.tts.elevenlabs import ElevenLabsTTS, ElevenLabsUnavailable

ROUTER_RATE = 24_000
FALLBACK_ENGINE = "kokoro"  # voix de base : locale et gratuite
PAUSE_S = 0.25  # entre l'annonce du secours et le fragment

VoiceEvent = Callable[[str, str], None]  # (« fallback » | « resume » | « profile », texte)
LocalFactory = Callable[[str], tuple[Any, str]]  # moteur -> (moteur non chargé, id de voix)


@dataclass(frozen=True)
class VoiceChoice:
    engine: str  # piper | kokoro | elevenlabs
    voice_id: str | None = None  # ElevenLabs seulement
    label: str = ""

    @property
    def describe(self) -> str:
        if self.engine == "elevenlabs":
            return f"ElevenLabs « {self.label or self.voice_id} »"
        return self.engine.capitalize()


def choose_voice(config: Any) -> VoiceChoice:
    """Voix du skin en cours (profil), sinon `tts.backend`."""
    profile = config.ui.profiles.get(config.ui.skin)
    if profile is None:
        engine = config.tts.backend
        voice_id = config.tts.elevenlabs.voice_id if engine == "elevenlabs" else None
        return VoiceChoice(engine, voice_id, "")
    if profile.voice == "elevenlabs":
        voice_id = profile.voice_id or config.tts.elevenlabs.voice_id
        return VoiceChoice("elevenlabs", voice_id, profile.voice_name)
    return VoiceChoice(profile.voice, None, profile.voice_name)


def resample(pcm: np.ndarray, rate: int, target: int = ROUTER_RATE) -> np.ndarray:
    if rate == target or len(pcm) == 0:
        return np.ascontiguousarray(pcm, dtype=np.float32)
    import soxr

    return np.ascontiguousarray(soxr.resample(pcm.astype(np.float32), rate, target), np.float32)


class _Resampled:
    """Moteur local vu à `ROUTER_RATE` (pour `PhraseCache`)."""

    sample_rate = ROUTER_RATE

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.name = engine.name

    def synthesize_sync(self, text: str) -> np.ndarray:
        pcm = self.engine.synthesize_sync(text)
        return resample(pcm, self.engine.sample_rate)


class VoiceRouter:
    """Implémente `TTS` (tts/base.py) et les phrases en cache (`PhraseSource`,
    `CachedPhrases`) pour `Speaker` et l'agent."""

    name = "voix"
    sample_rate = ROUTER_RATE

    def __init__(
        self,
        config: Any,
        local_factory: LocalFactory,
        eleven: ElevenLabsTTS | None,
        cache_dir: Path,
        on_event: VoiceEvent | None = None,
    ) -> None:
        self.config = config
        self._factory = local_factory
        self.eleven = eleven
        self._cache_dir = cache_dir
        self.on_event = on_event
        self.choice = choose_voice(config)
        self._local: dict[str, _Resampled] = {}
        self._local_phrases: dict[str, PhraseCache] = {}
        self._eleven_phrases: dict[str, dict[str, np.ndarray]] = {}
        self._eleven_loaded = False
        self.fallback_reason: str | None = None  # ElevenLabs remplacé par la voix de base
        self._notice_pending = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-tts")
        self._by_text = {_norm(text): key for key, text in PHRASES.items()}

    # -- chargement -----------------------------------------------------------------------

    @property
    def fallback_engine(self) -> str:
        return FALLBACK_ENGINE

    def _engine(self, name: str) -> _Resampled:
        """Moteur local chargé (au premier besoin) avec ses phrases fréquentes."""
        if name not in self._local:
            engine, voice_id = self._factory(name)
            logger.info("Chargement de la voix {}...", name)
            engine.load()
            wrapped = _Resampled(engine)
            phrases = PhraseCache(wrapped, self._cache_dir, f"{voice_id}|{ROUTER_RATE}")
            phrases.warm()
            self._local[name], self._local_phrases[name] = wrapped, phrases
        return self._local[name]

    def _fallback(self) -> _Resampled:
        try:
            return self._engine(FALLBACK_ENGINE)
        except TtsModelMissingError:
            backend = self.config.tts.backend
            if backend in ("elevenlabs", FALLBACK_ENGINE):
                raise
            logger.warning("Kokoro absent : voix de secours {}.", backend)
            return self._engine(backend)

    def _fallback_name(self) -> str:
        if FALLBACK_ENGINE in self._local:
            return FALLBACK_ENGINE
        return next(iter(self._local), FALLBACK_ENGINE)

    def _ensure(self, choice: VoiceChoice) -> None:
        if choice.engine != "elevenlabs":
            self._engine(choice.engine)
            return
        self._fallback()  # toujours prête : ElevenLabs peut tomber à tout moment
        if self.eleven is None:
            self._enter_fallback("ElevenLabs non configuré")
            return
        if not self._eleven_loaded:
            self.eleven.load()
            self._eleven_loaded = True
        self._warm_eleven(choice.voice_id or self.eleven.cfg.voice_id)

    def _warm_eleven(self, voice_id: str) -> None:
        """Phrases fréquentes dans la voix ElevenLabs : facturées une seule fois (cache
        disque), sinon laissées à la voix de base."""
        assert self.eleven is not None
        reason = self.eleven.unavailable_reason()
        if reason is not None:
            self._enter_fallback(reason)
            return
        phrases = self._eleven_phrases.setdefault(voice_id, {})
        for key, text in PHRASES.items():
            if key == FALLBACK_KEY or key in phrases:
                continue
            try:
                phrases[key] = self.eleven.synthesize_sync(text, voice_id)
            except ElevenLabsUnavailable as exc:
                self._enter_fallback(exc.reason)
                return
        self._leave_fallback()

    def load(self) -> None:
        """Bloquant (démarrage) : moteur du profil en cours, et Kokoro si c'est ElevenLabs."""
        self._ensure(self.choice)
        logger.info("Voix : {}.", self.choice.describe)

    def warm(self) -> None:
        """Phrases déjà préparées par `load` (gardé pour `Speaker.load`)."""

    # -- profils --------------------------------------------------------------------------

    def apply_sync(self, config: Any) -> bool:
        """Réglages relus : ElevenLabs reconfiguré, voix du skin en cours. True si la voix a
        changé. Bloquant (chargement d'un moteur) : thread de synthèse."""
        self.config = config
        if self.eleven is not None:
            key = config.secrets.elevenlabs_api_key
            self.eleven.configure(config.tts.elevenlabs, key.get_secret_value() if key else None)
            self._eleven_phrases.clear()  # voix ou modèle peut-être changés
        choice = choose_voice(config)
        changed = choice != self.choice
        self._ensure(choice)
        self.choice = choice
        if changed:
            logger.info("Voix : {} (profil « {} »).", choice.describe, config.ui.skin)
            self._emit("profile", choice.describe)
        return changed

    async def apply(self, config: Any) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.apply_sync, config)

    # -- secours --------------------------------------------------------------------------

    def _emit(self, kind: str, text: str) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, text)
            except Exception:
                logger.exception("Événement de voix non transmis.")

    def _enter_fallback(self, reason: str) -> None:
        if self.fallback_reason is None:
            logger.warning("Voix de base ({}) : {}.", self._fallback_name(), reason)
            self._notice_pending = True
            self.fallback_reason = reason
            self._set_eleven_state("kokoro")
            self._emit("fallback", reason)
        else:
            self.fallback_reason = reason

    def _leave_fallback(self) -> None:
        if self.fallback_reason is not None:
            logger.info("ElevenLabs de nouveau disponible : retour sur sa voix.")
            self.fallback_reason = None
            self._notice_pending = False
            self._set_eleven_state("elevenlabs")
            self._emit("resume", self.choice.describe)

    def _set_eleven_state(self, state: str) -> None:
        if self.eleven is not None:
            self.eleven.state = state
            self.eleven.save_usage()

    # -- synthèse -------------------------------------------------------------------------

    def synthesize_sync(self, text: str) -> np.ndarray:
        choice = self.choice
        if choice.engine != "elevenlabs":
            return self._engine(choice.engine).synthesize_sync(text)
        if self.eleven is None:
            self._enter_fallback("ElevenLabs non configuré")
        elif (reason := self.eleven.unavailable_reason()) is not None:
            self._enter_fallback(reason)  # même une phrase en cache : jamais deux voix mêlées
        else:
            try:
                pcm = self.eleven.synthesize_sync(text, choice.voice_id)
            except ElevenLabsUnavailable as exc:
                self._enter_fallback(exc.reason)
            else:
                self._leave_fallback()
                return resample(pcm, self.eleven.sample_rate)
        pcm = self._fallback().synthesize_sync(text)
        if self._notice_pending:
            self._notice_pending = False
            notice = self._local_phrases[self._fallback_name()].get(FALLBACK_KEY)
            if notice is not None:
                pause = np.zeros(int(PAUSE_S * ROUTER_RATE), dtype=np.float32)
                pcm = np.concatenate([notice, pause, pcm])
        return pcm

    async def synthesize(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.synthesize_sync, text)

    # -- phrases en cache (boucle asyncio : rien de bloquant) ------------------------------

    def get(self, key: str) -> np.ndarray | None:
        choice = self.choice
        if choice.engine == "elevenlabs":
            if self.fallback_reason is None and choice.voice_id is not None:
                pcm = self._eleven_phrases.get(choice.voice_id, {}).get(key)
                if pcm is not None:
                    return resample(pcm, self.eleven.sample_rate if self.eleven else ROUTER_RATE)
            if self.fallback_reason is None:
                return None  # pas encore dite par ElevenLabs : synthèse normale
            phrases = self._local_phrases.get(self._fallback_name())
        else:
            phrases = self._local_phrases.get(choice.engine)
        return phrases.get(key) if phrases is not None else None

    def lookup(self, text: str) -> np.ndarray | None:
        key = self._by_text.get(_norm(text))
        if key is None or key == FALLBACK_KEY:
            return None
        if self._notice_pending and self.choice.engine == "elevenlabs":
            return None  # l'annonce du secours part avec le prochain fragment synthétisé
        return self.get(key)

    @staticmethod
    def text(key: str) -> str:
        return PHRASES[key]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        for wrapped in self._local.values():
            close = getattr(wrapped.engine, "close", None)
            if close is not None:
                close()
        if self.eleven is not None:
            self.eleven.close()


def build_router(config: Any, cache_dir: Path, usage_path: Path, **kwargs: Any) -> VoiceRouter:
    """Routeur de la config, non chargé (`Speaker.load` le charge)."""
    from mikata.tts.base import create_tts

    key = config.secrets.elevenlabs_api_key
    eleven = ElevenLabsTTS(
        config.tts.elevenlabs,
        key.get_secret_value() if key else None,
        cache_dir,
        usage_path,
        transport=kwargs.pop("transport", None),
    )
    factory = kwargs.pop("local_factory", None) or (lambda name: create_tts(config, name))
    return VoiceRouter(config, factory, eleven, cache_dir, **kwargs)
