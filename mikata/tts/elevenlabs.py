"""Voix ElevenLabs (ajout phase 11) : service payant en ligne, rien n'est embarqué.

- Synthèse fragment par fragment, comme Piper et Kokoro : POST
  /v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000 (PCM 16 bits mono reçu en
  flux), modèle `tts.elevenlabs.model`, `language_code` = `tts.elevenlabs.language`.
- Cache disque (data/tts_cache/elevenlabs/) : une phrase déjà dite n'est jamais refacturée.
  Clé : voix + modèle + langue + format + texte normalisé (NFC, espaces et apostrophes
  unifiés). Au-delà de `cache_max_mb`, plus rien n'est ajouté (rien n'est supprimé).
- Crédits (`credits.CreditTracker`) : solde relevé au démarrage puis toutes les
  `subscription_refresh_every` synthèses, déduit localement entre deux relevés ; coût par
  caractère mesuré au démarrage (au plus toutes les `calibration_hours`).
- Indisponible (`ElevenLabsUnavailable`) : crédit sous le seuil ou plafond atteint (jusqu'au
  renouvellement), clé refusée (jusqu'à la relecture des réglages), quota épuisé (jusqu'au
  renouvellement), réseau ou serveur en panne (`retry_after_s`), trop de requêtes. Le routeur
  de voix (tts/router.py) passe alors sur Kokoro et revient ici dès que c'est rétabli.

La clé (ELEVENLABS_API_KEY, .env) part seulement dans l'en-tête `xi-api-key` : jamais dans
un journal ni un message.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mikata.config import ElevenLabsConfig
from mikata.tts.base import trim_silence
from mikata.tts.credits import CreditTracker, read_usage_file, write_usage

CACHE_SUBDIR = "elevenlabs"
CALIBRATION_TEXT = (
    "Bonjour, c'est MIKATA. Je mesure le coût de ma voix avant de commencer la journée."
)
CREDIT_RECHECK_S = 3600.0  # crédit sous le seuil : solde relu au plus toutes les heures
BUSY_RETRY_S = 30.0  # 429 sans Retry-After
COST_HEADERS = ("character-cost", "x-character-cost")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "«": '"', "»": '"'})


def normalize_text(text: str) -> str:
    """Texte envoyé à ElevenLabs et clé du cache : deux écritures de la même phrase
    (espaces, apostrophe typographique) ne sont facturées qu'une fois."""
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


class ElevenLabsUnavailable(Exception):
    """ElevenLabs ne peut pas servir ce fragment ; `reason` est dit dans l'interface."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(
        self,
        cfg: ElevenLabsConfig,
        api_key: str | None,
        cache_dir: Path,
        usage_path: Path,
        *,
        transport: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.sample_rate = int(cfg.output_format.split("_")[1])
        self._key = api_key
        self._cache_dir = cache_dir / CACHE_SUBDIR
        self._usage_path = usage_path
        self._transport = transport
        self._clock = clock
        self._client: Any = None
        self.credits = CreditTracker(
            cfg.monthly_credit_cap, cfg.min_remaining_credits, cfg.estimated_credits_per_char
        )
        self._blocked_until = 0.0  # math.inf : jusqu'à la relecture des réglages
        self._blocked_reason: str | None = None
        self._next_credit_check = 0.0
        self._bad_voices: set[str] = set()
        self._cache_bytes: int | None = None
        self._cache_full_logged = False
        self._no_balance_logged = False
        self.requests = 0  # synthèses facturées depuis le démarrage
        self.cache_hits = 0
        self.state = "elevenlabs"  # ou « kokoro » : tenu par le routeur, écrit dans le compteur

    # -- réglages -------------------------------------------------------------------------

    def configure(self, cfg: ElevenLabsConfig, api_key: str | None) -> None:
        """Réglages relus (interface) : plafond et seuils à chaud, clé peut-être changée.
        Une pause en cours est levée : la prochaine synthèse réessaie (et y retourne si la
        cause est toujours là, sans nouvelle annonce)."""
        if api_key != self._key or cfg.base_url != self.cfg.base_url:
            self._close_client()
        self._unblock()
        self.cfg, self._key = cfg, api_key
        self.sample_rate = int(cfg.output_format.split("_")[1])
        self.credits.cap = cfg.monthly_credit_cap
        self.credits.min_remaining = cfg.min_remaining_credits
        self.credits.estimated_rate = cfg.estimated_credits_per_char
        self._bad_voices.clear()
        self._next_credit_check = 0.0
        self.save_usage()

    def _http(self) -> Any:
        if self._client is None:
            import httpx2

            self._client = httpx2.Client(
                base_url=self.cfg.base_url,
                headers={"xi-api-key": self._key or "", "accept": "audio/pcm, application/json"},
                timeout=httpx2.Timeout(self.cfg.timeout_s, connect=min(5.0, self.cfg.timeout_s)),
                transport=self._transport,
            )
        return self._client

    def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def close(self) -> None:
        self._close_client()

    # -- démarrage ------------------------------------------------------------------------

    def load(self) -> None:
        """Relevé du solde et mesure du coût (bloquant : thread de démarrage). Jamais
        d'exception : en cas d'échec, la voix de base prend le relais."""
        self.credits.restore(read_usage_file(self._usage_path))
        if not self._key:
            self._block("clé ELEVENLABS_API_KEY absente de .env", math.inf)
            self.save_usage()
            return
        try:
            self.refresh_subscription()
            if self._needs_calibration() and self.credits.block_reason() is None:
                self._calibrate()
        except ElevenLabsUnavailable as exc:
            logger.warning("ElevenLabs indisponible au démarrage : {}", exc.reason)
        self.log_balance()
        self.save_usage()

    def _needs_calibration(self) -> bool:
        hours = self.cfg.calibration_hours
        if hours <= 0:
            return False
        at = self.credits.measured_at
        return at is None or self._clock() - at > hours * 3600

    def _calibrate(self) -> None:
        """Une courte synthèse entre deux relevés : coût réel par caractère (le tarif peut
        changer : promotion, nouveau modèle)."""
        logger.info("ElevenLabs : mesure du coût réel ({} caractères)...", len(CALIBRATION_TEXT))
        self._request(CALIBRATION_TEXT, self.cfg.voice_id)
        self.refresh_subscription()
        if self.credits.measured_rate is None:
            logger.warning(
                "ElevenLabs : coût non mesurable pour l'instant (compteur pas à jour) ; "
                "estimation {} crédit par caractère jusqu'au prochain relevé.",
                self.credits.estimated_rate,
            )

    def log_balance(self) -> None:
        c = self.credits
        if not c.known:
            return
        reset = (
            datetime.fromtimestamp(c.reset_unix).strftime("%d/%m") if c.reset_unix else "inconnu"
        )
        logger.info(
            "ElevenLabs : {:.0f} crédits utilisés sur {:.0f} (plafond {:.0f}), {:.0f} "
            "utilisables, coût {} {:.4f} crédit par caractère, environ {} réponses ; "
            "renouvellement le {}.",
            c.used,
            c.limit,
            c.cap,
            c.remaining,
            "mesuré" if c.measured_rate is not None else "estimé",
            c.rate,
            c.answers_left(),
            reset,
        )

    def save_usage(self) -> None:
        write_usage(
            self._usage_path,
            self.credits.summary(
                state=self.state,
                reason=self.unavailable_reason(check=False),
                requests=self.requests,
                cache_hits=self.cache_hits,
                model=self.cfg.model,
            ),
        )

    # -- solde --------------------------------------------------------------------------

    def refresh_subscription(self) -> None:
        try:
            response = self._send("GET", "/v1/user/subscription", reading_balance=True)
        except _NoBalancePermission:
            if not self._no_balance_logged:
                logger.warning(
                    "ElevenLabs : la clé n'a pas le droit de lire le solde (permission « User » "
                    "en lecture) : solde et plafond non suivis. Ajoute ce droit à la clé."
                )
                self._no_balance_logged = True
            self._next_credit_check = self._clock() + CREDIT_RECHECK_S
            return
        try:
            data = response.json()
            count, limit = float(data["character_count"]), float(data["character_limit"])
        except (ValueError, KeyError, TypeError) as exc:
            raise self._block_unavailable("réponse illisible de l'abonnement", "network") from exc
        reset = data.get("next_character_count_reset_unix")
        self.credits.refresh(count, limit, float(reset) if reset else None)
        now = self._clock()
        self._next_credit_check = now + CREDIT_RECHECK_S  # relu au renouvellement, au plus tard
        if self.credits.reset_unix is not None and self.credits.reset_unix > now:
            self._next_credit_check = min(self._next_credit_check, self.credits.reset_unix)
        blocked = self._blocked_reason is not None and self._blocked_until != math.inf
        if blocked and self.credits.block_reason() is None:  # quota rechargé ou renouvelé
            self._unblock()

    # -- état -----------------------------------------------------------------------------

    def _block(self, reason: str, until: float) -> None:
        if reason != self._blocked_reason:
            logger.warning("ElevenLabs en pause : {}.", reason)
        self._blocked_reason, self._blocked_until = reason, until

    def _unblock(self) -> None:
        self._blocked_reason, self._blocked_until = None, 0.0

    def _block_unavailable(self, reason: str, kind: str, retry_s: float | None = None) -> Exception:
        now = self._clock()
        if kind == "auth":
            until = math.inf
        elif kind == "quota":
            reset = self.credits.reset_unix
            until = reset if reset and reset > now else now + CREDIT_RECHECK_S
        else:
            until = now + (retry_s if retry_s is not None else self.cfg.retry_after_s)
        self._block(reason, until)
        return ElevenLabsUnavailable(reason)

    def unavailable_reason(self, *, check: bool = True) -> str | None:
        """Pourquoi ElevenLabs ne sert pas en ce moment (None : il sert). Avec `check`, un
        blocage échu est levé et le solde relu si c'est lui qui bloquait (réseau possible :
        appeler hors de la boucle asyncio)."""
        now = self._clock()
        if self._blocked_reason is not None:
            if now < self._blocked_until or not check:
                return self._blocked_reason
            self._unblock()
            self._next_credit_check = 0.0  # renouvellement ou panne finie : solde relu
        reason = self.credits.block_reason()
        reset = self.credits.reset_unix
        renewed = reset is not None and now >= reset
        due = now >= self._next_credit_check  # jamais plus d'un relevé par heure ainsi
        if check and due and (renewed or reason is not None):
            try:
                self.refresh_subscription()
            except ElevenLabsUnavailable as exc:
                return exc.reason
            reason = self.credits.block_reason()
        if not self._key:
            return "clé ELEVENLABS_API_KEY absente de .env"
        return reason

    # -- synthèse -------------------------------------------------------------------------

    def cache_path(self, text: str, voice_id: str) -> Path:
        key = f"{voice_id}|{self.cfg.model}|{self.cfg.language}|{self.cfg.output_format}|{text}"
        return self._cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.pcm"

    def cached(self, text: str, voice_id: str) -> np.ndarray | None:
        path = self.cache_path(normalize_text(text), voice_id)
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        return trim_silence(_to_float(raw), self.sample_rate)

    def synthesize_sync(self, text: str, voice_id: str | None = None) -> np.ndarray:
        """PCM float32 mono à `sample_rate` ; `ElevenLabsUnavailable` si ElevenLabs ne peut
        pas servir (le routeur prend alors la voix de base)."""
        voice = voice_id or self.cfg.voice_id
        clean = normalize_text(text)
        if not clean:
            return np.zeros(0, dtype=np.float32)
        if voice in self._bad_voices:
            voice = self._backup_voice(voice)
        pcm = self.cached(clean, voice)
        if pcm is not None:
            self.cache_hits += 1
            return pcm
        reason = self.unavailable_reason()
        if reason is not None:
            raise ElevenLabsUnavailable(reason)
        try:
            return self._request(clean, voice)
        except _VoiceNotFound:
            backup = self._backup_voice(voice)
        pcm = self.cached(clean, backup)
        if pcm is not None:
            self.cache_hits += 1
            return pcm
        try:
            return self._request(clean, backup)
        except _VoiceNotFound:
            raise ElevenLabsUnavailable("voix ElevenLabs introuvables sur le compte") from None

    def _backup_voice(self, voice: str) -> str:
        backup = self.cfg.backup_voice_id
        if not backup or backup == voice or backup in self._bad_voices:
            raise ElevenLabsUnavailable(f"voix ElevenLabs « {voice} » introuvable sur le compte")
        return backup

    def _request(self, text: str, voice: str) -> np.ndarray:
        path = f"/v1/text-to-speech/{voice}/stream"
        body = {"text": text, "model_id": self.cfg.model, "language_code": self.cfg.language}
        params = {"output_format": self.cfg.output_format}
        try:
            raw, headers = self._stream(path, body, params)
        except _VoiceNotFound:
            if voice not in self._bad_voices:
                logger.warning(
                    "ElevenLabs : voix « {} » introuvable ; voix de secours « {} ».",
                    voice,
                    self.cfg.backup_voice_id,
                )
            self._bad_voices.add(voice)
            raise
        self.requests += 1
        self.credits.record(len(text), _cost(headers))
        cache = self.cache_path(text, voice)
        self._store(cache, raw)
        if self.credits.synth_since_refresh >= self.cfg.subscription_refresh_every:
            with contextlib.suppress(ElevenLabsUnavailable):  # le fragment, lui, est arrivé
                self.refresh_subscription()
        self.save_usage()
        return trim_silence(_to_float(raw), self.sample_rate)

    def _store(self, path: Path, raw: bytes) -> None:
        if self.cfg.cache_max_mb <= 0:
            return
        if self._cache_bytes is None:
            try:
                self._cache_bytes = sum(p.stat().st_size for p in self._cache_dir.glob("*.pcm"))
            except OSError:
                self._cache_bytes = 0
        if self._cache_bytes + len(raw) > self.cfg.cache_max_mb * 1_000_000:
            if not self._cache_full_logged:
                logger.info("Cache ElevenLabs plein ({} Mo) : plus d'ajout.", self.cfg.cache_max_mb)
                self._cache_full_logged = True
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(raw)
            tmp.replace(path)
            self._cache_bytes += len(raw)
        except OSError as exc:
            logger.warning("Cache ElevenLabs non écrit : {}", exc)

    # -- HTTP -----------------------------------------------------------------------------

    def _send(self, method: str, path: str, *, reading_balance: bool = False, **kwargs: Any) -> Any:
        import httpx2

        try:
            response = self._http().request(method, path, **kwargs)
        except httpx2.HTTPError as exc:
            raise self._block_unavailable(
                f"connexion à ElevenLabs impossible ({type(exc).__name__})", "network"
            ) from exc
        if response.status_code >= 400:
            detail = _detail(response.content)
            no_right = response.status_code in (401, 403) and detail[0] == "missing_permissions"
            if reading_balance and no_right:  # la synthèse, elle, peut marcher
                raise _NoBalancePermission(detail[1])
            self._raise_for(response.status_code, detail, response.headers)
        return response

    def _stream(self, path: str, body: dict[str, Any], params: dict[str, str]) -> tuple[bytes, Any]:
        import httpx2

        chunks: list[bytes] = []
        try:
            with self._http().stream("POST", path, json=body, params=params) as response:
                if response.status_code >= 400:
                    content = response.read()
                    self._raise_for(response.status_code, _detail(content), response.headers)
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                headers = response.headers
        except httpx2.HTTPError as exc:
            raise self._block_unavailable(
                f"connexion à ElevenLabs impossible ({type(exc).__name__})", "network"
            ) from exc
        raw = b"".join(chunks)
        return raw[: len(raw) // 2 * 2], headers

    def _raise_for(self, status: int, detail: tuple[str, str], headers: Any) -> None:
        code, message = detail
        logger.debug("ElevenLabs : HTTP {} {} {}", status, code, message[:200])
        if status in (400, 404) and "voice" in code:
            raise _VoiceNotFound(code)
        if code in ("quota_exceeded", "payment_required") or status == 402:
            self.credits.used = self.credits.limit  # épuisé, quoi qu'en dise le dernier relevé
            raise self._block_unavailable("crédit ElevenLabs épuisé (quota)", "quota")
        if status in (401, 403):
            raise self._block_unavailable(f"clé ElevenLabs refusée ({code or status})", "auth")
        if status == 429:
            retry = _retry_after(headers)
            raise self._block_unavailable(
                f"ElevenLabs saturé (HTTP 429{' ' + code if code else ''})",
                "busy",
                retry if retry is not None else BUSY_RETRY_S,
            )
        label = f"HTTP {status} {code}".strip()
        raise self._block_unavailable(f"erreur ElevenLabs ({label})", "network")


class _VoiceNotFound(Exception):
    pass


class _NoBalancePermission(Exception):
    pass


def _detail(content: bytes) -> tuple[str, str]:
    """(statut, message) d'une erreur ElevenLabs : {"detail": {"status", "message"}}."""
    import json

    try:
        data = json.loads(content.decode("utf-8", "replace"))
    except ValueError:
        return "", ""
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("status") or detail.get("code") or ""), str(
            detail.get("message") or ""
        )
    if isinstance(detail, str):
        return "", detail
    return "", ""


def _cost(headers: Any) -> float | None:
    for name in COST_HEADERS:
        value = headers.get(name)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after")
    try:
        return max(1.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _to_float(raw: bytes) -> np.ndarray:
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
    return (pcm / 32768.0).astype(np.float32)
