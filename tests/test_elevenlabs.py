"""Phase 11 : voix ElevenLabs avec l'API simulée (httpx2.MockTransport, aucun appel réel),
cache de phrases, crédits, et retour automatique sur Kokoro."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from pydantic import SecretStr

from mikata.config import Config, ElevenLabsConfig
from mikata.tts.cache import PHRASES
from mikata.tts.credits import CreditTracker
from mikata.tts.elevenlabs import (
    CALIBRATION_TEXT,
    ElevenLabsTTS,
    ElevenLabsUnavailable,
    normalize_text,
)
from mikata.tts.router import ROUTER_RATE, VoiceRouter, choose_voice

KEY = "el-test-0123456789"  # gitleaks:allow
LUCIE, THEO, EMILIE = "YxrwjAKoUKULGd0g8K9Y", "NyxenPOqNyllHIzSoPbJ", "d3AXX0BlgJHYFCuH9X88"
RATE = 0.055  # crédits par caractère facturés par la fausse API


class FakeApi:
    """API ElevenLabs simulée : abonnement, synthèse en flux, erreurs à la demande."""

    def __init__(self) -> None:
        self.count = 1000.0
        self.limit = 40_000.0
        self.reset = time.time() + 20 * 86400
        self.rate = RATE
        self.cost_header = False
        self.down = False
        self.error: tuple[int, dict[str, Any], dict[str, str]] | None = None
        self.missing: set[str] = set()
        self.synth: list[tuple[str, str]] = []  # (voix, texte)
        self.requests: list[httpx2.Request] = []
        self.subscription_reads = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.headers["xi-api-key"] == KEY
        if self.down:
            raise httpx2.ConnectError("réseau coupé", request=request)
        path = request.url.path
        if path == "/v1/user/subscription":
            self.subscription_reads += 1
            return httpx2.Response(
                200,
                json={
                    "character_count": round(self.count),
                    "character_limit": self.limit,
                    "next_character_count_reset_unix": self.reset,
                    "tier": "starter",
                },
            )
        assert path.startswith("/v1/text-to-speech/") and path.endswith("/stream")
        voice = path.split("/")[3]
        if self.error is not None:
            status, detail, headers = self.error
            return httpx2.Response(status, json={"detail": detail}, headers=headers)
        if voice in self.missing:
            return httpx2.Response(
                404, json={"detail": {"status": "voice_not_found", "message": "absente"}}
            )
        body = json.loads(request.content)
        assert request.url.params["output_format"] == "pcm_24000"
        assert body["model_id"] == "eleven_v4_turbo" and body["language_code"] == "fr"
        text = body["text"]
        self.synth.append((voice, text))
        cost = len(text) * self.rate
        self.count += cost
        samples = np.full(240 * max(1, len(text) // 10), 8000, dtype="<i2")  # 10 ms par 10 car.
        headers = {"character-cost": f"{cost:.3f}"} if self.cost_header else {}
        return httpx2.Response(200, content=samples.tobytes(), headers=headers)


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


def _cfg(**overrides: Any) -> ElevenLabsConfig:
    return ElevenLabsConfig(**overrides)


def _eleven(
    tmp_path: Path, api: FakeApi, clock: Clock | None = None, key: str | None = KEY, **cfg: Any
) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        _cfg(**cfg),
        key,
        tmp_path / "tts_cache",
        tmp_path / "elevenlabs_usage.json",
        transport=httpx2.MockTransport(api),
        clock=clock or Clock(),
    )


def test_normalized_text_is_the_cache_key() -> None:
    assert normalize_text("  C’est   fait.\n") == "C'est fait."
    assert normalize_text("Il fait 20 degrés") == "Il fait 20 degrés"
    assert normalize_text("été") == "été"  # NFC


def test_streamed_pcm_and_a_repeated_sentence_is_billed_once(tmp_path: Path) -> None:
    api = FakeApi()
    tts = _eleven(tmp_path, api)
    pcm = tts.synthesize_sync("Bonjour, je suis Lucie.", LUCIE)
    assert pcm.dtype == np.float32 and len(pcm) > 0 and tts.sample_rate == 24_000
    assert abs(float(pcm.mean()) - 8000 / 32768) < 1e-3
    again = tts.synthesize_sync("Bonjour,  je suis Lucie. ", LUCIE)  # autre écriture
    assert np.array_equal(pcm, again)
    assert api.synth == [(LUCIE, "Bonjour, je suis Lucie.")]  # facturée une seule fois
    assert tts.requests == 1 and tts.cache_hits == 1
    tts.synthesize_sync("Bonjour, je suis Lucie.", THEO)  # autre voix : autre clé
    assert api.synth[-1] == (THEO, "Bonjour, je suis Lucie.")
    # Le cache survit au redémarrage : aucun nouvel appel.
    fresh = _eleven(tmp_path, api)
    fresh.synthesize_sync("Bonjour, je suis Lucie.", LUCIE)
    assert len(api.synth) == 2
    # Autre modèle : autre clé de cache.
    _eleven(tmp_path, api, model="eleven_v4_turbo").cache_path("x", LUCIE)
    assert tts.cache_path("x", LUCIE) != tts.cache_path("x", THEO)


def test_cache_stops_growing_at_its_limit(tmp_path: Path) -> None:
    api = FakeApi()
    tts = _eleven(tmp_path, api, cache_max_mb=0.0001)  # 100 octets
    tts.synthesize_sync("Une phrase assez longue pour dépasser la limite.", LUCIE)
    tts.synthesize_sync("Une phrase assez longue pour dépasser la limite.", LUCIE)
    assert len(api.synth) == 2  # pas gardée : rien n'est supprimé pour faire de la place
    assert not list((tmp_path / "tts_cache" / "elevenlabs").glob("*.pcm"))


def test_startup_reads_the_balance_and_measures_the_real_cost(tmp_path: Path) -> None:
    api = FakeApi()
    tts = _eleven(tmp_path, api)
    tts.load()
    assert api.synth == [(LUCIE, CALIBRATION_TEXT)]  # une synthèse de calibrage
    c = tts.credits
    # Mesuré, pas supposé (le compteur de l'abonnement est entier : arrondi sur 82 caractères).
    assert c.measured_rate == pytest.approx(RATE, abs=0.01)
    assert c.limit == 40_000 and c.used == pytest.approx(api.count, abs=1)
    usage = json.loads((tmp_path / "elevenlabs_usage.json").read_text(encoding="utf-8"))
    assert usage["rate_measured"] == c.measured_rate
    assert usage["remaining"] == pytest.approx(30_000 - api.count, abs=1)  # sous le plafond
    assert usage["answers_left"] == int(usage["remaining"] // (150 * usage["rate"]))
    assert usage["state"] == "elevenlabs" and usage["reason"] is None
    assert KEY not in (tmp_path / "elevenlabs_usage.json").read_text(encoding="utf-8")
    # Mesure récente : pas de nouveau calibrage au démarrage suivant.
    again = _eleven(tmp_path, api)
    again.load()
    assert len(api.synth) == 1 and again.credits.measured_rate == c.measured_rate


def test_cost_header_is_used_when_the_api_gives_it(tmp_path: Path) -> None:
    api = FakeApi()
    api.cost_header = True
    api.rate = 0.5
    tts = _eleven(tmp_path, api, calibration_hours=0)
    tts.load()
    assert api.synth == []  # calibrage désactivé
    before = tts.credits.used
    tts.synthesize_sync("Une réponse de cinquante caractères à peu près, oui.", LUCIE)
    assert tts.credits.measured_rate == pytest.approx(0.5)
    assert tts.credits.used == pytest.approx(before + 0.5 * 52, abs=0.1)


def test_balance_is_deduced_between_two_readings(tmp_path: Path) -> None:
    api = FakeApi()
    tts = _eleven(tmp_path, api, calibration_hours=0, subscription_refresh_every=3)
    tts.load()
    assert api.subscription_reads == 1
    for i in range(2):
        tts.synthesize_sync(f"Phrase numéro {i}, assez longue pour compter un peu.", LUCIE)
    assert api.subscription_reads == 1  # déduit localement (estimation prudente : 1 / car.)
    assert tts.credits.used == pytest.approx(1000 + 2 * 51, abs=2)
    tts.synthesize_sync("Phrase numéro 2, assez longue pour compter un peu.", LUCIE)
    assert api.subscription_reads == 2  # relevé toutes les 3 synthèses
    assert tts.credits.used == pytest.approx(api.count, abs=1)
    assert tts.credits.measured_rate == pytest.approx(RATE, abs=0.01)  # écart des relevés


def test_credit_tracker_reasons() -> None:
    c = CreditTracker(cap=30_000, min_remaining=500, estimated_rate=1.0)
    assert c.block_reason() is None and c.remaining is None
    c.refresh(29_999, 40_000, None)
    assert c.block_reason() is None
    c.record(10, 2.0)
    assert "plafond" in (c.block_reason() or "")
    c.cap = 40_000
    c.refresh(39_600, 40_000, None)
    assert "presque épuisé" in (c.block_reason() or "")
    c.refresh(0, 40_000, None)  # renouvellement
    assert c.block_reason() is None and c.answers_left() == (39_500 // 150)


# -- retour sur Kokoro ----------------------------------------------------------------------


class FakeLocal:
    """Moteur local simulé (Kokoro ou Piper) : un niveau reconnaissable."""

    def __init__(self, name: str, rate: int, level: float) -> None:
        self.name, self.sample_rate, self.level = name, rate, level
        self.loaded = False
        self.calls: list[str] = []

    def load(self) -> None:
        self.loaded = True

    def synthesize_sync(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.full(self.sample_rate // 10, self.level, dtype=np.float32)

    def close(self) -> None:
        pass


KOKORO_LEVEL, PIPER_LEVEL, ELEVEN_LEVEL = 0.1, 0.2, 8000 / 32768


def _level(pcm: np.ndarray) -> float:
    return float(np.median(pcm[-ROUTER_RATE // 20 :]))


def _config(skin: str = "femme", backend: str = "piper", key: str | None = KEY) -> Config:
    from mikata.config import Secrets

    data: dict[str, Any] = {
        "secrets": Secrets.model_construct(elevenlabs_api_key=SecretStr(key) if key else None),
        "tts": {"backend": backend},
        "ui": {
            "skin": skin,
            "profiles": {
                "default": {"voice": "kokoro"},
                "gars": {"voice": "elevenlabs", "voice_id": THEO, "voice_name": "Théo"},
                "femme": {"voice": "elevenlabs", "voice_id": LUCIE, "voice_name": "Lucie"},
            },
        },
    }
    return Config.model_validate(data)


class Rig:
    def __init__(self, tmp_path: Path, skin: str = "femme", key: str | None = KEY, **cfg: Any):
        self.api = FakeApi()
        self.clock = Clock()
        self.events: list[tuple[str, str]] = []
        self.locals: dict[str, FakeLocal] = {}
        cfg.setdefault("calibration_hours", 0)
        self.eleven = _eleven(tmp_path, self.api, self.clock, key=key, **cfg)
        self.router = VoiceRouter(
            _config(skin),
            self.factory,
            self.eleven,
            tmp_path / "tts_cache",
            on_event=lambda kind, text: self.events.append((kind, text)),
        )

    def factory(self, name: str) -> tuple[FakeLocal, str]:
        rate, level = (24_000, KOKORO_LEVEL) if name == "kokoro" else (22_050, PIPER_LEVEL)
        engine = self.locals.setdefault(name, FakeLocal(name, rate, level))
        return engine, f"{name}|test"

    def say(self, text: str) -> float:
        """Niveau de la fin du fragment : quelle voix a parlé."""
        return _level(self.router.synthesize_sync(text))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


def test_elevenlabs_profile_speaks_with_elevenlabs(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.router.load()
    rig.router.warm()
    assert rig.locals["kokoro"].loaded  # le secours est prêt dès le démarrage
    assert {voice for voice, _ in rig.api.synth} == {LUCIE}
    warmed = {text for _, text in rig.api.synth}
    assert warmed == {t for k, t in PHRASES.items() if k != "voice_fallback"}
    assert rig.say("Il fait beau aujourd'hui.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    done = rig.router.get("done")
    assert done is not None and done[-1] == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.events == [] and rig.router.sample_rate == ROUTER_RATE


def _fallback_then_resume(rig: Rig, cause: str) -> None:
    pcm = rig.router.synthesize_sync("Voici la météo du jour.")
    assert pcm[-1] == pytest.approx(KOKORO_LEVEL), cause  # Kokoro a dit le fragment
    notice = rig.locals["kokoro"].calls
    assert "Je passe sur ma voix de base." in notice, cause
    assert len(pcm) > ROUTER_RATE // 10, cause  # annonce + pause + fragment
    assert rig.kinds() == ["fallback"], cause
    assert rig.router.lookup("C'est fait.")[-1] == pytest.approx(KOKORO_LEVEL)  # type: ignore[index]
    second = rig.router.synthesize_sync("Et demain aussi.")
    assert len(second) == ROUTER_RATE // 10, cause  # annoncé une seule fois
    usage = json.loads(rig.eleven._usage_path.read_text(encoding="utf-8"))
    assert usage["state"] == "kokoro" and usage["reason"]


def test_fallback_when_the_connection_fails_then_back(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.router.load()
    rig.api.down = True
    _fallback_then_resume(rig, "réseau")
    rig.api.down = False
    assert rig.say("Toujours Kokoro pendant la pause.") == pytest.approx(KOKORO_LEVEL)
    rig.clock.now += 301  # retry_after_s
    assert rig.say("ElevenLabs est revenu.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.kinds() == ["fallback", "resume"]


@pytest.mark.parametrize(
    ("status", "code", "headers", "cause"),
    [
        (401, "invalid_api_key", {}, "refusée"),
        (429, "too_many_concurrent_requests", {"retry-after": "12"}, "saturé"),
        (401, "quota_exceeded", {}, "épuisé"),
        (500, "", {}, "erreur ElevenLabs"),
    ],
)
def test_fallback_on_api_errors(
    tmp_path: Path, status: int, code: str, headers: dict[str, str], cause: str
) -> None:
    rig = Rig(tmp_path)
    rig.router.load()
    rig.api.error = (status, {"status": code, "message": "non"}, headers)
    _fallback_then_resume(rig, cause)
    assert cause in (rig.router.fallback_reason or "")
    rig.api.error = None
    if status == 401 and code == "invalid_api_key":  # jusqu'à une nouvelle clé
        rig.clock.now += 86_400
        assert rig.say("Toujours la voix de base.") == pytest.approx(KOKORO_LEVEL)
        rig.eleven.configure(rig.eleven.cfg, KEY)  # réglages relus (nouvelle clé dans .env)
    elif status == 429:
        rig.clock.now += 13  # Retry-After
    elif code == "quota_exceeded":  # jusqu'au renouvellement
        rig.clock.now += 3 * 86400
        assert rig.say("Pas encore renouvelé.") == pytest.approx(KOKORO_LEVEL)
        rig.api.count, rig.api.reset = 0, rig.api.reset + 30 * 86400
        rig.clock.now = rig.eleven.credits.reset_unix or 0  # renouvellement
    else:
        rig.clock.now += 301
    assert rig.say("Retour sur ElevenLabs.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.kinds() == ["fallback", "resume"]


def test_fallback_when_credit_is_low_until_renewal(tmp_path: Path) -> None:
    rig = Rig(tmp_path, monthly_credit_cap=40_000)
    rig.api.count = 39_700  # moins de min_remaining_credits (500)
    rig.router.load()
    assert rig.api.synth == []  # rien de facturé : phrases et fragments en Kokoro
    assert rig.kinds() == ["fallback"]
    assert "presque épuisé" in (rig.router.fallback_reason or "")
    pcm = rig.router.synthesize_sync("Première réponse du jour.")
    assert pcm[-1] == pytest.approx(KOKORO_LEVEL) and len(pcm) > ROUTER_RATE // 10  # annonce
    rig.api.count = 0  # renouvellement
    rig.clock.now = rig.api.reset + 1
    rig.api.reset += 30 * 86400
    assert rig.say("Le mois est renouvelé.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.kinds() == ["fallback", "resume"]
    reads = rig.api.subscription_reads
    rig.api.reset = rig.clock.now - 10  # l'API tarde à donner la nouvelle date
    rig.eleven.refresh_subscription()
    for i in range(5):
        rig.say(f"Fragment {i}.")
    assert rig.api.subscription_reads == reads + 1  # pas un relevé par fragment


def test_fallback_when_the_monthly_cap_is_reached(tmp_path: Path) -> None:
    rig = Rig(tmp_path, subscription_refresh_every=100)
    rig.router.load()
    rig.router.warm()
    assert rig.kinds() == []
    rig.eleven.credits.cap = (rig.eleven.credits.used or 0) + 100
    # Déduction locale (estimation 1 crédit par caractère) : le plafond est vite atteint.
    for i in range(4):
        rig.router.synthesize_sync(f"Réponse {i} : il fait beau et chaud.")
    assert rig.kinds() == ["fallback"]
    assert "plafond" in (rig.router.fallback_reason or "")
    assert rig.say("Plus rien de facturé.") == pytest.approx(KOKORO_LEVEL)


def test_missing_key_means_kokoro_without_any_request(tmp_path: Path) -> None:
    rig = Rig(tmp_path, key=None)
    rig.api.requests.clear()
    rig.router.load()
    assert rig.say("Bonjour.") == pytest.approx(KOKORO_LEVEL)
    assert rig.api.requests == [] and "ELEVENLABS_API_KEY" in (rig.router.fallback_reason or "")


def test_missing_voice_uses_the_backup_voice(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.api.missing = {LUCIE}
    rig.router.load()
    assert rig.say("Bonjour.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.api.synth[-1] == (EMILIE, "Bonjour.")
    rig.api.missing = {LUCIE, EMILIE}
    assert rig.say("Plus aucune voix.") == pytest.approx(KOKORO_LEVEL)
    assert rig.kinds() == ["fallback"]


def test_a_key_that_cannot_read_the_balance_still_speaks(tmp_path: Path) -> None:
    api = FakeApi()
    real = api.__call__

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/user/subscription":
            detail = {"status": "missing_permissions", "message": "user_read"}
            return httpx2.Response(401, json={"detail": detail})
        return real(request)

    tts = ElevenLabsTTS(
        _cfg(calibration_hours=0), KEY, tmp_path, tmp_path / "u.json",
        transport=httpx2.MockTransport(handler),
    )  # fmt: skip
    tts.load()
    assert tts.unavailable_reason() is None and not tts.credits.known
    assert len(tts.synthesize_sync("Bonjour.", LUCIE)) > 0


def test_unavailable_reason_is_never_the_key(tmp_path: Path) -> None:
    api = FakeApi()
    api.error = (401, {"status": "invalid_api_key", "message": f"bad {KEY}"}, {})
    tts = _eleven(tmp_path, api)
    with pytest.raises(ElevenLabsUnavailable) as caught:
        tts.synthesize_sync("Bonjour.", LUCIE)
    assert KEY not in caught.value.reason


# -- profils : skin + voix, à chaud ----------------------------------------------------------


def test_choose_voice_follows_the_skin() -> None:
    assert choose_voice(_config("gars")).voice_id == THEO
    assert choose_voice(_config("femme")).label == "Lucie"
    assert choose_voice(_config("default")).engine == "kokoro"
    assert choose_voice(_config("sakura", backend="piper")).engine == "piper"  # sans profil
    unnamed = Config.model_validate(
        {"ui": {"skin": "x", "profiles": {"x": {"voice": "elevenlabs"}}}}
    )
    assert choose_voice(unnamed).voice_id == ElevenLabsConfig().voice_id


def test_switching_skin_switches_voice_without_restart(tmp_path: Path) -> None:
    rig = Rig(tmp_path, skin="default")
    rig.router.load()
    assert rig.api.requests == []  # Mika : Kokoro, ElevenLabs pas même contacté
    assert rig.say("Bonjour.") == pytest.approx(KOKORO_LEVEL)
    assert rig.router.apply_sync(_config("gars"))
    assert rig.say("Bonjour.") == pytest.approx(ELEVEN_LEVEL, abs=1e-3)
    assert rig.api.synth[-1] == (THEO, "Bonjour.")
    assert rig.router.apply_sync(_config("femme"))
    rig.say("Bonjour.")
    assert rig.api.synth[-1] == (LUCIE, "Bonjour.")
    assert not rig.router.apply_sync(_config("femme"))  # même voix : rien
    piper = _config("sakura", backend="piper")  # skin sans profil : tts.backend
    assert rig.router.apply_sync(piper)
    pcm = rig.router.synthesize_sync("Bonjour.")
    assert _level(pcm) == pytest.approx(PIPER_LEVEL, abs=1e-3)
    assert len(pcm) == pytest.approx(ROUTER_RATE // 10, abs=2)  # 22 050 Hz rééchantillonné
    assert [k for k in rig.kinds()] == ["profile"] * 3
    assert "Théo" in rig.events[0][1]
