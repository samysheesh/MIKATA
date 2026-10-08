"""Compteur de crédits ElevenLabs (ajout phase 11), sans réseau : testable.

Le solde vient de GET /v1/user/subscription (`character_count` = crédits consommés sur le
mois, `character_limit` = crédits du mois, `next_character_count_reset_unix` = date du
renouvellement). Entre deux relevés, chaque synthèse est déduite localement : coût annoncé
par l'API s'il est donné, sinon caractères envoyés x coût mesuré par caractère. Le coût
par caractère n'est jamais supposé : il est mesuré (relevé avant/après une synthèse de
calibrage, puis à chaque relevé) ; tant qu'il ne l'est pas, une valeur prudente sert
d'estimation (`tts.elevenlabs.estimated_credits_per_char`) et l'interface l'indique.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

USAGE_FILE = "data/elevenlabs_usage.json"
TYPICAL_ANSWER_CHARS = 150  # pour « réponses restantes »
MIN_MEASURE_CHARS = 40  # en dessous, un écart de relevés ne dit rien du coût par caractère


@dataclass
class CreditTracker:
    cap: float  # plafond de dépense mensuel (sur `character_count`)
    min_remaining: float  # crédit restant minimal avant de passer sur la voix de base
    estimated_rate: float  # crédits par caractère tant que rien n'est mesuré
    used: float | None = None  # crédits consommés ce mois (relevé + déductions locales)
    limit: float | None = None
    reset_unix: float | None = None
    measured_rate: float | None = None
    measured_at: float | None = None
    _count_at_refresh: float | None = field(default=None, repr=False)
    _chars_since_refresh: int = field(default=0, repr=False)
    _priced_since_refresh: bool = field(default=False, repr=False)
    synth_since_refresh: int = 0

    @property
    def rate(self) -> float:
        return self.measured_rate if self.measured_rate is not None else self.estimated_rate

    @property
    def known(self) -> bool:
        """Le solde a été relevé au moins une fois."""
        return self.used is not None and self.limit is not None

    def record(self, chars: int, cost: float | None = None) -> float:
        """Une synthèse facturée : `cost` annoncé par l'API, sinon estimé. Renvoie le coût
        déduit."""
        if cost is not None and chars >= MIN_MEASURE_CHARS:
            self._set_rate(cost / chars, "coût annoncé par l'API")
            self._priced_since_refresh = True
        spent = cost if cost is not None else chars * self.rate
        if self.used is not None:
            self.used += spent
        self._chars_since_refresh += chars
        self.synth_since_refresh += 1
        return spent

    def refresh(self, count: float, limit: float, reset_unix: float | None) -> None:
        """Relevé de l'abonnement. Mesure le coût par caractère depuis le relevé précédent
        (si l'API ne l'annonce pas et que le compteur n'a pas été remis à zéro)."""
        previous = self._count_at_refresh
        chars = self._chars_since_refresh
        if (
            previous is not None
            and not self._priced_since_refresh
            and chars >= MIN_MEASURE_CHARS
            and count >= previous
        ):
            delta = count - previous
            if delta > 0:  # compteur pas encore à jour : 0, on ne conclut rien
                self._set_rate(delta / chars, "écart des relevés de l'abonnement")
        self.used, self.limit, self.reset_unix = float(count), float(limit), reset_unix
        self._count_at_refresh = float(count)
        self._chars_since_refresh = 0
        self._priced_since_refresh = False
        self.synth_since_refresh = 0

    def _set_rate(self, rate: float, how: str) -> None:
        if self.measured_rate is None or abs(rate - self.measured_rate) > 1e-4:
            logger.info("ElevenLabs : coût mesuré {:.4f} crédit par caractère ({}).", rate, how)
        self.measured_rate = rate
        self.measured_at = time.time()

    @property
    def remaining(self) -> float | None:
        """Crédits encore utilisables : ce qui reste du mois, sous le plafond choisi."""
        if not self.known:
            return None
        assert self.used is not None and self.limit is not None
        return max(0.0, min(self.limit - self.used, self.cap - self.used))

    def block_reason(self) -> str | None:
        """Pourquoi ElevenLabs ne doit plus servir (None : il peut)."""
        if not self.known:
            return None
        assert self.used is not None and self.limit is not None
        if self.used >= self.cap:
            return f"plafond mensuel atteint ({self.used:.0f} / {self.cap:.0f} crédits)"
        if self.limit - self.used < self.min_remaining:
            left = max(0.0, self.limit - self.used)
            return f"crédit du mois presque épuisé ({left:.0f} restants)"
        return None

    def answers_left(self) -> int | None:
        remaining = self.remaining
        if remaining is None:
            return None
        if self.limit is not None and self.used is not None:
            remaining = max(0.0, min(remaining, self.limit - self.used - self.min_remaining))
        return int(remaining // (TYPICAL_ANSWER_CHARS * self.rate))

    def restore(self, data: dict[str, Any]) -> None:
        """Coût mesuré gardé d'un démarrage à l'autre (data/elevenlabs_usage.json)."""
        rate, at = data.get("rate_measured"), data.get("measured_at")
        if isinstance(rate, int | float) and rate > 0 and isinstance(at, int | float):
            self.measured_rate, self.measured_at = float(rate), float(at)

    def summary(self, **extra: Any) -> dict[str, Any]:
        return {
            "used": self.used,
            "limit": self.limit,
            "cap": self.cap,
            "remaining": self.remaining,
            "reset_unix": self.reset_unix,
            "rate": self.rate,
            "rate_measured": self.measured_rate,
            "measured_at": self.measured_at,
            "answers_left": self.answers_left(),
            "updated": time.time(),
            **extra,
        }


def write_usage(path: Path, summary: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Compteur ElevenLabs non écrit ({}) : {}", path, exc)


def read_usage_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
