"""Tokens consommés et coût estimé (ajout phase 4).

Une ligne de log par commande (entrée / sortie / cache / réflexion, coût), et un cumul gardé
dans `data/llm_usage.json` (réécrit en entier à chaque commande, jamais supprimé) pour suivre
le crédit restant d'une session à l'autre.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from mikata.config import LlmConfig
from mikata.llm.base import Usage

MTOK = 1_000_000


def cost_usd(usage: Usage, cfg: LlmConfig) -> float:
    """Coût d'un décompte : entrée hors cache + entrée en cache + sortie (réflexion comprise)."""
    uncached = max(0, usage.prompt - usage.cached)
    return (
        uncached * cfg.price_input_usd_per_mtok
        + usage.cached * cfg.price_cached_usd_per_mtok
        + usage.completion * cfg.price_output_usd_per_mtok
    ) / MTOK


@dataclass
class UsageTotals:
    cost_usd: float = 0.0
    commands: int = 0
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    since: str = ""


class UsageTracker:
    def __init__(self, cfg: LlmConfig, path: Path) -> None:
        self.cfg = cfg
        self.path = path
        self._lock = threading.Lock()
        self.totals = self._load()

    def _load(self) -> UsageTotals:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return UsageTotals(
                **{k: data[k] for k in UsageTotals.__dataclass_fields__ if k in data}
            )
        except FileNotFoundError:
            return UsageTotals(since=datetime.now().isoformat(timespec="seconds"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Compteur de tokens illisible ({}), repart de zéro : {}", self.path, exc)
            return UsageTotals(since=datetime.now().isoformat(timespec="seconds"))

    def _save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".part")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.totals.__dict__, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("Compteur de tokens non enregistré : {}", exc)

    def record(self, label: str, usage: Usage | None, calls: int) -> float:
        """Ajoute le décompte d'une commande, écrit la ligne de log ; renvoie son coût."""
        if usage is None:
            logger.info(
                "Tokens {} : décompte non fourni par le serveur ({} appel(s) LLM).", label, calls
            )
            return 0.0
        cost = cost_usd(usage, self.cfg)
        with self._lock:
            totals = self.totals
            totals.cost_usd += cost
            totals.commands += 1
            totals.prompt += usage.prompt
            totals.completion += usage.completion
            totals.cached += usage.cached
            self._save()
            remaining = max(0.0, self.cfg.credit_usd - totals.cost_usd)
            average = totals.cost_usd / totals.commands
        left = f", ≈ {remaining / average:.0f} commandes restantes" if average > 0 else ""
        logger.info(
            "Tokens {} : entrée {} (dont cache {}), sortie {} (dont réflexion {}), {} appel(s) ; "
            "coût ≈ {:.5f} $ ; cumul {:.4f} $ sur {:.2f} $ de crédit ({:.1f} %{}).",
            label,
            usage.prompt,
            usage.cached,
            usage.completion,
            usage.reasoning,
            calls,
            cost,
            totals.cost_usd,
            self.cfg.credit_usd,
            100 * totals.cost_usd / self.cfg.credit_usd if self.cfg.credit_usd else 0.0,
            left,
        )
        return cost
