"""Agent v2 (phase 13e) : la passe de nuit — verrouiller, corriger, fusionner, purger.

La passe traite les savoir-faire en attente (vérrouillés par la batterie de référence),
corrige les savoir-faire rétrogradés, fusionne les doublons, purge les traces et écrit un
rapport. Budget borné : la passe s'arrête dès que son coût dépasse 0,05 $.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from mikata.config import Agent2Config

DREAM_BUDGET_USD = 0.05
IDLE_S = 1200  # 20 minutes sans commande
RETRY_S = 1800  # 30 minutes entre deux tentatives
MAX_TRIES = 3
DREAM_DIR = "data/agent2/dream"


class DreamReport(BaseModel):
    learned: list[str] = []
    fixed: list[str] = []
    merged: list[str] = []
    refused: list[str] = []
    cost_usd: float = 0.0


async def dream(agent2: Any, *, now: datetime) -> DreamReport:
    """La passe elle-même (l'ordonnancement est dans `dream_loop`)."""
    cfg: Agent2Config = agent2.legacy.config.agent2
    learner = agent2.learner
    report = DreamReport()
    if learner is None:
        return report

    def spent() -> float:
        return report.cost_usd

    def over_budget() -> bool:
        return spent() >= DREAM_BUDGET_USD

    # 1. Savoir-faire en attente (pending), chacun verrouillé par la batterie.
    for name, meta, code, recipe in learner.pending():
        if over_budget():
            logger.info("Passe de nuit : budget dépassé, le reste attendra.")
            break
        adopted, cost = await learner.adopt(name, meta, code, recipe, usage=agent2.legacy.usage)
        report.cost_usd += cost
        (report.learned if adopted else report.refused).append(name)

    # 2. Savoir-faire rétrogradés : une version corrigée, verrouillée elle aussi.
    for name in list(learner.store.skills):
        if over_budget():
            break
        meta = learner.store.skills[name]
        if meta.status != "demoted":
            continue
        fixed, cost = await learner.fix(name, usage=agent2.legacy.usage)
        report.cost_usd += cost
        (report.fixed if fixed else report.refused).append(name)

    # 3. Fusion : deux savoir-faire actifs de même capacité, canal et paramètres.
    report.merged = learner.merge_duplicates()

    # 4. Purge des traces.
    agent2.traces.purge(cfg.traces_days)

    # 5. Rapport.
    root = Path(DREAM_DIR)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{now:%Y-%m-%d}.json"
    path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Passe de nuit : {} appris, {} corrigés, {} fusionnés, {} refusés ({:.4f} $).",
        len(report.learned),
        len(report.fixed),
        len(report.merged),
        len(report.refused),
        report.cost_usd,
    )
    return report


async def dream_loop(
    agent2: Any,
    *,
    now_fn: Callable[[], datetime],
    busy: Callable[[], bool],
    last_command_at: Callable[[], float],
    retry_s: float = RETRY_S,
    idle_s: float = IDLE_S,
    tries: int = MAX_TRIES,
    clock: Any = time,
) -> None:
    """La tâche d'arrière-plan : attend l'heure, exige 20 min d'inactivité, tente 3 fois."""
    cfg: Agent2Config = agent2.legacy.config.agent2
    now = now_fn()
    target = now.replace(hour=cfg.dream_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target.replace(day=now.day + 1)
    await asyncio.sleep(max(0.0, (target - now).total_seconds()))
    for attempt in range(1, tries + 1):
        idle_for = clock.monotonic() - last_command_at()
        if busy() or idle_for < idle_s:
            logger.info("Passe de nuit reportée ({}e essai) : activité récente.", attempt)
            if attempt < tries:
                await asyncio.sleep(retry_s)
            continue
        try:
            await dream(agent2, now=now_fn())
        except Exception:
            logger.exception("Passe de nuit impossible.")
        return
    logger.info("Passe de nuit abandonnée : 3 essais sans créneau calme.")
