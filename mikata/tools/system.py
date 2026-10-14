"""Outils système (§7) : get_time, system_status, set_timer ★, set_reminder ★, lock ★,
et (phase 5) shutdown ★ / restart ★ / sleep ★.

Extinction, redémarrage, veille : MIKATA annonce l'action, puis laisse
`guard.shutdown_cancel_s` (5 s) pour dire « stop », qui l'annule (§7). Pas une
confirmation : sans réponse, l'action a lieu.

Minuteurs et rappels : tâches asyncio en mémoire (perdues si MIKATA redémarre). À
l'échéance : son « timer », phrase dite dans le casque et notification Windows.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from mikata.action.effects import Effect
from mikata.llm.prompts import format_now
from mikata.tools.base import BaseTool, NoParams, ToolContext, ToolError, current_turn
from mikata.tools.registry import register_tool

MAX_DELAY = timedelta(days=7)
_TIME_RE = re.compile(r"^\s*(?:(\d{4})-(\d{2})-(\d{2})[ T])?(\d{1,2})\s*[:h]\s*(\d{2})?\s*$")


def format_duration(seconds: float) -> str:
    """Durée dite à voix haute : « 1 heure 30 minutes », « 45 secondes »."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    for value, unit in ((hours, "heure"), (minutes, "minute"), (secs, "seconde")):
        if value:
            parts.append(f"{value} {unit}{'s' if value > 1 else ''}")
    return " ".join(parts) or "0 seconde"


def format_clock(moment: datetime, now: datetime) -> str:
    """« à 9 h 05 », « demain à 9 h 05 » ou « le jeudi 9 octobre 2026 à 9 h 05 »."""
    hour = f"{moment.hour} h {moment.minute:02d}"
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"à {hour}"
    if days == 1:
        return f"demain à {hour}"
    return f"le {format_now(moment).rsplit(',', 1)[0]} à {hour}"


def parse_when(at: str, now: datetime) -> datetime:
    """« 18:30 », « 18h30 », « 9h » ou « 2026-10-08 09:00 » -> date et heure à venir."""
    match = _TIME_RE.match(at)
    if match is None:
        raise ValueError(f"heure illisible « {at} » (attendu HH:MM ou AAAA-MM-JJ HH:MM)")
    year, month, day, hour, minute = match.groups()
    try:
        if year:
            moment = datetime(int(year), int(month), int(day), int(hour), int(minute or 0))
        else:
            moment = now.replace(hour=int(hour), minute=int(minute or 0), second=0, microsecond=0)
            if moment <= now:  # heure passée aujourd'hui : demain
                moment += timedelta(days=1)
    except ValueError as exc:
        raise ValueError(f"heure invalide « {at} »") from exc
    return moment


async def _alarm(ctx: ToolContext, delay_s: float, spoken: str, title: str) -> None:
    """Attend puis prévient : phrase dans le casque + notification Windows."""
    await ctx.sleep(delay_s)
    logger.info("{} : {}", title, spoken)
    if ctx.notifier is not None:
        with contextlib.suppress(Exception):
            await ctx.run_blocking(ctx.notifier.notify, f"MIKATA — {title}", spoken)
    if ctx.announce is not None:
        try:
            await ctx.announce(spoken)
        except Exception:
            logger.exception("Annonce impossible : {}", spoken)


def _require_announce(ctx: ToolContext) -> None:
    if ctx.announce is None:
        raise ToolError("minuteurs et rappels indisponibles dans ce mode (MIKATA non lancé)")


@register_tool
class GetTime(BaseTool[NoParams]):
    name = "get_time"
    sdk = "system.time"
    effect = Effect.READ
    expected_s = 0.1
    description = "Donne la date et l'heure exactes (à la seconde)."
    params = NoParams

    async def run(self, args: NoParams) -> str:
        now = self.ctx.clock()
        return f"Nous sommes le {format_now(now)}:{now:%S}."


def collect_status(home: Path | None = None) -> str:
    """CPU, RAM, disque, GPU (NVML, sauf macOS) ; bloquant (~0,3 s)."""
    import sys

    import psutil

    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    lines = [
        f"Processeur : {cpu:.0f} % d'utilisation.",
        f"Mémoire : {mem.used / 1e9:.1f} Go utilisés sur {mem.total / 1e9:.0f} Go "
        f"({mem.percent:.0f} %).",
    ]
    try:
        disk = psutil.disk_usage((home or Path.home()).anchor or "/")
        lines.append(
            f"Disque système : {disk.free / 1e9:.0f} Go libres ({disk.percent:.0f} % pris)."
        )
    except OSError:
        pass
    if sys.platform == "darwin":
        # Pas de NVML sur macOS : on tente quand même pynvml (eGPU simulé dans les
        # tests), sinon mémoire résidente du processus à la place.
        try:
            import pynvml

            lines.append(_nvml_line(pynvml))
        except Exception:
            rss = psutil.Process().memory_info().rss
            lines.append(
                "Carte graphique : informations indisponibles. "
                f"(macOS sans NVML ; mémoire résidente MIKATA : {rss / 1e6:.0f} Mo.)"
            )
        lines.append("Température du processeur : non mesurable sans droits administrateur.")
        return "\n".join(lines)
    try:
        import pynvml

        lines.append(_nvml_line(pynvml))
    except Exception:
        lines.append("Carte graphique : informations indisponibles.")
    lines.append("Température du processeur : non mesurable sans droits administrateur.")
    return "\n".join(lines)


def _nvml_line(pynvml: Any) -> str:
    """Ligne d'état du premier GPU NVIDIA (NVML déjà importé)."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else str(name)
        usage = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return (
            f"Carte graphique {name} : {usage.gpu} %, {gpu_mem.used / 1e9:.1f} Go de "
            f"mémoire vidéo sur {gpu_mem.total / 1e9:.0f} Go, {temp} degrés."
        )
    finally:
        pynvml.nvmlShutdown()


@register_tool
class SystemStatus(BaseTool[NoParams]):
    name = "system_status"
    sdk = "system.status"
    effect = Effect.READ
    expected_s = 0.5
    description = (
        "État du PC : utilisation du processeur, de la mémoire, du disque, et carte "
        "graphique (charge, mémoire vidéo, température)."
    )
    params = NoParams

    async def run(self, args: NoParams) -> str:
        return await asyncio.to_thread(collect_status)


class TimerParams(BaseModel):
    seconds: int = Field(ge=1, le=86_400, description="Durée en secondes (5 minutes = 300)")
    label: str | None = Field(default=None, description="Ce que le minuteur concerne (facultatif)")


@register_tool
class SetTimer(BaseTool[TimerParams]):
    name = "set_timer"
    sdk = "system.timer"
    effect = Effect.ACT
    expected_s = 0.1
    description = "Lance un minuteur ; MIKATA prévient à voix haute quand il sonne."
    params = TimerParams
    terminal = True

    async def run(self, args: TimerParams) -> str:
        _require_announce(self.ctx)
        duration = format_duration(args.seconds)
        spoken = f"Le minuteur de {duration} est terminé."
        if args.label:
            spoken = f"Le minuteur {args.label} de {duration} est terminé."
        self.ctx.spawn(_alarm(self.ctx, args.seconds, spoken, "Minuteur"), f"minuteur {duration}")
        end = self.ctx.clock() + timedelta(seconds=args.seconds)
        return f"Minuteur de {duration} lancé, fin à {end:%H:%M:%S}."


class ReminderParams(BaseModel):
    text: str = Field(min_length=1, description="Ce qu'il faut rappeler")
    at: str | None = Field(
        default=None, description="Heure locale « HH:MM » ou date et heure « AAAA-MM-JJ HH:MM »"
    )
    in_minutes: int | None = Field(
        default=None, ge=1, le=10_080, description="Ou bien : dans combien de minutes"
    )

    @model_validator(mode="after")
    def _one_moment(self) -> ReminderParams:
        if (self.at is None) == (self.in_minutes is None):
            raise ValueError("donner soit at, soit in_minutes")
        return self


@register_tool
class SetReminder(BaseTool[ReminderParams]):
    name = "set_reminder"
    sdk = "system.reminder"
    effect = Effect.ACT
    expected_s = 0.1
    description = "Programme un rappel à une heure donnée ou dans N minutes (dans les 7 jours)."
    params = ReminderParams
    terminal = True

    async def run(self, args: ReminderParams) -> str:
        _require_announce(self.ctx)
        now = self.ctx.clock()
        if args.in_minutes is not None:
            moment = now + timedelta(minutes=args.in_minutes)
        else:
            assert args.at is not None
            try:
                moment = parse_when(args.at, now)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        delay = (moment - now).total_seconds()
        if delay <= 0:
            raise ToolError(f"{moment:%Y-%m-%d %H:%M} est déjà passé")
        if moment - now > MAX_DELAY:
            raise ToolError("rappel trop lointain (7 jours au plus)")
        text = args.text.strip().rstrip(".")
        spoken = f"Rappel : {text}."
        self.ctx.spawn(_alarm(self.ctx, delay, spoken, "Rappel"), f"rappel {moment:%H:%M}")
        return f"Rappel programmé {format_clock(moment, now)} : {text}."


@register_tool
class Lock(BaseTool[NoParams]):
    name = "lock"
    sdk = "system.lock"
    effect = Effect.ACT
    description = "Verrouille la session Windows (écran de connexion)."
    params = NoParams
    terminal = True

    async def run(self, args: NoParams) -> str:
        desktop = self.ctx.require_desktop()
        await self.ctx.run_blocking(desktop.lock)
        return "Session verrouillée."


class _PowerTool(BaseTool[NoParams]):
    """Annonce, délai pour dire « stop », puis l'action (desktop.power)."""

    params = NoParams
    terminal = True
    action: str
    announce: str  # « Extinction » -> « Extinction dans 5 secondes. Dis stop pour annuler. »
    cancelled: str  # dit si l'utilisateur annule

    def time_limit(self) -> float:
        guard = self.ctx.config.guard
        return guard.shutdown_cancel_s + guard.confirm_timeout_s + 30.0

    async def run(self, args: NoParams) -> str:
        guard = self.ctx.require_guard()
        desktop = self.ctx.require_desktop()
        proceed = await guard.countdown(
            self.announce, self.cancelled, current_turn(), self.ctx.sleep
        )
        if not proceed:
            return f"{self.cancelled} (l'utilisateur a dit stop)"
        await self.ctx.run_blocking(desktop.power, self.action)
        return f"Demande envoyée à Windows : {self.action}."


@register_tool
class Shutdown(_PowerTool):
    name = "shutdown"
    sdk = "system.shutdown"
    effect = Effect.CONSEQUENTIAL
    expected_s = 6.0
    announces = True  # « Extinction dans 5 secondes... » : pas d'accusé en plus
    description = "Éteint le PC (après 5 s pendant lesquelles « stop » annule)."
    action = "shutdown"
    announce = "Extinction"
    cancelled = "Extinction annulée."


@register_tool
class Restart(_PowerTool):
    name = "restart"
    sdk = "system.restart"
    effect = Effect.CONSEQUENTIAL
    expected_s = 6.0
    announces = True  # « Extinction dans 5 secondes... » : pas d'accusé en plus
    description = "Redémarre le PC (après 5 s pendant lesquelles « stop » annule)."
    action = "restart"
    announce = "Redémarrage"
    cancelled = "Redémarrage annulé."


@register_tool
class Sleep(_PowerTool):
    name = "sleep"
    sdk = "system.sleep"
    effect = Effect.CONSEQUENTIAL
    expected_s = 6.0
    announces = True  # « Extinction dans 5 secondes... » : pas d'accusé en plus
    description = "Met le PC en veille (après 5 s pendant lesquelles « stop » annule)."
    action = "sleep"
    announce = "Mise en veille"
    cancelled = "Mise en veille annulée."
