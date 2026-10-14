"""Épisodes (phase 16e, docs/AGENT_CORE.md §3.7) : ce que MIKATA a fait récemment, tiré des
traces (`learn/trace.py`), pour une demande qui renvoie au passé (« remets la même vidéo
qu'hier », « qu'est-ce que tu as fait ce matin ? »).

Sans appel au modèle : les traces sont filtrées par des mots en commun avec la demande,
puis les plus récentes d'abord. La réponse dite n'est reprise que pour un tour qui n'a rien
lu de non fiable : un texte piégé lu ailleurs ne revient pas par ce chemin.
"""

from __future__ import annotations

import re
from datetime import datetime

from mikata.learn.skills import fold, words
from mikata.learn.trace import TurnTrace

MAX_EPISODES = 4
ANSWER_CHARS = 90
_PAST = re.compile(
    r"\b(?:hier|avant hier|tout a l heure|ce matin|cette nuit|ce soir|cet apres midi|"
    r"la derniere fois|l autre jour|la semaine derniere|comme d habitude|deja|encore|"
    r"meme .{1,40} que|yesterday|last time|earlier|again)\b"
)


def refers_to_past(command: str) -> bool:
    """La demande renvoie-t-elle à ce que MIKATA a fait avant ?"""
    return _PAST.search(" ".join(fold(command).split())) is not None


def _when(at: str, now: datetime) -> str:
    try:
        moment = datetime.fromisoformat(at)
    except ValueError:
        return at
    hour = f"{moment.hour} h {moment.minute:02d}"
    days = (now.date() - moment.date()).days
    if days == 0:
        return f"aujourd'hui à {hour}"
    if days == 1:
        return f"hier à {hour}"
    return f"le {moment:%d/%m} à {hour}"


def episodes(command: str, traces: list[TurnTrace], now: datetime) -> list[str]:
    """Les épisodes les plus utiles pour cette demande, du plus pertinent au moins."""
    wanted = words(command)
    candidates = [t for t in traces if t.outcome == "ok" and t.request]
    scored = sorted(
        candidates,
        key=lambda t: (len(wanted & words(t.request)), t.at),
        reverse=True,
    )
    lines = []
    for trace in scored[:MAX_EPISODES]:
        line = f"{_when(trace.at, now)} : « {trace.request} »"
        answer = trace.answer.strip()
        if answer and not trace.tainted and answer != "C'est fait.":
            line += f" -> « {answer[:ANSWER_CHARS]} »"
        lines.append(line)
    return lines


def render(lines: list[str]) -> str:
    if not lines:
        return ""
    return "Ce que MIKATA a fait récemment (la demande y renvoie peut-être) :\n" + "\n".join(
        f"- {line}" for line in lines
    )
