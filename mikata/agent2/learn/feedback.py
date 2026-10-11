"""Agent v2 (phase 13d) : entendre une correction et punir ce qui s'est trompé.

La commande suivante, dans les 60 s d'une tâche réussie, peut être une correction :
« non, sur YouTube », « pas sur Spotify, sur YouTube », « pas celle-là », « c'est pas ça »,
« je voulais le générique ». L'agent rejoue alors la dernière demande corrigée.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from mikata.agent2.schema import Trace

_CORRECTION_WINDOW_S = 60.0

_PATTERNS: list[tuple[str, str, int | None]] = [
    # (regex insensible à la casse, kind, n° du groupe de la valeur)
    (r"pas\s+sur\s+(.+?)\s*,\s*sur\s+(.+)", "channel", 2),
    (r"non\s*[,:]?\s+sur\s+(.+)", "channel", 1),
    (r"pas\s+cell?e[\s-]+l[àa]", "wrong", None),
    (r"c['’]est\s+pas\s+[çc]a", "wrong", None),
    (r"je\s+voulais\s+(.+)", "item", 1),
]


class Correction(BaseModel):
    kind: Literal["channel", "item", "wrong"]
    value: str | None = None


def detect(command: str, last: Trace | None, now: datetime) -> Correction | None:
    """La correction portée par la commande, ou None si ce n'en est pas une."""
    if last is None or last.outcome != "ok":
        return None
    if (now - last.started).total_seconds() > _CORRECTION_WINDOW_S:
        return None
    for pattern, kind, group in _PATTERNS:
        match = re.search(pattern, command, re.I)
        if match is None:
            continue
        value = match.group(group).strip() if group is not None else None
        return Correction(kind=kind, value=value)
    return None
