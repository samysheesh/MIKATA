"""Agent v2 (phase 13b) : les préférences de canal, apprises par l'usage.

Fichier `data/agent2/preferences.json`, indexé par `verbe|catégorie d'objet` — jamais par
objet précis. Écriture atomique, lecture tolérante (fichier absent ou abîmé = vide).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_CONFIRM_STEP = 0.1  # une confirmation sans correction monte doucement
_CORRECT_PENALTY = 0.5  # une correction fait chuter l'ancien canal
_CORRECT_SCORE = 0.95  # et impose le nouveau

PREFS_FILE = "data/agent2/preferences.json"


class PreferenceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.channels: dict[str, dict[str, float]] = self._load()

    def _load(self) -> dict[str, dict[str, float]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("channels", {})
        except (OSError, ValueError, TypeError, AttributeError):
            return {}
        loaded: dict[str, dict[str, float]] = {}
        for key, scores in raw.items():
            if isinstance(key, str) and isinstance(scores, dict):
                loaded[key] = {
                    str(name): float(score)
                    for name, score in scores.items()
                    if isinstance(score, (int, float))
                }
        return loaded

    def save(self) -> None:
        """Écriture atomique : `.part` puis `replace`."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_suffix(".json.part")
        payload = {
            "version": 1,
            "channels": self.channels,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        part.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        part.replace(self.path)

    @staticmethod
    def key(verb: str, object_kind: str) -> str:
        return f"{verb}|{object_kind.strip().lower()}"

    def scores(self, verb: str, object_kind: str) -> dict[str, float]:
        return dict(self.channels.get(self.key(verb, object_kind), {}))

    def confirm(self, verb: str, object_kind: str, channel: str) -> None:
        """Un choix accepté : le canal monte doucement, plafonné à 1."""
        key = self.key(verb, object_kind)
        entry = self.channels.setdefault(key, {})
        entry[channel] = min(1.0, entry.get(channel, 0.0) + _CONFIRM_STEP)
        self.save()

    def correct(self, verb: str, object_kind: str, wrong: str, right: str) -> None:
        """Une correction : l'ancien canal chute, le nouveau s'impose."""
        key = self.key(verb, object_kind)
        entry = self.channels.setdefault(key, {})
        entry[wrong] = max(0.0, entry.get(wrong, 0.0) - _CORRECT_PENALTY)
        entry[right] = _CORRECT_SCORE
        self.save()

    def describe(self, limit: int = 10) -> str:
        """Texte court pour les prompts : les préférences les plus marquées d'abord."""
        lines: list[str] = []
        for key in sorted(self.channels):
            strongest = sorted(self.channels[key].items(), key=lambda item: item[1], reverse=True)
            for name, score in strongest[:2]:
                if score >= 0.6:
                    lines.append(f"{key} -> {name} ({score:.1f})")
                    break
        return "\n".join(lines[:limit])
