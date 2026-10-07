"""Mémoire de conversation (§6).

Court terme (phase 3) : historique en RAM, remis à zéro après `session_reset_min` sans
interaction ou sur « nouvelle conversation ». Plafond ~`history_max_tokens` : les échanges
les plus anciens sautent en premier. Long terme (`remember` / `forget`) : phase 6.

Plusieurs commandes peuvent tourner en parallèle : chacune lit un instantané de l'historique
au départ et y ajoute son échange à la fin.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

CHARS_PER_TOKEN = 3.5  # estimation pour du français, sans tokeniseur
_RESET_RE = re.compile(r"^\W*nouvelle\s+conversation\W*$", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 4  # + surcoût du message


def is_reset_command(text: str) -> bool:
    return _RESET_RE.match(text) is not None


class ShortTermMemory:
    def __init__(
        self,
        max_tokens: int = 6000,
        reset_after_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_tokens = max_tokens
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._exchanges: list[tuple[str, str]] = []
        self._last_activity = clock()

    def _expire(self) -> None:
        if self._clock() - self._last_activity > self.reset_after_s:
            self._exchanges.clear()

    def messages(self) -> list[dict[str, Any]]:
        """Historique à envoyer au LLM (instantané)."""
        self._expire()
        out: list[dict[str, Any]] = []
        for user, assistant in self._exchanges:
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": assistant})
        return out

    def add(self, user: str, assistant: str) -> None:
        self._expire()
        self._exchanges.append((user, assistant))
        self._last_activity = self._clock()
        while self._exchanges and self.tokens > self.max_tokens:
            self._exchanges.pop(0)

    def touch(self) -> None:
        self._expire()
        self._last_activity = self._clock()

    def clear(self) -> None:
        self._exchanges.clear()
        self._last_activity = self._clock()

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(u) + estimate_tokens(a) for u, a in self._exchanges)

    def __len__(self) -> int:
        return len(self._exchanges)
