"""Phrases d'attente (phase 15c, réglage `agent.narration`) : quand une demande dure, MIKATA
dit où il en est, depuis le cache, sans appel au modèle."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from loguru import logger

from mikata.agent.script_tool import SCRIPT_TOOL
from mikata.agent.turn import SpeechSink
from mikata.llm.base import ToolCallReq
from mikata.tts.cache import PHRASES

# Phrase d'attente (clé de PHRASES) selon l'étape en cours.
_ACTIVITY_PREFIXES = (
    ("browser_", "browser"),
    ("web_search", "search"),
    ("fetch_page", "search"),
    ("find_", "files"),
    ("list_folder", "files"),
    ("read_file", "files"),
    ("uia_", "app"),
)
_SDK_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\(")


def activity_of(calls: Sequence[ToolCallReq]) -> str:
    """L'étape lancée par ces appels : browser, search, files, app ou work."""
    names: list[str] = []
    for call in calls:
        if call.name == SCRIPT_TOOL:  # un programme : les fonctions qu'il appelle
            names.extend(_SDK_CALL.findall(str(call.arguments.get("code", ""))))
        else:
            names.append(call.name)
    for prefix, kind in _ACTIVITY_PREFIXES:
        if any(name.startswith(prefix) for name in names):
            return kind
    return "work"


class Narrator:
    """Destination des phrases d'une commande qui retient quand MIKATA a parlé pour la
    dernière fois. `run` dit une phrase d'attente (en cache, sans LLM) si rien n'a été dit
    `after_s` secondes après le début, selon l'étape en cours (`activity`), puis « ça prend
    un peu plus de temps » après un nouveau silence de 2 × `after_s`. Jamais pendant une
    écoute (confirmation)."""

    MAX_LINES = 2
    POLL_S = 0.2

    def __init__(
        self,
        speech: SpeechSink,
        say_phrase: Callable[[SpeechSink, str], None],
        after_s: float,
    ) -> None:
        self._speech = speech
        self._say_phrase = say_phrase
        self.after_s = after_s
        self.activity = "thinking"
        self.listening = False
        self.lines: list[str] = []  # phrases d'attente dites (clés de PHRASES)
        self._started = time.monotonic()
        self._last_said: float | None = None

    def say(self, text: str) -> None:
        self._last_said = time.monotonic()
        self._speech.say(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        self._last_said = time.monotonic()
        self._speech.say_audio(pcm, sample_rate)

    def __getattr__(self, name: str) -> Any:  # drain, cancelled... de la vraie destination
        return getattr(self._speech, name)

    def starting(self, activity: str, expected_s: float, threshold_s: float) -> None:
        """Phase 16c : une action part. Si sa durée prévue dépasse le seuil et que rien
        n'a encore été dit dans ce tour, MIKATA l'annonce maintenant, en même temps qu'elle
        part (jamais avant : une phrase d'attente systématique coupe la parole)."""
        self.activity = activity
        if expected_s < threshold_s or self._last_said is not None or self.lines:
            return
        if self.listening:
            return
        key = f"wait_{activity}" if f"wait_{activity}" in PHRASES else "wait_work"
        logger.info(
            "Accusé : « {} » (action prévue {:.1f} s, étape : {}).",
            PHRASES[key],
            expected_s,
            activity,
        )
        self.lines.append(key)
        self._say_phrase(self, key)

    def due(self, now: float) -> str | None:
        """La phrase d'attente à dire maintenant, ou None."""
        if self.listening or len(self.lines) >= self.MAX_LINES:
            return None
        if self._last_said is None:
            if now - self._started >= self.after_s:
                return f"wait_{self.activity}"
            return None
        if now - self._last_said >= 2 * self.after_s and "wait_long" not in self.lines:
            return "wait_long"
        return None

    async def run(self) -> None:
        while len(self.lines) < self.MAX_LINES:
            await asyncio.sleep(self.POLL_S)
            key = self.due(time.monotonic())
            if key is None:
                continue
            key = key if key in PHRASES else "wait_work"
            logger.info("Narration : « {} » (étape : {}).", PHRASES[key], self.activity)
            self.lines.append(key)
            self._say_phrase(self, key)


def listening(
    narrator: Narrator, listen: Callable[[float], Awaitable[str]] | None
) -> Callable[[float], Awaitable[str]] | None:
    """L'écoute d'une réponse (confirmation) suspend les phrases d'attente."""
    if listen is None:
        return None

    async def wrapped(timeout_s: float) -> str:
        narrator.listening = True
        try:
            return await listen(timeout_s)
        finally:
            narrator.listening = False

    return wrapped
