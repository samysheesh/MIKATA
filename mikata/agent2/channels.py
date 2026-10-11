"""Agent v2 (phase 13b) : le registre des canaux et le choix par score.

Un canal déclare ce qu'il sait faire (capacités) au lieu que le LLM le devine ; le choix
combine préférence apprise, disponibilité réelle, taux de réussite passé et coût.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from mikata.agent2.preferences import PreferenceStore
from mikata.agent2.schema import Intent, Verb

_NAME_REPLACEMENTS = {
    "youtube": "YouTube",
    "spotify": "Spotify",
    "discord": "Discord",
    "browser": "le navigateur",
    "files": "les fichiers",
    "system": "le système",
    "apps": "les applications",
    "web": "le web",
}


class Channel(BaseModel):
    name: str
    kind: str  # "app" | "site" | "system"
    capabilities: list[str]  # "play.video", "play.music", "send.message", "find.file",
    # "open.app", "control.system", "ask.web", "create.document"
    app_names: list[str] = []  # noms d'applis installées qui rendent le canal disponible
    url: str | None = None  # pour kind == "site"
    base_cost: float = 0.5  # 0 (rapide, fiable) à 1 (lent, fragile)


BUILTIN_CHANNELS: list[Channel] = [
    Channel(
        name="youtube",
        kind="site",
        capabilities=["play.video", "play.music"],
        url="https://www.youtube.com",
    ),
    Channel(
        name="spotify",
        kind="app",
        capabilities=["play.music", "play.podcast"],
        app_names=["Spotify"],
    ),
    Channel(
        name="discord",
        kind="app",
        capabilities=["send.message", "send.file"],
        app_names=["Discord"],
    ),
    Channel(name="browser", kind="site", capabilities=["ask.web", "open.site"]),
    Channel(name="files", kind="system", capabilities=["find.file", "organize.file", "open.file"]),
    Channel(name="system", kind="system", capabilities=["control.system", "schedule.timer"]),
    Channel(name="apps", kind="system", capabilities=["open.app"]),
    Channel(name="web", kind="system", capabilities=["ask.web"]),
]

_BY_NAME = {channel.name: channel for channel in BUILTIN_CHANNELS}

# verb + mots de la catégorie d'objet -> capacité. Table fixe en Python.
_PLAY_VIDEO_WORDS = ("vidéo", "video", "clip", "opening", "ending", "trailer", "film", "anime")
_PLAY_MUSIC_WORDS = ("musique", "chanson", "album", "playlist")
_SEND_FILE_WORDS = ("fichier", "document", "pdf")


def capability_for(intent: Intent) -> str:
    verb = intent.verb
    kind = intent.object_kind.strip().lower()
    if verb is Verb.PLAY:
        if any(word in kind for word in _PLAY_VIDEO_WORDS):
            return "play.video"
        if "podcast" in kind:
            return "play.podcast"
        if any(word in kind for word in _PLAY_MUSIC_WORDS):
            return "play.music"
        return "play.generic"
    if verb is Verb.SEND:
        if any(word in kind for word in _SEND_FILE_WORDS):
            return "send.file"
        return "send.message"
    if verb is Verb.FIND:
        return "find.file"
    if verb is Verb.ORGANIZE:
        return "organize.file"
    if verb is Verb.OPEN:
        if any(word in kind for word in ("application", "appli", "logiciel", "jeu")):
            return "open.app"
        return "open.site"
    if verb is Verb.ASK:
        return "ask.web"
    if verb is Verb.CONTROL:
        return "control.system"
    if verb is Verb.SCHEDULE:
        return "schedule.timer"
    if verb in (Verb.CREATE, Verb.EDIT):
        return "create.document"
    return f"{verb.value}.generic"


def available(channel: Channel, installed: Sequence[str]) -> bool:
    if channel.kind in ("site", "system"):
        return True
    lowered = [name.casefold() for name in installed]
    return any(any(name.casefold() in entry for entry in lowered) for name in channel.app_names)


def display_name(name: str) -> str:
    return _NAME_REPLACEMENTS.get(name, name)


def choose(
    intent: Intent,
    prefs: PreferenceStore,
    installed: Sequence[str],
    success_rates: dict[str, float],
) -> tuple[Channel, bool]:
    """(canal, annoncer). annoncer = True si aucune préférence ≥ 0.6 n'a décidé."""
    capability = capability_for(intent)
    if intent.channel_hint:
        hint = intent.channel_hint
        channel = _BY_NAME.get(hint.casefold())
        if channel is not None:
            return channel, False
        # Canal imposé mais inconnu : on le traite comme une appli générique.
        return Channel(name=hint, kind="app", capabilities=[capability], app_names=[hint]), False
    candidates = [
        channel
        for channel in BUILTIN_CHANNELS
        if capability in channel.capabilities and available(channel, installed)
    ]
    if not candidates:
        return _BY_NAME["browser"], True  # repli universel : le navigateur
    pref_scores = prefs.scores(intent.verb.value, intent.object_kind)
    scored = [
        (
            channel,
            2.0 * pref_scores.get(channel.name, 0.0)
            + 1.0 * success_rates.get(channel.name, 0.5)
            - 0.5 * channel.base_cost,
        )
        for channel in candidates
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    best = scored[0][0]
    announce = pref_scores.get(best.name, 0.0) < 0.6
    return best, announce
