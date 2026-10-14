"""Sessions média de Windows (phase 17e) : ce que joue chaque appli, et sa pause.

Windows tient une session de contrôle média (SMTC, « Global System Media Transport
Controls ») par appli qui joue : Spotify, un navigateur, le lecteur multimédia... Chacune
dit son état (lecture, pause) et son titre, et se commande à part — les touches média,
elles, vont à l'appli que Windows juge courante (« pause » pouvait arrêter YouTube au lieu
de Spotify). Sert l'interface : le titre en cours et Pause / Reprendre d'une appli tenue
en arrière-plan. Mesuré le 14/10 : Spotify caché y est, avec son titre, en 0,14 s.

`winrt` est importé paresseusement (Windows), simulé dans les tests. Appels bloquants :
une boucle asyncio privée, dans le thread des outils (`ToolContext.run_blocking`).
"""

from __future__ import annotations

import asyncio
import unicodedata
from dataclasses import dataclass
from typing import Any

_PLAYING = 4  # GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING


@dataclass(frozen=True)
class MediaSession:
    app: str  # identifiant de l'appli pour Windows (« Spotify.exe », ou celui d'une appli du Store)
    playing: bool
    title: str = ""
    artist: str = ""
    can_pause: bool = False
    can_play: bool = False

    @property
    def label(self) -> str:
        """« Jet Lag · Tiakola » ; "" sans titre."""
        return " · ".join(part for part in (self.title, self.artist) if part)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def matching(app: str, found: list[MediaSession]) -> MediaSession | None:
    """La session de l'appli `app` (« Spotify » trouve « Spotify.exe » comme l'identifiant
    d'une appli du Store), sinon None."""
    wanted = _fold(app).strip()
    for session in found:
        if wanted and wanted in _fold(session.app):
            return session
    return None


def _manager() -> Any:
    from winrt.windows.media.control import (  # type: ignore[import-not-found,unused-ignore]
        GlobalSystemMediaTransportControlsSessionManager,
    )

    return GlobalSystemMediaTransportControlsSessionManager


async def _read() -> list[tuple[Any, MediaSession]]:
    manager = await _manager().request_async()
    found: list[tuple[Any, MediaSession]] = []
    for session in manager.get_sessions():
        info = session.get_playback_info()
        try:
            props = await session.try_get_media_properties_async()
            title, artist = str(props.title or ""), str(props.artist or "")
        except Exception:  # appli qui ne publie pas son titre
            title, artist = "", ""
        controls = info.controls
        found.append(
            (
                session,
                MediaSession(
                    app=str(session.source_app_user_model_id or ""),
                    playing=int(info.playback_status) == _PLAYING,
                    title=title,
                    artist=artist,
                    can_pause=bool(controls.is_pause_enabled),
                    can_play=bool(controls.is_play_enabled),
                ),
            )
        )
    return found


def sessions() -> list[MediaSession]:
    """Les sessions média ouvertes ; [] si Windows ne les donne pas (jamais une erreur)."""
    try:
        return [info for _, info in asyncio.run(_read())]
    except Exception:
        return []


def toggle(app: str) -> str | None:
    """Pause si l'appli `app` joue, sinon reprise : « pause », « lecture », ou None si
    elle n'a pas de session (rien ne joue, ou l'appli ne se laisse pas commander)."""

    async def run() -> str | None:
        for session, info in await _read():
            if matching(app, [info]) is None:
                continue
            if info.playing and info.can_pause:
                await session.try_pause_async()
                return "pause"
            if info.can_play:
                await session.try_play_async()
                return "lecture"
        return None

    try:
        return asyncio.run(run())
    except Exception:
        return None
