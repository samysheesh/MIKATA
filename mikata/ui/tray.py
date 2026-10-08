"""Icône de la zone de notification (pystray), docs/ARCHITECTURE.md §8.

pystray tourne dans son propre thread (boucle de messages Win32) ; le menu « Quitter »
rappelle l'orchestrateur via un callback thread-safe. pystray est importé paresseusement :
il n'est installé que sous Windows et il est mocké dans les tests.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw

from mikata.app import State

ICON_SIZE = 64

# Couleur selon l'état : gris repos, bleu écoute, orange réflexion, vert parole, rouge erreur.
STATE_COLORS: dict[State, str] = {
    State.IDLE: "#8a8a8a",
    State.LISTENING: "#2f7de1",
    State.FOLLOW_UP: "#2f7de1",
    State.TRANSCRIBING: "#f28c28",
    State.THINKING: "#f28c28",
    State.ACTING: "#f28c28",
    State.SPEAKING: "#2fb350",
}
ERROR_COLOR = "#d93838"

STATE_LABELS: dict[State, str] = {
    State.IDLE: "repos",
    State.LISTENING: "écoute",
    State.FOLLOW_UP: "suivi",
    State.TRANSCRIBING: "transcription",
    State.THINKING: "réflexion",
    State.ACTING: "action",
    State.SPEAKING: "parole",
}


def make_icon_image(color: str, size: int = ICON_SIZE) -> Image.Image:
    """Disque coloré marqué d'un « M »."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size // 16
    draw.ellipse((margin, margin, size - margin, size - margin), fill=color)
    draw.text((size / 2, size / 2), "M", fill="white", anchor="mm", font_size=size * 0.55)
    return image


class TrayIcon:
    def __init__(self, title: str = "MIKATA") -> None:
        self._title = title
        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._on_quit: Callable[[], None] | None = None

    def start(
        self, on_quit: Callable[[], None], on_action: Callable[[str], None] | None = None
    ) -> None:
        """`on_action` (ajout phase 8) : « settings », « mascot », « pause » ; sans lui, le
        menu n'a que « Quitter » (interface désactivée)."""
        import pystray  # Windows uniquement

        self._on_quit = on_quit
        items = []
        if on_action is not None:
            items = [
                pystray.MenuItem("Réglages…", lambda *_: on_action("settings"), default=True),
                pystray.MenuItem("Afficher la mascotte", lambda *_: on_action("mascot")),
                pystray.MenuItem("Pause / reprise de l'écoute", lambda *_: on_action("pause")),
                pystray.Menu.SEPARATOR,
            ]
        menu = pystray.Menu(*items, pystray.MenuItem("Quitter", self._quit_clicked))
        self._icon = pystray.Icon(
            "mikata",
            icon=make_icon_image(STATE_COLORS[State.IDLE]),
            title=self._tooltip(State.IDLE),
            menu=menu,
        )
        self._thread = threading.Thread(target=self._run, name="mikata-tray", daemon=True)
        self._thread.start()
        logger.info("Icône de la zone de notification affichée.")

    def _run(self) -> None:
        try:
            self._icon.run()
        except Exception:
            logger.exception("L'icône de la zone de notification s'est arrêtée sur une erreur.")

    def _tooltip(self, state: State) -> str:
        return f"{self._title} — {STATE_LABELS[state]}"

    def _quit_clicked(self, icon: Any = None, item: Any = None) -> None:
        logger.info("Menu « Quitter » cliqué.")
        if self._on_quit is not None:
            self._on_quit()

    def set_state(self, state: State) -> None:
        if self._icon is None:
            return
        self._icon.icon = make_icon_image(STATE_COLORS[state])
        self._icon.title = self._tooltip(state)

    def set_error(self) -> None:
        if self._icon is not None:
            self._icon.icon = make_icon_image(ERROR_COLOR)
            self._icon.title = f"{self._title} — erreur"

    def stop(self) -> None:
        """Retire l'icône et attend la fin de son thread. Idempotent."""
        icon, self._icon = self._icon, None
        if icon is not None:
            icon.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Le thread de l'icône ne s'est pas arrêté à temps.")
            self._thread = None
