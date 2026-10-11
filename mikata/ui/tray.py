"""Icône de la zone de notification (pystray), docs/ARCHITECTURE.md §8.

pystray est importé paresseusement (mocké dans les tests) ; le menu « Quitter »
rappelle l'orchestrateur via un callback thread-safe.
- win32 : boucle de messages Win32 dans le thread « mikata-tray ».
- darwin : backend Cocoa ; AppKit exige le thread principal et sa boucle NSRunLoop.
  L'icône est donc lancée « détachée » sur le thread principal (`run_detached`) et la
  boucle Cocoa est pompée depuis la boucle asyncio (thread principal) : sans ce pompage,
  le menu de la barre de menus ne serait jamais distribué. Si `start` est appelé hors du
  thread principal, un avertissement est journalisé et l'icône retombe sur son thread (le
  menu peut ne pas s'afficher). `stop` reste idempotent.

Le menu « Afficher la mascotte » est le moyen de récupérer une mascotte masquée : il reste
toujours présent, et le libellé/tooltip de l'icône signale l'état masqué quand le fichier
`data/ui_state.json` (préférences de la mascotte) le lui apprend.
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw

from mikata.app import State
from mikata.ui.prefs import STATE_FILE, load_prefs

ICON_SIZE = 64
PUMP_INTERVAL_S = 0.05  # darwin : pompage de la boucle Cocoa (~20 Hz)
PUMP_BLOCK_S = 0.002  # darwin : temps maximal passé dans NSRunLoop à chaque pompage
HIDDEN_POLL_S = 1.0  # relecture de ui_state.json au plus une fois par seconde

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


def set_macos_app_icon(image: Image.Image, size: int = 256) -> bool:
    """Icône de l'application dans le Dock / l'alt-tab macOS (best-effort, darwin).

    `python -m mikata` affiche sinon l'icône Python générique. Réutilise le disque « M »
    (converti en `NSImage` par AppKit, import paresseux). No-op (journalisé en debug) hors
    macOS ou si AppKit manque : la plateforme Windows n'est pas touchée.
    """
    if sys.platform != "darwin":
        return False
    try:
        import AppKit
        import Foundation
    except ImportError as exc:  # AppKit absent : on n'insiste pas
        logger.debug("Icône d'application macOS indisponible : {}", exc)
        return False
    try:
        picture = image.convert("RGBA")
        if picture.size != (size, size):
            picture = picture.resize((size, size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        data = Foundation.NSData.dataWithBytes_length_(buffer.getvalue(), buffer.tell())
        ns_image = AppKit.NSImage.alloc().initWithData_(data)
        AppKit.NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
        return True
    except Exception as exc:  # jamais fatal : l'icône resterait celle de Python
        logger.debug("Icône d'application macOS non appliquée : {}", exc)
        return False


def pump_cocoa(block_s: float = PUMP_BLOCK_S) -> int:
    """Distribue les événements AppKit en attente (darwin, thread principal uniquement).

    pystray tourne « détaché » (pas de `NSApplication.run`) : sans ce pompage, les clics
    sur l'icône de la barre de menus ne seraient jamais distribués. Renvoie le nombre
    d'événements traités ; 0 hors macOS, sans AppKit, ou si rien n'est en attente.
    """
    if sys.platform != "darwin":
        return 0
    try:
        import AppKit
        import Foundation
    except ImportError as exc:  # AppKit absent : rien à pomper
        logger.debug("Boucle Cocoa indisponible : {}", exc)
        return 0
    application = AppKit.NSApplication.sharedApplication()
    deadline = Foundation.NSDate.dateWithTimeIntervalSinceNow_(block_s)
    dispatched = 0
    while True:
        event = application.nextEventMatchingMask_untilDate_inMode_dequeue_(
            AppKit.NSEventMaskAny,
            deadline,
            Foundation.NSDefaultRunLoopMode,
            True,
        )
        if event is None:
            break
        application.sendEvent_(event)
        dispatched += 1
    return dispatched


class TrayIcon:
    def __init__(self, title: str = "MIKATA", state_path: Path | None = None) -> None:
        self._title = title
        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._on_quit: Callable[[], None] | None = None
        # Préférences de la mascotte (masquée ?) : fichier relatif au dossier de travail.
        self.state_path = state_path if state_path is not None else Path(STATE_FILE)
        self._state = State.IDLE
        self._error = False
        self._mascot_hidden = False
        self._hidden_checked_at = 0.0

    def start(
        self, on_quit: Callable[[], None], on_action: Callable[[str], None] | None = None
    ) -> None:
        """`on_action` (ajout phase 8) : « settings », « mascot », « pause » ; sans lui, le
        menu n'a que « Quitter » (interface désactivée)."""
        import pystray  # backend Win32 sous Windows, Cocoa sous macOS

        self._on_quit = on_quit
        items = []
        if on_action is not None:
            items = [
                pystray.MenuItem("Réglages…", lambda *_: on_action("settings"), default=True),
                pystray.MenuItem("Afficher la mascotte", lambda *_: on_action("mascot")),
                pystray.MenuItem("Pause / reprise de l'écoute", lambda *_: on_action("pause")),
                # Phase 12f : le navigateur de travail tourne caché (la lecture y est
                # fiable) — ce menu le ferme sans gestionnaire des tâches.
                pystray.MenuItem(
                    "Fermer le navigateur de MIKATA",
                    lambda *_: on_action("browser_close"),
                ),
                pystray.Menu.SEPARATOR,
            ]
        menu = pystray.Menu(*items, pystray.MenuItem("Quitter", self._quit_clicked))
        self._icon = pystray.Icon(
            "mikata",
            icon=make_icon_image(STATE_COLORS[State.IDLE]),
            title=self._tooltip(State.IDLE),
            menu=menu,
        )
        if sys.platform == "darwin":
            self._start_cocoa()
        else:
            self._thread = threading.Thread(target=self._run, name="mikata-tray", daemon=True)
            self._thread.start()
        logger.info("Icône de la zone de notification affichée.")

    # -- darwin : thread principal + pompage Cocoa ------------------------------------------

    def _start_cocoa(self) -> None:
        """Lance l'icône sur le thread principal (`run_detached`) et prépare le pompage."""
        if threading.current_thread() is not threading.main_thread():
            logger.warning(
                "Icône Cocoa démarrée hors du thread principal : le menu macOS exige la "
                "boucle NSRunLoop du thread principal, il risque de ne pas s'afficher."
            )
            self._thread = threading.Thread(target=self._run, name="mikata-tray", daemon=True)
            self._thread.start()
            return
        self._icon.run_detached()  # prépare l'élément de barre, non bloquant (thread principal)
        set_macos_app_icon(make_icon_image(STATE_COLORS[State.IDLE]))  # Dock / alt-tab
        self._start_watch()

    def _start_watch(self) -> None:
        """Boucle de fond (thread principal) : pompe la boucle Cocoa et suit l'état masqué
        de la mascotte. Le menubar Cocoa a besoin de ce pompage ; le suivi de l'état masqué
        sert le même usage (récupérer la mascotte depuis la barre de menus)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # hors boucle asyncio : pas de pompage ni de suivi
            logger.warning("Boucle asyncio absente : le menu macOS peut ne pas répondre.")
            return
        self._watch_task = loop.create_task(self._watch_loop(), name="mikata-tray-surveille")

    async def _watch_loop(self) -> None:
        while not self._stopping:
            pump_cocoa()  # darwin : distribue les clics de la barre de menus ; 0 ailleurs
            self._refresh_mascot_hidden()
            await asyncio.sleep(PUMP_INTERVAL_S)

    def _run(self) -> None:
        # win32 : boucle de messages Win32 ; darwin : boucle NSRunLoop (Cocoa).
        try:
            self._icon.run()
        except Exception:
            logger.exception("L'icône de la zone de notification s'est arrêtée sur une erreur.")

    # -- libellé ----------------------------------------------------------------------------

    def _tooltip(self, state: State) -> str:
        return f"{self._title} — {STATE_LABELS[state]}"

    def _hidden_tooltip(self) -> str:
        return f"{self._title} — mascotte masquée (menu : Afficher la mascotte)"

    def _apply_title(self) -> None:
        if self._icon is None:
            return
        if self._error:
            self._icon.title = f"{self._title} — erreur"
        elif self._mascot_hidden:
            self._icon.title = self._hidden_tooltip()
        else:
            self._icon.title = self._tooltip(self._state)

    def _quit_clicked(self, icon: Any = None, item: Any = None) -> None:
        logger.info("Menu « Quitter » cliqué.")
        if self._on_quit is not None:
            self._on_quit()

    def set_state(self, state: State) -> None:
        self._state = state
        self._error = False
        if self._icon is None:
            return
        self._icon.icon = make_icon_image(STATE_COLORS[state])
        self._apply_title()

    def set_error(self) -> None:
        self._error = True
        if self._icon is not None:
            self._icon.icon = make_icon_image(ERROR_COLOR)
            self._apply_title()

    def set_mascot_hidden(self, hidden: bool) -> None:
        """Mascotte masquée : l'icône de la barre de menus devient le seul moyen de la
        faire revenir (« Afficher la mascotte ») ; le tooltip le dit."""
        if hidden == self._mascot_hidden:
            return
        self._mascot_hidden = hidden
        self._apply_title()
        if hidden:
            logger.info("Mascotte masquée : « Afficher la mascotte » dans la barre de menus.")

    def _refresh_mascot_hidden(self) -> None:
        """Répercute l'état masqué écrit par l'interface (data/ui_state.json), au plus une
        fois par seconde. `load_prefs` ne lève jamais : fichier absent = préférences par
        défaut."""
        now = time.monotonic()
        if now - self._hidden_checked_at < HIDDEN_POLL_S:
            return
        self._hidden_checked_at = now
        self.set_mascot_hidden(load_prefs(self.state_path).hidden)

    def watch_mascot_state(self, path: Path) -> None:
        """Suit le fichier de préférences de la mascotte (data/ui_state.json) pour refléter
        l'état masqué. Appelé par l'orchestrateur avec un chemin résolu depuis `base_dir`."""
        self.state_path = path

    def stop(self) -> None:
        """Retire l'icône et attend la fin de son thread. Idempotent."""
        self._stopping = True
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        icon, self._icon = self._icon, None
        if icon is not None:
            icon.stop()
            if sys.platform == "darwin":
                pump_cocoa()  # traite l'événement de retrait posté par pystray
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Le thread de l'icône ne s'est pas arrêté à temps.")
            self._thread = None
