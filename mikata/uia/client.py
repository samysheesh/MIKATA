"""Client d'accessibilité UIA de MIKATA (phase 12a).

- win32 : tant que ce client écoute, Windows répond « vrai » à
  `UiaClientsAreListening()` : les applications Chromium/Electron (Discord,
  VS Code...) exposent alors leur arbre d'accessibilité complet, sans drapeau de
  lancement. Un thread dédié s'abonne aux changements de focus et attend l'arrêt ;
  aucun événement n'est traité (on écoute, c'est tout).
- darwin : pas d'équivalent (AX est toujours actif si l'autorisation
  « Accessibilité » est accordée) ; `start` vérifie juste que `macuiauto` est
  importable et journalise la marche à suivre sinon. Note : pour Chromium,
  préférez CDP (arbre complet sans droits) quand c'est possible.
Hors Windows/macOS ou si l'abonnement échoue : rien ne se passe (le drapeau
`--force-renderer-accessibility` de scripts/discord_accessibility.ps1 reste le secours).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Any, ClassVar

from loguru import logger

from mikata.uia.state import _com_init, load_uia_interfaces


class UiaAccessibilityClient:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if sys.platform == "darwin":
            self._check_ax()
            return
        if sys.platform != "win32":
            return
        self._thread = threading.Thread(target=self._run, name="mikata-uia-client", daemon=True)
        self._thread.start()

    def _check_ax(self) -> None:
        """darwin : AX n'a pas besoin d'écoute ; vérifie le backend + les droits."""
        try:
            import macuiauto  # type: ignore[import-not-found]

            logger.info(
                "Accessibilité macOS : backend AX prêt ({}). Si les fenêtres restent "
                "illisibles, accordez l'autorisation Accessibilité puis relancez.",
                getattr(macuiauto, "__name__", "macuiauto"),
            )
        except ImportError:
            logger.warning(
                "Accessibilité macOS indisponible : macuiauto non installé "
                "(et Chromium : préférez CDP)."
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        _com_init()
        client, handler = self._subscribe()
        if client is None:
            return
        logger.info("Client d'accessibilité UIA actif : les applications se laissent lire.")
        self._stop.wait()
        with contextlib.suppress(Exception):
            client.RemoveFocusChangedEventHandler(handler)

    @staticmethod
    def _subscribe() -> tuple[Any, Any]:
        """Client IUIAutomation + abonnement aux changements de focus, ou (None, None).

        Trois essais : un processus mort sans désabonnement (arrêt forcé) peut faire
        échouer le premier abonnement quelques secondes.
        """
        for attempt in range(3):
            try:
                import comtypes

                uia = load_uia_interfaces()
                client = comtypes.CoCreateInstance(
                    uia.CUIAutomation._reg_clsid_,
                    interface=uia.IUIAutomation,
                    clsctx=comtypes.CLSCTX_INPROC_SERVER,
                )

                class _FocusHandler(comtypes.COMObject):
                    _com_interfaces_: ClassVar[list[object]] = [
                        uia.IUIAutomationFocusChangedEventHandler
                    ]

                    def HandleFocusChangedEvent(self, sender: object) -> None:
                        pass  # on écoute, c'est tout

                handler = _FocusHandler()
                client.AddFocusChangedEventHandler(None, handler)
                return client, handler
            except Exception as exc:
                if attempt == 2:
                    logger.warning("Client d'accessibilité UIA indisponible : {}", exc)
                    return None, None
                time.sleep(1.0)
        return None, None
