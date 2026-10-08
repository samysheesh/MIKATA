"""Client d'accessibilité UIA de MIKATA (phase 12a).

Tant que ce client écoute, Windows répond « vrai » à `UiaClientsAreListening()` : les
applications Chromium/Electron (Discord, VS Code...) exposent alors leur arbre
d'accessibilité complet, sans drapeau de lancement. Un thread dédié s'abonne aux
changements de focus et attend l'arrêt ; aucun événement n'est traité (on écoute, c'est
tout). Hors Windows ou si l'abonnement échoue : rien ne se passe (le drapeau
`--force-renderer-accessibility` de scripts/discord_accessibility.ps1 reste le secours).
"""

from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any, ClassVar

from loguru import logger

from mikata.uia.state import _com_init, _com_uninit


class UiaAccessibilityClient:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or sys.platform != "win32":
            return
        self._thread = threading.Thread(target=self._run, name="mikata-uia-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        _com_init()
        try:
            client, handler = self._subscribe()
            if client is None:
                return
            logger.info("Client d'accessibilité UIA actif : les applications se laissent lire.")
            self._stop.wait()
            with contextlib.suppress(Exception):
                client.RemoveFocusChangedEventHandler(handler)
        finally:
            _com_uninit()

    @staticmethod
    def _subscribe() -> tuple[Any, Any] | None:
        """Client IUIAutomation + abonnement aux changements de focus, ou (None, None)."""
        try:
            import comtypes
            from comtypes.gen import UIAutomationClient as uia

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
            logger.warning("Client d'accessibilité UIA indisponible : {}", exc)
            return None, None
