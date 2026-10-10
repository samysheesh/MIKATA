"""Notifications (§8) : rappels et minuteurs (ajout phase 4).

- win32 : `win11toast`, importé paresseusement.
- darwin : `pync` si présent, sinon `osascript -e 'display notification ...'`.
  Jamais `win11toast` sous macOS : c'est un module Windows (déclaré
  `sys_platform == 'win32'` dans pyproject.toml).
Une notification ratée est journalisée, jamais bloquante.
"""

from __future__ import annotations

import subprocess

from loguru import logger

from mikata.plat import platform


class ToastNotifier:
    """Implémente `tools.base.Notifier` ; à appeler hors de la boucle asyncio (bloquant)."""

    def notify(self, title: str, body: str) -> None:
        if platform() == "darwin":
            self._notify_darwin(title, body)
            return
        self._notify_win32(title, body)

    def _notify_win32(self, title: str, body: str) -> None:
        try:
            from win11toast import notify

            notify(title, body)
        except Exception as exc:
            logger.warning("Notification Windows impossible : {}", exc)

    def _notify_darwin(self, title: str, body: str) -> None:
        try:
            import pync  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            try:
                pync.notify(body, title=title)
            except Exception as exc:
                logger.warning("Notification macOS impossible : {}", exc)
            return
        # Repli : osascript natif (jamais win11toast, module Windows).
        self._notify_osascript(title, body)

    def _notify_osascript(self, title: str, body: str) -> None:
        try:
            script = f"display notification {body!r} with title {title!r}"
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except Exception as exc:
            logger.warning("Notification macOS impossible : {}", exc)
