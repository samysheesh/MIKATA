"""Notifications Windows (§8) : rappels et minuteurs (ajout phase 4).

`win11toast` est importé paresseusement (Windows uniquement) ; une notification ratée est
journalisée, jamais bloquante.
"""

from __future__ import annotations

from loguru import logger


class ToastNotifier:
    """Implémente `tools.base.Notifier` ; à appeler hors de la boucle asyncio (bloquant)."""

    def notify(self, title: str, body: str) -> None:
        try:
            from win11toast import notify

            notify(title, body)
        except Exception as exc:
            logger.warning("Notification Windows impossible : {}", exc)
