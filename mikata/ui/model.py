"""État affiché par la mascotte (ajout phase 8), sans Qt : testable avec des événements
simulés.

`MascotModel` reçoit les messages de MIKATA (`ui.protocol`) et répond à deux questions :
quel état dessiner maintenant (`visual`) et que mettre dans la bulle (`bubble`).

États dessinés (`VISUAL_STATES`) :
- repos `idle` (il dort), `listening`, `thinking` (transcription et réflexion), `acting`
  (outils ; un thème peut le dessiner comme `thinking`), `speaking`, `follow_up` (fenêtre de
  suivi, avec le temps restant), `mic_off` (micro en pause, occupé, ou MIKATA absent) ;
- passagers, joués une fois par-dessus : `wake` (mot d'activation), `success` (« C'est
  fait »), `error`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

VISUAL_STATES = (
    "idle",
    "wake",
    "listening",
    "thinking",
    "acting",
    "speaking",
    "success",
    "error",
    "follow_up",
    "mic_off",
)
# Libellés français (menus, réglages, captures).
STATE_LABELS = {
    "idle": "repos (il dort)",
    "wake": "mot d'activation",
    "listening": "écoute",
    "thinking": "réflexion",
    "acting": "action",
    "speaking": "parole",
    "success": "c'est fait",
    "error": "erreur",
    "follow_up": "fenêtre de suivi",
    "mic_off": "micro coupé",
}
TRANSIENT_S = {"wake": 0.9, "success": 1.6, "error": 2.4}
TRANSIENT_PRIORITY = ("error", "success", "wake")
# État de MIKATA (app.State) -> état dessiné.
FROM_MIKATA = {
    "idle": "idle",
    "listening": "listening",
    "transcribing": "thinking",
    "thinking": "thinking",
    "acting": "acting",
    "speaking": "speaking",
    "follow_up": "follow_up",
}


@dataclass(frozen=True)
class Bubble:
    heard: str  # ce que MIKATA a compris
    said: str  # ce qu'il dit
    note: str  # message d'état (MIKATA absent, erreur...)


class MascotModel:
    def __init__(self, bubble_s: float = 8.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.bubble_s = bubble_s
        self._clock = clock
        self.connected = False
        self.mikata_state = "idle"
        self.mic = "ok"
        self.follow_up_s = 6.0
        self._follow_up_at: float | None = None
        self._transient: dict[str, float] = {}  # état passager -> fin
        self._exchange: int | None = None
        self._heard = ""
        self._said = ""
        self._note = ""
        self._text_at = float("-inf")
        self._visual = "mic_off"
        self._visual_since = clock()

    # -- messages ---------------------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        now = self._clock()
        if kind == "hello":
            self.connected = True
            self._note = ""
            self.mic = str(message.get("mic", "ok"))
            self._set_mikata_state(str(message.get("state", "idle")), message, now)
        elif kind == "state":
            self._set_mikata_state(str(message.get("state", "idle")), message, now)
        elif kind == "wake":
            self._start("wake", now)
            self._new_exchange(None, now)
        elif kind == "heard":
            self._new_exchange(message.get("id"), now)
            self._heard = str(message.get("text", ""))
        elif kind == "say":
            if message.get("id") != self._exchange or self._exchange is None:
                self._new_exchange(message.get("id"), now)
            self._said = f"{self._said} {message.get('text', '')}".strip()
            self._text_at = now
        elif kind == "result":
            outcome = message.get("outcome")
            if outcome == "success":
                self._start("success", now)
            elif outcome == "error":
                self._start("error", now)
            elif outcome == "ignored":  # phrase pas pour MIKATA : rien à afficher
                self._heard = self._said = ""
        elif kind == "error":
            self._start("error", now)
            self._note = f"Erreur : {message.get('message', '')}".strip()
            self._text_at = now
        elif kind == "mic":
            self.mic = str(message.get("status", "ok"))
        elif kind == "bye":
            self.disconnected()

    def disconnected(self) -> None:
        self.connected = False
        self.mikata_state = "idle"
        self._transient.clear()
        self._note = "MIKATA n'est pas lancé."

    def note(self, text: str) -> None:
        """Message local de l'interface, affiché dans la bulle."""
        self._note = text
        self._text_at = self._clock()

    # -- lecture ----------------------------------------------------------------------------

    def visual(self) -> str:
        now = self._clock()
        state = self._compute(now)
        if state != self._visual:
            self._visual, self._visual_since = state, now
        return state

    def visual_age(self) -> float:
        """Secondes depuis le dernier changement d'état dessiné (animations « pop »)."""
        self.visual()
        return self._clock() - self._visual_since

    def follow_up_left(self) -> float | None:
        """Part restante de la fenêtre de suivi (1 -> 0), None hors fenêtre de suivi."""
        if self.mikata_state != "follow_up" or self._follow_up_at is None:
            return None
        elapsed = self._clock() - self._follow_up_at
        return max(0.0, 1.0 - elapsed / max(self.follow_up_s, 0.1))

    def bubble(self) -> Bubble | None:
        """Contenu de la bulle, ou None si elle doit être cachée."""
        now = self._clock()
        busy = self.connected and self.mikata_state != "idle"
        recent = now - self._text_at < self.bubble_s
        if not self.connected and self._note:
            return Bubble("", "", self._note)
        if (self._heard or self._said or self._note) and (busy or recent):
            return Bubble(self._heard, self._said, self._note)
        return None

    # -- interne ----------------------------------------------------------------------------

    def _set_mikata_state(self, state: str, message: dict[str, Any], now: float) -> None:
        if state == "follow_up" and self.mikata_state != "follow_up":
            self._follow_up_at = now
            self.follow_up_s = float(message.get("follow_up_s", self.follow_up_s) or 6.0)
        self.mikata_state = state if state in FROM_MIKATA else "idle"

    def _new_exchange(self, exchange: Any, now: float) -> None:
        self._exchange = exchange
        self._heard = self._said = self._note = ""
        self._text_at = now

    def _start(self, transient: str, now: float) -> None:
        self._transient[transient] = now + TRANSIENT_S[transient]

    def _compute(self, now: float) -> str:
        if not self.connected:
            return "mic_off"
        for name in TRANSIENT_PRIORITY:
            if self._transient.get(name, float("-inf")) > now:
                return name
        if self.mic != "ok" and self.mikata_state == "idle":
            return "mic_off"
        return FROM_MIKATA.get(self.mikata_state, "idle")


def demo_messages(state: str) -> list[dict[str, Any]]:
    """Messages de MIKATA qui amènent la mascotte dans l'état dessiné `state` (aperçu d'un
    skin, captures, tests) ; ils repartent du repos, micro actif."""
    start: list[dict[str, Any]] = [
        {"type": "mic", "status": "ok"},
        {"type": "state", "state": "idle"},
    ]
    special: dict[str, dict[str, Any]] = {
        "wake": {"type": "wake", "score": 0.9},
        "success": {"type": "result", "outcome": "success"},
        "error": {"type": "error", "message": "exemple d'erreur"},
        "mic_off": {"type": "mic", "status": "paused"},
    }
    return [*start, special.get(state, {"type": "state", "state": state})]
