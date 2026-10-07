"""Règles de déclenchement du mot d'activation (§4), indépendantes du détecteur.

Score ≥ seuil sur N trames consécutives, puis pause (réfractaire) après chaque détection.
"""

from __future__ import annotations


class WakeTrigger:
    def __init__(self, threshold: float, consecutive_frames: int, refractory_s: float) -> None:
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.refractory_s = refractory_s
        self._streak = 0
        self._quiet_until = float("-inf")

    def update(self, score: float, t: float) -> bool:
        """Score de la trame capturée à l'instant `t` (monotone). Vrai = détection."""
        if t < self._quiet_until:
            self._streak = 0
            return False
        self._streak = self._streak + 1 if score >= self.threshold else 0
        if self._streak >= self.consecutive_frames:
            self.suppress(t)
            return True
        return False

    def suppress(self, t: float) -> None:
        """Ouvre une période réfractaire (après une détection ou le raccourci clavier)."""
        self._streak = 0
        self._quiet_until = t + self.refractory_s
