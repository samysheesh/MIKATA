"""Rendez-vous de tâches concurrentes, pour tester le parallélisme sans mesurer le temps
(ajout phase 9).

Chaque appel concurrent attend que tous les autres aient commencé. Si le code les traite en
parallèle, le rendez-vous s'ouvre aussitôt et le test ne dépend d'aucune horloge ni de
l'ordre d'exécution. S'il les traite l'un après l'autre, le premier attend le second qui
ne viendra qu'après lui : au bout de `timeout_s` (jamais atteint quand le code est
correct), le rendez-vous est marqué raté et le test échoue avec un message clair.
"""

from __future__ import annotations

import asyncio


class Rendezvous:
    def __init__(self, parties: int = 2, timeout_s: float = 5.0) -> None:
        self.parties = parties
        self.timeout_s = timeout_s
        self.arrived: list[str] = []  # dans l'ordre d'arrivée
        self.missed: list[str] = []  # appels qui ont attendu les autres en vain
        self._all_here = asyncio.Event()

    async def meet(self, label: str) -> None:
        self.arrived.append(label)
        if len(self.arrived) >= self.parties:
            self._all_here.set()
        try:
            await asyncio.wait_for(self._all_here.wait(), self.timeout_s)
        except TimeoutError:
            self.missed.append(label)

    def check(self) -> None:
        assert not self.missed, (
            f"{self.missed} a attendu les autres appels en vain : ils sont traités l'un après "
            f"l'autre au lieu d'en parallèle (arrivés : {self.arrived})"
        )
        assert len(self.arrived) == self.parties, self.arrived
