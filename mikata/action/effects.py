"""Espace d'action (docs/AGENT_CORE.md §3.3, §3.5) : effet d'un outil et confiance d'une valeur.

Deux notions déclarées par chaque outil et lues par les politiques (`mikata.action.policy`) :

- son **effet** sur le PC, du plus anodin au plus grave ;
- la **confiance** de ce qu'il renvoie : une page web ou le contenu d'un fichier peuvent
  contenir un texte écrit par quelqu'un d'autre que l'utilisateur.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Effect(StrEnum):
    READ = "lire"  # ne change rien
    ACT = "agir"  # change l'état du PC de façon locale et réversible
    CONSEQUENTIAL = "conséquent"  # sortant, persistant ou difficile à défaire
    DESTRUCTIVE = "destructif"  # interdit, toujours


class Trust(IntEnum):
    """Niveau de confiance d'une valeur ; combiner deux valeurs garde le moins fiable."""

    TRUSTED = 0  # dit par l'utilisateur, écrit dans le script, config
    LOCAL = 1  # état du système : fenêtres, volume, noms de fichiers
    UNTRUSTED = 2  # contenu lu : web, fichiers, presse-papiers, texte d'une appli

    def join(self, other: Trust) -> Trust:
        return self if self >= other else other

    @property
    def label(self) -> str:
        return {0: "fiable", 1: "local", 2: "non fiable"}[int(self)]


def join(*levels: Trust) -> Trust:
    """Le moins fiable des niveaux donnés (TRUSTED s'il n'y en a aucun)."""
    worst = Trust.TRUSTED
    for level in levels:
        worst = worst.join(level)
    return worst
