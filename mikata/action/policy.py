"""Politiques d'action (docs/AGENT_CORE.md §3.5) : autoriser, faire confirmer ou refuser.

La décision est prise **par le code**, au moment d'appeler un outil, d'après trois choses :

- l'**effet** que l'outil a déclaré (`lire`, `agir`, `conséquent`, `destructif`) ;
- la **provenance des arguments** de l'appel, suivie par l'interpréteur ;
- la **provenance du tour** : le modèle qui décide a-t-il lu du contenu non fiable ?

Principe (Beurer-Kellner et al., 2025) : une fois du contenu non fiable lu, il doit être
impossible que ce contenu déclenche une action conséquente. Le garde-fou historique
(`mikata.agent.guard` : liste noire, dossiers autorisés, corbeille) reste derrière, comme
filet : il s'applique dans chaque outil, quoi que décide cette politique.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mikata.action.effects import Effect, Trust, join
from mikata.action.sdk import SdkEntry


class Verdict(StrEnum):
    ALLOW = "autorisé"
    CONFIRM = "à confirmer"
    DENY = "refusé"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""


ALLOW = Decision(Verdict.ALLOW)
# Outils dont l'effet persiste au-delà de la commande : un contenu lu ne les alimente jamais.
PERSISTENT_TOOLS = frozenset({"remember", "remember_skill"})
# Ce que l'utilisateur entend quand MIKATA demande confirmation.
PLAIN_ACTION = {
    "files": "modifier un fichier",
    "keyboard": "taper au clavier",
    "mouse": "utiliser la souris",
    "shell": "lancer une commande",
    "memory": "modifier ce que je retiens",
    "skills": "modifier mes savoir-faire",
    "system": "éteindre ou redémarrer le PC",
    "browser": "écrire dans une page web",
    "ui": "écrire dans une application",
}


def decide(entry: SdkEntry, arg_trust: Trust, turn_trust: Trust, *, full_access: bool) -> Decision:
    """Verdict pour un appel. `full_access` (réglage `guard.full_access`) supprime les
    confirmations, jamais les refus."""
    if entry.effect is Effect.DESTRUCTIVE:
        return Decision(Verdict.DENY, "action destructive, toujours interdite")
    if entry.effect in (Effect.READ, Effect.ACT):
        return ALLOW
    if arg_trust is Trust.UNTRUSTED:
        if entry.namespace == "shell":
            return Decision(
                Verdict.DENY, "commande construite à partir d'un contenu lu : jamais lancée"
            )
        if entry.tool in PERSISTENT_TOOLS:
            return Decision(Verdict.DENY, "un contenu lu ne devient pas un souvenir")
    if join(arg_trust, turn_trust) is Trust.UNTRUSTED and not full_access:
        return Decision(Verdict.CONFIRM, "cette action dépend d'un contenu lu ailleurs")
    return ALLOW


def confirm_question(entry: SdkEntry) -> str:
    """Question posée à voix haute avant une action à confirmer."""
    action = PLAIN_ACTION.get(entry.namespace, "faire une action sensible")
    return f"Je m'apprête à {action}, d'après un contenu que je viens de lire. Tu confirmes ?"
