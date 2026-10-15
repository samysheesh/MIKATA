"""Phase 17f : l'effet d'une action dans une appli, constaté dans sa fenêtre.

Le 15/10, MIKATA a dit « c'est fait » sans rien regarder : « Play » cliqué sur 18 résultats
de suite, « tu es déjà connecté » après une déconnexion. Réfléchir plus n'y change rien ;
il faut regarder ce que l'action a fait. Pour toutes les applis, sans rien leur coder :

- avant un clic ou une saisie, le contrôleur garde ce que la fenêtre montrait : les noms
  et types de ses contrôles, déjà lus pour trouver la cible (rien de plus à lire) ;
- un clic qui se voit sur-le-champ (un interrupteur basculé, un bouton qui change de nom,
  « Play » devenu « Pause ») est confirmé tout de suite ;
- sinon, à la fin du tour, **pendant que MIKATA parle**, la boucle agent relit la fenêtre
  et compare, quelques secondes au plus (`uia.verify_timeout_s`). Rien n'a changé (aucun
  contrôle apparu ni disparu, ni titre changé, ni fenêtre ouverte) : l'action n'a pas pris,
  MIKATA le dit et le modèle reprend.

Les nombres ne comptent pas : un minuteur (« durée de l'appel 41 minutes »), un ping
(« 3 ms ») ou une position de lecture changent tout seuls.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from mikata.tools.base import UiWindowInfo

Snapshot = Counter[tuple[str, str]]  # (nom sans nombres, type) -> combien
_DIGITS = re.compile(r"\d+")
SHOWN_CHANGES = 4  # contrôles apparus ou disparus cités dans un constat


def mask(text: str) -> str:
    """Le nom sans ses nombres ni ses blancs en trop."""
    return " ".join(_DIGITS.sub("#", text).split())


@dataclass
class PendingEffect:
    """Une action dont l'effet reste à constater (ou déjà constaté : `confirmed`)."""

    window: UiWindowInfo
    action: str  # « clic sur « Déconnecter » »
    before: Snapshot  # la fenêtre juste avant
    windows_before: frozenset[int]  # fenêtres ouvertes juste avant (poignées)
    at: float  # time.monotonic() de l'action
    confirmed: bool = False  # l'effet s'est vu tout de suite
    partial: bool = False  # `before` ne couvre qu'une partie de la fenêtre


@dataclass
class EffectReport:
    """Ce que les actions d'une fenêtre ont changé, relu après coup."""

    window: UiWindowInfo
    actions: list[str]
    appeared: list[str] = field(default_factory=list)
    disappeared: list[str] = field(default_factory=list)
    new_windows: list[str] = field(default_factory=list)
    title: str = ""  # nouveau titre de la fenêtre, s'il a changé
    gone: bool = False  # la fenêtre s'est fermée

    @property
    def changed(self) -> bool:
        return bool(self.appeared or self.disappeared or self.new_windows or self.title) or (
            self.gone
        )

    def summary(self) -> str:
        """Le constat en une phrase, pour le modèle et la trace."""
        what = " ; ".join(self.actions)
        if not self.changed:
            return f"après {what} dans « {self.window.title} », rien n'a changé"
        parts: list[str] = []
        if self.gone:
            parts.append("la fenêtre s'est fermée")
        if self.title:
            parts.append(f"titre devenu « {self.title} »")
        if self.appeared:
            parts.append("apparu : " + ", ".join(f"« {n} »" for n in self.appeared))
        if self.disappeared:
            parts.append("disparu : " + ", ".join(f"« {n} »" for n in self.disappeared))
        if self.new_windows:
            parts.append("fenêtre ouverte : " + ", ".join(f"« {t} »" for t in self.new_windows))
        return f"après {what} dans « {self.window.title} » : " + " ; ".join(parts)


def compare(
    pending: list[PendingEffect],
    after: Snapshot | None,
    title_after: str | None,
    windows_after: dict[int, str],
) -> EffectReport:
    """Le constat pour une fenêtre : `pending` (ses actions, dans l'ordre), `after` (sa
    relecture ; None si elle s'est fermée), son titre et les fenêtres ouvertes maintenant.
    On compare à la fenêtre d'avant la **première** action : un menu ouvert puis refermé
    par l'action suivante ne compte pas, seul le résultat compte."""
    first = pending[0]
    report = EffectReport(first.window, [p.action for p in pending])
    if after is None:
        report.gone = True
        return report
    before = first.before
    appeared = after - before
    disappeared = before - after
    if not first.partial:
        report.appeared = _names(appeared)
    report.disappeared = _names(disappeared)
    if title_after is not None and mask(title_after) != mask(first.window.title):
        report.title = title_after
    report.new_windows = [
        title
        for handle, title in windows_after.items()
        if title and handle not in first.windows_before
    ][:SHOWN_CHANGES]
    return report


def _names(changes: Snapshot) -> list[str]:
    names = [name for (name, _kind), _count in changes.most_common() if name]
    return names[:SHOWN_CHANGES]
