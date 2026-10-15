"""Phase 17f : l'effet d'une action dans une appli, constaté dans sa fenêtre pendant que
MIKATA parle (mikata/uia/effects.py). Le 15/10, elle disait « c'est fait » sans regarder :
18 clics sur « Play », « tu es déjà connecté » après une déconnexion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from test_agent_learning import Learner
from test_uia_matching import FakeDesktop

from mikata.llm.base import ToolCallReq
from mikata.tools.base import UiWindowInfo
from mikata.uia.controller import UiController
from mikata.uia.effects import PendingEffect, compare, mask

DISCORD = UiWindowInfo("#general | samania - Discord", "Discord")
SHORT = {"verify_timeout_s": 0.2}

# -- le constat --------------------------------------------------------------------------


def _pending(names: list[str], action: str = "clic sur « Rejoindre »") -> PendingEffect:
    before = Counter((mask(n), "bouton") for n in names)
    return PendingEffect(DISCORD, action, before, frozenset({1}), 0.0)


def test_numbers_do_not_count_as_a_change() -> None:
    """Un minuteur, un ping, une position de lecture changent tout seuls."""
    before = ["Champions league (salon vocal), cupi, durée de l'appel 41 minutes", "3 ms"]
    after = Counter(
        (mask(n), "bouton")
        for n in ["Champions league (salon vocal), cupi, durée de l'appel 42 minutes", "12 ms"]
    )
    report = compare([_pending(before)], after, None, {1: "Discord"})
    assert not report.changed
    assert report.summary() == (
        "après clic sur « Rejoindre » dans « #general | samania - Discord », rien n'a changé"
    )


def test_what_appeared_disappeared_or_opened_is_told() -> None:
    after = Counter([(mask("Voix connectée"), "texte"), (mask("Déconnecter"), "bouton")])
    report = compare([_pending(["Rejoindre"])], after, None, {1: "Discord", 7: "Paramètres"})
    assert report.changed
    assert report.appeared == ["Voix connectée", "Déconnecter"]
    assert report.disappeared == ["Rejoindre"] and report.new_windows == ["Paramètres"]
    assert "apparu : « Voix connectée », « Déconnecter »" in report.summary()
    assert compare([_pending(["A"])], None, None, {}).changed  # fermée : ça a bougé


def test_only_the_result_of_several_actions_counts() -> None:
    """Un menu ouvert puis refermé par le clic suivant : on compare à avant la première."""
    first, second = _pending(["Menu"], "clic sur « Menu »"), _pending(["Menu", "Copier"])
    report = compare([first, second], Counter([(mask("Menu"), "bouton")]), None, {1: "x"})
    assert not report.changed and report.actions == ["clic sur « Menu »", "clic sur « Rejoindre »"]


# -- le contrôleur : ce que montrait la fenêtre avant, ce qu'elle montre après -------------


class Live:
    """Un arbre qui change quand on clique : `effects[nom] = (ajoutés, retirés)`."""

    def __init__(self, names: list[str], effects: dict[str, tuple[list[str], list[str]]]) -> None:
        self.names, self.effects = list(names), effects
        self.states: dict[str, bool] = {}

    def attach(self, window: UiWindowInfo) -> Any:
        return window

    def walk(self, root: Any, max_nodes: int) -> list[str]:
        return list(self.names)

    def node_name(self, node: str) -> str:
        return node if node in self.names else ""

    def node_kind(self, node: str) -> str:
        return "bouton"

    def toggle_state(self, node: str) -> bool | None:
        return self.states.get(node)

    def click(self, node: str, name: str) -> str:
        added, removed = self.effects.get(name, ([], []))
        self.names = [n for n in self.names if n not in removed] + added
        if node in self.states:
            self.states[node] = not self.states[node]
        return f"« {name} » cliqué."


def _controller(backend: Live) -> UiController:
    controller = object.__new__(UiController)  # sans thread d'état ni client UIA réel
    controller._backend = backend  # type: ignore[assignment]
    controller._desktop = FakeDesktop()  # type: ignore[assignment]
    return controller


def test_a_click_that_changes_nothing_is_seen_as_such() -> None:
    backend = Live(
        ["Champions league (salon vocal), cupi", "Paramètres", "Rendre muet"],
        {"Paramètres": (["Mon compte"], [])},
    )
    backend.states["Rendre muet"] = False
    ui = _controller(backend)
    window = UiWindowInfo("Discord", "Discord")
    ui.click(window, "Champions league (salon vocal)", None, 1)  # Discord l'ignore
    ui.click(window, "Rendre muet", None, 1)  # l'interrupteur bascule : vu sur-le-champ
    first, toggled = ui.take_pending()
    assert not first.confirmed and toggled.confirmed and ui.take_pending() == []
    (report,) = ui.effect_of([first])
    assert not report.changed
    ui.click(window, "Paramètres", None, 1)
    (opened,) = ui.effect_of(ui.take_pending())
    assert opened.appeared == ["Mon compte"]


# -- la boucle agent : vérifier pendant que MIKATA parle --------------------------------

CLICK = [ToolCallReq("a", "uia_click", {"control": "Rendre muet", "window": "Discord"})]


def test_an_action_without_visible_effect_is_said_then_redone(tmp_path: Path) -> None:
    learner = Learner(tmp_path, CLICK, CLICK, uia=SHORT)
    learner.ui.inert = {"Rendre muet": 1}  # le premier clic ne prend pas
    result = learner.respond("coupe mon micro sur Discord")
    assert result.answer == "C'est fait. Attends, ça n'a pas pris. Voilà, c'est bon."
    note = learner.system(1)[-1]
    assert "rien n'a changé" in note and "clic sur « Rendre muet »" in note
    assert len([v for n, v in learner.ui.log if n == "click"]) == 2
    assert result.no_effect == [] and any("rien n'a changé" in e for e in result.effects)


def test_still_nothing_after_the_retry_is_said_and_kept(tmp_path: Path) -> None:
    learner = Learner(tmp_path, CLICK, CLICK, uia=SHORT)
    learner.ui.inert = {"Rendre muet": 2}
    result = learner.respond("coupe mon micro sur Discord")
    assert result.answer.endswith("Je ne vois toujours aucun changement, regarde de ton côté.")
    assert result.no_effect and "rien n'a changé" in result.no_effect[0]
    (trace,) = learner.traces.recent(5)
    assert any(failure.startswith("sans effet") for failure in trace.failures)


def test_a_visible_effect_or_the_setting_off_changes_nothing(tmp_path: Path) -> None:
    learner = Learner(tmp_path, CLICK)
    assert learner.respond("coupe mon micro sur Discord").answer == "C'est fait."
    (tmp_path / "off").mkdir()
    off = Learner(tmp_path / "off", CLICK, uia={"verify_effects": False})
    off.ui.inert = {"Rendre muet": 1}
    assert off.respond("coupe mon micro sur Discord").answer == "C'est fait."
    assert not any(name == "effect_of" for name, _ in off.ui.log)
