"""Retours du 14/10 (phase 14g) : trouver la fenêtre Discord par son nom, et donner les noms
proches quand un contrôle n'existe pas (le modèle devinait « Muet », « Mute »...)."""

from __future__ import annotations

from typing import Any

import pytest

from mikata.tools.base import ToolError, UiWindowInfo, WindowInfo
from mikata.uia.controller import UiController


class FakeDesktop:
    def windows(self) -> list[WindowInfo]:
        return [
            WindowInfo(1, "Reprise du projet critique", "WindowsTerminal"),
            WindowInfo(2, "#general | samania - Discord", "Discord"),
        ]

    def foreground_window(self) -> WindowInfo | None:
        return self.windows()[0]


class FakeBackend:
    NAMES = ("Rendre muet", "Mettre en sourdine", "Paramètres utilisateur", "#general")

    def attach(self, window: UiWindowInfo) -> Any:
        return window

    def walk(self, root: Any, max_nodes: int) -> list[str]:
        return list(self.NAMES)

    def node_name(self, node: str) -> str:
        return node

    def node_kind(self, node: str) -> str:
        return "bouton"


def _controller() -> UiController:
    controller = object.__new__(UiController)  # sans thread d'état ni client UIA réel
    controller._backend = FakeBackend()  # type: ignore[assignment]
    controller._desktop = FakeDesktop()  # type: ignore[assignment]
    return controller


def test_a_window_is_found_by_a_word_of_its_title_or_its_process() -> None:
    controller = _controller()
    assert controller.window("Discord", None).title == "#general | samania - Discord"
    assert controller.window("discord", None).handle == 2
    assert controller.window(None, "discord").handle == 2
    with pytest.raises(ToolError, match="aucune fenêtre ne ressemble à « Photoshop »"):
        controller.window("Photoshop", None)


def test_a_missing_control_lists_the_closest_names() -> None:
    controller = _controller()
    window = controller.window("Discord", None)
    with pytest.raises(ToolError) as error:
        controller._find(window, "Muet", None, 1)
    assert "contrôle introuvable : « Muet »" in str(error.value)
    assert "noms proches : « Rendre muet »" in str(error.value)
    assert controller._find(window, "Rendre muet", None, 1) == "Rendre muet"


def test_both_backends_implement_the_whole_interface() -> None:
    """Toute opération appelée par le contrôleur est déclarée par la base et fournie par
    chaque moteur : un moteur incomplet ne se construit pas."""
    from mikata.uia.controller import AxBackend, UiaBackend

    for backend in (UiaBackend, AxBackend):
        assert not backend.__abstractmethods__, backend.__abstractmethods__
        backend(FakeDesktop())  # type: ignore[arg-type]


def test_an_ax_click_that_fails_is_reported_as_a_failure() -> None:
    """Le moteur macOS disait « cliqué » quand AXPress échouait mais que l'attribut
    existait : lire l'attribut ne clique pas."""
    from mikata.uia.controller import AxBackend

    class Node:
        def perform(self, action: str) -> None:
            raise RuntimeError("AXError -25205")

        def get(self, attribute: str) -> Any:
            return object()  # l'attribut existe

    with pytest.raises(ToolError, match="n'a pas répondu au clic"):
        AxBackend(FakeDesktop()).click(Node(), "Envoyer")  # type: ignore[arg-type]


def test_a_node_that_vanishes_during_the_read_is_skipped() -> None:
    """Juste après un clic, Discord redessine sa fenêtre : un nœud déjà parcouru n'existe
    plus et lire son nom lève une erreur COM (0x80040201, relevée le 14/10). Il est sauté :
    la lecture de la fenêtre continue au lieu d'échouer."""
    from mikata.uia.controller import UiaBackend, _discord_voice_from

    class Gone:
        @property
        def Name(self) -> str:
            raise OSError(-2147220991, "Un événement n'a pu invoquer aucun des abonnés.")

        ControlTypeName = Name

        def GetChildren(self) -> list[Any]:
            return []

    class Node:
        def __init__(self, name: str, kind: str, children: list[Any]) -> None:
            self.Name, self.ControlTypeName, self._children = name, kind, children

        def GetChildren(self) -> list[Any]:
            return self._children

    root = Node("Discord", "WindowControl", [Gone(), Node("Rendre muet", "TextControl", [])])
    backend = UiaBackend(FakeDesktop())  # type: ignore[arg-type]
    assert backend.node_name(Gone()) == "" and backend.node_kind(Gone()) == "contrôle"
    assert backend.visible_text(root) == "Rendre muet"
    assert _discord_voice_from([Gone()]) == (None, [])


def test_a_name_with_a_prefix_is_another_control_not_a_typo() -> None:
    """Phase 17e : « Mute » cliquait « Unmute » (approché à 80) et rallumait le son."""

    class Player(FakeBackend):
        NAMES = ("Unmute", "Unfollow", "Rechercher")

    controller = _controller()
    controller._backend = Player()  # type: ignore[assignment]
    window = controller.window("Discord", None)
    for wanted in ("Mute", "Follow"):
        with pytest.raises(ToolError, match="contrôle introuvable"):
            controller._find(window, wanted, None, 1)
    assert controller._find(window, "Recherche", None, 1) == "Rechercher"  # faute tolérée


def test_a_gesture_right_after_a_read_reuses_that_read() -> None:
    """Phase 17f : relire Discord prend 0,4 à 0,7 s (15/10), et une compétence clique juste
    après avoir lu la fenêtre. La lecture sert au geste qui suit, 2 s au plus ; un contrôle
    qui n'y est pas encore (la fenêtre a changé) fait relire ; après un clic, on relit."""

    class Counting(FakeBackend):
        def __init__(self) -> None:
            self.walks = 0
            self.names = list(FakeBackend.NAMES)

        def walk(self, root: Any, max_nodes: int) -> list[str]:
            self.walks += 1
            return list(self.names)

        def toggle_state(self, node: str) -> bool | None:
            return None

        def click(self, node: str, name: str) -> str:
            return f"« {name} » cliqué."

    backend = Counting()
    controller = _controller()
    controller._backend = backend  # type: ignore[assignment]
    window = UiWindowInfo("Discord", "Discord")
    controller.controls(window, 200)
    assert controller._find(window, "Rendre muet", None, 1) == "Rendre muet"
    assert backend.walks == 1  # la lecture a servi
    backend.names.append("Quitter le salon")  # apparu après la lecture
    assert controller._find(window, "Quitter le salon", None, 1) == "Quitter le salon"
    assert backend.walks == 2
    assert controller.click(window, "Rendre muet", None, 1) == "« Rendre muet » cliqué."
    assert backend.walks == 2
    controller._find(window, "Rendre muet", None, 1)
    assert backend.walks == 3  # après un clic, la fenêtre est relue
    controller.controls(window, 200)
    controller._find(UiWindowInfo("Spotify", "Spotify"), "Rendre muet", None, 1)
    assert backend.walks == 5  # une autre fenêtre
    assert controller._walk is not None
    controller._walk.at -= 3  # lue il y a plus de 2 s
    controller._find(window, "Rendre muet", None, 1)
    assert backend.walks == 6
