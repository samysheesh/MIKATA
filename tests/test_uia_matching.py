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
