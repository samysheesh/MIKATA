"""Phase 16g : lancer un titre dans l'appli Spotify du PC, sans navigateur ni vision.

L'arbre de test reproduit celui relevé sur le vrai Spotify le 14/10 (appli en anglais) : la
ligne en tête des résultats a un bouton « Play » tout court, les suivantes « Play <titre> » ;
la carte d'un autre titre et le lecteur ont aussi un « Play »."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tool_fakes import FakeDesktop, FakeUi, make_ctx

from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolError, UiControlInfo, UiWindowInfo
from mikata.tools.registry import ToolRegistry
from mikata.uia.controller import UiController, _BackendBase


class Node:
    def __init__(self, name: str, kind: str, children: list[Node] | None = None) -> None:
        self.name, self.kind, self.children, self.parent = name, kind, children or [], None
        for child in self.children:
            child.parent = self


class TreeBackend(_BackendBase):
    def __init__(self, root: Node) -> None:
        super().__init__(FakeDesktop())  # type: ignore[arg-type]
        self.root, self.clicked = root, []

    def attach(self, window: UiWindowInfo) -> Any:
        return self.root

    def children(self, node: Any) -> list[Any]:
        return list(node.children)

    def parent(self, node: Any) -> Any:
        return node.parent

    def node_name(self, node: Any) -> str:
        return node.name

    def node_kind(self, node: Any) -> str:
        return node.kind

    def control_text(self, node: Any) -> str:
        return node.name

    def document_text(self, root: Any) -> str:
        return ""

    def visible_text(self, root: Any) -> str:
        return ""

    def click(self, node: Any, name: str) -> str:
        self.clicked.append(node)
        return f"« {name} » cliqué."

    def set_text(self, node: Any, control: str, text: str) -> str:
        return ""


def _row(title: str, artists: str, button: str) -> Node:
    return Node(
        title,
        "élément",
        [Node(title, "groupe", [Node(button, "bouton"), Node(f"Song • {artists}", "texte")])],
    )


def spotify_tree() -> tuple[Node, dict[str, Node]]:
    jet_lag = _row("Jet Lag", "Tiakola, Jorja Smith", "Play")
    evasion = _row("Jet lag (Evasion #2)", "Autre Artiste", "Play Jet lag (Evasion #2)")
    other = _row("Jet Lag", "Quelqu'un d'autre", "Play Jet Lag")
    radio = _row("Big Jet Plane (Radio Edit)", "Angus & Julia Stone", "Play")
    # Cartes dont le bouton porte titre et artiste (playlists d'utilisateurs, paroles).
    cards = Node("", "groupe", [Node("", "groupe", [Node("Play Jet Lag Tiakola", "bouton")])])
    overbooking = Node("Overbooking", "groupe", [Node("", "groupe", [Node("Play", "bouton")])])
    footer = Node(
        "Player controls", "groupe", [Node("Play", "bouton"), Node("Tiakola - WpointM", "lien")]
    )
    grid = Node("Search results", "liste", [jet_lag, evasion, other, radio])
    root = Node("Spotify", "fenêtre", [Node("", "groupe", [overbooking, cards, grid]), footer])
    buttons = {
        "jet_lag": jet_lag.children[0].children[0],
        "other": other.children[0].children[0],
        "radio": radio.children[0].children[0],
        "card": cards.children[0].children[0],
    }
    return root, buttons


def _controller(root: Node) -> tuple[UiController, TreeBackend]:
    controller = object.__new__(UiController)
    backend = TreeBackend(root)
    controller._backend, controller._desktop = backend, FakeDesktop()  # type: ignore[assignment]
    return controller, backend


WINDOW = UiWindowInfo("Spotify Free", "Spotify")
PLAY = ["Play", "Lire"]


def test_the_play_button_of_the_row_titled_and_by_the_artist() -> None:
    root, buttons = spotify_tree()
    controller, backend = _controller(root)
    controller.click_in(WINDOW, "Jet Lag", PLAY, ["Tiakola"])
    assert backend.clicked == [buttons["jet_lag"]]  # ni une carte, ni le lecteur, ni l'autre


def test_the_artist_picks_among_rows_of_the_same_title() -> None:
    root, buttons = spotify_tree()
    controller, backend = _controller(root)
    controller.click_in(WINDOW, "jet lag", PLAY, ["Quelqu'un d'autre"])
    assert backend.clicked == [buttons["other"]]  # son bouton « Play Jet Lag »


def test_without_artist_the_first_row_of_that_title(tmp_path: Path) -> None:
    root, buttons = spotify_tree()
    controller, backend = _controller(root)
    controller.click_in(WINDOW, "Jet Lag", PLAY)
    assert backend.clicked == [buttons["jet_lag"]]


def test_a_row_whose_title_only_starts_alike_comes_last() -> None:
    root, buttons = spotify_tree()
    controller, backend = _controller(root)
    controller.click_in(WINDOW, "Big Jet Plane", PLAY, ["Angus"])
    assert backend.clicked == [buttons["radio"]]


def test_an_artist_absent_from_every_row_is_refused() -> None:
    root, _ = spotify_tree()
    controller, backend = _controller(root)
    with pytest.raises(ToolError, match="aucun élément « Jet Lag » qui mentionne « Damso »"):
        controller.click_in(WINDOW, "Jet Lag", PLAY, ["Damso"])
    assert backend.clicked == []


def test_near_ignores_the_name_of_the_control_itself() -> None:
    """Une carte « Play Jet Lag Tiakola » n'est pas « près de Tiakola » à cause de son
    seul nom : le « Play » de la vraie ligne passe avant."""
    root, buttons = spotify_tree()
    controller, backend = _controller(root)
    controller.click(WINDOW, ["Play", "Play Jet Lag Tiakola"], "bouton", 1, ["Jet Lag", "Tiakola"])
    assert backend.clicked == [buttons["jet_lag"]]


class SpotifyUi(FakeUi):
    """Le Spotify simulé. Au 1er coup d'œil, la page de la recherche précédente (qui
    contient déjà le titre : piège de la page périmée) ; au 2e, le champ de recherche porte
    la nouvelle recherche (« Jet+Lag+Tiakola ») ; le lecteur passe à « Pause » une fois un
    titre lancé."""

    def __init__(self, playing_after_click: bool = True, rows: bool = True) -> None:
        super().__init__(windows=[UiWindowInfo("Spotify Free", "Spotify")])
        self.looks, self.started, self.playing_after_click = 0, False, playing_after_click
        self.rows = rows
        self.query = "Jet+Lag+Tiakola"

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        self.looks += 1
        box = "Mélo+Décalé" if self.looks < 2 else self.query
        player = "Pause" if self.started else "Play"
        return [
            UiControlInfo("What do you want to play?", "liste déroulante", box),
            # Sans ligne : le nom n'est qu'un lien (une carte d'artiste, comme « Tiakola »).
            UiControlInfo("Jet Lag", "élément" if self.rows else "lien"),
            UiControlInfo("Play", "bouton"),
            UiControlInfo("Play Jet lag (Evasion #2)", "bouton"),
            UiControlInfo(player, "bouton"),
        ]

    def click_in(
        self, window: Any, item: str, buttons: list[str], contains: list[str] | None = None
    ) -> str:
        self.log.append(("click_in", (item, buttons, contains, self.looks)))
        if not self.rows:
            raise ToolError(f"aucun élément « {item} »")
        self.started = self.playing_after_click
        return "cliqué"

    def click(self, window: Any, control: Any, kind: Any, nth: int, near: Any = None) -> str:
        self.log.append(("click", (control, kind)))
        self.started = self.playing_after_click
        return "cliqué"

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        if control == "Pause" and not self.started:
            raise ToolError("contrôle introuvable : « Pause »")
        return "Pause"


def _play(tmp_path: Path, ui: SpotifyUi, **args: Any) -> tuple[Any, FakeDesktop]:
    desktop = FakeDesktop()
    registry = ToolRegistry(make_ctx(tmp_path, desktop=desktop, uia=ui))
    result = asyncio.run(registry.execute(ToolCallReq("1", "spotify_play", args)))
    return result, desktop


def test_spotify_play_opens_the_app_search_and_plays_the_right_row(tmp_path: Path) -> None:
    ui = SpotifyUi()
    result, desktop = _play(tmp_path, ui, titre="Jet Lag", artiste="Tiakola")
    assert ("open", "spotify:search:Jet%20Lag%20Tiakola") in desktop.log  # l'appli, encodé
    clicks = [entry for entry in ui.log if entry[0] == "click_in"]
    # Pas sur la page périmée du 1er coup d'œil : quand le champ porte la recherche.
    assert clicks == [("click_in", ("Jet Lag", ["Play", "Lire"], ["Tiakola"], 2))]
    assert result.ok and result.content == "« Jet Lag » lancé dans Spotify."


def test_spotify_play_says_when_playback_was_not_seen(tmp_path: Path) -> None:
    ui = SpotifyUi(playing_after_click=False)
    ui.query = "Jet+Lag"
    result, _ = _play(tmp_path, ui, titre="Jet Lag")
    assert result.ok and "sans voir la lecture démarrer" in result.content


def test_spotify_play_reports_a_title_it_cannot_find(tmp_path: Path) -> None:
    class Empty(SpotifyUi):
        def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
            return [UiControlInfo("Search", "champ")]

    result, _ = _play(tmp_path, Empty(), titre="Titre inexistant")
    assert not result.ok and "« Titre inexistant » introuvable dans Spotify" in result.content


def test_an_artist_alone_plays_its_card(tmp_path: Path) -> None:
    """« mets Tiakola » : un artiste est une carte, pas une ligne ; sans artiste précisé, le
    bouton au nom exact « Play Tiakola » est pris."""
    ui = SpotifyUi(rows=False)
    ui.query = "Jet+Lag"
    result, _ = _play(tmp_path, ui, titre="Jet Lag")
    assert ("click", (["Play Jet Lag", "Lire Jet Lag"], "bouton")) in ui.log
    assert result.ok


def test_no_fallback_to_another_artists_version(tmp_path: Path) -> None:
    result, _ = _play(tmp_path, SpotifyUi(rows=False), titre="Jet Lag", artiste="Tiakola")
    assert not result.ok and "« Jet Lag » de Tiakola introuvable" in result.content
