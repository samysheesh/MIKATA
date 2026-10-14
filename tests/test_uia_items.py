"""Phases 16g-17a : gestes génériques d'appli (élément de liste, clic dans un élément,
attente, liens d'applis).

L'arbre de test reproduit celui relevé sur le vrai Spotify le 14/10 (appli en anglais) : la
ligne en tête des résultats a un bouton « Play » tout court, les suivantes « Play <titre> » ;
des cartes « Play Jet Lag Tiakola » (playlists, paroles), la carte d'un autre titre et le
lecteur ont aussi un « Play ». Rien n'y est propre à Spotify : ce sont des listes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

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


# -- éléments de liste, attente, clic dans un élément (phase 17a) ---------------------------


def test_items_are_the_outer_rows_with_their_texts_and_buttons() -> None:
    root, _ = spotify_tree()
    controller, _ = _controller(root)
    rows = controller.items(WINDOW, "tiakola", 10)
    assert [r.name for r in rows] == ["Jet Lag"]  # filtré par les mots, sans ses cellules
    assert rows[0].texts == ("Song • Tiakola, Jorja Smith",) and rows[0].buttons == ("Play",)
    assert [r.name for r in controller.items(WINDOW, None, 2)] == [
        "Jet Lag",
        "Jet lag (Evasion #2)",
    ]


def test_find_text_stops_at_the_first_match() -> None:
    root, _ = spotify_tree()
    controller, _ = _controller(root)
    assert controller.find_text(WINDOW, "evasion") == "Jet lag (Evasion #2)"
    assert controller.find_text(WINDOW, "Damso") is None


def _run_tool(tmp_path: Path, name: str, args: dict[str, Any], **ctx: Any) -> Any:
    registry = ToolRegistry(make_ctx(tmp_path, **ctx))
    return asyncio.run(registry.execute(ToolCallReq("1", name, args)))


def test_uia_items_lists_rows_for_the_model_and_scripts(tmp_path: Path) -> None:
    ui = FakeUi(
        controls=[
            UiControlInfo("Jet Lag", "élément", "Song • Tiakola, Jorja Smith"),
            UiControlInfo("Play", "bouton"),
        ]
    )
    result = _run_tool(tmp_path, "uia_items", {"process": "Spotify", "query": "jet"}, uia=ui)
    assert result.ok and "- « Jet Lag » — Song • Tiakola, Jorja Smith" in result.content
    assert result.data == {
        "window": "Discord",
        "rows": [{"name": "Jet Lag", "texts": ["Song • Tiakola, Jorja Smith"], "buttons": []}],
    }


def test_uia_wait_returns_once_the_text_shows(tmp_path: Path) -> None:
    class Slow(FakeUi):
        looks = 0

        def find_text(self, window: UiWindowInfo, text: str) -> str | None:
            Slow.looks += 1
            return "What do you want to play? : Jet+Lag" if Slow.looks >= 3 else None

    ok = _run_tool(tmp_path, "uia_wait", {"process": "Spotify", "text": "Jet Lag"}, uia=Slow())
    assert ok.ok and Slow.looks == 3 and "Jet+Lag" in ok.content
    late = _run_tool(
        tmp_path,
        "uia_wait",
        {"process": "Spotify", "text": "absent", "timeout_s": 0.5},
        uia=FakeUi(),
    )
    assert not late.ok and "« absent » n'est pas apparu en 0.5 s" in late.content


def test_uia_click_with_item_clicks_inside_that_row(tmp_path: Path) -> None:
    ui = FakeUi()
    args = {"process": "Spotify", "control": "Play", "item": "Jet Lag", "near": "Tiakola"}
    assert _run_tool(tmp_path, "uia_click", args, uia=ui).ok
    assert ("click_in", ("Discord", "Jet Lag", ("Play",), ("Tiakola",))) in ui.log


# -- liens d'applis (phase 17a) -------------------------------------------------------------


def test_open_app_link_encodes_and_opens_in_the_background(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    args = {"link": "spotify:search:Mélo Décalé Tiakola", "background": True}
    result = _run_tool(tmp_path, "open_app_link", args, desktop=desktop)
    assert result.ok and result.content == "Lien ouvert dans Spotify en arrière-plan."
    assert ("open", "spotify:search:M%C3%A9lo%20D%C3%A9cal%C3%A9%20Tiakola") in desktop.log
    assert ("background", "Spotify") in desktop.log  # seule l'appli visée est minimisée
    again = _run_tool(
        tmp_path, "open_app_link", {"link": "spotify:search:Jet%20Lag"}, desktop=desktop
    )
    assert ("open", "spotify:search:Jet%20Lag") in desktop.log and again.ok  # pas d'encodage double


@pytest.mark.parametrize(
    ("link", "reason"),
    [
        ("ms-msdt:/id PCWDiagnostic", "aucune appli installée n'ouvre les liens « ms-msdt: »"),
        ("search-ms:query=x", "aucune appli installée n'ouvre les liens « search-ms: »"),
        ("inconnu:truc", "aucune appli installée n'ouvre les liens « inconnu: »"),
        ("https://youtube.com", "c'est une page web"),
        ("pas un lien", "pas un lien d'appli"),
    ],
)
def test_open_app_link_refuses_what_no_installed_app_declares(
    tmp_path: Path, link: str, reason: str
) -> None:
    desktop = FakeDesktop()
    result = _run_tool(tmp_path, "open_app_link", {"link": link}, desktop=desktop)
    assert not result.ok and reason in result.content
    assert not any(entry[0] == "open" for entry in desktop.log)


def test_app_links_lists_the_apps_and_their_schemes(tmp_path: Path) -> None:
    result = _run_tool(tmp_path, "app_links", {}, desktop=FakeDesktop())
    assert result.ok and "- Spotify : spotify:" in result.content
    assert "- Paramètres Windows : ms-settings:" in result.content


def test_denied_schemes_never_count_as_app_links() -> None:
    from mikata.tools.desktop import _linkable

    for scheme in (
        "ms-msdt",
        "search-ms",
        "explorer.assocprotocol.search-ms",
        "discord-1443349082578882741",
    ):
        assert not _linkable(scheme), scheme
    assert _linkable("spotify") and _linkable("com.epicgames.launcher")


def test_an_item_given_as_shown_by_uia_items_is_understood() -> None:
    """Le modèle recopie la ligne entière ; ce qu'elle affiche départage deux « Général »."""
    from mikata.tools.uia import split_item

    assert split_item("« Général » — Salon vocal • 2 membres [boutons : Rejoindre]") == (
        "Général",
        "Salon vocal • 2 membres",
    )
    assert split_item("Jet Lag — Song • Tiakola | Explicit") == ("Jet Lag", "Song • Tiakola")
    assert split_item("Jet Lag") == ("Jet Lag", "")


def test_a_link_opened_in_the_background_is_only_a_step(tmp_path: Path) -> None:
    shown = _run_tool(
        tmp_path, "open_app_link", {"link": "ms-settings:bluetooth"}, desktop=FakeDesktop()
    )
    step = _run_tool(
        tmp_path,
        "open_app_link",
        {"link": "spotify:search:x", "background": True},
        desktop=FakeDesktop(),
    )
    assert shown.terminal and not step.terminal


def test_a_background_link_waits_for_the_view_it_asked_for(tmp_path: Path) -> None:
    """Sans attente, la lecture suivante pouvait tomber sur la page d'avant (14/10)."""

    class Loading(FakeUi):
        looks: ClassVar[list[str]] = []

        def find_text(self, window: UiWindowInfo, text: str) -> str | None:
            Loading.looks.append(text)
            return "What do you want to play? : Mélo+Décalé" if len(Loading.looks) >= 3 else None

    args = {"link": "spotify:search:Mélo Décalé", "background": True}
    result = _run_tool(tmp_path, "open_app_link", args, desktop=FakeDesktop(), uia=Loading())
    assert result.ok and Loading.looks == ["Mélo Décalé"] * 3  # le texte du lien, décodé
    shown = _run_tool(
        tmp_path,
        "open_app_link",
        {"link": "ms-settings:bluetooth"},
        desktop=FakeDesktop(),
        uia=Loading(),
    )
    assert shown.ok and len(Loading.looks) == 3  # montré à l'utilisateur : pas d'attente


def test_uia_items_waits_a_little_for_rows_that_are_still_loading(tmp_path: Path) -> None:
    class Filling(FakeUi):
        reads = 0

        def items(self, window: UiWindowInfo, query: str | None, limit: int) -> list[Any]:
            from mikata.tools.base import UiItemInfo

            Filling.reads += 1
            return [UiItemInfo("Jet Lag")] if Filling.reads >= 3 else []

    result = _run_tool(
        tmp_path, "uia_items", {"process": "Spotify", "query": "Jet Lag"}, uia=Filling()
    )
    assert result.ok and Filling.reads == 3


# -- interrupteurs (phase 17d) --------------------------------------------------------------


class _Toggle:
    def __init__(self, state: int) -> None:
        self.ToggleState = state

    def Toggle(self) -> None:
        self.ToggleState = 1 - self.ToggleState


class _MuteButton:
    Name = "Rendre muet"
    ControlTypeName = "ButtonControl"

    def __init__(self) -> None:
        self.toggle = _Toggle(0)

    def GetTogglePattern(self) -> _Toggle:
        return self.toggle

    def GetInvokePattern(self) -> None:
        raise RuntimeError("pas d'Invoke")


def test_a_switch_tells_its_state_so_it_is_not_clicked_again() -> None:
    """Le 14/10 : « Rendre muet » garde son nom une fois activé ; MIKATA relisait le nom,
    croyait le micro ouvert, recliquait, et rendait le micro."""
    from mikata.uia.controller import UiaBackend

    backend = UiaBackend(FakeDesktop())  # type: ignore[arg-type]
    button = _MuteButton()
    assert backend.control_text(button) == "Rendre muet : désactivé"
    assert backend.click(button, "Rendre muet") == "« Rendre muet » basculé : maintenant activé."
    assert backend.control_text(button) == "Rendre muet : activé"


def test_controls_show_the_state_of_switches() -> None:
    class Switches(TreeBackend):
        def toggle_state(self, node: Any) -> bool | None:
            return True if node.name == "Rendre muet" else None

    root = Node("Discord", "fenêtre", [Node("Rendre muet", "bouton"), Node("Paramètres", "bouton")])
    controller = object.__new__(UiController)
    controller._backend, controller._desktop = Switches(root), FakeDesktop()  # type: ignore[assignment]
    infos = {c.name: c.text for c in controller.controls(WINDOW, 10)}
    assert infos == {"Rendre muet": "activé", "Paramètres": ""}
