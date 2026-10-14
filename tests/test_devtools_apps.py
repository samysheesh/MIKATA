"""Phase 17e : applis Chromium (Electron, CEF) tenues cachées et pilotées par DevTools.

L'arbre de test reprend la forme de celui relevé sur le vrai Spotify le 14/10 (appli en
anglais, page de résultats après Entrée), sans aucune donnée personnelle : une ligne de
résultat a ses cellules (« Jet Lag », « Explicit Song · Tiakola, Jorja Smith »), un bouton
« Play » et « More options for Jet Lag » ; des nœuds ignorés et des conteneurs sans nom
l'entourent. Rien n'y est propre à Spotify : le même code sert toute appli listée."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from tool_fakes import FakeDesktop, FakeUi, make_config, make_ctx

from mikata.browser import apps as apps_mod
from mikata.browser.apps import DevtoolsApps, debug_port, fold, free_port, main_page
from mikata.browser.cdp import CdpClosed, CdpError, CdpSyncClient, Target, list_targets_sync
from mikata.config import Config
from mikata.llm.base import ToolCallReq
from mikata.tools.base import ToolError, UiWindowInfo
from mikata.tools.registry import ToolRegistry
from mikata.uia.controller import UiController
from mikata.uia.devtools import _CLICK_FN, _FOCUS_FN, DevtoolsBackend, build_tree

HIDDEN = UiWindowInfo("Spotify", "Spotify", 0, "devtools")

# -- arbre d'accessibilité de test -----------------------------------------------------------


def ax(node_id: int, role: str, name: str = "", children: list[int] | None = None, **extra: Any):
    raw: dict[str, Any] = {
        "nodeId": str(node_id),
        "role": {"type": "role", "value": role},
        "name": {"type": "computedString", "value": name},
        "childIds": [str(c) for c in children or []],
        "backendDOMNodeId": 1000 + node_id,
    }
    if extra.get("ignored"):
        raw["ignored"] = True
    if "value" in extra:
        raw["value"] = {"type": "string", "value": extra["value"]}
    if "props" in extra:
        raw["properties"] = [
            {"name": key, "value": {"type": "token", "value": value}}
            for key, value in extra["props"].items()
        ]
    return raw


def results_page() -> list[dict[str, Any]]:
    """La page de résultats : champ de recherche, interrupteur, deux lignes « Jet Lag »
    (la bonne cite Tiakola), le lecteur et son bouton « Play »."""
    nodes = [
        ax(1, "RootWebArea", "Spotify - Web Player", [2, 3]),
        ax(2, "generic", "", [4, 5, 6], ignored=True),  # ignoré : transparent
        ax(3, "generic", "", [30]),  # sans nom : transparent
        ax(4, "combobox", "What do you want to play?", value="Jet Lag Tiakola",
           props={"editable": "plaintext"}),
        ax(5, "switch", "Shuffle", props={"checked": "false"}),
        ax(6, "grid", "Songs", [10, 20]),
        # Ligne 1 : un autre artiste.
        ax(10, "row", "Jet Lag", [11, 12, 13]),
        ax(11, "gridcell", "Play Jet Lag", [14]),
        ax(14, "button", "Play Jet Lag"),
        ax(12, "gridcell", "Song · Autre Groupe", [15, 16]),
        ax(15, "StaticText", "Song · Autre Groupe", [17]),
        ax(17, "InlineTextBox", "Song · Autre Groupe"),
        ax(16, "link", "Autre Groupe"),
        ax(13, "button", "More options for Jet Lag"),
        # Ligne 2 : la bonne.
        ax(20, "row", "Jet Lag", [21, 22, 23]),
        ax(21, "button", "Play"),
        ax(22, "gridcell", "Explicit Song · Tiakola, Jorja Smith", [24, 25, 26, 27]),
        ax(24, "image", "Explicit"),
        ax(25, "StaticText", "Song"),
        ax(26, "link", "Tiakola"),
        ax(27, "StaticText", ","),
        ax(23, "button", "More options for Jet Lag"),
        # Le lecteur.
        ax(30, "region", "Now playing bar", [31, 32]),
        ax(31, "button", "Play"),
        ax(32, "button", "Mute"),
    ]  # fmt: skip
    return nodes


class FakeApps:
    """Le gestionnaire DevTools, simulé : rend l'arbre, enregistre les commandes."""

    def __init__(self, nodes: list[dict[str, Any]] | None = None) -> None:
        self.nodes = nodes if nodes is not None else results_page()
        self.trees: list[list[dict[str, Any]]] = []  # arbres successifs, puis le dernier
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.partial: dict[str, Any] = {}

    def serves(self, name: str | None) -> str | None:
        return "Spotify" if name and fold(name).startswith("spotify") else None

    def window(self, app: str) -> UiWindowInfo | None:
        return HIDDEN

    def call(self, process: str, method: str, **params: Any) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "Accessibility.getFullAXTree":
            if self.trees:
                self.nodes = self.trees.pop(0)
            return {"nodes": self.nodes}
        if method == "DOM.resolveNode":
            return {"object": {"objectId": f"obj-{params['backendNodeId']}"}}
        if method == "Runtime.callFunctionOn":
            return {"result": {"value": True}}
        if method == "Accessibility.getPartialAXTree":
            return self.partial
        return {}

    def evaluate(self, process: str, expression: str) -> Any:
        self.calls.append(("evaluate", {"expression": expression}))
        return "Texte de la page"

    def methods(self) -> list[str]:
        return [method for method, _ in self.actions()]

    def actions(self) -> list[tuple[str, dict[str, Any]]]:
        return [call for call in self.calls if call[0] != "Accessibility.getFullAXTree"]

    def reads(self) -> int:
        return sum(1 for method, _ in self.calls if method == "Accessibility.getFullAXTree")


@pytest.fixture(autouse=True)
def quick_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Après Entrée, l'attente de la vue ne dort pas dans les tests."""
    from mikata.uia import devtools

    monkeypatch.setattr(devtools, "SETTLE_STEP_S", 0.0)
    monkeypatch.setattr(devtools, "SETTLE_QUIET_S", 0.0)


def _controller(apps: FakeApps) -> UiController:
    controller = object.__new__(UiController)  # sans thread d'état ni client UIA réel
    controller._backend = None  # type: ignore[assignment]  # jamais utilisé ici
    controller._desktop = FakeDesktop()  # type: ignore[assignment]
    controller._apps = apps
    controller._devtools = DevtoolsBackend(None, apps)  # type: ignore[arg-type]
    return controller


# -- arbre et moteur ---------------------------------------------------------------------------


def test_the_tree_skips_ignored_nodes_and_text_fragments() -> None:
    root = build_tree(results_page(), "Spotify")
    assert [child.role for child in root.children] == [
        "combobox",
        "switch",
        "grid",
        "region",
    ]  # nœuds ignorés et conteneurs sans nom traversés, dans l'ordre du document
    static = root.children[2].children[0].children[1].children[0]
    assert static.role == "StaticText" and static.children == []  # InlineTextBox retiré
    assert static.parent is root.children[2].children[0].children[1]


def test_kinds_follow_the_accessibility_roles() -> None:
    backend = DevtoolsBackend(None, FakeApps())  # type: ignore[arg-type]
    root = backend.attach(HIDDEN)
    field, switch, grid, _ = root.children
    assert backend.node_kind(field) == "champ"  # champ de recherche à suggestions
    assert backend.node_kind(switch) == "case" and backend.toggle_state(switch) is False
    assert backend.node_kind(grid) == "liste"
    assert backend.node_kind(grid.children[0]) == "élément"
    assert backend.control_text(field) == "Jet Lag Tiakola"


def test_items_show_the_artist_of_each_row_without_repeats() -> None:
    controller = _controller(FakeApps())
    items = controller.items(HIDDEN, "Jet Lag", 10)
    assert [item.name for item in items] == ["Jet Lag", "Jet Lag"]
    assert items[1].texts == ("Explicit Song · Tiakola, Jorja Smith",)  # pas « Tiakola », « , »...
    assert items[1].buttons == ("Play", "More options for Jet Lag")
    assert items[0].texts == ("Song · Autre Groupe",)  # la cellule du bouton n'est pas un texte


def test_click_in_presses_the_play_button_of_the_right_row_by_its_dom_node() -> None:
    apps = FakeApps()
    result = _controller(apps).click_in(HIDDEN, "Jet Lag", ["Play"], ["Tiakola"])
    assert result == "« Play » cliqué."
    assert apps.calls[1] == ("DOM.resolveNode", {"backendNodeId": 1021})  # la ligne 2
    method, params = apps.calls[2]
    assert method == "Runtime.callFunctionOn" and params["functionDeclaration"] == _CLICK_FN
    assert apps.calls[3] == ("Runtime.releaseObject", {"objectId": "obj-1021"})


def test_set_text_types_like_the_keyboard_then_submits_with_enter() -> None:
    apps = FakeApps()
    result = _controller(apps).set_text(HIDDEN, "What do you want to play?", "Jet Lag", True)
    assert result == "« What do you want to play? » rempli et validé."
    assert apps.methods() == [
        "DOM.resolveNode",
        "Runtime.callFunctionOn",
        "Runtime.releaseObject",
        "Input.insertText",
        "Input.dispatchKeyEvent",
        "Input.dispatchKeyEvent",
    ]
    actions = apps.actions()
    assert actions[1][1]["functionDeclaration"] == _FOCUS_FN
    assert actions[3][1] == {"text": "Jet Lag"}
    down, up = actions[4][1], actions[5][1]
    assert down["type"] == "keyDown" and down["key"] == "Enter" and down["text"] == "\r"
    assert up["type"] == "keyUp" and "text" not in up


def suggestions() -> list[dict[str, Any]]:
    """La liste de suggestions ouverte sous le champ : des « Jet Lag » sans artiste."""
    return [
        ax(1, "RootWebArea", "Spotify", [4, 6]),
        ax(4, "combobox", "What do you want to play?", props={"editable": "plaintext"}),
        ax(6, "listbox", "", [10]),
        ax(10, "row", "Jet Lag", [11]),
        ax(11, "button", "Play Jet Lag"),
    ]


def test_after_enter_the_view_is_read_once_the_results_have_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mikata.uia import devtools

    monkeypatch.setattr(devtools, "SETTLE_QUIET_S", 10.0)
    apps = FakeApps()
    closed = [ax(1, "RootWebArea", "Spotify", [4]), suggestions()[1]]
    # Lecture du champ, avant Entrée, puis : suggestions fermées, résultats, résultats.
    apps.trees = [suggestions(), suggestions(), closed, results_page(), results_page()]
    _controller(apps).set_text(HIDDEN, "What do you want to play?", "Jet Lag", True)
    assert apps.reads() == 5 and not apps.trees  # rendu la main dès que les lignes tiennent
    items = _controller(apps).items(HIDDEN, "Jet Lag", 10)
    assert any("Tiakola" in " ".join(item.texts) for item in items)


def test_after_enter_nothing_moving_is_not_waited_for() -> None:
    apps = FakeApps()
    _controller(apps).set_text(HIDDEN, "What do you want to play?", "Jet Lag", True)
    assert apps.reads() == 3  # le champ, avant Entrée, une lecture : rien n'a bougé


def test_without_submit_the_field_is_only_filled() -> None:
    apps = FakeApps()
    _controller(apps).set_text(HIDDEN, "What do you want to play?", "Jet Lag")
    assert "Input.dispatchKeyEvent" not in apps.methods()


def test_a_switch_reports_its_new_state_after_the_click() -> None:
    apps = FakeApps()
    apps.partial = {"nodes": [ax(5, "switch", "Shuffle", props={"checked": "true"})]}
    result = _controller(apps).click(HIDDEN, "Shuffle", None, 1)
    assert result == "« Shuffle » basculé : maintenant activé."


def test_a_node_without_dom_element_is_an_error() -> None:
    nodes = results_page()
    for node in nodes:
        if node["nodeId"] == "32":
            node.pop("backendDOMNodeId")
    with pytest.raises(ToolError, match="aucun élément de la page"):
        _controller(FakeApps(nodes)).click(HIDDEN, "Mute", "bouton", 1)


def test_controls_and_reading_go_through_devtools() -> None:
    apps = FakeApps()
    controller = _controller(apps)
    controls = controller.controls(HIDDEN, 50)
    field = next(c for c in controls if c.kind == "champ")
    assert field.text == "Jet Lag Tiakola"
    assert any(c.name == "Shuffle" and c.text == "désactivé" for c in controls)
    assert controller.read(HIDDEN, None, 100) == "Texte de la page"
    assert controller.find_text(HIDDEN, "tiakola") == "What do you want to play? : Jet Lag Tiakola"


def test_only_listed_apps_are_routed_to_devtools() -> None:
    controller = _controller(FakeApps())
    assert controller.window(None, "Spotify") == HIDDEN
    assert controller.window("Spotify Free", None) == HIDDEN

    class Nowhere(FakeApps):
        def window(self, app: str) -> UiWindowInfo | None:
            return None  # occupée : repli sur UIA

    fallback = _controller(Nowhere())
    window = fallback.window(None, "Spotify")
    assert window.engine == "" and window.handle == 11  # la fenêtre de FakeDesktop
    other = controller.window(None, "brave")
    assert other.engine == "" and other.process == "brave"


# -- gestionnaire : lancer, rejoindre, relancer, refuser -------------------------------------


class Proc:
    def __init__(self, pid: int, cmdline: list[str], exe: str = r"C:\Apps\Spotify.exe") -> None:
        self.pid, self._cmdline, self._exe = pid, cmdline, exe
        self.info = {"name": "Spotify.exe"}

    def cmdline(self) -> list[str]:
        return self._cmdline

    def exe(self) -> str:
        return self._exe

    def name(self) -> str:
        return "Spotify.exe"


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Processus, son et lancements simulés."""
    state: dict[str, Any] = {
        "procs": [],
        "busy": None,
        "launched": [],
        "killed": [],
        "visible": [],
    }
    monkeypatch.setattr(apps_mod, "_processes", lambda app: list(state["procs"]))
    monkeypatch.setattr(apps_mod, "_busy", lambda pids: state["busy"])
    monkeypatch.setattr(apps_mod, "free_port", lambda: 50123)
    monkeypatch.setattr(apps_mod, "_terminate", lambda procs: state["killed"].append(len(procs)))
    monkeypatch.setattr(
        apps_mod, "_launch_hidden", lambda exe, port: state["launched"].append((exe, port))
    )
    monkeypatch.setattr(apps_mod, "_visible_windows", lambda pids: list(state["visible"]))
    monkeypatch.setattr(apps_mod, "_hide_windows", lambda pids: None)
    return state


def test_an_app_already_running_with_the_port_is_joined(system: dict[str, Any]) -> None:
    system["procs"] = [
        Proc(1, ["Spotify.exe", "--remote-debugging-port=41000"]),
        Proc(2, ["Spotify.exe", "--type=renderer"]),
    ]
    apps = DevtoolsApps(["Spotify"], None)
    assert apps._port("Spotify") == (41000, "Spotify", False)
    assert system["launched"] == [] and system["killed"] == []


def test_an_idle_app_without_the_port_is_relaunched_hidden(system: dict[str, Any]) -> None:
    system["procs"] = [Proc(1, ["Spotify.exe"]), Proc(2, ["Spotify.exe", "--type=gpu-process"])]
    apps = DevtoolsApps(["Spotify"], None)
    assert apps._port("Spotify") == (50123, "Spotify", True)
    assert system["killed"] == [2] and system["launched"] == [(r"C:\Apps\Spotify.exe", 50123)]


@pytest.mark.parametrize("reason", ["en train de jouer", "devant l'utilisateur", "son illisible"])
def test_a_busy_app_is_never_relaunched(system: dict[str, Any], reason: str) -> None:
    system["procs"] = [Proc(1, ["Spotify.exe"])]
    system["busy"] = reason
    apps = DevtoolsApps(["Spotify"], None)
    assert apps._port("Spotify") is None
    assert system["killed"] == [] and system["launched"] == []


def test_relaunch_can_be_turned_off(system: dict[str, Any]) -> None:
    system["procs"] = [Proc(1, ["Spotify.exe"])]
    assert DevtoolsApps(["Spotify"], None, relaunch=False)._port("Spotify") is None
    assert system["killed"] == []


def test_a_closed_app_is_launched_hidden_from_the_program_seen_before(
    system: dict[str, Any],
) -> None:
    apps = DevtoolsApps(["Spotify"], None)
    assert apps._port("Spotify") is None  # jamais vu, pas de raccourci : rien à lancer
    system["procs"] = [Proc(1, ["Spotify.exe", "--remote-debugging-port=41000"])]
    apps._port("Spotify")  # vu en cours d'exécution : son programme est retenu
    system["procs"] = []
    assert apps._port("Spotify") == (50123, "Spotify", True)
    assert system["launched"] == [(r"C:\Apps\Spotify.exe", 50123)]


class FakeClient:
    def __init__(self, fail_once: bool = False) -> None:
        self.connected = True
        self.fail_once = fail_once
        self.sent: list[str] = []

    def command(self, method: str, **params: Any) -> dict[str, Any]:
        if self.fail_once:
            self.fail_once, self.connected = False, False
            raise CdpClosed("connexion à l'appli fermée")
        self.sent.append(method)
        return {"ok": method}

    def close(self) -> None:
        self.connected = False


class Building:
    """Une page qui se construit : ses résumés successifs, puis le dernier pour toujours."""

    summaries: ClassVar[list[str]] = []
    closed: ClassVar[int] = 0

    def __init__(self, ws_url: str, timeout_s: float = 10.0) -> None:
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def evaluate(self, expression: str) -> Any:
        return Building.summaries.pop(0) if len(Building.summaries) > 1 else Building.summaries[0]

    def close(self) -> None:
        Building.closed += 1


@pytest.mark.parametrize(
    ("hide", "summaries", "reads_left"),
    [
        (True, ["", "3|1", "50|20", "54|25", "54|25"], 0),  # lancée : attend deux résumés égaux
        (False, ["54|25", "55|25"], 1),  # rejointe : déjà prête
    ],
)
def test_a_launched_app_is_ready_once_its_page_stops_changing(
    system: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    hide: bool,
    summaries: list[str],
    reads_left: int,
) -> None:
    target = Target("Spotify", "https://xpui.app.spotify.com/index.html", "ws://x")
    monkeypatch.setattr(apps_mod, "list_targets_sync", lambda port, timeout: [target])
    monkeypatch.setattr(apps_mod, "CdpSyncClient", Building)
    monkeypatch.setattr(apps_mod.time, "sleep", lambda seconds: None)
    Building.summaries = list(summaries)
    found = DevtoolsApps(["Spotify"], None)._page("Spotify", 41000, 30.0, hide)
    assert found is not None and found[0] is target
    assert len(Building.summaries) == max(reads_left, 1)


def test_an_app_that_never_settles_is_given_up(
    system: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apps_mod, "list_targets_sync", lambda port, timeout: [])
    monkeypatch.setattr(apps_mod.time, "sleep", lambda seconds: None)
    assert DevtoolsApps(["Spotify"], None)._page("Spotify", 41000, 0.0, True) is None


def test_window_and_commands_reconnect_after_a_lost_connection(
    system: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    system["procs"] = [Proc(1, ["Spotify.exe", "--remote-debugging-port=41000"])]
    clients = [FakeClient(fail_once=True), FakeClient()]
    target = Target("Spotify Free", "https://xpui.app.spotify.com/index.html", "ws://x")
    monkeypatch.setattr(
        DevtoolsApps, "_page", lambda self, app, port, timeout, hide: (target, clients.pop(0))
    )
    apps = DevtoolsApps(["Spotify"], None)
    assert apps.window("Spotify") == UiWindowInfo("Spotify Free", "Spotify", 0, "devtools")
    assert apps.call("spotify.exe", "Accessibility.getFullAXTree") == {
        "ok": "Accessibility.getFullAXTree"
    }
    assert clients == []  # la connexion tombée a été rétablie une fois
    with pytest.raises(ToolError, match="pas une appli pilotée"):
        apps.call("Discord", "Runtime.evaluate")


def test_quit_closes_only_a_hidden_app_it_drives(system: dict[str, Any]) -> None:
    apps = DevtoolsApps(["Spotify"], None)
    system["procs"] = [Proc(1, ["Spotify.exe"])]
    assert apps.quit("Spotify") is None  # lancée par l'utilisateur, sans le port
    system["procs"] = [Proc(1, ["Spotify.exe", "--remote-debugging-port=41000"])]
    system["visible"] = [99]
    assert apps.quit("Spotify") is None  # visible : la fermeture normale s'en charge
    assert apps.background_apps() == []
    system["visible"] = []
    assert apps.background_apps() == ["Spotify"]
    assert apps.quit("Discord") is None
    assert apps.quit("spotify") == "Spotify fermé." and system["killed"] == [1]


def test_names_ports_and_pages() -> None:
    apps = DevtoolsApps(["Spotify", " "], None)
    assert apps.serves("Spotify.exe") == "Spotify" and apps.serves("Spotify Free") == "Spotify"
    assert apps.serves("Spotifyy") is None and apps.serves(None) is None
    assert debug_port([["a.exe"], ["a.exe", "--remote-debugging-port=9333"]]) == 9333
    assert debug_port([["a.exe", "--remote-debugging-port=abc"]]) is None
    targets = [
        Target("", "chrome://omnibox-popup.top-chrome/", "ws://1", "browser_ui"),
        Target("Omni", "chrome://omnibox", "ws://2"),
        Target("Spotify", "https://xpui.app.spotify.com/index.html", "ws://3"),
    ]
    page = main_page(targets)
    assert page is not None and page.ws_url == "ws://3"
    assert 0 < free_port() < 65536


def test_no_app_is_driven_this_way_by_default() -> None:
    assert Config().uia.devtools_apps == []


# -- outils ------------------------------------------------------------------------------------


def _run_tool(tmp_path: Path, name: str, args: dict[str, Any], **ctx: Any) -> Any:
    config = make_config(tmp_path, uia={"devtools_apps": ["Spotify"]})
    registry = ToolRegistry(make_ctx(tmp_path, config=config, **ctx))
    return asyncio.run(registry.execute(ToolCallReq("1", name, args)))


def test_a_link_never_brings_a_hidden_app_to_the_front(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    ui = FakeUi(windows=[HIDDEN])
    args = {"link": "spotify:search:Jet Lag", "background": True}
    result = _run_tool(tmp_path, "open_app_link", args, desktop=desktop, uia=ui)
    assert not result.ok and "tourne caché" in result.content and "uia_set_text" in result.content
    assert not any(entry[0] == "open" for entry in desktop.log)
    shown = _run_tool(
        tmp_path, "open_app_link", {"link": "spotify:search:x"}, desktop=desktop, uia=ui
    )
    assert shown.ok  # à montrer : le lien s'ouvre, l'utilisateur veut la voir


def test_a_link_still_opens_an_app_driven_by_uia(tmp_path: Path) -> None:
    desktop = FakeDesktop()
    args = {"link": "spotify:search:Jet Lag", "background": True}
    result = _run_tool(tmp_path, "open_app_link", args, desktop=desktop, uia=FakeUi())
    assert result.ok and any(entry[0] == "open" for entry in desktop.log)


def test_close_app_quits_a_hidden_app(tmp_path: Path) -> None:
    desktop = FakeDesktop(windows=[])
    ui = FakeUi(windows=[HIDDEN])
    result = _run_tool(tmp_path, "close_app", {"name": "Spotify"}, desktop=desktop, uia=ui)
    assert result.ok and result.content == "Spotify fermé."
    missing = _run_tool(tmp_path, "close_app", {"name": "Discord"}, desktop=desktop, uia=ui)
    assert not missing.ok and "aucune fenêtre ouverte" in missing.content


def test_uia_set_text_can_submit(tmp_path: Path) -> None:
    ui = FakeUi()
    args = {"process": "Spotify", "control": "Search", "text": "Jet Lag", "submit": True}
    result = _run_tool(tmp_path, "uia_set_text", args, uia=ui)
    assert result.ok and ("set_text", ("Discord", "Search", "Jet Lag", "validé")) in ui.log


def test_the_platform_engines_do_not_submit_silently() -> None:
    from mikata.uia.controller import UiaBackend

    with pytest.raises(ToolError, match="pas validé"):
        UiaBackend(FakeDesktop()).submit(object(), "Recherche")  # type: ignore[arg-type]


# -- client DevTools bloquant ------------------------------------------------------------------


@pytest.fixture
def ws_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Faux Chromium : répond aux commandes, envoie un événement avant chaque réponse."""
    from websockets.sync.server import serve

    received: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        for raw in ws:
            message = json.loads(raw)
            received.append(message)
            ws.send(json.dumps({"method": "Page.frameNavigated", "params": {}}))
            if message["method"] == "Boom":
                ws.send(json.dumps({"id": message["id"], "error": {"message": "boom"}}))
            elif message["method"] == "Silence":
                continue
            elif message["method"] == "Close":
                ws.close()
                return
            else:
                result = {"result": {"value": message["params"].get("expression")}}
                ws.send(json.dumps({"id": message["id"], "result": result}))

    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.socket.getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}", received
    finally:
        server.shutdown()


def test_sync_client_roundtrip_skips_events_and_reports_errors(
    ws_server: tuple[str, list[dict[str, Any]]],
) -> None:
    url, received = ws_server
    client = CdpSyncClient(url, timeout_s=10.0)  # large : le PC peut être chargé
    client.connect()
    assert client.evaluate("1 + 1") == "1 + 1"
    assert received[0]["method"] == "Runtime.evaluate" and received[0]["id"] == 1
    with pytest.raises(CdpError, match="boom"):
        client.command("Boom")
    client.close()
    assert not client.connected


def test_sync_client_times_out_and_notices_a_closed_connection(
    ws_server: tuple[str, list[dict[str, Any]]],
) -> None:
    url, _ = ws_server
    client = CdpSyncClient(url, timeout_s=10.0)
    client.connect()
    client._timeout = 0.3  # la réponse qui ne vient jamais : attente courte
    with pytest.raises(CdpError, match="pas répondu à temps"):
        client.command("Silence")
    with pytest.raises(CdpClosed):
        client.command("Close")
    assert not client.connected
    with pytest.raises(CdpClosed):
        client.command("Runtime.evaluate")


def test_list_targets_sync_reads_the_local_endpoint() -> None:
    entries = [
        {"title": "Spotify", "url": "https://xpui/", "type": "page", "id": "A",
         "webSocketDebuggerUrl": "ws://127.0.0.1/page/A"},
        {"title": "sans ws", "url": "x", "type": "page"},
    ]  # fmt: skip

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps(entries).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        targets = list_targets_sync(httpd.server_address[1])
    finally:
        httpd.shutdown()
    assert targets == [Target("Spotify", "https://xpui/", "ws://127.0.0.1/page/A", "page", "A")]
    with pytest.raises(CdpError, match="injoignable"):
        list_targets_sync(free_port(), 0.5)
