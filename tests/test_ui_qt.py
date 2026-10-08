"""Phase 8 : fenêtres Qt de l'interface, hors écran (QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import write_test_config
from scripts_loader import load_script

from mikata.ui.model import VISUAL_STATES, MascotModel
from mikata.ui.prefs import MascotPrefs, load_prefs

pytestmark = pytest.mark.usefixtures("qapp")
ROOT = Path(__file__).resolve().parent.parent
# Fausses clés, pour les tests (le contrôle de secrets de la CI les ignore).
SECRET = "sk-test-0123456789-ne-doit-pas-paraitre"  # gitleaks:allow
NEW_KEY = "sk-nouvelle-1234567890wxyz"  # gitleaks:allow


def _wait(condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class Clock:
    def __init__(self) -> None:
        self.t = 50.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def mascot(tmp_path: Path) -> Iterator[Any]:
    from mikata.ui.qt.mascot import MascotWindow

    clock = Clock()
    model = MascotModel(clock=clock)
    model.handle({"type": "hello", "state": "idle", "mic": "ok"})
    sent: list[dict[str, Any]] = []
    opened: list[bool] = []

    def send(message: dict[str, Any]) -> bool:
        sent.append(message)
        return True

    window = MascotWindow(
        model, MascotPrefs(), tmp_path / "ui_state.json", None, send, lambda: opened.append(True),
        clock=clock,
    )  # fmt: skip
    window.sent, window.opened, window.clock = sent, opened, clock  # type: ignore[attr-defined]
    window.start()
    yield window
    window.bubble.close()
    window.close()


# -- mascotte -------------------------------------------------------------------------------


def test_window_is_a_frameless_overlay_that_never_takes_focus(mascot: Any) -> None:
    from PySide6.QtCore import Qt

    flags = mascot.windowFlags()
    for flag in (
        Qt.WindowType.FramelessWindowHint,
        Qt.WindowType.WindowStaysOnTopHint,
        Qt.WindowType.Tool,
        Qt.WindowType.WindowDoesNotAcceptFocus,
    ):
        assert flags & flag, flag
    assert mascot.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert mascot.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert not flags & Qt.WindowType.WindowTransparentForInput
    assert mascot.bubble.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert mascot.isVisible() and mascot.size().width() == 128


def test_first_start_goes_bottom_right_of_the_primary_screen(mascot: Any) -> None:
    from PySide6.QtGui import QGuiApplication

    area = QGuiApplication.primaryScreen().availableGeometry()
    assert mascot.geometry().right() < area.right() and mascot.geometry().bottom() < area.bottom()
    assert mascot.x() > area.center().x() and mascot.y() > area.center().y()


def _mouse(widget: Any, kind: str, local: tuple[int, int], pressed: bool) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}  # fmt: skip
    left = Qt.MouseButton.LeftButton
    point = QPointF(*local)
    event = QMouseEvent(
        types[kind],
        point,
        QPointF(widget.mapToGlobal(point.toPoint())),
        Qt.MouseButton.NoButton if kind == "move" else left,
        left if pressed else Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def test_drag_moves_and_remembers_position_and_screen(mascot: Any) -> None:
    mascot.move(300, 200)
    _mouse(mascot, "press", (10, 10), True)
    _mouse(mascot, "move", (-90, -40), True)  # 100 px à gauche, 50 px plus haut
    _mouse(mascot, "release", (-90, -40), False)
    assert (mascot.x(), mascot.y()) == (200, 150)
    saved = load_prefs(mascot.prefs_path)
    assert (saved.x, saved.y) == (200, 150) and saved.screen == mascot.screen().name()
    assert mascot.opened == []  # glisser n'ouvre rien


def test_double_click_opens_the_settings(mascot: Any) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseDClick(mascot, Qt.MouseButton.LeftButton)
    assert mascot.opened == [True]


def test_position_is_restored_and_a_missing_screen_falls_back(tmp_path: Path) -> None:
    from PySide6.QtGui import QGuiApplication

    from mikata.ui.qt.mascot import MascotWindow

    screen = QGuiApplication.primaryScreen()
    model = MascotModel()
    prefs = MascotPrefs(screen=screen.name(), x=40, y=60, size="petite")
    window = MascotWindow(model, prefs, tmp_path / "s.json", None, lambda m: True, lambda: None)
    assert (window.x(), window.y(), window.width()) == (40, 60, 96)
    gone = MascotPrefs(screen="\\\\.\\DISPLAY9", x=40, y=60)  # écran débranché
    other = MascotWindow(model, gone, tmp_path / "s.json", None, lambda m: True, lambda: None)
    assert (other.x(), other.y()) != (40, 60)
    off = MascotPrefs(screen=screen.name(), x=100_000, y=-500)  # hors de l'écran : ramenée
    third = MascotWindow(model, off, tmp_path / "s.json", None, lambda m: True, lambda: None)
    assert screen.availableGeometry().contains(third.geometry())
    for w in (window, other, third):
        w.bubble.close()
        w.close()


def test_size_change_keeps_the_feet_in_place(mascot: Any) -> None:
    mascot.move(400, 300)
    bottom, center = mascot.geometry().bottom(), mascot.geometry().center().x()
    mascot.set_size("grande")
    assert mascot.width() == 192 and mascot.geometry().bottom() == bottom
    assert abs(mascot.geometry().center().x() - center) <= 1
    assert load_prefs(mascot.prefs_path).size == "grande"
    mascot.set_size("immense")  # inconnue : ignorée
    assert mascot.width() == 192


def test_hide_and_reveal(mascot: Any) -> None:
    from PySide6.QtCore import Qt

    mascot.set_hidden(True)
    assert not mascot.isVisible() and load_prefs(mascot.prefs_path).hidden
    mascot.reveal()  # « Afficher la mascotte » depuis l'icône ou les réglages
    assert mascot.isVisible() and not load_prefs(mascot.prefs_path).hidden
    assert not mascot.windowFlags() & Qt.WindowType.WindowTransparentForInput  # jamais


def test_old_click_through_preference_is_ignored(tmp_path: Path) -> None:
    """Le mode traversable a été retiré : une ancienne préférence ne le réactive pas."""
    from PySide6.QtCore import Qt

    from mikata.ui.qt.mascot import MascotWindow

    path = tmp_path / "ui_state.json"
    path.write_text(json.dumps({"size": "petite", "click_through": True}), encoding="utf-8")
    prefs = load_prefs(path)
    assert prefs.size == "petite" and not hasattr(prefs, "click_through")
    window = MascotWindow(MascotModel(), prefs, path, None, lambda m: True, lambda: None)
    assert not window.windowFlags() & Qt.WindowType.WindowTransparentForInput
    window.bubble.close()
    window.close()


def _actions(menu: Any) -> dict[str, Any]:
    found = {}
    for action in menu.actions():
        if action.menu() is not None:
            found.update({f"{action.text()}/{a.text()}": a for a in action.menu().actions()})
        elif action.text():
            found[action.text()] = action
    return found


def test_context_menu(mascot: Any) -> None:
    actions = _actions(mascot.build_menu())
    assert list(actions) == [
        "Taille/Petite (96 px)",
        "Taille/Moyenne (128 px)",
        "Taille/Grande (192 px)",
        "Bulle de texte",
        "Mettre en pause l'écoute",
        "Réglages…",
        "Masquer",
    ]
    assert actions["Taille/Moyenne (128 px)"].isChecked()
    assert actions["Bulle de texte"].isChecked()
    actions["Taille/Petite (96 px)"].trigger()
    assert mascot.width() == 96
    actions["Mettre en pause l'écoute"].trigger()
    assert mascot.sent == [{"type": "pause", "paused": True}]
    actions["Réglages…"].trigger()
    assert mascot.opened == [True]
    actions["Bulle de texte"].trigger()
    assert not load_prefs(mascot.prefs_path).bubble
    actions["Masquer"].trigger()
    assert not mascot.isVisible()
    mascot.model.disconnected()
    assert not _actions(mascot.build_menu())["Mettre en pause l'écoute"].isEnabled()


def test_bubble_follows_the_mascot(mascot: Any) -> None:
    from PySide6.QtGui import QGuiApplication

    mascot.model.handle({"type": "heard", "id": 1, "text": "quelle heure est-il"})
    mascot.model.handle({"type": "say", "id": 1, "text": "Il est quinze heures."})
    mascot.move(400, 400)
    mascot.tick()
    bubble = mascot.bubble
    assert bubble.isVisible() and bubble.tail_down
    assert bubble.geometry().bottom() < mascot.y()
    assert abs(bubble.geometry().center().x() - mascot.geometry().center().x()) <= 1
    top = QGuiApplication.primaryScreen().availableGeometry().top()
    mascot.move(400, top)  # en haut de l'écran : la bulle passe dessous
    mascot.tick()
    assert not bubble.tail_down and bubble.y() > mascot.geometry().bottom()
    mascot.set_bubble(False)
    mascot.tick()
    assert not bubble.isVisible()
    mascot.set_bubble(True)
    mascot.clock.t += 20  # bulle_s écoulé, au repos
    mascot.tick()
    assert not bubble.isVisible()


def _single_image_skin(root: Path, key: str = "rond") -> None:
    """Skin d'une image par état (un disque), animé par les effets par défaut."""
    from PIL import Image, ImageDraw

    folder = root / key
    folder.mkdir(parents=True)
    states = {}
    for state in VISUAL_STATES:
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse((4, 4, 60, 60), fill=(200, 60, 60, 255))
        image.save(folder / f"{state}.png")
        states[state] = {"file": f"{state}.png"}
    manifest = {"name": "Rond", "author": "test", "license": "CC0", "states": states}
    (folder / "skin.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_every_state_paints_with_every_kind_of_skin(mascot: Any, tmp_path: Path) -> None:
    from mikata.ui.model import demo_messages
    from mikata.ui.skins import load_skin

    load_script("export_mascot_skin").export("images", size=32, fps=4, root=tmp_path)
    _single_image_skin(tmp_path)
    for skin in (None, load_skin(tmp_path, "images"), load_skin(tmp_path, "rond")):
        mascot.set_skin(skin)
        for state in VISUAL_STATES:
            for message in demo_messages(state):
                mascot.model.handle(message)
            assert mascot.model.visual() == state
            image = mascot.grab().toImage()
            center = image.pixelColor(image.width() // 2, image.height() // 2 + 6)
            assert center.alpha() > 200, (skin, state)  # le personnage est dessiné
            assert image.pixelColor(0, 0).alpha() == 0  # le coin reste transparent
            mascot.clock.t += 5  # fin des états joués une fois


def test_single_image_skin_is_animated_by_its_effects(mascot: Any, tmp_path: Path) -> None:
    from mikata.ui.skins import load_skin

    _single_image_skin(tmp_path)
    mascot.set_skin(load_skin(tmp_path, "rond"))
    mascot.model.handle({"type": "state", "state": "speaking"})  # rebond + lueur
    first = mascot.grab().toImage()
    mascot.clock.t += 0.17  # au sommet du rebond (3 rebonds/s)
    second = mascot.grab().toImage()
    assert first != second
    mascot.set_skin(None)  # retour à Mika, sans redémarrer
    assert mascot.painter.skin is None


# -- réglages -------------------------------------------------------------------------------


class FakeTester:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start_mic(self, name: str) -> None:
        self.calls.append(("mic", name))

    def level(self) -> float | None:
        return 0.42

    def stop_mic(self) -> None:
        self.calls.append(("stop", ""))

    def play_output(self, name: str) -> None:
        self.calls.append(("out", name))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "config.yaml", tmp_path / "config.yaml")
    (tmp_path / ".env").write_text(f"LLM_API_KEY={SECRET}\n", encoding="utf-8")
    (tmp_path / "data" / "logs").mkdir(parents=True)
    (tmp_path / "data" / "logs" / "mikata.log").write_text(
        "2026-10-08 09:12:01.204 | INFO     | MainThread | mikata.app:run:352 - MIKATA démarré.\n"
        "2026-10-08 09:12:02.000 | DEBUG    | MainThread | mikata.app:x:1 - détail\n"
        "2026-10-08 09:12:03.000 | WARNING  | MainThread | mikata.app:y:2 - Micro occupé.\n",
        encoding="utf-8",
    )
    usage = {"cost_usd": 1.25, "commands": 40, "since": "2026-10-01T08:00:00"}
    (tmp_path / "data" / "llm_usage.json").write_text(json.dumps(usage), encoding="utf-8")
    shutil.copytree(ROOT / "assets" / "mascot", tmp_path / "assets" / "mascot")
    return tmp_path


def _settings(base: Path, sent: list[dict[str, Any]], connected: bool = True, **kw: Any) -> Any:
    from mikata.ui.qt.settings import SettingsWindow

    def send(message: dict[str, Any]) -> bool:
        sent.append(message)
        return connected

    devices = (["Micro (RØDE NT-USB+)"], ["Casque"], None)
    return SettingsWindow(
        base / "config.yaml", base, send, devices=lambda: devices, tester=FakeTester(), **kw
    )


def test_settings_show_the_current_values(base: Path) -> None:
    window = _settings(base, [])
    values = window.values()
    assert values["wake.threshold"] == 0.5 and values["agent.follow_up_s"] == 6
    assert values["audio.input_device"] == "NT-USB" and values["llm.reasoning_effort"] == "none"
    assert "ui.skin" not in values and values["ui.mascot"] is True  # onglet Skins : immédiat
    assert window.tabs.count() == 7
    window.close()


def test_save_writes_only_changed_lines_and_reloads_mikata(base: Path) -> None:
    before = (base / "config.yaml").read_text(encoding="utf-8").splitlines()
    sent: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    window = _settings(base, sent, on_applied=applied.append)
    window.fields["wake.threshold"].setValue(0.62)
    window.fields["audio.input_device"].setCurrentText("Micro (RØDE NT-USB+)")
    assert window.save() == ["audio.input_device", "wake.threshold"]
    assert sent == [{"type": "reload_config"}]
    status = window.status.text()
    assert "Tout de suite : Seuil du mot d'activation." in status
    assert "Au redémarrage : Micro." in status
    after = (base / "config.yaml").read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert len(changed) == 2 and all(b.split("#", 1)[1] == a.split("#", 1)[1] for a, b in changed)
    assert list((base / "data" / "backups").glob("*-config/config.yaml"))
    assert applied == [{"audio.input_device": "Micro (RØDE NT-USB+)", "wake.threshold": 0.62}]
    assert window.save() == [] and window.status.text() == "Rien à enregistrer."
    window.close()


def test_save_without_mikata_says_so(base: Path) -> None:
    window = _settings(base, [], connected=False)
    window.fields["agent.follow_up_s"].setValue(4)
    assert window.save() == ["agent.follow_up_s"]
    assert "MIKATA n'est pas lancé" in window.status.text()
    window.on_config_reply({"type": "config", "error": "wake.threshold : trop grand"})
    assert "refuse" in window.status.text()
    window.close()


def test_secrets_are_masked_everywhere(base: Path) -> None:
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QLineEdit, QPlainTextEdit

    window = _settings(base, [])
    assert window.secret_labels["LLM_API_KEY"].text() == "••••••••" + SECRET[-4:]
    assert window.secret_labels["SEARCH_API_KEY"].text() == "non définie"
    texts = [w.text() for w in window.findChildren(QLabel)]
    texts += [w.text() for w in window.findChildren(QLineEdit)]
    texts += [w.text() for w in window.findChildren(QAbstractButton)]
    texts += [w.toPlainText() for w in window.findChildren(QPlainTextEdit)]
    texts += [w.itemText(i) for w in window.findChildren(QComboBox) for i in range(w.count())]
    assert not any(SECRET[:-4] in text for text in texts)
    window.close()


def test_replace_a_key(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QInputDialog

    window = _settings(base, [])
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (NEW_KEY, True))
    window._replace_secret("LLM_API_KEY")
    assert f"LLM_API_KEY={NEW_KEY}" in (base / ".env").read_text(encoding="utf-8")
    assert window.secret_labels["LLM_API_KEY"].text() == "••••••••wxyz"
    assert "sk-nouvelle" not in window.status.text()
    window.close()


def test_cost_audio_tests_and_live_log(base: Path) -> None:
    window = _settings(base, [])
    assert (
        "1.2500 $" in window.cost_label.text() and "3.75 $ sur 5.00 $" in window.cost_label.text()
    )
    assert window.credit_bar.value() == 750
    text = window.log_view.toPlainText()
    assert "09:12:01  INFO      MIKATA démarré." in text and "détail" not in text
    window.log_filter.setCurrentIndex(0)
    assert "détail" in window.log_view.toPlainText()
    window.log_filter.setCurrentIndex(2)
    assert window.log_view.toPlainText() == "09:12:03  WARNING   Micro occupé."
    with (base / "data" / "logs" / "mikata.log").open("a", encoding="utf-8") as log:
        log.write("2026-10-08 09:13:00.000 | ERROR    | MainThread | mikata.app:z:3 - Panne.\n")
    window._poll()
    assert window.log_view.toPlainText().endswith("ERROR     Panne.")
    window.mic_button.setChecked(True)  # bouton « Tester le micro »
    window._poll()
    assert window.mic_level.value() == 42
    window._test_output()
    window.close()
    assert window._tester.calls == [("mic", "NT-USB"), ("out", "default"), ("stop", "")]


# -- processus de l'interface ---------------------------------------------------------------


@pytest.fixture
def controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from mikata.config import load_config
    from mikata.ui.qt import app as ui_app
    from mikata.ui.qt.settings import SettingsWindow

    path = write_test_config(tmp_path / "config.yaml")
    monkeypatch.setattr(ui_app, "SettingsWindow", lambda *a, **k: SettingsWindow(
        *a, **k, devices=lambda: ([], [], "audio indisponible"), tester=FakeTester()))  # fmt: skip
    quits: list[bool] = []
    monkeypatch.setattr(ui_app.QApplication, "quit", lambda: quits.append(True))
    ctl = ui_app.UiController(load_config(path), path)
    ctl.quits = quits  # type: ignore[attr-defined]
    yield ctl
    ctl.client.close()
    if ctl.settings is not None:
        ctl.settings.close()
    ctl.mascot.bubble.close()
    ctl.mascot.close()


def test_controller_routes_mikata_messages(controller: Any) -> None:
    assert controller.model.visual() == "mic_off"  # MIKATA pas encore connecté
    controller.on_message({"type": "hello", "state": "idle", "mic": "ok"})
    assert controller.model.visual() == "idle"
    controller.on_message({"type": "open_settings"})
    assert controller.settings is not None and controller.settings.isVisible()
    controller.on_message({"type": "config", "error": "wake.threshold : trop grand"})
    assert "refuse" in controller.settings.status.text()
    controller.mascot.set_hidden(True)
    controller.on_message({"type": "show_mascot"})
    assert controller.mascot.isVisible()
    controller.apply({"ui.bubble_s": 3, "ui.skin": "inexistant"})  # skin absent : Mika
    assert controller.model.bubble_s == 3.0 and controller.mascot.painter.skin is None
    controller.on_message({"type": "bye"})
    assert controller.quits == [True] and controller.model.visual() == "mic_off"


def test_client_and_bridge_talk_over_the_local_channel(tmp_path: Path) -> None:
    """Vraie connexion : UiBridge (asyncio, côté MIKATA) <-> MikataClient (Qt)."""
    from mikata.ui.bridge import UiBridge
    from mikata.ui.qt.client import MikataClient

    channel_path = tmp_path / "data" / "ui_channel.json"
    commands: list[dict[str, Any]] = []
    loop = asyncio.new_event_loop()
    bridge = UiBridge(channel_path, commands.append, lambda: {"state": "idle", "mic": "ok"})
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(bridge.start(), loop).result(3)
        client = MikataClient(channel_path)
        received: list[dict[str, Any]] = []
        lost: list[bool] = []
        client.message.connect(received.append)
        client.disconnected.connect(lambda: lost.append(True))
        client.connect_to_mikata()
        assert _wait(lambda: bool(received)) and received[0]["type"] == "hello"
        assert _wait(lambda: bridge.connected)
        loop.call_soon_threadsafe(bridge.publish, {"type": "heard", "id": 1, "text": "bonjour"})
        assert _wait(lambda: len(received) == 2) and received[1]["text"] == "bonjour"
        assert client.send({"type": "pause", "paused": True})
        assert _wait(lambda: commands == [{"type": "pause", "paused": True}])
        asyncio.run_coroutine_threadsafe(bridge.close(), loop).result(3)
        assert _wait(lambda: bool(lost)) and received[-1]["type"] == "bye"
        assert not client.send({"type": "reload_config"})  # MIKATA parti : rien n'est envoyé
        client.close()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()


def test_client_waits_for_mikata(tmp_path: Path) -> None:
    from mikata.ui.qt.client import MikataClient

    client = MikataClient(tmp_path / "absent.json")
    client.connect_to_mikata()  # pas de fichier : réessai plus tard, sans erreur
    assert not client.is_connected and client._retry.isActive()
    client.close()
    assert not client._retry.isActive()


def test_short_log_lines() -> None:
    from mikata.ui.qt.settings import short_line

    line = "2026-10-08 09:12:01.204 | INFO     | MainThread | mikata.app:run:352 - Prêt - oui."
    assert short_line(line) == "09:12:01  INFO      Prêt - oui."
    assert short_line("Traceback (most recent call last):") == "Traceback (most recent call last):"


def test_screenshot_script(tmp_path: Path) -> None:
    script = load_script("ui_screenshots")
    written = script.render(tmp_path)
    names = sorted(p.name for p in written)
    assert names == sorted(
        ["mascotte_etats.png", "mascotte_bulle.png", "mascotte_menu.png"]
        + [f"reglages_{tab}.png" for tab in script.SETTINGS_TABS]
        + ["skin_detourage.png", "skin_images_etats.png"]
    )
    assert len(script.SETTINGS_TABS) == 7  # un nom par onglet des réglages
    assert all(p.stat().st_size > 1000 for p in written)
