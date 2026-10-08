"""Phase 11 : profils de personnage (skin + voix) appliqués à chaud par MIKATA, compteur
ElevenLabs et profils écrits dans config.yaml par l'interface et scripts/eleven_voices.py."""

from __future__ import annotations

import asyncio
import json
import shutil
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import edit_config
from test_app_ui import FakeBridge, _with_ui

from mikata.app import App, is_live_setting
from mikata.config import Config, load_config
from mikata.ui.settings_store import (
    SECRET_NAMES,
    EditorError,
    read_secrets,
    set_profile,
    voice_usage_text,
)

ROOT = Path(__file__).resolve().parent.parent
THEO = "NyxenPOqNyllHIzSoPbJ"


class FakeRouter:
    def __init__(self) -> None:
        self.applied: list[Config] = []
        self.fail = False

    async def apply(self, config: Config) -> bool:
        if self.fail:
            raise FileNotFoundError("models/kokoro-v1.0.onnx")
        self.applied.append(config)
        return True


def _app(config_file: Path) -> tuple[App, FakeRouter, FakeBridge]:
    router = FakeRouter()
    app = App(load_config(config_file), speaker=types.SimpleNamespace(tts=router))  # type: ignore[arg-type]
    bridge = FakeBridge()
    _with_ui(app, bridge)
    return app, router, bridge


def test_voice_settings_apply_live() -> None:
    for key in ("ui.skin", "ui.profiles", "tts.backend", "tts.elevenlabs"):
        assert is_live_setting(key), key
    assert not is_live_setting("tts.kokoro_voice")


def test_changing_skin_changes_the_voice_live(config_file: Path) -> None:
    app, router, bridge = _app(config_file)
    edit_config(config_file, ui={"skin": "gars", "profiles": {"gars": {"voice": "elevenlabs",
                "voice_id": THEO, "voice_name": "Théo"}}})  # fmt: skip
    asyncio.run(app.reload_config())
    (reply,) = bridge.of("config")
    assert reply["error"] is None and sorted(reply["applied"]) == ["ui.profiles", "ui.skin"]
    assert [c.ui.skin for c in router.applied] == ["gars"]
    edit_config(config_file, agent={"follow_up_s": 3})  # rien à voir avec la voix
    asyncio.run(app.reload_config())
    assert len(router.applied) == 1


def test_a_voice_that_cannot_load_is_reported(config_file: Path) -> None:
    app, router, bridge = _app(config_file)
    router.fail = True
    edit_config(config_file, tts={"backend": "kokoro"})
    asyncio.run(app.reload_config())
    (reply,) = bridge.of("config")
    assert "kokoro-v1.0.onnx" in reply["error"]


def test_voice_events_reach_the_interface(config_file: Path) -> None:
    app, _, bridge = _app(config_file)

    async def scenario() -> None:
        app.bus.bind_loop(asyncio.get_running_loop())
        app.voice_event("fallback", "crédit du mois presque épuisé (120 restants)")  # autre thread
        app.voice_event("resume", "ElevenLabs « Lucie »")
        await asyncio.sleep(0.01)

    asyncio.run(scenario())
    fallback, resume = bridge.of("voice")
    assert fallback["engine"] == "kokoro" and fallback["event"] == "fallback"
    assert fallback["message"].startswith("Je passe sur ma voix de base : crédit")
    assert resume["engine"] == "elevenlabs" and "Lucie" in resume["message"]


def test_mascot_shows_the_voice_change() -> None:
    from mikata.ui.model import MascotModel

    model = MascotModel(clock=lambda: 10.0)
    model.handle({"type": "hello", "state": "idle"})
    model.handle({"type": "voice", "event": "fallback", "message": "Je passe sur ma voix de base."})
    assert model.bubble() is not None and "voix de base" in model.bubble().note  # type: ignore[union-attr]


def test_default_config_has_the_three_profiles() -> None:
    config = load_config(ROOT / "config.yaml")
    profiles = config.ui.profiles
    assert profiles["default"].voice == "kokoro"
    assert (profiles["gars"].voice_id, profiles["gars"].voice_name) == (THEO, "Théo")
    assert profiles["femme"].voice_name == "Lucie"
    assert config.tts.elevenlabs.model == "eleven_v4_turbo"
    assert config.tts.elevenlabs.monthly_credit_cap == 30_000
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "api_key" not in text.replace("ELEVENLABS_API_KEY", "")


# -- config.yaml : profils écrits ligne par ligne --------------------------------------------


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    shutil.copy2(ROOT / "config.yaml", path)
    return path


def test_set_profile_replaces_adds_and_removes_one_line(config_copy: Path) -> None:
    before = config_copy.read_text(encoding="utf-8")
    new = {"voice": "elevenlabs", "voice_id": "abc123", "voice_name": "Hugo"}
    after = set_profile(before, "gars", new)
    a, b = before.splitlines(), after.splitlines()
    changed = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
    assert len(changed) == 1 and changed[0][1].strip().startswith("gars: {voice: elevenlabs")
    assert yaml.safe_load(after)["ui"]["profiles"]["gars"] == new
    added = set_profile(after, "chat", {"voice": "kokoro"})
    assert len(added.splitlines()) == len(b) + 1
    assert yaml.safe_load(added)["ui"]["profiles"]["chat"] == {"voice": "kokoro"}
    assert yaml.safe_load(added)["wake"] == yaml.safe_load(before)["wake"]
    removed = set_profile(added, "chat", None)
    assert removed == after
    with pytest.raises(EditorError):
        set_profile(before, "../x", {"voice": "kokoro"})


def test_set_profile_creates_the_section() -> None:
    text = "ui:\n  skin: default   # skin\n"
    out = set_profile(text, "gars", {"voice": "elevenlabs", "voice_id": THEO})
    assert yaml.safe_load(out)["ui"] == {"skin": "default", "profiles": {"gars": {
        "voice": "elevenlabs", "voice_id": THEO}}}  # fmt: skip
    assert out.startswith("ui:\n  skin: default   # skin\n")
    Config.model_validate(yaml.safe_load(out))


# -- onglet « Coût et clés » ------------------------------------------------------------------


def test_voice_usage_text(tmp_path: Path) -> None:
    path = tmp_path / "elevenlabs_usage.json"
    text, share = voice_usage_text(path)
    assert "pas encore de relevé" in text and share is None
    path.write_text(json.dumps({
        "used": 4200, "limit": 40000, "cap": 30000, "remaining": 25800, "rate": 0.055,
        "rate_measured": 0.055, "answers_left": 3066, "reset_unix": 1_800_000_000,
        "state": "elevenlabs", "reason": None}), encoding="utf-8")  # fmt: skip
    text, share = voice_usage_text(path)
    assert "4200 crédits utilisés ce mois sur 40000 (plafond choisi : 30000)" in text
    assert "restants utilisables : 25800" in text and "3066 réponses" in text
    assert "coût mesuré 0.055" in text and share == pytest.approx(0.86)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(state="kokoro", reason="plafond mensuel atteint", rate_measured=None)
    path.write_text(json.dumps(data), encoding="utf-8")
    text, _ = voice_usage_text(path)
    assert "Voix de base (Kokoro) en ce moment : plafond mensuel atteint" in text
    assert "pas encore mesuré" in text


def test_elevenlabs_key_is_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    secret = "sk_0123456789abcdef0123"  # gitleaks:allow
    env = tmp_path / ".env"
    env.write_text(f"ELEVENLABS_API_KEY={secret}\n", encoding="utf-8")
    assert "ELEVENLABS_API_KEY" in SECRET_NAMES
    shown = read_secrets(env)["ELEVENLABS_API_KEY"]
    assert secret not in shown and shown.endswith("0123")


def test_settings_window_shows_the_elevenlabs_counter(
    qapp: Any, tmp_path: Path, config_copy: Path
) -> None:
    from mikata.ui.qt.settings import SettingsWindow

    sent: list[dict[str, Any]] = []
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "elevenlabs_usage.json").write_text(json.dumps({
        "used": 100, "limit": 40000, "cap": 30000, "remaining": 29900, "rate": 0.055,
        "rate_measured": 0.055, "answers_left": 3600, "state": "elevenlabs"}))  # fmt: skip
    window = SettingsWindow(config_copy, tmp_path, lambda m: sent.append(m) or True)
    assert "29900" in window.voice_label.text() and window.voice_bar.value() > 990
    window.on_voice(
        {"type": "voice", "event": "fallback", "message": "Je passe sur ma voix de base."}
    )
    assert window.status.text() == "Je passe sur ma voix de base."
    window.close()


def test_choosing_a_skin_in_settings_tells_mikata(
    qapp: Any, tmp_path: Path, config_copy: Path
) -> None:
    from test_ui_skins_tab import _install, _row

    from mikata.ui.qt.settings import SettingsWindow

    _install(tmp_path / "assets" / "mascot", "gars", "Gars")
    sent: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    window = SettingsWindow(
        config_copy, tmp_path, lambda m: sent.append(m) or True, on_applied=applied.append
    )
    tab = window.skins_tab
    assert "Voix : ElevenLabs « Théo »" in tab.describe("gars")
    tab.list.setCurrentRow(_row(tab, "gars"))
    assert applied == [{"ui.skin": "gars"}]  # la mascotte change tout de suite
    assert sent == [{"type": "reload_config"}]  # et MIKATA prend la voix du profil
    assert window.status.text() == "Skin « Gars » appliqué (voix ElevenLabs « Théo »)."
    assert load_config(config_copy).ui.skin == "gars"
    window.close()
