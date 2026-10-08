"""Phase 8 : écriture de config.yaml, secrets, coût et journal pour la fenêtre de réglages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from mikata.app import LIVE_SETTINGS, is_live_setting
from mikata.config import Config, load_config
from mikata.ui.settings_store import (
    SETTINGS,
    SPECS,
    ConfigEditor,
    EditorError,
    LogTail,
    mask,
    read_secrets,
    read_usage,
    set_value,
    write_secret,
    yaml_scalar,
)

ROOT = Path(__file__).resolve().parent.parent
# Fausses clés, pour les tests (le contrôle de secrets de la CI les ignore).
OLD_KEY = "sk-live-0123456789abcdef"  # gitleaks:allow
NEW_KEY = "sk-nouvelle-cle-9876543210"  # gitleaks:allow


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    shutil.copy2(ROOT / "config.yaml", path)
    return path


def _diff(before: str, after: str) -> list[tuple[str, str]]:
    a, b = before.splitlines(), after.splitlines()
    assert len(a) == len(b)
    return [(x, y) for x, y in zip(a, b, strict=True) if x != y]


def test_only_the_edited_lines_change(config_file: Path) -> None:
    before = config_file.read_text(encoding="utf-8")
    editor = ConfigEditor(config_file)
    editor.set("wake.threshold", 0.62)
    editor.set("agent.follow_up_s", 4)
    editor.set("llm.reasoning_effort", "low")
    editor.set("audio.input_device", "Micro (RØDE NT-USB+)")
    assert editor.changed() == [
        "agent.follow_up_s",
        "audio.input_device",
        "llm.reasoning_effort",
        "wake.threshold",
    ]
    editor.save()
    after = config_file.read_text(encoding="utf-8")
    changed = _diff(before, after)
    assert len(changed) == 4
    for old, new in changed:  # commentaire gardé, à la même colonne si la valeur y tient
        assert old[old.index("#") :] == new[new.index("#") :]
        assert new.index("#") == max(old.index("#"), new.index(" #") + 1)
    assert '  input_device: "Micro (RØDE NT-USB+)" #' in after
    assert "  threshold: 0.62 " in after
    config = load_config(config_file)
    assert config.wake.threshold == 0.62 and config.agent.follow_up_s == 4
    assert config.llm.reasoning_effort == "low"
    assert config.audio.input_device == "Micro (RØDE NT-USB+)"
    assert config.vad.threshold == 0.5  # l'autre « threshold » (section vad) n'a pas bougé


def test_wake_section_untouched_unless_edited(config_file: Path) -> None:
    before = config_file.read_text(encoding="utf-8")
    editor = ConfigEditor(config_file)
    editor.set("ui.bubble_s", 12)
    editor.save()
    changed = _diff(before, config_file.read_text(encoding="utf-8"))
    assert [new.split("#")[0].strip() for _, new in changed] == ["bubble_s: 12"]


def test_same_value_rewrites_nothing(config_file: Path) -> None:
    before = config_file.read_bytes()
    editor = ConfigEditor(config_file)
    editor.set("agent.follow_up_s", 6)
    assert editor.changed() == []
    editor.save()
    assert config_file.read_bytes() == before


def test_invalid_value_is_refused_and_file_kept(config_file: Path) -> None:
    before = config_file.read_bytes()
    editor = ConfigEditor(config_file)
    editor.set("wake.threshold", 3)
    error = editor.validate()
    assert error is not None and "wake.threshold" in error
    with pytest.raises(EditorError):
        editor.save()
    assert config_file.read_bytes() == before


def test_backup_and_bom(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"\xef\xbb\xbfwake:\r\n  threshold: 0.5   # seuil\r\n")
    editor = ConfigEditor(path)
    editor.set("wake.threshold", 0.7)
    backup = editor.save(tmp_path / "backups")
    assert backup is not None and backup.read_bytes().endswith(b"0.5   # seuil\r\n")
    assert path.read_bytes() == b"\xef\xbb\xbfwake:\r\n  threshold: 0.7   # seuil\r\n"


def test_missing_key_or_section_is_added() -> None:
    text = "wake:\n  threshold: 0.5\n\n# fin\nagent:\n  follow_up_s: 6\n"
    added = set_value(text, "wake.verifier_threshold", 0.4)
    assert added.startswith("wake:\n  threshold: 0.5\n  verifier_threshold: 0.4\n\n# fin\n")
    added = set_value(text, "ui.skin", "chat")
    assert added.endswith("agent:\n  follow_up_s: 6\nui:\n  skin: chat\n")
    assert yaml.safe_load(added)["ui"] == {"skin": "chat"}


def test_multi_line_values_and_secrets_are_refused(config_file: Path) -> None:
    text = "agent:\n  cancel_phrases:\n    - laisse tomber\n    - annule\n"
    with pytest.raises(EditorError):
        set_value(text, "agent.cancel_phrases", "x")
    with pytest.raises(EditorError):
        set_value(text, "nopoint", 1)
    with pytest.raises(EditorError):
        ConfigEditor(config_file).set("llm.llm_api_key", "sk-123")


def test_null_and_hash_in_strings_survive() -> None:
    text = 'audio:\n  input_device: null   # micro\n  output_device: "a # b"  # sortie\n'
    out = set_value(text, "audio.input_device", "x # y")
    assert out.splitlines()[1] == '  input_device: "x # y" # micro'
    out = set_value(out, "audio.output_device", None)
    assert out.splitlines()[2] == "  output_device: null     # sortie"  # colonne gardée
    assert yaml.safe_load(out)["audio"] == {"input_device": "x # y", "output_device": None}


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 6, 0.5, 1e-05, 0.123456, "none", "default", "yes", "on", "null",
     "1.5", "Micro (USB)", "a: b", "", "C:\\voix", "l'été"],
)  # fmt: skip
def test_yaml_scalar_round_trips(value: object) -> None:
    rendered = yaml_scalar(value)
    assert "\n" not in rendered
    loaded = yaml.safe_load(f"k: {rendered}")["k"]
    if isinstance(value, float):
        assert loaded == pytest.approx(value, abs=1e-6)
    else:
        assert loaded == value and type(loaded) is type(value)


def test_specs_match_what_mikata_applies_live() -> None:
    """La fenêtre annonce « tout de suite » exactement pour ce que reload_config applique."""
    for spec in SETTINGS:
        assert spec.live == is_live_setting(spec.key), spec.key
        section, _, name = spec.key.partition(".")
        assert hasattr(getattr(Config(), section), name), spec.key
    assert {k for k in LIVE_SETTINGS if k in SPECS} == {s.key for s in SETTINGS if s.live}


def test_editor_get_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("wake:\n  threshold: 0.4\n", encoding="utf-8")
    editor = ConfigEditor(path)
    assert editor.get("wake.threshold") == 0.4
    assert editor.get("ui.bubble_s") == Config().ui.bubble_s


def test_secrets_are_never_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("SEARCH_API_KEY", "court")
    env = tmp_path / ".env"
    secret = OLD_KEY
    env.write_text(f"# mes clés\nLLM_API_KEY={secret}\n", encoding="utf-8")
    shown = read_secrets(env)
    assert shown["LLM_API_KEY"] == "••••••••cdef"
    assert shown["SEARCH_API_KEY"] == "•••••••• (variable d'environnement)"
    assert all(secret not in v and "court" not in v for v in shown.values())
    assert mask(None) == mask("") == "non définie"
    write_secret(env, "LLM_API_KEY", f"  {NEW_KEY}  ")
    text = env.read_text(encoding="utf-8")
    assert text.startswith("# mes clés\n") and f"LLM_API_KEY={NEW_KEY}" in text
    assert secret not in text
    for bad_name, bad_value in (("AUTRE", "x"), ("LLM_API_KEY", ""), ("LLM_API_KEY", "a b")):
        with pytest.raises(EditorError):
            write_secret(env, bad_name, bad_value)


def test_usage_summary(tmp_path: Path) -> None:
    path = tmp_path / "llm_usage.json"
    assert read_usage(path, 5.0).remaining_usd == 5.0
    path.write_text(json.dumps({"cost_usd": 1.25, "commands": 40, "since": "2026-10-01"}))
    usage = read_usage(path, 5.0)
    assert (usage.cost_usd, usage.commands, usage.remaining_usd) == (1.25, 40, 3.75)


def test_log_tail_follows_rotation(tmp_path: Path) -> None:
    log = tmp_path / "mikata.log"
    tail = LogTail(log, backlog_bytes=10)
    assert tail.poll() == []
    log.write_text("ancienne ligne\nfin\n", encoding="utf-8")
    assert tail.poll() == ["fin"]  # seulement les derniers octets au départ
    with log.open("a", encoding="utf-8") as file:
        file.write("une ligne\nà moi")
    assert tail.poll() == ["une ligne"]
    with log.open("a", encoding="utf-8") as file:
        file.write("tié\n")
    assert tail.poll() == ["à moitié"]
    log.write_text("neuf\n", encoding="utf-8")  # rotation : fichier plus court
    assert tail.poll() == ["neuf"]
