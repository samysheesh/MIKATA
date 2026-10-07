from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mikata.config import (
    PROJECT_ROOT,
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
)


def _write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _repo_yaml() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_repo_config_is_valid_and_matches_section_10(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.audio.input_device == "NT-USB"
    assert config.audio.output_device == "default"
    assert config.wake.model == Path("models/hey_jarvis.onnx")
    assert config.wake.verifier is None
    assert config.wake.threshold == 0.5
    assert config.wake.consecutive_frames == 1
    assert config.wake.hotkey == "<ctrl>+<alt>+m"
    assert config.vad.end_silence_ms == 500
    assert config.vad.speculative_ms == 250
    assert config.stt.model == "large-v3-turbo"
    assert config.stt.device == "cuda"
    assert config.stt.compute_type == "float16"
    assert config.llm.max_tool_rounds == 8
    assert config.tts.backend == "piper"
    assert config.agent.follow_up_s == 6
    assert config.guard.confirm_after_web is False
    assert config.logging.rotation_mb == 5
    assert config.logging.retention_files == 5
    # ajouts phase 1
    assert config.audio.input_channel is None
    assert config.audio.earcon_volume == 0.5
    assert config.wake.log_days == 7
    assert config.logging.stats_interval_s == 60
    # ajouts phase 2
    assert config.vad.model == Path("models/silero_vad.onnx")
    assert config.vad.threshold == 0.5
    assert config.stt.wake_words == ["mikata", "mi kata", "micata", "mikado", "jarvis"]


def test_repo_config_keys_are_exactly_the_model_fields() -> None:
    raw = _repo_yaml()
    fields = set(Config.model_fields) - {"secrets"}
    assert set(raw) == fields
    for section, values in raw.items():
        assert set(values) == set(Config.model_fields[section].annotation.model_fields)


def test_defaults_equal_repo_config(config_file: Path) -> None:
    assert load_config(config_file).model_dump() == Config().model_dump()


def test_empty_file_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path).model_dump() == Config().model_dump()


def test_base_dir_and_resolve(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.base_dir == config_file.parent
    assert config.resolve(config.wake.model) == config_file.parent / "models" / "hey_jarvis.onnx"
    assert config.resolve("~") == Path.home()
    absolute = config_file.parent / "x.onnx"
    assert config.resolve(absolute) == absolute


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(tmp_path / "absent.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("audio: [\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Impossible de lire"):
        load_config(path)


def test_not_a_mapping(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.yaml", ["a", "b"])
    with pytest.raises(ConfigError, match="sections"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "loc"),
    [
        ("wake", "threshold", 1.5, "wake.threshold"),
        ("wake", "consecutive_frames", 0, "wake.consecutive_frames"),
        ("stt", "device", "tpu", "stt.device"),
        ("stt", "compute_type", "int4", "stt.compute_type"),
        ("tts", "backend", "espeak", "tts.backend"),
        ("llm", "base_url", "ftp://x", "llm.base_url"),
        ("vad", "speculative_ms", 600, "vad"),
        ("audio", "pre_roll_s", -1, "audio.pre_roll_s"),
        ("logging", "level", "BAVARD", "logging.level"),
        ("audio", "input_channel", 0, "audio.input_channel"),
        ("audio", "earcon_volume", 1.5, "audio.earcon_volume"),
        ("vad", "threshold", 1.0, "vad.threshold"),
        ("stt", "wake_words", [], "stt.wake_words"),
    ],
)
def test_invalid_values(tmp_path: Path, section: str, key: str, value: object, loc: str) -> None:
    raw = _repo_yaml()
    raw[section][key] = value
    path = _write(tmp_path / "config.yaml", raw)
    with pytest.raises(ConfigError, match="Configuration invalide") as info:
        load_config(path)
    assert loc in str(info.value)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    raw = _repo_yaml()
    raw["wake"]["treshold"] = 0.4  # faute de frappe
    raw["inconnu"] = {}
    path = _write(tmp_path / "config.yaml", raw)
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert "wake.treshold" in str(info.value)
    assert "inconnu" in str(info.value)


def test_secrets_in_yaml_are_rejected(tmp_path: Path) -> None:
    raw = _repo_yaml()
    raw["llm_api_key"] = "sk-123"
    path = _write(tmp_path / "config.yaml", raw)
    with pytest.raises(ConfigError, match=r"\.env"):
        load_config(path)


def test_logging_level_case_insensitive(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.yaml", {"logging": {"level": "debug"}})
    assert load_config(path).logging.level == "DEBUG"


def test_secrets_from_dotenv(config_file: Path) -> None:
    env = config_file.parent / ".env"
    env.write_text("﻿LLM_API_KEY=sk-secret\nSEARCH_API_KEY=\nAUTRE=1\n", encoding="utf-8")
    config = load_config(config_file)
    assert config.secrets.llm_api_key is not None
    assert config.secrets.llm_api_key.get_secret_value() == "sk-secret"
    assert config.secrets.search_api_key is None  # vide = absente
    assert "sk-secret" not in repr(config)
    assert "secrets" not in config.model_dump()


def test_secrets_from_environment(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_API_KEY", "brave-key")
    config = load_config(config_file)
    assert config.secrets.search_api_key is not None
    assert config.secrets.search_api_key.get_secret_value() == "brave-key"


def test_no_dotenv(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.secrets.llm_api_key is None


def test_llm_placeholder(config_file: Path, tmp_path: Path) -> None:
    assert load_config(config_file).llm.is_placeholder
    path = _write(
        tmp_path / "other.yaml",
        {"llm": {"base_url": "https://api.example.com/v1", "model": "deepseek-flash"}},
    )
    assert not load_config(path).llm.is_placeholder


def test_resolve_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert resolve_config_path() == PROJECT_ROOT / "config.yaml"
    monkeypatch.setenv("MIKATA_CONFIG", str(tmp_path / "a.yaml"))
    assert resolve_config_path() == tmp_path / "a.yaml"
    assert resolve_config_path(tmp_path / "b.yaml") == tmp_path / "b.yaml"


def test_invalid_fixture_is_invalid() -> None:
    with pytest.raises(ConfigError):
        load_config(PROJECT_ROOT / "tests" / "fixtures" / "config_invalide.yaml")
