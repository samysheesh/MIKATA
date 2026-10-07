"""Configuration de MIKATA : `config.yaml` (réglages) + `.env` (secrets).

Voir docs/ARCHITECTURE.md §10. Les noms de sections et de clés suivent ce document ;
la section `logging` est un ajout de la phase 0 (rotation des logs, §10 « Logs »).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ENV_VAR = "MIKATA_CONFIG"
SECRET_KEYS = ("llm_api_key", "search_api_key")


class ConfigError(Exception):
    """Configuration absente, illisible ou invalide (message en français)."""


class _Section(BaseModel):
    # Une clé inconnue (faute de frappe) est une erreur, pas un réglage ignoré en silence.
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioConfig(_Section):
    input_device: str = "NT-USB"
    output_device: str = "default"
    pre_roll_s: float = Field(default=2.0, gt=0, le=10)


class WakeConfig(_Section):
    model: Path = Path("models/hey_jarvis.onnx")
    verifier: Path | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    consecutive_frames: int = Field(default=2, ge=1)
    refractory_s: float = Field(default=2.0, ge=0)
    hotkey: str = "<ctrl>+<alt>+m"
    collect_negatives: bool = False


class VadConfig(_Section):
    end_silence_ms: int = Field(default=500, gt=0)
    speculative_ms: int = Field(default=250, gt=0)
    max_utterance_s: float = Field(default=15, gt=0)
    no_speech_timeout_s: float = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _speculative_before_end(self) -> VadConfig:
        if self.speculative_ms >= self.end_silence_ms:
            raise ValueError("speculative_ms doit être inférieur à end_silence_ms")
        return self


class SttConfig(_Section):
    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = "float16"
    beam_size: int = Field(default=1, ge=1)
    initial_prompt: str = "MIKATA, Spotify, Discord, Steam, Chrome, VS Code"


class LlmConfig(_Section):
    base_url: str = "https://..."
    model: str = "..."
    max_tool_rounds: int = Field(default=8, ge=1)
    request_timeout_s: float = Field(default=20, gt=0)

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url doit commencer par http:// ou https://")
        return value

    @property
    def is_placeholder(self) -> bool:
        """Vrai tant que l'endpoint CommandCode n'est pas renseigné (avant la phase 3)."""
        return "..." in self.base_url or self.model.strip() in ("", "...")


class TtsConfig(_Section):
    backend: Literal["piper", "kokoro"] = "piper"
    voice: Path = Path("models/fr_FR-voice.onnx")
    first_chunk_min_words: int = Field(default=6, ge=1)


class AgentConfig(_Section):
    follow_up_s: float = Field(default=6, ge=0)
    session_reset_min: float = Field(default=10, gt=0)
    thinking_sound_after_s: float = Field(default=1.2, ge=0)


class GuardConfig(_Section):
    allowed_dirs: list[Path] = Field(default_factory=lambda: [Path("~")])
    backup_days: int = Field(default=14, ge=1)
    shutdown_cancel_s: float = Field(default=5, ge=0)
    confirm_after_web: bool = False


class LoggingConfig(_Section):
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"] = "INFO"
    file: Path = Path("data/logs/mikata.log")
    rotation_mb: float = Field(default=5, gt=0)
    retention_files: int = Field(default=5, ge=1)

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class Secrets(BaseSettings):
    """Secrets lus dans les variables d'environnement puis dans `.env`."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8-sig",  # tolère le BOM ajouté par certains éditeurs Windows
        extra="ignore",
        frozen=True,
    )

    llm_api_key: SecretStr | None = None
    search_api_key: SecretStr | None = None

    @field_validator("llm_api_key", "search_api_key", mode="after")
    @classmethod
    def _empty_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value


class Config(_Section):
    """Configuration complète et validée."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)

    _base_dir: Path = PrivateAttr(default=PROJECT_ROOT)

    @property
    def base_dir(self) -> Path:
        """Dossier du fichier de config : base des chemins relatifs."""
        return self._base_dir

    def resolve(self, path: Path | str) -> Path:
        """Chemin absolu : `~` développé, relatif résolu depuis `base_dir`."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self._base_dir / p).resolve()


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Chemin du fichier de config : argument, sinon $MIKATA_CONFIG, sinon `config.yaml`
    à la racine du projet."""
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR) or PROJECT_ROOT / "config.yaml"
    return Path(path).expanduser().resolve()


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(racine)"
        lines.append(f"  - {loc} : {err['msg']}")
    return "\n".join(lines)


def load_config(path: Path | str | None = None, env_file: Path | str | None = None) -> Config:
    """Charge et valide `config.yaml` + `.env` (par défaut à côté du fichier de config).

    Lève `ConfigError` avec un message lisible si quelque chose ne va pas.
    """
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Impossible de lire {config_path} :\n{exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} doit contenir des sections clé: valeur.")

    leaked = [key for key in SECRET_KEYS if key in raw]
    if leaked:
        raise ConfigError(
            f"Secret(s) trouvé(s) dans {config_path.name} : {', '.join(leaked)}. "
            "Les secrets vont dans .env (voir .env.example)."
        )

    env_path = Path(env_file) if env_file is not None else config_path.parent / ".env"
    try:
        secrets = Secrets(_env_file=env_path if env_path.is_file() else None)  # type: ignore[call-arg]
        config = Config.model_validate({**raw, "secrets": secrets})
    except ValidationError as exc:
        raise ConfigError(
            f"Configuration invalide ({config_path}) :\n{_format_validation_error(exc)}"
        ) from exc

    config._base_dir = config_path.parent
    return config
