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


class StartupError(Exception):
    """Démarrage impossible pour une raison matérielle (GPU, DLL...), message en français."""


class _Section(BaseModel):
    # Une clé inconnue (faute de frappe) est une erreur, pas un réglage ignoré en silence.
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioConfig(_Section):
    input_device: str = "NT-USB"
    output_device: str = "default"
    pre_roll_s: float = Field(default=2.0, gt=0, le=10)
    # Ajouts phase 1 : canal du micro (1 = premier canal ; null = moyenne des canaux)
    # et volume des sons d'interface.
    input_channel: int | None = Field(default=None, ge=1)
    earcon_volume: float = Field(default=0.5, ge=0, le=1)


class WakeConfig(_Section):
    model: Path = Path("models/hey_jarvis.onnx")
    verifier: Path | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    consecutive_frames: int = Field(default=1, ge=1)
    refractory_s: float = Field(default=1.0, ge=0)
    hotkey: str = "<ctrl>+<alt>+m"
    collect_negatives: bool = False
    log_days: int = Field(default=7, ge=1)  # ajout phase 1 : purge de data/wake_logs/
    # Ajouts phase 7. Vérificateur de ta voix (§4) : consulté quand le modèle atteint
    # `threshold` ; sous ce seuil de probabilité, la détection est refusée.
    verifier_threshold: float = Field(default=0.5, ge=0, le=1)
    # Collecte de la vie courante (collect_negatives) : arrêt après N heures (~115 Mo / h).
    negatives_max_hours: float = Field(default=12, gt=0)
    # Ajouts phase 9. Raccourci clavier quand « MIKATA » n'a pas été entendu : les dernières
    # secondes d'audio vont dans data/wakeword/misses/ (prochain entraînement), purgées
    # (corbeille) après `miss_days` jours.
    save_misses: bool = True
    miss_days: int = Field(default=30, ge=1)


class VadConfig(_Section):
    end_silence_ms: int = Field(default=500, gt=0)
    speculative_ms: int = Field(default=250, gt=0)
    max_utterance_s: float = Field(default=15, gt=0)
    no_speech_timeout_s: float = Field(default=4, gt=0)
    # Ajouts phase 2 : modèle Silero VAD (ONNX) et seuil de probabilité de voix.
    model: Path = Path("models/silero_vad.onnx")
    threshold: float = Field(default=0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def _speculative_before_end(self) -> VadConfig:
        if self.speculative_ms >= self.end_silence_ms:
            raise ValueError("speculative_ms doit être inférieur à end_silence_ms")
        return self


class SttConfig(_Section):
    model: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = "int8_float16"
    beam_size: int = Field(default=1, ge=1)
    # Vocabulaire, puis amorces de suites courtes (phase 6b : « et à Tokyo ? » était lu
    # « Yeah, Tokyo »). La langue est forcée à « fr » dans stt/whisper_local.py.
    initial_prompt: str = (
        "MIKATA, Bloc-notes, Spotify, Discord, Steam, Chrome, VS Code, PowerShell. "
        "Et à Tokyo ? Et demain ?"
    )
    # Ajouts phase 6 : noms des applications installées ajoutés au prompt de Whisper, dans
    # la limite de prompt_max_chars (Whisper ne garde que ~220 tokens de prompt).
    prompt_apps: bool = True
    prompt_max_chars: int = Field(default=500, ge=50, le=800)
    # Ajout phase 2 : formes du mot d'activation retirées du texte transcrit (§3).
    # « jarvis » couvre le modèle de développement hey_jarvis (retirable en phase 7).
    wake_words: list[str] = Field(
        default_factory=lambda: ["mikata", "mi kata", "micata", "mikado", "jarvis"], min_length=1
    )


ReasoningLevel = Literal["none", "low", "medium", "high"]


class LlmConfig(_Section):
    base_url: str = "https://..."
    model: str = "..."
    max_tool_rounds: int = Field(default=8, ge=1)
    request_timeout_s: float = Field(default=20, gt=0)
    turn_timeout_s: float = Field(default=60, gt=0)  # ajout phase 3 : §6, 60 s par requête
    # Ajout phase 6 : rien reçu du serveur dans ce délai -> échec immédiat, nouvelle connexion
    # et relance (au lieu d'attendre request_timeout_s).
    first_chunk_timeout_s: float = Field(default=8, gt=0)
    # Ajouts phase 4. Niveau de réflexion du modèle (`reasoning_effort`) : « none » par
    # défaut (premier mot plus tôt) ; null = paramètre non envoyé (choix du fournisseur).
    reasoning_effort: ReasoningLevel | None = "none"
    # Mot-clé dit en début ou en fin de commande -> niveau de réflexion pour cette commande
    # (détecté par le code, retiré du texte ; variantes de « réfléchis » acceptées).
    reasoning_keywords: dict[Literal["low", "medium", "high"], list[str]] = Field(
        default_factory=lambda: {
            "low": ["réfléchis"],
            "medium": ["réfléchis bien"],
            "high": ["réfléchis à fond"],
        }
    )
    # Filet : un outil en échec, ou plus de N étapes d'outils -> au moins « low ».
    reasoning_fallback_steps: int = Field(default=2, ge=1)
    include_usage: bool = True  # demande le décompte des tokens en fin de flux
    price_input_usd_per_mtok: float = Field(default=0.30, ge=0)
    price_output_usd_per_mtok: float = Field(default=1.20, ge=0)
    price_cached_usd_per_mtok: float = Field(default=0.006, ge=0)
    credit_usd: float = Field(default=5.0, ge=0)

    @field_validator("reasoning_keywords")
    @classmethod
    def _keywords(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        return {level: [k.strip() for k in words if k.strip()] for level, words in value.items()}

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url doit commencer par http:// ou https://")
        return value

    @property
    def is_placeholder(self) -> bool:
        """Vrai tant que l'endpoint CommandCode n'est pas renseigné."""
        return "..." in self.base_url or self.model.strip() in ("", "...")


class ToolsConfig(_Section):
    """Ajout phase 4 : exécution des outils (§6, §7)."""

    timeout_s: float = Field(default=15, gt=0)  # par appel d'outil
    result_max_chars: int = Field(default=2000, ge=200)  # résultat renvoyé au LLM (§6)
    shell_timeout_s: float = Field(default=30, gt=0)  # ajout phase 5 : run_powershell (§7)


class WebConfig(_Section):
    """Ajout phase 4 : outils web (§7)."""

    search_provider: Literal["tavily", "brave"] = "tavily"  # clé : SEARCH_API_KEY (.env)
    max_results: int = Field(default=5, ge=1, le=10)
    default_city: str | None = None  # ville de la météo si aucune n'est dite


class BrowserConfig(_Section):
    """Ajout phase 12b : contrôle du navigateur (Brave/Chromium) par le protocole
    DevTools. Le port ne s'active qu'au lancement : MIKATA lance Brave elle-même quand il
    est fermé, et refuse poliment (sans jamais le fermer) s'il tourne déjà sans le port."""

    enabled: bool = True  # false : les outils browser_* répondent « désactivé »
    cdp_port: int = Field(default=9222, ge=1024, le=65535)
    fallback_uia: bool = True  # Brave ouvert sans le port : suggérer les outils UIA
    page_text_max_chars: int = Field(default=4000, ge=200, le=20000)  # texte lu, tronqué
    click_timeout_s: float = Field(default=5, gt=0, le=30)  # délai d'une commande DevTools


class SkillsConfig(_Section):
    """Ajout phase 12c : savoir-faire appris (mémoire procédurale, data/skills/skills.json).
    Une commande multi-étapes réussie est distillée en procédure ; la suivante qui
    ressemble la rejoue sans exploration (ou avec le plan si elle a des paramètres)."""

    enabled: bool = True  # false : pas de bibliothèque (outils skills indisponibles)
    max_skills: int = Field(default=30, ge=1, le=200)
    confidence: float = Field(default=70, ge=0, le=100)  # score flou commande -> procédure
    auto_learn: bool = True  # mémorise en arrière-plan les commandes multi-étapes réussies
    auto_learn_min_rounds: int = Field(default=2, ge=1, le=10)  # au moins N étapes d'outils
    max_failures: int = Field(default=3, ge=1, le=20)  # échecs au rejeu avant désactivation
    max_steps: int = Field(default=10, ge=1, le=50)  # étapes rejouées au plus


VoiceEngine = Literal["piper", "kokoro", "elevenlabs"]


class ElevenLabsConfig(_Section):
    """Ajout phase 11 : voix ElevenLabs (service payant en ligne, clé ELEVENLABS_API_KEY dans
    .env). Retour automatique sur Kokoro (local, gratuit) si le crédit, le plafond ou la
    connexion manquent ; retour sur ElevenLabs au renouvellement (mikata/tts/elevenlabs.py)."""

    model: str = "eleven_v4_turbo"
    language: str = "fr"
    voice_id: str = "YxrwjAKoUKULGd0g8K9Y"  # voix si aucun profil n'en donne (Lucie)
    backup_voice_id: str | None = "d3AXX0BlgJHYFCuH9X88"  # voix introuvable : celle-ci (Émilie)
    output_format: Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"] = "pcm_24000"
    base_url: str = "https://api.elevenlabs.io"
    timeout_s: float = Field(default=10, gt=0)
    # Crédits : plafond de dépense mensuel (sur le compteur de l'abonnement), crédit restant
    # minimal, relevé de l'abonnement toutes les N synthèses (déduit localement entre deux).
    monthly_credit_cap: float = Field(default=30_000, ge=0)
    min_remaining_credits: float = Field(default=500, ge=0)
    subscription_refresh_every: int = Field(default=20, ge=1)
    # Coût par caractère tant qu'il n'a pas été mesuré (prudent : 1 crédit / caractère).
    estimated_credits_per_char: float = Field(default=1.0, gt=0)
    calibration_hours: float = Field(default=24, ge=0)  # 0 : jamais de mesure au démarrage
    retry_after_s: float = Field(default=300, gt=0)  # après une panne réseau ou serveur
    cache_max_mb: float = Field(default=300, ge=0)  # phrases gardées sur le disque (0 : aucune)


class TtsConfig(_Section):
    backend: VoiceEngine = "piper"
    voice: Path = Path("models/fr_FR-siwis-medium.onnx")  # voix Piper (+ .onnx.json)
    first_chunk_min_words: int = Field(default=6, ge=1)
    # Ajouts phase 3 : Kokoro (ONNX), vitesse de parole, threads CPU (§10 : 4 max).
    kokoro_model: Path = Path("models/kokoro-v1.0.onnx")
    kokoro_voices: Path = Path("models/voices-v1.0.bin")
    kokoro_voice: str = "ff_siwis"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    cpu_threads: int = Field(default=4, ge=1, le=16)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)  # ajout phase 11


class AgentConfig(_Section):
    follow_up_s: float = Field(default=6, ge=0)
    session_reset_min: float = Field(default=10, gt=0)
    thinking_sound_after_s: float = Field(default=1.2, ge=0)
    history_max_tokens: int = Field(default=6000, ge=500)  # ajout phase 3 : §6, ~6 000 tokens
    # Ajouts phase 6. Historique gardé sur disque (data/history.json) entre deux démarrages,
    # et `history_persist_h` heures sans interaction au lieu de session_reset_min (durée
    # modifiable à la voix). Au-delà de history_max_tokens, les plus anciens échanges sont
    # résumés par le LLM (history_summarize) au lieu d'être supprimés.
    history_persist: bool = True
    history_persist_h: float = Field(default=168, gt=0)
    history_summarize: bool = True
    # Fenêtre de suivi dans le bruit : une phrase dont la transcription est moins fiable que
    # ce seuil (0 à 1) est ignorée sans réponse ; puis le LLM juge si elle s'adresse à MIKATA.
    follow_up_min_confidence: float = Field(default=0.2, ge=0, le=1)
    follow_up_use_llm_gate: bool = True
    # Commande entière qui annule (dite après « MIKATA » pendant une réponse ou pendant la
    # fenêtre de suivi) : jamais envoyée au LLM.
    cancel_phrases: list[str] = Field(
        default_factory=lambda: [
            "stop", "arrête", "arrête-toi", "annule", "annule ça", "annule tout",
            "laisse tomber", "oublie", "oublie ça", "tais-toi", "chut", "silence",
            "c'est bon", "ça ira", "ça suffit", "c'est tout", "non", "non rien", "rien",
            "non merci", "pas grave",
        ]
    )  # fmt: skip
    # Ajouts phase 6b : arrêt vocal **sans** mot d'activation pendant que MIKATA parle. Une
    # phrase courte qui est entièrement l'une de voice_stop_phrases coupe la réponse en
    # cours ; toute autre parole est ignorée. Liste plus courte que cancel_phrases : un
    # « non » ou un « c'est bon » dit à quelqu'un d'autre ne doit pas couper MIKATA.
    voice_stop: bool = True
    voice_stop_phrases: list[str] = Field(
        default_factory=lambda: ["stop", "arrête", "arrête-toi", "tais-toi", "ça suffit", "chut"],
        min_length=1,
    )
    voice_stop_silence_ms: int = Field(default=160, ge=80, le=1000)  # fin de la phrase d'arrêt
    voice_stop_max_s: float = Field(default=2.0, gt=0, le=10)  # au-delà : pas un « stop »


class GuardConfig(_Section):
    allowed_dirs: list[Path] = Field(default_factory=lambda: [Path("~")], min_length=1)
    backup_days: int = Field(default=14, ge=1)
    shutdown_cancel_s: float = Field(default=5, ge=0)
    confirm_after_web: bool = False
    # Ajouts phase 5 : confirmation orale des commandes risquées (liste grise), délai de
    # réponse, taille maximale envoyée à la corbeille (au-delà, Windows supprimerait
    # définitivement : refus).
    confirm_risky: bool = True
    confirm_timeout_s: float = Field(default=6, gt=0)
    trash_max_mb: float = Field(default=4096, gt=0)


class UiaConfig(_Section):
    """Ajout phase 12a : pilotage fin des applications par l'arbre d'accessibilité Windows
    (lire les contrôles, cliquer par nom). L'état ambiant (présence vocale Discord...) est
    gardé en RAM seulement, jamais loggé."""

    enabled: bool = True  # false : les outils UIA répondent « désactivé »
    state_poll_s: float = Field(default=2, ge=0)  # état ambiant ; 0 = jamais rafraîchi
    max_controls: int = Field(default=200, ge=10, le=2000)  # contrôles listés par uia_window
    max_text_chars: int = Field(default=2000, ge=100, le=20000)  # texte lu, tronqué
    # Client d'accessibilité à l'écoute (UiaClientsAreListening) : les applications
    # Chromium/Electron (Discord...) exposent leur arbre complet sans drapeau de lancement.
    accessibility_client: bool = True


class LoggingConfig(_Section):
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"] = "INFO"
    file: Path = Path("data/logs/mikata.log")
    rotation_mb: float = Field(default=5, gt=0)
    retention_files: int = Field(default=5, ge=1)
    stats_interval_s: float = Field(default=60, ge=0)  # ajout phase 1 : ligne CPU (0 = jamais)
    actions_file: Path = Path("data/actions.log")  # ajout phase 5 : journal des actions (§7)

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class UiConfig(_Section):
    """Ajout phase 8 : interface (mascotte, réglages) dans un processus à part, reliée à
    MIKATA par un canal local (mikata/ui/protocol.py)."""

    enabled: bool = True  # canal local ouvert : mascotte et réglages utilisables
    mascot: bool = True  # lancer la mascotte au démarrage de MIKATA
    skin: str = "femme"  # dossier assets/mascot/<skin>/ (default : Mika, dessinée en code)
    bubble_s: float = Field(default=8, ge=1, le=120)  # bulle visible après la dernière phrase
    # Ajout phase 11 : profil = skin de la mascotte + voix. Changer de skin change la voix,
    # à chaud. Un skin sans profil parle avec tts.backend.
    profiles: dict[str, VoiceProfile] = Field(default_factory=dict)


class VoiceProfile(_Section):
    voice: VoiceEngine = "kokoro"
    voice_id: str | None = None  # ElevenLabs : identifiant de voix (pas un secret)
    voice_name: str = ""  # pour les messages et les réglages


class Secrets(BaseSettings):
    """Secrets lus dans les variables d'environnement puis dans `.env`."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8-sig",  # tolère le BOM ajouté par certains éditeurs Windows
        extra="ignore",
        frozen=True,
    )

    llm_api_key: SecretStr | None = None
    search_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None  # ajout phase 11

    @field_validator("llm_api_key", "search_api_key", "elevenlabs_api_key", mode="after")
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
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    uia: UiaConfig = Field(default_factory=UiaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
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
