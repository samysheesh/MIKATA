"""Données de la fenêtre de réglages (ajout phase 8), sans Qt : testables.

- `ConfigEditor` : modifie des valeurs de config.yaml **ligne par ligne**. ruamel.yaml
  situe chaque clé (ligne, colonne) ; seule la valeur de cette ligne est remplacée. Le reste
  du fichier (commentaires, ordre, listes sur plusieurs lignes, `null`) reste identique à
  l'octet près, ce qu'un aller-retour YAML complet ne garantit pas. Le texte obtenu est
  validé par `Config` avant écriture ; l'ancien fichier part dans data/backups/.
- `SETTINGS` : réglages exposés, et s'ils s'appliquent tout de suite (`live`) ou au
  prochain démarrage (cohérent avec `app.LIVE_SETTINGS`, vérifié par les tests).
- Secrets : lus et écrits dans .env, jamais affichés en clair (`mask`).
- Coût LLM cumulé (data/llm_usage.json) et journal en direct (`LogTail`).
- Ajouts phase 11 : compteur de crédits ElevenLabs (data/elevenlabs_usage.json, écrit par
  MIKATA) et profils de personnage (`set_profile` : une ligne `ui.profiles.<skin>`).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mikata.config import Config

BACKUP_DIR = "data/backups"  # = agent.guard.BACKUP_DIR : même purge (guard.backup_days)
USAGE_FILE = "data/llm_usage.json"


class EditorError(ValueError):
    """Modification impossible (clé introuvable, valeur sur plusieurs lignes, config
    invalide) ; message en français."""


@dataclass(frozen=True)
class SettingSpec:
    key: str  # « section.clé » de config.yaml
    label: str
    live: bool  # True : appliqué tout de suite ; False : au prochain démarrage
    help: str = ""


SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec("audio.input_device", "Micro", False, "sous-chaîne du nom du périphérique"),
    SettingSpec("audio.output_device", "Sortie audio", False, "« default » : sortie de Windows"),
    SettingSpec("wake.threshold", "Seuil du mot d'activation", True, "voir eval_wakeword.py"),
    SettingSpec("wake.verifier_threshold", "Seuil du vérificateur de voix", True),
    SettingSpec("llm.reasoning_effort", "Réflexion par défaut", True, "none, low, medium, high"),
    SettingSpec("agent.follow_up_s", "Fenêtre de suivi (s)", True, "0 : désactivée"),
    SettingSpec("agent.history_persist_h", "Durée de l'historique (h)", True),
    SettingSpec("ui.skin", "Skin de la mascotte", True),
    SettingSpec("ui.bubble_s", "Bulle visible après la dernière phrase (s)", True),
    SettingSpec("ui.mascot", "Mascotte au démarrage de MIKATA", False),
    SettingSpec(
        "guard.full_access",
        "Accès complet",
        False,
        "plus aucune confirmation après une page web ou un fichier lu",
    ),
)
SPECS = {spec.key: spec for spec in SETTINGS}

SECRET_NAMES = {
    "LLM_API_KEY": "Clé du LLM",
    "SEARCH_API_KEY": "Clé de recherche web",
    "ELEVENLABS_API_KEY": "Clé ElevenLabs",  # ajout phase 11
}
_YAML_WORDS = {"true", "false", "yes", "no", "on", "off", "null", "~", "y", "n"}
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")


def yaml_scalar(value: Any) -> str:
    """Valeur YAML d'une ligne, relue à l'identique par PyYAML (pas d'exposant : PyYAML lit
    « 1e-05 » comme du texte)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    text = str(value)
    if _PLAIN.match(text) and text.lower() not in _YAML_WORDS:
        return text
    return json.dumps(text, ensure_ascii=False)


def _comment_start(line: str, after: int) -> int | None:
    """Colonne du « # » qui commence un commentaire (hors guillemets) après `after`."""
    quote = None
    for i in range(after, len(line)):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            return i
    return None


def _ruamel_doc(text: str) -> Any:
    from ruamel.yaml import YAML

    return YAML(typ="rt").load(text)


def set_value(text: str, key: str, value: Any) -> str:
    """`text` (config.yaml) avec « section.clé » = `value`, le reste inchangé."""
    section, _, name = key.partition(".")
    if not section or not name:
        raise EditorError(f"clé « {key} » : « section.clé » attendu")
    doc = _ruamel_doc(text) or {}
    lines = text.splitlines(keepends=True)
    rendered = yaml_scalar(value)
    if section not in doc:  # section absente : ajoutée à la fin
        tail = "" if not lines or lines[-1].endswith("\n") else "\n"
        return f"{text}{tail}{section}:\n  {name}: {rendered}\n"
    block = doc[section]
    if block is None or not hasattr(block, "lc"):
        raise EditorError(f"section « {section} » : une liste de clés est attendue")
    if name not in block:  # clé absente : ajoutée à la fin de la section
        start = doc.lc.key(section)[0]
        end = start + 1
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if lines[i][:1] not in (" ", "\t", "\n", "\r", "#") and stripped:
                break
            if stripped and not stripped.startswith("#"):
                end = i + 1
        indent = block.lc.key(next(iter(block)))[1] if len(block) else 2
        lines.insert(end, f"{' ' * indent}{name}: {rendered}\n")
        return "".join(lines)
    row, col = block.lc.key(name)
    line = lines[row]
    newline = line[len(line.rstrip("\r\n")) :]
    body = line[: len(line) - len(newline)]
    colon = body.index(":", col + len(name)) + 1
    if row + 1 < len(lines):  # valeur sur plusieurs lignes : refusée (pas un réglage simple)
        nxt = lines[row + 1]
        continued = nxt.strip() and not nxt.lstrip().startswith("#")
        if continued and len(nxt) - len(nxt.lstrip()) > col:
            raise EditorError(f"« {key} » s'étend sur plusieurs lignes : modifie-la à la main")
    comment_at = _comment_start(body, colon)
    head = f"{body[:colon]} {rendered}"
    if comment_at is None:
        return "".join([*lines[:row], head + newline, *lines[row + 1 :]])
    pad = max(1, comment_at - len(head))
    return "".join(
        [*lines[:row], head + " " * pad + body[comment_at:] + newline, *lines[row + 1 :]]
    )


def flow_mapping(values: dict[str, Any]) -> str:
    """{voice: elevenlabs, voice_id: abc} : un profil sur une ligne."""
    return "{" + ", ".join(f"{k}: {yaml_scalar(v)}" for k, v in values.items()) + "}"


def set_profile(text: str, skin: str, profile: dict[str, Any] | None) -> str:
    """`text` (config.yaml) avec le profil `ui.profiles.<skin>` (une ligne, en style
    `{clé: valeur}`) remplacé, ajouté, ou retiré si `profile` est None. Le reste du fichier
    est inchangé."""
    if not _PLAIN.match(skin):
        raise EditorError(f"nom de skin « {skin} » invalide")
    doc = _ruamel_doc(text) or {}
    lines = text.splitlines(keepends=True)
    ui = doc.get("ui")
    profiles = ui.get("profiles") if ui is not None and hasattr(ui, "lc") else None
    rendered = flow_mapping(profile) if profile is not None else None
    if profiles is None or not hasattr(profiles, "lc"):
        if rendered is None:
            return text
        if ui is not None and "profiles" in ui:
            raise EditorError("ui.profiles n'est pas une liste de profils : modifie-la à la main")
        indent = ui.lc.key(next(iter(ui)))[1] if ui is not None and len(ui) else 2
        added = set_value(text, "ui.profiles", "PLACEHOLDER")
        return added.replace(
            f"{' ' * indent}profiles: PLACEHOLDER\n",
            f"{' ' * indent}profiles:\n{' ' * (indent + 2)}{skin}: {rendered}\n",
        )
    if skin in profiles:
        row, col = profiles.lc.key(skin)
        line = lines[row]
        nxt = lines[row + 1] if row + 1 < len(lines) else ""
        continued = nxt.strip() and not nxt.lstrip().startswith("#")
        if continued and len(nxt) - len(nxt.lstrip()) > col:
            raise EditorError(f"profil « {skin} » sur plusieurs lignes : à la main")
        if rendered is None:
            return "".join([*lines[:row], *lines[row + 1 :]])
        newline = line[len(line.rstrip("\r\n")) :]
        body = line[: len(line) - len(newline)]
        colon = body.index(":", col + len(skin)) + 1
        comment_at = _comment_start(body, colon)
        comment = f"  {body[comment_at:]}" if comment_at is not None else ""
        lines[row] = f"{body[:colon]} {rendered}{comment}{newline}"
        return "".join(lines)
    if rendered is None:
        return text
    last_row, col = profiles.lc.key(next(iter(profiles))) if len(profiles) else (None, 4)
    for name in profiles:
        last_row = max(last_row or 0, profiles.lc.key(name)[0])
    if last_row is None:
        raise EditorError("ui.profiles vide : ajoute le profil à la main")
    lines.insert(last_row + 1, f"{' ' * col}{skin}: {rendered}\n")
    return "".join(lines)


def _format_error(exc: ValidationError) -> str:
    return "\n".join(
        f"{'.'.join(str(p) for p in err['loc'])} : {err['msg']}" for err in exc.errors()
    )


class ConfigEditor:
    def __init__(self, path: Path) -> None:
        self.path = path
        raw = path.read_bytes()
        self._bom = raw.startswith(b"\xef\xbb\xbf")
        self.original = raw.decode("utf-8-sig")
        self.text = self.original

    def _data(self, text: str) -> dict[str, Any]:
        data = yaml.safe_load(io.StringIO(text)) or {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str) -> Any:
        """Valeur du fichier en cours d'édition, ou la valeur par défaut de MIKATA."""
        section, _, name = key.partition(".")
        block = self._data(self.text).get(section) or {}
        if name in block:
            return block[name]
        return getattr(getattr(Config(), section), name)

    def set(self, key: str, value: Any) -> None:
        if key.split(".")[-1] in ("llm_api_key", "search_api_key", "elevenlabs_api_key"):
            raise EditorError("les secrets vont dans .env, jamais dans config.yaml")
        self.text = set_value(self.text, key, value)

    def changed(self) -> list[str]:
        before, after = self._data(self.original), self._data(self.text)
        keys = set()
        for section in set(before) | set(after):
            a, b = before.get(section) or {}, after.get(section) or {}
            keys |= {f"{section}.{k}" for k in set(a) | set(b) if a.get(k) != b.get(k)}
        return sorted(keys)

    def validate(self) -> str | None:
        """Message d'erreur si le texte modifié n'est pas une configuration valide."""
        try:
            Config.model_validate(self._data(self.text))
        except yaml.YAMLError as exc:
            return f"YAML invalide : {exc}"
        except ValidationError as exc:
            return _format_error(exc)
        return None

    def save(self, backup_root: Path | None = None) -> Path | None:
        """Écrit config.yaml (atomique) après l'avoir sauvegardé ; renvoie la sauvegarde."""
        error = self.validate()
        if error is not None:
            raise EditorError(error)
        backup = None
        if backup_root is not None:
            backup = backup_root / f"{datetime.now():%Y%m%d-%H%M%S-%f}-config" / self.path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, backup)
        data = ("\ufeff" if self._bom else "") + self.text
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_bytes(data.encode("utf-8"))
        tmp.replace(self.path)
        self.original = self.text
        return backup


# -- secrets ---------------------------------------------------------------------------------


def mask(value: str | None) -> str:
    """« ••••••••a1b2 » : jamais la clé en clair (les 4 derniers caractères si elle est
    longue, pour la reconnaître)."""
    if not value:
        return "non définie"
    return "•" * 8 + (value[-4:] if len(value) >= 16 else "")


def read_secrets(env_path: Path) -> dict[str, str]:
    """Nom -> valeur masquée (fichier .env, sinon variable d'environnement)."""
    from dotenv import dotenv_values

    values = dotenv_values(env_path) if env_path.is_file() else {}
    shown = {}
    for name in SECRET_NAMES:
        value = values.get(name) or os.environ.get(name)
        source = "" if values.get(name) or not value else " (variable d'environnement)"
        shown[name] = mask(value) + source
    return shown


def write_secret(env_path: Path, name: str, value: str) -> None:
    """Remplace un secret dans .env (créé si besoin). La valeur n'est jamais journalisée."""
    from dotenv import set_key

    if name not in SECRET_NAMES:
        raise EditorError(f"secret inconnu : {name}")
    value = value.strip()
    if not value or any(c in value for c in "\r\n\"'# "):
        raise EditorError("clé invalide (vide, ou espaces, guillemets ou # dedans)")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(exist_ok=True)
    set_key(str(env_path), name, value, quote_mode="never")


# -- coût et journal -------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageSummary:
    cost_usd: float
    commands: int
    since: str
    credit_usd: float

    @property
    def remaining_usd(self) -> float:
        return self.credit_usd - self.cost_usd


def read_usage(path: Path, credit_usd: float) -> UsageSummary:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return UsageSummary(
        cost_usd=float(data.get("cost_usd", 0.0)),
        commands=int(data.get("commands", 0)),
        since=str(data.get("since", "")),
        credit_usd=credit_usd,
    )


VOICE_USAGE_FILE = "data/elevenlabs_usage.json"  # = tts.credits.USAGE_FILE


def voice_usage_text(path: Path) -> tuple[str, float | None]:
    """Compteur ElevenLabs pour l'onglet « Coût et clés » : (texte, part restante du
    plafond entre 0 et 1, None si inconnue)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("used") is None:
        return "ElevenLabs : pas encore de relevé (MIKATA le fait au démarrage).", None
    used, cap = float(data["used"]), float(data.get("cap") or 0)
    limit = float(data.get("limit") or 0)
    remaining = float(data.get("remaining") or 0)
    measured = data.get("rate_measured")
    rate = float(data.get("rate") or 0)
    rate_text = (
        f"coût mesuré {rate:.3f} crédit par caractère"
        if measured is not None
        else f"coût pas encore mesuré (estimé {rate:.3f} crédit par caractère)"
    )
    reset = data.get("reset_unix")
    reset_text = f", renouvellement le {datetime.fromtimestamp(reset):%d/%m}" if reset else ""
    state = data.get("state")
    if state == "kokoro":
        reason = data.get("reason") or "indisponible"
        state_text = f"Voix de base (Kokoro) en ce moment : {reason}."
    else:
        state_text = "Voix ElevenLabs active."
    answers = data.get("answers_left")
    lines = [
        f"ElevenLabs : {used:.0f} crédits utilisés ce mois sur {limit:.0f} "
        f"(plafond choisi : {cap:.0f}){reset_text}",
        f"Crédits restants utilisables : {remaining:.0f} — environ "
        f"{answers if answers is not None else '?'} réponses de 150 caractères ; {rate_text}.",
        state_text,
    ]
    share = remaining / cap if cap else 0.0
    return "\n".join(lines), min(max(share, 0.0), 1.0)


class LogTail:
    """Nouvelles lignes d'un fichier de log ; suit la rotation (fichier raccourci)."""

    def __init__(self, path: Path, backlog_bytes: int = 32 * 1024) -> None:
        self.path = path
        self._pos: int | None = None
        self._backlog = backlog_bytes
        self._partial = b""

    def poll(self) -> list[str]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if self._pos is None:
            self._pos = max(0, size - self._backlog)
            if self._pos:  # au milieu d'une ligne : on repart à la suivante
                with self.path.open("rb") as file:
                    file.seek(self._pos - 1)
                    self._pos += len(file.readline()) - 1
        if size < self._pos:  # rotation : nouveau fichier
            self._pos, self._partial = 0, b""
        if size == self._pos:
            return []
        with self.path.open("rb") as file:
            file.seek(self._pos)
            chunk = file.read(size - self._pos)
        self._pos = size
        data = self._partial + chunk
        *complete, self._partial = data.split(b"\n")
        return [line.decode("utf-8", errors="replace").rstrip("\r") for line in complete]


def list_audio_devices() -> tuple[list[str], list[str], str | None]:
    """(entrées, sorties, erreur) : noms des périphériques WASAPI (ou tous hors Windows)."""
    try:
        import sounddevice as sd

        from mikata.audio.devices import collect_devices
    except (ImportError, OSError) as exc:
        return [], [], f"audio indisponible : {exc}"
    devices = collect_devices(sd)
    wasapi = [d for d in devices if "WASAPI" in d.host_api] or devices
    inputs = sorted({d.name for d in wasapi if d.max_input_channels > 0})
    outputs = sorted({d.name for d in wasapi if d.max_output_channels > 0})
    return inputs, outputs, None
