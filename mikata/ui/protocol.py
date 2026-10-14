"""Canal local entre MIKATA et son interface (ajout phase 8).

L'interface (mascotte, réglages) tourne dans un **processus à part** : elle ne peut ni
ralentir la boucle audio ni faire tomber MIKATA. Les deux parlent par une connexion TCP sur
127.0.0.1 (port choisi par le système), en lignes JSON :

- MIKATA écrit `data/ui_channel.json` au démarrage : port, jeton aléatoire, pid. Seul un
  processus qui peut lire ce fichier (ta session) connaît le jeton.
- L'interface se connecte et envoie d'abord `{"type": "hello", "token": ...}` ; sans le bon
  jeton, la connexion est fermée.
- MIKATA → interface : `state`, `wake`, `heard`, `say`, `result`, `error`, `mic`, `config`, `voice`,
  `tasks`, `open_settings`, `show_mascot`, `bye` (voir `MIKATA_EVENTS`).
- Interface → MIKATA : `reload_config` (après écriture de config.yaml), `pause`, `restart`
  (redémarrer MIKATA : arrêt propre avec le code 75, relancé par la supervision ou par lui-même),
  `list_tasks`, `close_task` (onglet « En cours » des réglages : ce qui tourne en arrière-plan,
  fermable — ajout phase 13f), `browser` (phase 15a), `app` (phase 17e : une appli tenue
  en arrière-plan, `name` et `action` show | hide | pause | close).

Ce module n'importe ni Qt ni asyncio : il sert aux deux côtés et aux tests.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
CHANNEL_FILE = "data/ui_channel.json"
MAX_LINE_BYTES = 64 * 1024  # une ligne plus longue est refusée (texte tronqué avant envoi)
MAX_TEXT_CHARS = 2000

# Messages de MIKATA vers l'interface.
MIKATA_EVENTS = (
    "hello",  # version, état courant, pause
    "state",  # state : idle | listening | transcribing | thinking | acting | speaking | follow_up
    "wake",  # score, voice : mot d'activation détecté
    "heard",  # id, text : ce que MIKATA a compris
    "say",  # id, text : une phrase dite
    "result",  # id, outcome : success | ok | error | ignored | cancelled
    "error",  # message
    "mic",  # status : ok | paused | busy
    "config",  # applied, restart, error : réponse à reload_config
    "voice",  # event (fallback | resume | profile), engine, message : voix (phase 11)
    "tasks",  # tasks : [{id, label, running, visible, tabs}] — arrière-plan (13f, 15a) ;
    # phase 17e : id « app:<nom> », avec driven, media (playing | paused | ""), now
    "open_settings",
    "show_mascot",
    "bye",  # MIKATA s'arrête : l'interface se ferme
)
# Commandes de l'interface vers MIKATA.
UI_COMMANDS = (
    "hello",
    "reload_config",
    "pause",
    "restart",
    "list_tasks",
    "close_task",
    "browser",  # phase 15a : action (show | hide | pause | close_tab | start), tab
    "app",  # phase 17e : name, action (show | hide | pause | close)
)


class ProtocolError(ValueError):
    """Ligne illisible ou message inconnu."""


def encode(message: dict[str, Any]) -> bytes:
    """Une ligne JSON (UTF-8, terminée par \\n). Les textes trop longs sont tronqués."""
    clean = {k: (v[:MAX_TEXT_CHARS] if isinstance(v, str) else v) for k, v in message.items()}
    line = json.dumps(clean, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(line) >= MAX_LINE_BYTES:
        raise ProtocolError("message trop long")
    return line + b"\n"


def decode(line: bytes | str, allowed: tuple[str, ...]) -> dict[str, Any]:
    """Message d'une ligne ; `allowed` : types acceptés de ce côté."""
    try:
        message = json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"ligne illisible : {exc}") from exc
    if not isinstance(message, dict) or message.get("type") not in allowed:
        raise ProtocolError(f"message inattendu : {str(message)[:80]}")
    return message


@dataclass(frozen=True)
class Channel:
    port: int
    token: str
    pid: int

    @classmethod
    def new(cls, port: int) -> Channel:
        return cls(port=port, token=secrets.token_urlsafe(24), pid=os.getpid())


def write_channel(path: Path, channel: Channel) -> None:
    """Écriture atomique (l'interface peut lire le fichier à tout moment)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    data = {"port": channel.port, "token": channel.token, "pid": channel.pid}
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def read_channel(path: Path) -> Channel | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Channel(port=int(data["port"]), token=str(data["token"]), pid=int(data["pid"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
