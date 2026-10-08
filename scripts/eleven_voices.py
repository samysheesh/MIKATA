"""Voix ElevenLabs du compte : lister, chercher dans la bibliothèque, ajouter, essayer, et
choisir la voix d'un skin, sans écrire de code (ajout phase 11).

    uv run python scripts/eleven_voices.py list
    uv run python scripts/eleven_voices.py search --gender female --query "douce"
    uv run python scripts/eleven_voices.py add <public_owner_id> <voice_id> --name "Camille"
    uv run python scripts/eleven_voices.py try <voice_id> --text "Bonjour, je suis MIKATA."
    uv run python scripts/eleven_voices.py use femme <voice_id> --name "Camille"
    uv run python scripts/eleven_voices.py use default --kokoro

`list` : voix de ton compte (et les profils qui les utilisent). `search` : bibliothèque
publique (/v1/shared-voices), en français par défaut. `add` : copie une voix de la
bibliothèque dans ton compte (gratuit, limité en nombre selon l'abonnement). `try` : fait dire
une phrase (facturée, quelques crédits) dans data/eleven_essai_<voix>.wav. `use` : écrit le
profil `ui.profiles.<skin>` dans config.yaml (une ligne, le reste du fichier est gardé ;
ancienne version dans data/backups/) ; MIKATA l'applique au prochain choix du skin dans
l'onglet Skins, ou à son prochain démarrage.

La clé est lue dans .env (ELEVENLABS_API_KEY) et n'est jamais affichée.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRY_TEXT = "Bonjour, je suis MIKATA. Il fait vingt degrés, et ton rendez-vous est à quinze heures."


class VoiceError(RuntimeError):
    pass


def _client(config: Any, transport: Any = None) -> Any:
    import httpx2

    key = config.secrets.elevenlabs_api_key
    if key is None:
        raise VoiceError("ELEVENLABS_API_KEY absente de .env (voir .env.example).")
    return httpx2.Client(
        base_url=config.tts.elevenlabs.base_url,
        headers={"xi-api-key": key.get_secret_value()},
        timeout=httpx2.Timeout(30.0, connect=10.0),
        transport=transport,
    )


def _check(response: Any) -> Any:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status")
        raise VoiceError(f"ElevenLabs a refusé (HTTP {response.status_code}) : {detail or ''}")
    return response


def _labels(voice: dict[str, Any]) -> str:
    labels = voice.get("labels") or {}
    parts = [voice.get(k) or labels.get(k) for k in ("gender", "age", "accent", "language")]
    return ", ".join(str(p) for p in parts if p)


def list_voices(client: Any, config: Any) -> list[str]:
    voices = _check(client.get("/v1/voices")).json().get("voices", [])
    used: dict[str, list[str]] = {}
    for skin, profile in config.ui.profiles.items():
        if profile.voice == "elevenlabs":
            used.setdefault(profile.voice_id or config.tts.elevenlabs.voice_id, []).append(skin)
    lines = []
    for voice in sorted(voices, key=lambda v: str(v.get("name", "")).lower()):
        voice_id = voice.get("voice_id", "")
        skins = f"  <- profil(s) {', '.join(used[voice_id])}" if voice_id in used else ""
        lines.append(
            f"{voice.get('name', '?'):<24} {voice_id}  [{voice.get('category', '')}] "
            f"{_labels(voice)}{skins}"
        )
    return lines


def search_voices(client: Any, args: argparse.Namespace) -> list[str]:
    params: dict[str, Any] = {"page_size": args.limit, "language": args.language}
    if args.gender:
        params["gender"] = args.gender
    if args.query:
        params["search"] = args.query
    data = _check(client.get("/v1/shared-voices", params=params)).json()
    lines = []
    for voice in data.get("voices", []):
        description = str(voice.get("description") or "").replace("\n", " ")[:80]
        lines.append(
            f"{voice.get('name', '?'):<24} owner={voice.get('public_owner_id', '')} "
            f"voice={voice.get('voice_id', '')}  {_labels(voice)}"
            + (f"\n    {description}" if description else "")
            + (f"\n    écoute : {voice['preview_url']}" if voice.get("preview_url") else "")
        )
    return lines


def add_voice(client: Any, owner: str, voice_id: str, name: str) -> str:
    response = _check(client.post(f"/v1/voices/add/{owner}/{voice_id}", json={"new_name": name}))
    return str(response.json().get("voice_id", voice_id))


def try_voice(client: Any, config: Any, voice_id: str, text: str) -> Path:
    import numpy as np

    from mikata.audio.wavfile import write_wav

    cfg = config.tts.elevenlabs
    response = _check(
        client.post(
            f"/v1/text-to-speech/{voice_id}/stream",
            params={"output_format": cfg.output_format},
            json={"text": text, "model_id": cfg.model, "language_code": cfg.language},
        )
    )
    raw = response.content
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
    path = config.resolve(f"data/eleven_essai_{voice_id}.wav")
    write_wav(path, pcm, int(cfg.output_format.split("_")[1]))
    return path


def use_voice(config_path: Path, skin: str, profile: dict[str, Any]) -> None:
    from mikata.ui.settings_store import BACKUP_DIR, ConfigEditor, set_profile

    editor = ConfigEditor(config_path)
    editor.text = set_profile(editor.text, skin, profile)
    editor.save(config_path.parent / BACKUP_DIR)


def main(argv: Sequence[str] | None = None, transport: Any = None) -> int:
    from mikata.config import ConfigError, load_config, resolve_config_path
    from mikata.ui.settings_store import EditorError

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="voix de ton compte")
    search = sub.add_parser("search", help="chercher dans la bibliothèque publique")
    search.add_argument("--language", default="fr")
    search.add_argument("--gender", choices=("female", "male", "neutral"))
    search.add_argument("--query", help="mots de la description (ex. « douce », « radio »)")
    search.add_argument("--limit", type=int, default=20)
    add = sub.add_parser("add", help="ajouter une voix de la bibliothèque à ton compte")
    add.add_argument("owner", help="public_owner_id (donné par search)")
    add.add_argument("voice_id")
    add.add_argument("--name", required=True, help="nom de la voix dans ton compte")
    essai = sub.add_parser("try", help="faire dire une phrase (facturée)")
    essai.add_argument("voice_id")
    essai.add_argument("--text", default=TRY_TEXT)
    use = sub.add_parser("use", help="voix d'un skin (ui.profiles de config.yaml)")
    use.add_argument("skin", help="dossier du skin (assets/mascot/<skin>) ou default")
    use.add_argument("voice_id", nargs="?")
    use.add_argument("--name", default="", help="nom affiché (ex. « Lucie »)")
    use.add_argument("--kokoro", action="store_true", help="voix locale Kokoro")
    use.add_argument("--piper", action="store_true", help="voix locale Piper")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    try:
        config = load_config(config_path)
        if args.command == "use":
            if args.kokoro or args.piper:
                profile: dict[str, Any] = {"voice": "kokoro" if args.kokoro else "piper"}
            elif args.voice_id:
                profile = {"voice": "elevenlabs", "voice_id": args.voice_id}
                if args.name:
                    profile["voice_name"] = args.name
            else:
                parser.error("use : donne un voice_id, ou --kokoro / --piper")
            use_voice(config_path, args.skin, profile)
            print(f"Profil « {args.skin} » : {profile} (écrit dans {config_path.name}).")
            print("Choisis ce skin dans l'onglet Skins pour l'entendre tout de suite.")
            return 0
        with _client(config, transport) as client:
            if args.command == "list":
                lines = list_voices(client, config)
            elif args.command == "search":
                lines = search_voices(client, args) or ["Aucune voix trouvée."]
            elif args.command == "add":
                voice_id = add_voice(client, args.owner, args.voice_id, args.name)
                lines = [
                    f"Voix « {args.name} » ajoutée : {voice_id}",
                    f"Pour un skin : uv run python scripts/eleven_voices.py use <skin> {voice_id} "
                    f'--name "{args.name}"',
                ]
            else:
                lines = [f"Écoute : {try_voice(client, config, args.voice_id, args.text)}"]
    except (ConfigError, VoiceError, EditorError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # réseau
        print(f"ElevenLabs injoignable : {type(exc).__name__}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.exit(main())
