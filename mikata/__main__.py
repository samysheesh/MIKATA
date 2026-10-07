"""Point d'entrée : `python -m mikata` (ou `pythonw -m mikata` sans console)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from mikata import __version__
from mikata.app import App
from mikata.config import (
    Config,
    ConfigError,
    LoggingConfig,
    StartupError,
    load_config,
    resolve_config_path,
)
from mikata.logging_setup import setup_logging, shutdown_logging
from mikata.ui.tray import TrayIcon

if TYPE_CHECKING:
    from mikata.agent.loop import Agent
    from mikata.audio.player import Speaker
    from mikata.tools.base import ToolContext

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_STARTUP = 3  # matériel indisponible (GPU, DLL CUDA...)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mikata", description="Assistant vocal MIKATA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="chemin de config.yaml (défaut : $MIKATA_CONFIG, sinon celui du projet)",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="valide la configuration puis quitte (code 0 si valide, 2 sinon)",
    )
    parser.add_argument(
        "--no-tray", action="store_true", help="sans icône de notification (arrêt par Ctrl+C)"
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="sans micro, sortie ni mot d'activation"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="sans LLM ni voix : la commande transcrite est seulement journalisée (phase 2)",
    )
    parser.add_argument(
        "--ask",
        metavar="TEXTE",
        default=None,
        help="pose une question au LLM, la dit dans le casque et quitte (sans micro)",
    )
    parser.add_argument("--version", action="version", version=f"MIKATA {__version__}")
    return parser.parse_args(argv)


def log_config_summary(config: Config) -> None:
    logger.info("Configuration : {}", config.base_dir)
    logger.info(
        "Audio : entrée « {} », sortie « {} »",
        config.audio.input_device,
        config.audio.output_device,
    )
    logger.info("Mot d'activation : {} (seuil {})", config.wake.model, config.wake.threshold)
    logger.info(
        "STT : {} sur {} ({})", config.stt.model, config.stt.device, config.stt.compute_type
    )
    logger.info("TTS : {}", config.tts.backend)
    if config.llm.is_placeholder:
        logger.warning("LLM non renseigné dans config.yaml (llm.base_url / llm.model).")
    else:
        logger.info(
            "LLM : {} @ {} (réflexion par défaut : {})",
            config.llm.model,
            config.llm.base_url,
            config.llm.reasoning_effort or "choix du fournisseur",
        )
    logger.info("Recherche web : {}", config.web.search_provider)
    for name, value in (
        ("LLM_API_KEY", config.secrets.llm_api_key),
        ("SEARCH_API_KEY", config.secrets.search_api_key),
    ):
        logger.info("{} : {}", name, "définie" if value is not None else "absente (.env)")


WHISPER_DIR = "models/whisper"  # cache des modèles faster-whisper (scripts/download_models.py)


TTS_CACHE_DIR = "data/tts_cache"
USAGE_FILE = "data/llm_usage.json"  # cumul des tokens et du coût
MEMORY_FILE = "data/memory.json"  # faits mémorisés (remember / forget)


def build_speaker(config: Config, player: Any) -> Speaker:
    """Voix (Piper ou Kokoro) + cache de phrases + file de phrases, non chargés."""
    from mikata.audio.player import Speaker
    from mikata.tts.base import create_tts
    from mikata.tts.cache import PhraseCache

    tts, voice_id = create_tts(config)
    return Speaker(tts, player, PhraseCache(tts, config.resolve(TTS_CACHE_DIR), voice_id))


def build_tool_context(config: Config) -> ToolContext:
    """Services des outils : bureau Windows, volume, notifications, web, mémoire ;
    phase 5 : garde-fou, clavier et souris, lancement de commandes."""
    from mikata.agent.guard import Guard
    from mikata.agent.memory import LongTermMemory
    from mikata.tools.base import ToolContext
    from mikata.tools.shell import SubprocessRunner
    from mikata.tools.web import create_http_client

    ctx = ToolContext(
        config,
        http=create_http_client(),
        memory=LongTermMemory(config.resolve(MEMORY_FILE)),
        guard=Guard(config, folders=_known_folders()),  # phase 5 : garde-fou
        shell=SubprocessRunner(),
    )
    key = config.secrets.search_api_key
    ctx.search_api_key = key.get_secret_value() if key is not None else None
    if sys.platform == "win32":
        from mikata.tools.desktop import WindowsDesktop
        from mikata.tools.input import PynputInput
        from mikata.tools.media import PycawAudio
        from mikata.ui.notify import ToastNotifier

        ctx.desktop, ctx.audio, ctx.notifier = WindowsDesktop(), PycawAudio(), ToastNotifier()
        ctx.input = PynputInput()
    else:
        logger.warning("Hors Windows : outils du bureau, du volume et des notifications inactifs.")
    return ctx


def _known_folders() -> dict[str, Path]:
    """Vrais dossiers Bureau, Documents... (Windows) ; vide ailleurs ou en cas d'échec."""
    if sys.platform != "win32":
        return {}
    from mikata.tools.desktop import known_folders

    try:
        return known_folders()
    except Exception as exc:
        logger.warning(
            "Dossiers personnels introuvables ({}) : chemins sous le dossier perso.", exc
        )
        return {}


def build_agent(config: Config, phrases: Any = None, *, tools: bool = True) -> Agent:
    """Client LLM + outils + boucle agent. Refuse (ConfigError) sans clé API ou endpoint."""
    from mikata.agent.loop import Agent
    from mikata.llm.client import OpenAICompatClient, check_llm_config
    from mikata.llm.usage import UsageTracker

    api_key = check_llm_config(config.llm, config.secrets)
    usage = UsageTracker(config.llm, config.resolve(USAGE_FILE))
    llm = OpenAICompatClient(config.llm, api_key)
    if not tools:
        return Agent(llm, config, phrases=phrases, usage=usage)

    from mikata.tools.apps import open_apps
    from mikata.tools.registry import ToolRegistry

    ctx = build_tool_context(config)
    registry = ToolRegistry(ctx)
    if config.secrets.search_api_key is None:
        logger.warning("SEARCH_API_KEY absente (.env) : web_search répondra « indisponible ».")
    desktop = ctx.desktop

    async def running_apps() -> list[str]:
        assert desktop is not None
        return open_apps(await ctx.run_blocking(desktop.windows))

    return Agent(
        llm,
        config,
        phrases=phrases,
        tools=registry,
        long_memory=ctx.memory,
        usage=usage,
        open_apps=running_apps if desktop is not None else None,
    )


def find_output_device(config: Config) -> Any:
    from mikata.audio.devices import collect_devices, default_device_index, find_device

    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError(f"Bibliothèque audio PortAudio introuvable : {exc}") from exc
    return find_device(
        collect_devices(sd),
        config.audio.output_device,
        "output",
        default_device_index(sd, "output"),
    )


def build_app(config: Config, *, tray: bool, audio: bool, llm: bool = True) -> App:
    """Assemble l'orchestrateur avec les composants réels (imports lourds paresseux)."""
    if llm and audio:  # avant de charger quoi que ce soit : refus immédiat et clair
        from mikata.llm.client import check_llm_config

        check_llm_config(config.llm, config.secrets)
    icon = TrayIcon() if tray else None
    if not audio:
        return App(config, tray=icon)

    from mikata.audio.capture import AudioCapture
    from mikata.audio.devices import collect_devices, default_device_index, find_device
    from mikata.audio.earcons import Earcons
    from mikata.audio.player import AudioPlayer
    from mikata.governor import CpuMonitor, lower_priority
    from mikata.stt.whisper_local import WhisperSTT
    from mikata.ui.hotkeys import HotkeyListener
    from mikata.vad.silero import SileroVAD
    from mikata.wake.oww import OpenWakeWordDetector

    lower_priority()
    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError(f"Bibliothèque audio PortAudio introuvable : {exc}") from exc

    devices = collect_devices(sd)
    mic = find_device(
        devices, config.audio.input_device, "input", default_device_index(sd, "input")
    )
    headset = find_device(
        devices, config.audio.output_device, "output", default_device_index(sd, "output")
    )
    if config.audio.input_channel is not None and config.audio.input_channel > min(
        mic.max_input_channels, 8
    ):
        raise ConfigError(
            f"audio.input_channel = {config.audio.input_channel} : « {mic.name} » n'a que "
            f"{mic.max_input_channels} canal(aux)."
        )
    player = AudioPlayer(headset)
    verifier = config.resolve(config.wake.verifier) if config.wake.verifier else None
    detector = OpenWakeWordDetector(config.resolve(config.wake.model), verifier)
    logger.info("Mot d'activation chargé : {}", detector.name)
    vad = SileroVAD(config.resolve(config.vad.model), config.vad.threshold)
    speaker = build_speaker(config, player) if llm else None
    agent = build_agent(config, speaker.cache if speaker else None) if llm else None
    if not llm:
        logger.warning("Mode --no-llm : les commandes sont transcrites et journalisées seulement.")
    app = App(
        config,
        tray=icon,
        audio=AudioCapture(mic, channel=config.audio.input_channel),
        detector=detector,
        player=player,
        earcons=Earcons(
            config.resolve("assets/sounds"), player.sample_rate, config.audio.earcon_volume
        ),
        hotkeys=HotkeyListener(config.wake.hotkey) if config.wake.hotkey else None,
        cpu=CpuMonitor(),
        vad=vad,
        stt=WhisperSTT(config.stt, download_root=config.resolve(WHISPER_DIR)),
        agent=agent,
        speaker=speaker,
    )
    if agent is not None and agent.tools is not None:
        agent.tools.ctx.announce = app.announce  # minuteurs et rappels dits dans le casque
    return app


async def ask(config: Config, question: str, *, audio: bool) -> str:
    """Un tour de conversation sans micro : question -> LLM -> voix dans le casque."""
    import time

    from mikata.audio.player import AudioPlayer

    loop = asyncio.get_running_loop()
    speaker = None
    if audio:
        player = AudioPlayer(find_output_device(config))
        player.start(loop)
        speaker = build_speaker(config, player)
        logger.info("Chargement de la voix ({})...", config.tts.backend)
        await asyncio.to_thread(speaker.load)
    agent = build_agent(config, speaker.cache if speaker else None)
    try:
        # Comme au mot d'activation : connexion TLS ouverte et index des applis prêt avant
        # que la commande parte (le temps mesuré est celui d'un vrai usage).
        await agent.prewarm()
        start = time.monotonic()
        if speaker is None:
            from mikata.agent.loop import SilentSpeech

            result = await agent.respond(question, SilentSpeech(), label="--ask")
            first_sound = None
        else:
            stream = speaker.open("ask")
            result = await agent.respond(question, stream, label="--ask")
            stream.close()
            await stream.wait()
            first_sound = stream.first_sound_at
        token_ms = (result.first_token_at - start) * 1000 if result.first_token_at else None
        sound_ms = (first_sound - start) * 1000 if first_sound else None
        logger.info(
            "--ask : premier token {} ms, premier son {} ms, réflexion {} ; chemin : {} ; "
            "issue : {}.",
            f"{token_ms:.0f}" if token_ms is not None else "—",
            f"{sound_ms:.0f}" if sound_ms is not None else "—",
            result.effort_label,
            result.path,
            result.outcome,
        )
        return result.answer
    finally:
        await agent.aclose()
        if speaker is not None:
            await speaker.aclose()
            speaker.player.close()  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args.config)
    # Logs provisoires (valeurs par défaut) pour tracer une éventuelle erreur de config,
    # y compris sous pythonw.exe où il n'y a pas de console.
    setup_logging(LoggingConfig(), base_dir=config_path.parent)
    try:
        logger.info("MIKATA {} — démarrage (Python {})", __version__, sys.version.split()[0])
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            logger.error("{}", exc)
            return EXIT_CONFIG

        log_file = setup_logging(config.logging, base_dir=config.base_dir)
        logger.info("Logs : {}", log_file)
        log_config_summary(config)
        if args.check_config:
            logger.success("Configuration valide : {}", config_path)
            return EXIT_OK

        if args.ask is not None:
            try:
                answer = asyncio.run(ask(config, args.ask, audio=not args.no_audio))
            except ConfigError as exc:
                logger.error("{}", exc)
                return EXIT_CONFIG
            print(answer)
            return EXIT_OK

        try:
            app = build_app(
                config, tray=not args.no_tray, audio=not args.no_audio, llm=not args.no_llm
            )
        except ConfigError as exc:
            logger.error("{}", exc)
            return EXIT_CONFIG
        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            logger.info("Interruption clavier (Ctrl+C).")
        except StartupError as exc:
            logger.error("{}", exc)
            return EXIT_STARTUP
        return EXIT_OK
    except Exception:
        logger.exception("Erreur fatale.")
        return EXIT_ERROR
    finally:
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
