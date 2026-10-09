"""Point d'entrée : `python -m mikata` (ou `pythonw -m mikata` sans console)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Sequence
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
EXIT_RESTART = 75  # « Redémarrer » dans l'interface (cf. supervise.RESTART_CODE)


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
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="relance MIKATA s'il plante (démarrage automatique, scripts/install_autostart.ps1)",
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
    from mikata.tts.router import choose_voice

    logger.info("TTS : {} (skin « {} »)", choose_voice(config).describe, config.ui.skin)
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
        ("ELEVENLABS_API_KEY", config.secrets.elevenlabs_api_key),
    ):
        logger.info("{} : {}", name, "définie" if value is not None else "absente (.env)")


WHISPER_DIR = "models/whisper"  # cache des modèles faster-whisper (scripts/download_models.py)


TTS_CACHE_DIR = "data/tts_cache"
USAGE_FILE = "data/llm_usage.json"  # cumul des tokens et du coût
MEMORY_FILE = "data/memory.json"  # faits mémorisés (remember / forget)
SKILLS_FILE = "data/skills/skills.json"  # savoir-faire appris (phase 12c)


def build_speaker(config: Config, player: Any) -> Speaker:
    """Voix + phrases fréquentes + file de phrases, non chargées. Phase 11 : routeur de voix
    (profil du skin en cours : Piper, Kokoro ou ElevenLabs, Kokoro en secours)."""
    from mikata.audio.player import Speaker
    from mikata.tts.credits import USAGE_FILE as VOICE_USAGE_FILE
    from mikata.tts.router import build_router

    router = build_router(config, config.resolve(TTS_CACHE_DIR), config.resolve(VOICE_USAGE_FILE))
    return Speaker(router, player, router)


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
    if config.skills.enabled:  # phase 12c : savoir-faire appris (mémoire procédurale)
        from mikata.skills.library import SkillLibrary

        ctx.skills = SkillLibrary(
            config.resolve(SKILLS_FILE), config.skills.max_skills, config.skills.max_failures
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
        if config.uia.enabled:  # phase 12a : pilotage fin des applications (arbre UIA)
            from mikata.uia.controller import UiController

            ctx.uia = UiController(config, ctx.desktop)
            if config.uia.state_poll_s > 0:
                logger.info(
                    "Pilotage des applications : actif (état ambiant toutes les {} s).",
                    config.uia.state_poll_s,
                )
        if config.browser.enabled:  # phase 12b : navigateur piloté (CDP, connexion paresseuse)
            from mikata.browser.controller import BrowserController

            ctx.browser = BrowserController(config, ctx.desktop)
        if config.files_index.enabled:  # phase 12d : index local des fichiers (MFT + USN)
            from mikata.files.index import FileIndex

            ctx.index = FileIndex(config)
            ctx.index.start()
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

    from mikata.tools.apps import installed_app_names, open_apps
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
        skills=ctx.skills,
        usage=usage,
        open_apps=running_apps if desktop is not None else None,
        installed_apps=(
            (lambda: installed_app_names(desktop.cached_apps())) if desktop is not None else None
        ),
    )


def app_vocabulary(agent: Agent | None) -> Callable[[], list[str]] | None:
    """Noms des applications installées pour le prompt de Whisper (ajout phase 6) : l'index
    du menu Démarrer déjà en mémoire, construit au préchauffage."""
    desktop = agent.tools.ctx.desktop if agent is not None and agent.tools is not None else None
    if desktop is None:
        return None
    from mikata.tools.apps import installed_app_names

    def names() -> list[str]:
        return installed_app_names(desktop.cached_apps())

    return names


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


def build_ui(config: Config) -> dict[str, Any]:
    """Phase 8 : canal vers l'interface (mascotte, réglages) et son lanceur ; {} si
    `ui.enabled` est faux. L'interface tourne dans un autre processus (`python -m mikata.ui`)."""
    if not config.ui.enabled:
        return {}
    from mikata.ui.bridge import UiBridge, launch_ui
    from mikata.ui.protocol import CHANNEL_FILE

    holder: dict[str, App] = {}

    def on_command(command: dict[str, Any]) -> None:
        holder["app"]._on_ui_command(command)

    def snapshot() -> dict[str, Any]:
        return holder["app"]._ui_snapshot() if "app" in holder else {}

    bridge = UiBridge(config.resolve(CHANNEL_FILE), on_command, snapshot)
    return {
        "ui": bridge,
        "ui_launcher": lambda argv: launch_ui(config.base_dir, argv),
        "_holder": holder,
    }


def build_app(
    config: Config,
    *,
    tray: bool,
    audio: bool,
    llm: bool = True,
    config_path: Path | None = None,
) -> App:
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
    detector = OpenWakeWordDetector(
        config.resolve(config.wake.model),
        verifier,
        threshold=config.wake.threshold,
        verifier_threshold=config.wake.verifier_threshold,
    )
    logger.info(
        "Mot d'activation chargé : {}{}",
        detector.name,
        f" + vérificateur {verifier.name}" if verifier is not None else " (sans vérificateur)",
    )
    vad = SileroVAD(config.resolve(config.vad.model), config.vad.threshold)
    speaker = build_speaker(config, player) if llm else None
    agent = build_agent(config, speaker.cache if speaker else None) if llm else None
    if not llm:
        logger.warning("Mode --no-llm : les commandes sont transcrites et journalisées seulement.")
    stt = WhisperSTT(
        config.stt, download_root=config.resolve(WHISPER_DIR), vocabulary=app_vocabulary(agent)
    )
    ui = build_ui(config)
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
        stt=stt,
        agent=agent,
        speaker=speaker,
        config_path=config_path,
        **{k: v for k, v in ui.items() if not k.startswith("_")},
    )
    if "_holder" in ui:
        ui["_holder"]["app"] = app
    if agent is not None and agent.tools is not None:
        agent.tools.ctx.announce = app.announce  # minuteurs et rappels dits dans le casque
    if speaker is not None and hasattr(speaker.tts, "on_event"):
        speaker.tts.on_event = app.voice_event  # secours Kokoro, retour ElevenLabs (phase 11)
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
        logger.info("Chargement de la voix...")
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
    """Point d'entrée. Un redémarrage demandé dans l'interface sort avec le code 75 : sous
    supervision, le superviseur relance ; sinon on se relance soi-même (processus détaché)."""
    code = _main(argv)
    if code != EXIT_RESTART:
        return code
    from mikata.supervise import SUPERVISED_ENV, relaunch_self

    if os.environ.get(SUPERVISED_ENV):
        return EXIT_RESTART
    relaunch_self(sys.argv[1:] if argv is None else argv)
    return EXIT_OK


def _main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args.config)
    if args.supervise:  # ajout phase 7 : processus de relance, MIKATA tourne dans un enfant
        from mikata.supervise import child_command, setup_supervisor_log, supervise

        setup_supervisor_log(config_path.parent / "data" / "logs")
        try:
            return supervise(child_command(sys.argv[1:] if argv is None else argv))
        except KeyboardInterrupt:
            return EXIT_OK
        finally:
            shutdown_logging()
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
                config,
                tray=not args.no_tray,
                audio=not args.no_audio,
                llm=not args.no_llm,
                config_path=config_path,
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
        return EXIT_RESTART if app.restart_requested else EXIT_OK
    except Exception:
        logger.exception("Erreur fatale.")
        return EXIT_ERROR
    finally:
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
