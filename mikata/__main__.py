"""Point d'entrée : `python -m mikata` (ou `pythonw -m mikata` sans console)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from mikata import __version__
from mikata.app import App
from mikata.config import Config, ConfigError, LoggingConfig, load_config, resolve_config_path
from mikata.logging_setup import setup_logging, shutdown_logging
from mikata.ui.tray import TrayIcon

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


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
        logger.warning("LLM non renseigné dans config.yaml (llm.base_url / llm.model) : phase 3.")
    else:
        logger.info("LLM : {} @ {}", config.llm.model, config.llm.base_url)
    for name, value in (
        ("LLM_API_KEY", config.secrets.llm_api_key),
        ("SEARCH_API_KEY", config.secrets.search_api_key),
    ):
        logger.info("{} : {}", name, "définie" if value is not None else "absente (.env)")


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

        app = App(config, tray=None if args.no_tray else TrayIcon())
        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            logger.info("Interruption clavier (Ctrl+C).")
        return EXIT_OK
    except Exception:
        logger.exception("Erreur fatale.")
        return EXIT_ERROR
    finally:
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
