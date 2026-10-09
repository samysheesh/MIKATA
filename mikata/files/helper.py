"""Helper élevé du journal USN (phase 12d) : le seul processus qui touche aux volumes.

MIKATA tourne en utilisateur standard ; Windows refuse alors la lecture de la MFT et du
journal USN (accès au volume = administrateur). Ce helper, installé comme tâche planifiée
avec élévation (un seul UAC à l'installation), fait le travail et dépose :

- `<data>/index/build.jsonl` : un dump complet (MFT) au démarrage et sur erreur ;
- `<data>/index/events.log` : les deltas du journal, en append (JSONL).

MIKATA ne lit que ces fichiers (voir `mikata.files.events`). Le protocole est un flux de
lignes : pas de socket, pas de partage de mémoire, rien d'autre à sécuriser.

Usage :
    python -m mikata.files.helper --data <dossier data de MIKATA> [--once]

`--once` : un dump puis sortie (vérification). Sans : boucle infinie (tâche planifiée).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from loguru import logger

from mikata.files import events, usn

DUMP = "build.jsonl"
EVENTS = "events.log"
LOG = "helper.log"
_POLL_S = 3.0
_APPEND_FLUSH = 64  # lignes par flush ; l'append reste atomique (loin sous 64 Ko)


def _atomic_lines(path: Path, lines: list[str]) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    part.replace(path)


def _open_append(path: Path) -> TextIO:
    return path.open("a", encoding="utf-8")


def dump_all(data_dir: Path) -> int:
    """Dump complet des volumes NTFS ; renvoie le nombre de lignes écrites."""
    lines: list[str] = []
    for volume in usn.list_ntfs_volumes():
        for path, entry in usn.build_paths(usn.enumerate_files(volume), volume):
            lines.append(events.dump_line(path, entry))
            if len(lines) >= 200_000:
                logger.info("Dump en cours : {} lignes.", len(lines))
    _atomic_lines(data_dir / DUMP, lines)
    return len(lines)


def watch(data_dir: Path) -> None:
    """Dump puis rejeu du journal, pour toujours (arrêt par fin de processus)."""
    started = time.monotonic()
    try:
        total = dump_all(data_dir)
        logger.info("Dump initial : {} entrées en {:.1f} s.", total, time.monotonic() - started)
    except Exception as exc:
        logger.error("Dump initial impossible : {}", exc)
        raise
    volumes = usn.list_ntfs_volumes()
    journal_ids = {volume: usn.query_journal(volume) for volume in volumes}
    pending: list[str] = []
    while True:
        changed = False
        for volume, (journal_id, start_usn) in list(journal_ids.items()):
            try:
                events_list, next_usn = usn.read_journal(
                    volume, start_usn, journal_id, timeout_s=0.0
                )
            except Exception as exc:
                logger.warning("Journal de {} illisible : {} — nouveau dump.", volume, exc)
                dump_all(data_dir)
                journal_ids[volume] = usn.query_journal(volume)
                continue
            journal_ids[volume] = (journal_id, next_usn)
            for event in events_list:
                if usn.relevant(event.reason):
                    pending.append(events.event_line(event))
                    changed = True
        if pending:
            try:
                with _open_append(data_dir / EVENTS) as handle:
                    for line in pending:
                        handle.write(line + "\n")
                    handle.flush()
            except OSError as exc:
                logger.error("Écriture de {} impossible : {}", EVENTS, exc)
            else:
                pending = []
        if changed and not pending:
            pass  # tout est écrit ; on retombe sur le sommeil
        time.sleep(_POLL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Helper USN de MIKATA (élevé).")
    parser.add_argument("--data", required=True, type=Path, help="dossier data/ de MIKATA")
    parser.add_argument("--once", action="store_true", help="un dump puis sortie")
    args = parser.parse_args(argv)
    data_dir = args.data
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.add(data_dir / LOG, rotation="1 MB", retention=2, level="INFO")
    try:
        if args.once:
            total = dump_all(data_dir)
            logger.info("Dump ponctuel : {} entrées.", total)
            return 0
        watch(data_dir)
    except Exception as exc:
        logger.error(
            "Helper arrêté : {} (sans élévation, Windows refuse l'accès aux volumes).", exc
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
