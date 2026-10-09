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
import os
import sys
import time
from pathlib import Path
from typing import TextIO

from loguru import logger

from mikata.files import events, usn

LOG = "helper.log"
_POLL_S = 3.0


def _open_append(path: Path) -> TextIO:
    return path.open("a", encoding="utf-8")


def dump_all(share_dir: Path) -> int:
    """Dump complet des volumes NTFS dans `share_dir` ; renvoie le nombre de lignes.

    Écrit en flux (jamais tout en RAM) et journalise la progression tous les 500 000.
    """
    part = share_dir / (events.DUMP + ".part")
    total = 0
    with part.open("w", encoding="utf-8") as handle:
        for volume in usn.list_ntfs_volumes():
            paths = 0
            for path, entry in usn.build_paths(usn.enumerate_files(volume), volume):
                handle.write(events.dump_line(path, entry) + "\n")
                paths += 1
                if paths % 500_000 == 0:
                    logger.info("Volume {} : {} entrées...", volume, paths)
            if paths == 0:
                logger.info("Volume {} : aucun chemin produit ({})", volume, usn.diagnose(volume))
            else:
                logger.info("Volume {} : {} entrées.", volume, paths)
            total += paths
    part.replace(share_dir / events.DUMP)
    return total


_DUMP_REUSE_S = 600.0  # un dump de moins de 10 min est réutilisé tel quel


def watch(share_dir: Path) -> None:
    """Dump puis rejeu du journal, pour toujours (arrêt par fin de processus)."""
    dump = share_dir / events.DUMP
    try:
        age = time.time() - dump.stat().st_mtime
    except OSError:
        age = _DUMP_REUSE_S + 1
    if age <= _DUMP_REUSE_S:
        logger.info("Dump récent ({} s) : reprise directe du journal.", int(age))
    else:
        started = time.monotonic()
        try:
            total = dump_all(share_dir)
            logger.info("Dump initial : {} entrées en {:.1f} s.", total, time.monotonic() - started)
        except Exception as exc:
            logger.error("Dump initial impossible : {}", exc)
            raise
    volumes = usn.list_ntfs_volumes()
    journal_ids: dict[str, tuple[int, int]] = {}
    for volume in volumes:
        try:
            journal_ids[volume] = usn.query_journal(volume)
        except Exception as exc:
            # Journal USN absent/désactivé : fsutil usn createjournal le répare ; en
            # attendant, le volume est suivi au prochain redump.
            logger.warning(
                "Journal USN de {} inactif ({}) — lance fsutil usn createjournal.", volume, exc
            )
    pending: list[str] = []
    while True:
        for volume, (journal_id, start_usn) in list(journal_ids.items()):
            try:
                events_list, next_usn = usn.read_journal(
                    volume, start_usn, journal_id, timeout_s=0.0
                )
            except Exception as exc:
                logger.warning("Journal de {} illisible : {} — nouveau dump.", volume, exc)
                dump_all(share_dir)
                try:
                    journal_ids[volume] = usn.query_journal(volume)
                except Exception:
                    del journal_ids[volume]
                continue
            journal_ids[volume] = (journal_id, next_usn)
            for event in events_list:
                if usn.relevant(event.reason):
                    pending.append(events.event_line(volume, event))
        if pending:
            try:
                with _open_append(share_dir / events.EVENTS) as handle:
                    for line in pending:
                        handle.write(line + "\n")
                    handle.flush()
            except OSError as exc:
                logger.error("Écriture de {} impossible : {}", events.EVENTS, exc)
            else:
                pending = []
        time.sleep(_POLL_S)


def main(argv: list[str] | None = None) -> int:
    if sys.stderr is None:  # pythonw : pas de console, loguru écrit dans helper.log
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115  # gardé ouvert, remplace stderr
    parser = argparse.ArgumentParser(description="Helper USN de MIKATA (élevé).")
    parser.add_argument("--data", required=True, type=Path, help="dossier data/ de MIKATA")
    parser.add_argument("--once", action="store_true", help="un dump puis sortie")
    args = parser.parse_args(argv)
    data_dir = args.data
    share_dir = data_dir / "index"
    data_dir.mkdir(parents=True, exist_ok=True)
    share_dir.mkdir(parents=True, exist_ok=True)
    logger.add(share_dir / LOG, rotation="1 MB", retention=2, level="INFO")
    try:
        if args.once:
            total = dump_all(share_dir)
            logger.info("Dump ponctuel : {} entrées.", total)
            return 0
        watch(share_dir)
    except Exception as exc:
        logger.error(
            "Helper arrêté : {} (l'élévation et le journal USN actif sont requis, "
            "voir scripts/install_usn_helper.ps1).",
            exc,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
