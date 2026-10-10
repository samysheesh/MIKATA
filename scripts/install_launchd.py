"""Génère l'agent launchd de MIKATA (macOS, équivalent de install_autostart.ps1).

Écrit `~/Library/LaunchAgents/fr.mikata.plist` (RunAtLoad + KeepAlive, LimitLoadToSessionType
Aqua pour la session graphique, sorties vers data/logs/launchd.{out,err}.log) puis affiche
les commandes `launchctl bootstrap` / `bootout` pour l'activer.

Usage :
    python scripts/install_launchd.py [--python EXE] [--data DIR] [--no-bootstrap]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LABEL = "fr.mikata"
PLIST_NAME = f"{LABEL}.plist"

_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>mikata</string>
        <string>--supervise</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{workdir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>LimitLoadToSessionType</key>
    <string>Aqua</string>
    <key>StandardOutPath</key>
    <string>{out_log}</string>
    <key>StandardErrorPath</key>
    <string>{err_log}</string>
</dict>
</plist>
"""


def render_plist(python: str, workdir: str, out_log: str, err_log: str) -> str:
    """Contenu du plist launchd (fonction pure, testable partout)."""
    return _TEMPLATE.format(
        label=LABEL, python=python, workdir=workdir, out_log=out_log, err_log=err_log
    )


def agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / PLIST_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Installe l'agent launchd de MIKATA (macOS).")
    parser.add_argument("--python", default=sys.executable, help="interpréteur Python à lancer")
    parser.add_argument("--data", default="data", type=Path, help="dossier data/ de MIKATA")
    parser.add_argument(
        "--no-bootstrap", action="store_true", help="écrire le plist sans l'activer"
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    data_dir = args.data if args.data.is_absolute() else repo / args.data
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    content = render_plist(
        args.python,
        str(repo),
        str(log_dir / "launchd.out.log"),
        str(log_dir / "launchd.err.log"),
    )
    dest = agent_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"Agent écrit : {dest}")

    uid = "gui/$(id -u)"
    print(f"Activer   : launchctl bootstrap {uid} {dest}")
    print(f"Désactiver: launchctl bootout {uid}/{LABEL}")
    if args.no_bootstrap:
        return 0
    if sys.platform != "darwin":
        print("(hors macOS : bootstrap ignoré, relancez ce script sur le Mac)")
        return 0
    result = subprocess.run(
        ["launchctl", "bootstrap", uid.replace("$(id -u)", str(_uid())), str(dest)],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    return 0


def _uid() -> int:
    try:
        import os

        return os.getuid()
    except AttributeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
