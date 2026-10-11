"""Config de travail : config du dépôt (main) + réglages perso connus + agent2 activé."""

import sys
import tempfile
from pathlib import Path

import yaml

BACKUP = Path(tempfile.gettempdir()) / "mikata-config-work.yaml"
CONFIG = Path(__file__).parent.parent / "config.yaml"
REPO = Path(__file__).parent.parent / "config.yaml"


def merge(base: dict, over: dict) -> dict:
    """Les clés de `over` qui existent dans `base` écrasent ; les autres sont ignorées."""
    for key, value in over.items():
        if key in base:
            if isinstance(value, dict) and isinstance(base[key], dict):
                merge(base[key], value)
            else:
                base[key] = value
    return base


def main() -> int:
    if not BACKUP.exists():
        print("Backup introuvable :", BACKUP)
        return 1
    # La base : la version commitée (git show HEAD:config.yaml) pour repartir du propre.
    import subprocess

    base = yaml.safe_load(
        subprocess.run(
            ["git", "show", "HEAD:config.yaml"],
            capture_output=True,
            text=True,
            cwd=REPO.parent,
        ).stdout
    )
    personal = yaml.safe_load(BACKUP.read_text(encoding="utf-8"))
    merged = merge(base, personal)
    merged["agent2"]["enabled"] = True
    CONFIG.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("Config prêt : base main + réglages perso connus, agent2.enabled: true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
