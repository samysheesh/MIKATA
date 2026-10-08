"""Fabrique un skin de la mascotte à partir d'images brutes (ajout phase 8).

    uv run python scripts/make_skin.py C:\\chemin\\images --name "Sakura" --author "Moi"
    uv run python scripts/make_skin.py C:\\chemin\\images --name "Sakura" --preview

Entrée : un dossier avec une image par état, nommée d'après l'état, en anglais ou en
français (idle.png ou repos.png, wake.png ou activation.png, listening.png ou ecoute.png,
thinking/reflexion, acting/action, speaking/parole, success/fait, error/erreur,
follow_up/suivi, mic_off/micro_coupe). PNG, JPEG, WebP, ou GIF/APNG animé. Des images
numérotées (idle_01.png, idle_02.png...) forment une animation ; idle_blink.png (yeux
fermés) sert au clignement. Seul le repos est obligatoire : un état absent reprend l'image de
repos.

Pour chaque image : fond uni ou damier retiré, bords adoucis, même échelle pour tous les
états, recadrage centré, redimensionnement (--size). Sortie : assets/mascot/<nom>/ avec
skin.json (nom, auteur, licence, effets par défaut à retoucher) et apercu.png (tous les
états sur un damier, pour vérifier le détourage). Guide : docs/MASCOT_SKIN_GUIDE.md.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKINS = PROJECT_ROOT / "assets" / "mascot"


def main(argv: Sequence[str] | None = None) -> int:
    from mikata.ui.skin_maker import (
        DEFAULT_FPS,
        DEFAULT_MARGIN,
        DEFAULT_SIZE,
        DEFAULT_TOLERANCE,
        collect_sources,
        make_skin,
    )
    from mikata.ui.skins import DEFAULT_SKIN, SkinError, skin_key

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", type=Path, help="dossier des images brutes (une par état)")
    parser.add_argument("--name", help="nom du skin (par défaut : nom du dossier d'images)")
    parser.add_argument("--author", default="", help="auteur (écrit dans skin.json)")
    parser.add_argument("--license", default="", help="licence (ex. « usage personnel »)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="côté des images (px)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="écart de couleur toléré pour le fond (plus haut : détourage plus large)",
    )
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN, help="marge autour du personnage (0-0.3)"
    )
    parser.add_argument(
        "--align",
        choices=("bottom", "center"),
        default="bottom",
        help="bottom : posé en bas (il ne « saute » pas d'un état à l'autre) ; center : centré",
    )
    parser.add_argument(
        "--holes",
        choices=("auto", "on", "off"),
        default="auto",
        help="trous fermés montrant le fond (entre un bras et le corps) : auto = damier seul",
    )
    parser.add_argument(
        "--shadow",
        choices=("clean", "remove", "keep"),
        default="clean",
        help="ombre au sol : clean = translucide et propre, remove = retirée, keep = telle quelle",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="images numérotées")
    parser.add_argument("--out", type=Path, default=SKINS, help="dossier des skins")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="remplace un skin du même nom (l'ancien à la corbeille)",
    )
    parser.add_argument("--preview", action="store_true", help="ouvre le résultat dans la mascotte")
    args = parser.parse_args(argv)

    if not args.images.is_dir():
        print(f"Dossier introuvable : {args.images}", file=sys.stderr)
        return 2
    name = args.name or args.images.name
    key = skin_key(name)
    if key == DEFAULT_SKIN:
        print("« default » est réservé à Mika : choisis un autre nom.", file=sys.stderr)
        return 2
    folder = args.out / key
    if folder.exists():
        if not args.replace:
            print(f"{folder} existe déjà : --replace pour le remplacer.", file=sys.stderr)
            return 2
        from send2trash import send2trash

        send2trash(str(folder))
        print(f"Ancien skin mis à la corbeille : {folder}")
    sources, ignored = collect_sources(args.images)
    for item in ignored:
        print(f"Ignoré (nom d'état inconnu) : {item}")
    try:
        report = make_skin(
            sources,
            folder,
            name=name,
            author=args.author,
            license=args.license,
            size=args.size,
            tolerance=args.tolerance,
            margin=min(max(args.margin, 0.0), 0.3),
            align=args.align,
            holes=args.holes,
            shadow=args.shadow,
            fps=args.fps,
        )
    except SkinError as exc:
        print(f"Skin non créé : {exc}", file=sys.stderr)
        return 1
    for line in report.lines:
        print(f"  {line}")
    print(f"Skin créé : {folder}")
    print(f"Vérifie le détourage : {folder / 'apercu.png'}")
    print(f"Pour l'utiliser : réglages > Skins > « {name} » (ou ui.skin: {key}).")
    if args.preview:
        from mikata.ui.qt.preview import preview_skin

        return preview_skin(args.out, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
