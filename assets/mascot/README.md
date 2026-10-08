# Skins de la mascotte

Chaque dossier de `assets/mascot/` est un **skin** : un manifeste `skin.json` (nom, auteur,
licence, image de chaque état) et ses images.

- `default/` : **Mika**, dessinée en code (`mikata/ui/qt/painter.py`). Elle ne contient
  aucune image et n'est soumise à aucun droit de tiers. Elle est toujours disponible et ne
  peut pas être supprimée.
- Les autres dossiers sont **tes** skins, ignorés par git. Tu les crées avec
  `scripts/make_skin.py` ou tu les importes depuis un .zip.

Pour changer de skin : **Réglages > Skins**, puis un clic sur le skin. Le changement est
immédiat et le choix est gardé dans `config.yaml` (`ui.skin`).

Pour créer un skin (images d'un générateur, détourage automatique, prompts d'exemple,
format du manifeste, effets d'animation, import et suppression) : voir
[`docs/MASCOT_SKIN_GUIDE.md`](../../docs/MASCOT_SKIN_GUIDE.md).

Mika en images, comme exemple du format « image par image » :
`uv run python scripts/export_mascot_skin.py mika-images`.
