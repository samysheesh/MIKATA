# Mascot skins

Each folder in `assets/mascot/` is a **skin**: a `skin.json` manifest (name, author,
license, image for each state) and its images.

- `default/`: **Mika**, drawn in code (`mikata/ui/qt/painter.py`). It contains no images
  and is subject to no third-party rights. It is always available and can't be
  deleted.
- The other folders are **your** skins, ignored by git. You create them with
  `scripts/make_skin.py` or import them from a .zip.

To switch skins: **Réglages > Skins** (Settings > Skins), then click the skin. The change
is immediate and the choice is kept in `config.yaml` (`ui.skin`).

To create a skin (images from a generator, automatic cut-out, example prompts, manifest
format, animation effects, importing and deleting): see
[`docs/MASCOT_SKIN_GUIDE.md`](../../docs/MASCOT_SKIN_GUIDE.md).

Mika as images, as an example of the frame-by-frame format:
`uv run python scripts/export_mascot_skin.py mika-images`.
