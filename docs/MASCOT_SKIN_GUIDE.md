# Creating a mascot skin

A **skin** changes how the mascot looks. It changes nothing else: the mascot goes through the
same states, at the same moments. This guide explains how to create your own with an image
generator (Nano Banana or another), starting from raw images. It then covers importing,
sharing and the manifest format.

In short:

1. Generate **one image per state** (10 at most; only idle is required).
2. Put them in one folder, named after the state (`repos.png`, `ecoute.png`...).
3. Run `scripts/make_skin.py`. It removes the background (solid or checkerboard), crops,
   resizes, softens the edges and writes the manifest.
4. Pick your skin in **Réglages > Skins** (Settings > Skins): the change is immediate.

## 1. The ten states

| State | File (either name) | When | What the character expresses | Default animation for a still image |
| --- | --- | --- | --- | --- |
| `idle` | `repos`, `idle` | MIKATA is waiting for "MIKATA" | **asleep**: eyes closed, peaceful | slow breathing |
| `wake` | `activation`, `wake` | "MIKATA" was just heard (0.9 s) | startled awake, surprised | small bounce + yellow glow |
| `listening` | `ecoute`, `listening` | it's listening to you | attentive, facing you | breathing, blinking, blue glow |
| `thinking` | `reflexion`, `thinking` | transcription, LLM thinking | thinking | swaying, blinking |
| `acting` | `action`, `acting` | a tool is working (opening an app, searching...) | busy, focused | bounce + sway |
| `speaking` | `parole`, `speaking` | it's answering out loud | talking | small bounces + green glow |
| `success` | `fait`, `success` | "C'est fait" ("Done", 1.6 s) | happy, success | bounce + green glow |
| `error` | `erreur`, `error` | an error (2.4 s) | oops, sorry | quick sway + red glow |
| `follow_up` | `suivi`, `follow_up` | 6 s to carry on without saying "MIKATA" again | waiting for what comes next | breathing + turquoise glow |
| `mic_off` | `micro_coupe`, `mic_off` | mic paused or busy, MIKATA not running | muted, withdrawn | very slow breathing |

File names are accepted with or without accents, in lower or upper case, and with `_` or
`-` ("Réflexion.png" works).

**A missing state** falls back to the idle image. A warning is written to
`data\logs\ui.log`; nothing crashes. So you can start with 2 or 3 images and add the rest
later.

**Optional extra images**:

- `ecoute_blink.png`, `repos_blink.png`...: the same image with **eyes closed**. It is shown
  for an instant to make a real blink. Without it, the blink is a slight vertical squash of
  the image.
- `parole_01.png`, `parole_02.png`...: several numbered images for one state. They are
  played frame by frame (8 frames/s, adjustable with `--fps`). An animated GIF or APNG
  works too.

## 2. Size and framing

- **Format**: square (1:1). Nano Banana outputs 1024 × 1024, which is perfect. Below
  512 px, the result will be blurry.
- **Character**: full body, front view, **centered**, about 75% of the height. Leave a
  margin all around: an arm that sticks out of the frame gets cut off.
- **Same framing for every state**: same character size, same position, same angle.
  `make_skin.py` applies the same scale to every state, so the character doesn't grow from
  one state to the next. It places them all at the same level, at the bottom of the frame.
- **Background**:
  - ideally **solid, with no gradient**, in a color the character doesn't use (pure white,
    or bright green #00FF00 if the character has a lot of white);
  - the **fake transparency checkerboard** is recognized too;
  - **no floor shadow** if possible. Otherwise, `make_skin.py` turns it into a translucent
    black shadow (`--shadow clean`), or removes it (`--shadow remove`).
- **Output**: 384 px images (`--size`). The mascot is displayed at 96, 128 or 192 px, and
  384 stays sharp even with Windows scaling at 150 or 200%.

## 3. Keeping the same character across states

This is the hard part with an image generator. What works:

1. **A fixed character sheet**: a paragraph describing the character, copied **word for
   word** at the start of every prompt (see the example below). Never rephrase it: every
   changed word can change a detail of the character.
2. **A reference image**: generate the **idle** image first, or a neutral pose if idle (eyes
   closed) hides too much of the face. For every other state:
   - give that image to Nano Banana with the prompt: "same character, same clothes, same
     proportions, same style; only change the expression and the pose";
   - Nano Banana can **edit** an image: that's far more faithful than a new generation from
     scratch.
3. **A single conversation** for the whole skin: the generator keeps the context. If a
   state drifts (hairstyle, colors), go back to the reference image rather than the
   previous image.
4. **Same background, same framing, same lighting** in every prompt (it's already in the
   sheet).
5. **Check them side by side**: `make_skin.py` writes `apercu.png`, a contact sheet of all
   the states. Regenerate the one that stands out.

## 4. Example: an anime-style chibi character

**Easiest with Gemini**: attach [`GEMINI_SKIN_BRIEF.md`](GEMINI_SKIN_BRIEF.md) in a new
conversation and write "Follow this brief." Gemini asks its questions, then generates the
states one by one, with the right file name for each.

### The character sheet

Copy it as is at the start of **every** prompt. The prompts are in English: image
generators usually follow them more faithfully. You can translate them into another
language; if you do, stay consistent from one prompt to the next.

```text
Chibi anime character sheet style, single character: Mika, a cheerful little girl with
2.5-head-tall chibi proportions, short teal bob haircut with one small ahoge, large round
amber eyes with white highlights, rosy cheeks, wearing an oversized cream hoodie with a
small blue lightning-bolt badge, big white over-ear headphones, black shorts, white sneakers.
Clean thick dark outlines, soft cel shading, flat vivid colors. Full body, front view,
centered, character fills about 75% of the image height with margin on all sides.
Plain pure white background (#FFFFFF), no ground, no floor shadow, no text, no watermark,
square 1:1 image.
```

### One prompt per state

The first one serves as the reference. For the next ones, attach the idle image (or the
listening one) and start with: *"Same character as the reference image, same outfit, same
proportions, same art style, same framing and background. Only change the pose and
expression:"*, followed by the sheet, then the state's line.

| File | State prompt (after the sheet) |
| --- | --- |
| `repos.png` | Asleep while standing, eyes gently closed, peaceful little smile, head slightly tilted, headphones resting around her neck, hands relaxed, three small floating "z" letters near her head. |
| `activation.png` | Just woke up with a start, eyes wide open and sparkling, mouth in a small "o" of surprise, one hand raised, hair ahoge standing straight up, a small yellow exclamation mark above her head. |
| `ecoute.png` | Listening attentively, headphones on her ears, one hand cupping an ear, eyes wide open looking at the viewer, slight curious smile, leaning a little forward. |
| `ecoute_blink.png` | Exactly the same image as the listening pose, only her eyes are closed in a blink. Change nothing else. |
| `reflexion.png` | Thinking hard, index finger on her chin, eyes looking up and to the side, small pout, three small thought bubbles rising above her head. |
| `action.png` | Busy working, determined expression, holding a small glowing tablet in both hands, sleeves pushed up, a tiny spinning gear icon next to her. |
| `parole.png` | Talking cheerfully to the viewer, mouth open mid-sentence, one hand gesturing openly, eyes bright, small sound-wave lines near her mouth. |
| `fait.png` | Proud and happy, giving a thumbs up, eyes closed in happy upside-down crescents (^ ^), big smile, a few small sparkles around her. |
| `erreur.png` | Oops moment, embarrassed apologetic smile, one hand behind her head, a big sweat drop on her forehead, eyes as small swirls, slightly slumped posture. |
| `suivi.png` | Waiting expectantly for the next request, leaning forward with hands clasped behind her back, eyebrows raised, small smile, a small question mark above her head. |
| `micro_coupe.png` | Quiet mode, both hands covering her mouth, headphones off and resting around her neck, half-closed calm eyes, slightly desaturated colors. |

Tips:

- If the background isn't perfectly solid, add "plain flat background, no gradient,
  no texture".
- If the character has a lot of white, ask for a "pure green (#00FF00)" background instead
  of white.
- If Nano Banana renders a gray and white checkerboard instead of a background, that's
  fine: `make_skin.py` recognizes it.

## 5. Building the skin

```powershell
uv run python scripts\make_skin.py C:\Users\me\Pictures\mika_chibi --name "Mika chibi" --author "Me" --license "personal use" --preview
```

- The skin is written to `assets\mascot\mika-chibi\`. It contains:
  - the cut-out images;
  - `skin.json`, with the default effects;
  - `apercu.png`: every state on a checkerboard. **Look at it** to check the cut-out.
- `--preview` opens the mascot straight away with this skin, plus a small window to switch
  from one state to another, or cycle through them. Nothing is saved to `config.yaml`.
  For a skin that's already installed: `uv run python -m mikata.ui --preview mika-chibi`.
- To use it afterwards: **Réglages > Skins** (Settings > Skins), click the skin, and it
  applies immediately. The choice is kept in `config.yaml` (`ui.skin`).

Useful options:

| Option | Effect | When to use it |
| --- | --- | --- |
| `--tolerance 38` | color difference tolerated for the background | background fringe left around → higher (50); parts of the character removed → lower (25) |
| `--holes auto\|on\|off` | enclosed holes (between an arm and the body) | solid background visible between the arms → `on`; the whites of the eyes disappear → `off` |
| `--shadow clean\|remove\|keep` | floor shadow | `clean`: translucent black shadow; `remove`: removed |
| `--size 384` | side length of the output images | 512 for a 4K screen |
| `--margin 0.06` | margin around the character | |
| `--align bottom\|center` | sitting at the bottom (default) or centered | `center` for a floating character |
| `--fps 8` | speed of the numbered images | |
| `--replace` | replaces a skin with the same name (the old one goes to the Recycle Bin) | after regenerating images |

An image that is **already transparent** (PNG with an alpha channel) is kept as is, without
any cut-out.

## 6. Tweaking the animation: `skin.json`

`make_skin.py` writes a complete manifest that you can edit with any text editor. After a
change, click the skin again in **Réglages > Skins** (Settings > Skins), or click
**Actualiser** (Refresh). Example:

```json
{
  "name": "Mika chibi",
  "author": "Me",
  "license": "personal use",
  "size": 384,
  "states": {
    "idle":      { "file": "idle.png", "effects": { "breathe": { "amplitude": 0.025, "speed": 0.22 } } },
    "listening": { "file": "listening.png", "blink_file": "listening_blink.png",
                   "effects": { "breathe": { "amplitude": 0.02, "speed": 0.5 },
                                "blink":   { "amplitude": 0, "speed": 0.3 },
                                "glow":    { "amplitude": 0.5, "speed": 1.0, "color": "#2f7de1" } } },
    "speaking":  { "file": "speaking.png", "frames": 4, "columns": 4, "fps": 10 },
    "wake":      { "file": "wake.gif", "loop": false }
  }
}
```

- `name`, `author`, `license`: shown in **Réglages > Skins** (Settings > Skins). Put
  whatever you like; nothing else depends on them.
- Per state:
  - `file`: the image, in the skin folder (subfolders allowed). Formats: PNG, JPEG,
    WebP, GIF, APNG.
  - `frames` and `columns`: an image strip (spritesheet) cut into a grid.
  - `fps`, or a GIF/APNG's own frame durations: speed of the frame-by-frame animation.
  - `loop`: `false` plays the animation once and keeps the last frame. That's the default
    for `wake`, `success` and `error`.
  - `blink_file`: eyes-closed image, used by the `blink` effect.
  - `effects`: see the table below. `{}` = no effect. If `effects` is missing:
    - a still image gets its state's default effects (table in section 1);
    - a frame-by-frame animation gets none.

| Effect | What it does | `amplitude` | `speed` |
| --- | --- | --- | --- |
| `breathe` | breathing: the image stretches up a little and settles back | fraction of the size, 0 to 0.2 (0.02 = 2%) | cycles per second, 0 to 10 |
| `bounce` | small bounce | jump height, 0 to 0.3 of the size | bounces per second |
| `blink` | blink: `blink_file` shown for an instant, otherwise a vertical squash | squash, 0 to 0.5 (0 with `blink_file`) | blinks per second (0.25 = every 4 s) |
| `glow` | colored glow behind the character, pulsing | intensity, 0 to 1 | pulses per second; `color`: "#rrggbb", defaults to the state's color |
| `sway` | swaying around the feet | angle in degrees, 0 to 30 | back-and-forth swings per second |

An out-of-range value is clamped to the limits, and an unknown effect is ignored. In both
cases, a warning is written to `data\logs\ui.log`.

## 7. Importing, sharing, deleting

- **Sharing a skin**: compress its folder into a .zip. Right-click the folder >
  **Compress to ZIP file** (Windows 11).
- **Importing**: **Réglages > Skins > Importer un .zip…** (Settings > Skins > Import a
  .zip…). The skin is checked **before** anything is written. It is rejected, with the
  reason shown, in these cases:
  - no `skin.json` or several, or a `skin.json` nested too deep (it must be at the root
    or inside a single folder);
  - a path that leaves the folder (`..`);
  - a file other than an image or text (`.json`, `.txt`, `.md`): an `.exe` or a
    script is rejected;
  - more than 300 files, more than 64 MB in total or more than 16 MB for a single file;
  - a missing or unreadable idle image;
  - a skin with the same name already installed.

  Once imported, the skin is applied right away.
- **Deleting**: **Réglages > Skins** (Settings > Skins), select the skin, then
  **Supprimer** (Delete). It goes to the **Recycle Bin** (recoverable). Mika, the default
  skin, can't be deleted. If it was the current skin, Mika comes back.
- Your skins aren't tracked by git (`.gitignore`): they stay on your PC.

## 8. Troubleshooting

| Problem | Solution |
| --- | --- |
| White or gray fringe around the character | `--tolerance 50`, or regenerate on a more contrasting background |
| The whites of the eyes or a light piece of clothing turned transparent | `--holes off`, or a green background (#00FF00) instead of white |
| Background still visible between the arms | `--holes on` |
| Checkerboard left under the feet | `--shadow remove` |
| The character changes size from one state to another | the framing differed at generation: regenerate the state, reminding it "same framing" |
| A state shows the idle image | file missing or misnamed: see the "Ignoré" (ignored) and "absent" (missing) lines printed by `make_skin.py`, and `data\logs\ui.log` |
| Skin shown in red in the list | unusable (reason in the tooltip and in `data\logs\ui.log`) |
