# Brief for Gemini: creating the images for a MIKATA skin

> **How to use it (for you)**
>
> 1. Open a **new conversation** at gemini.google.com.
> 2. Attach this file ("+" button > Upload files), then write: **"Follow this brief."**
> 3. Gemini asks you a few questions about the character, shows you the idle image, then
>    generates the other states one by one. Reply **"ok"** to move on to the next one, or
>    say what needs fixing.
> 4. Download each approved image at full size (click the image > Download) into a single
>    folder, under the name Gemini gives you (`repos.png`, `ecoute.png`...).
> 5. Build the skin on your PC:
>
>    ```powershell
>    uv run python scripts\make_skin.py C:\Users\<you>\Pictures\my_character --name "My character" --author "Me" --license "personal use" --preview
>    ```
>
> 6. In MIKATA, Réglages > Skins > Actualiser (Settings > Skins > Refresh), then click
>    your skin.
>
> Everything below is addressed to Gemini.

---

## Your role, Gemini

You help the user create **the same character in 10 poses**, for the desktop mascot of their
voice assistant MIKATA. Each image shows the character in one of the assistant's states.
A program will then cut out the background automatically, so the framing and the background
must be **identical** from one image to the next.

Speak with the user in their own language. Write your image prompts in English.

## Step 1: the character

Ask these questions in **a single message**, then wait for the answers:

1. Which character? (default: "Mika", a little anime-style chibi girl, teal bob haircut,
   big amber eyes, oversized cream hoodie, big white headphones)
2. Which style? (default: chibi anime, thick outlines, flat colors, soft shading)
3. Any colors or accessories to avoid or to add?

If the user answers "default", use the values above.

Then write a **character sheet** in English (5 to 8 lines) and show it to the user. It must
contain:

- appearance: hairstyle, eye color, clothes, accessories, proportions (2.5 heads tall for a
  chibi);
- drawing style;
- and always this block, word for word:
  `Full body, front view, centered, the character fills about 75% of the image height with
  margin on all sides. Plain pure white background (#FFFFFF), flat, no gradient, no texture,
  no ground, no floor shadow, no text, no watermark. Square 1:1 image.`

If the character has a lot of white, replace the white background in this block with
`Plain pure green background (#00FF00)`.

**Copy this sheet word for word at the start of every image prompt. Never rephrase
it.**

## Step 2: the reference image (idle)

Generate `repos.png`: sheet + *"Asleep while standing, eyes gently closed, peaceful little
smile, head slightly tilted, relaxed arms, three small floating "z" letters near the
head."*

Show the image and ask: "Do you like this character? ok / needs fixing". Regenerate until
the user says ok. **This image becomes the reference for all the following ones.**

## Step 3: the other states, one by one

For each state below, in this order:

1. Start again from the **approved idle image** as the reference (image editing, not a new
   generation from scratch). Your prompt starts with: *"Same character as the reference
   image, same outfit, same proportions, same art style, same framing, same size in the
   frame, same background. Only change the pose and expression:"*, followed by the sheet,
   then the state description.
2. Show the image with its **file name** in bold, and ask "ok / needs fixing".
3. Move on to the next state only after "ok".

| File | State description |
| --- | --- |
| `activation.png` | Just woke up with a start, eyes wide open and sparkling, small "o" mouth of surprise, one hand raised, a small yellow exclamation mark above the head. |
| `ecoute.png` | Listening attentively, headphones on the ears, one hand cupping an ear, eyes open looking at the viewer, slight curious smile, leaning a little forward. |
| `ecoute_blink.png` | Exactly the same image as the listening pose, only the eyes are closed in a blink. Change nothing else. |
| `reflexion.png` | Thinking hard, index finger on the chin, eyes looking up to the side, small pout, three small thought bubbles above the head. |
| `action.png` | Busy working, determined expression, holding a small glowing tablet in both hands, a tiny gear icon next to the character. |
| `parole.png` | Talking cheerfully to the viewer, mouth open mid-sentence, one hand gesturing openly, bright eyes, small sound-wave lines near the mouth. |
| `fait.png` | Proud and happy, thumbs up, eyes closed as happy crescents (^ ^), big smile, a few small sparkles around. |
| `erreur.png` | Oops moment, embarrassed apologetic smile, one hand behind the head, a big sweat drop, eyes as small swirls, slightly slumped. |
| `suivi.png` | Waiting for the next request, leaning forward with hands behind the back, eyebrows raised, small smile, a small question mark above the head. |
| `micro_coupe.png` | Quiet mode, both hands covering the mouth, headphones resting around the neck, half-closed calm eyes, slightly desaturated colors. |

For `ecoute_blink.png`, the reference is `ecoute.png`, not the idle image.

## Rules for every image

- **Square 1:1** format, the **whole** character, centered, **same size and same position**
  in every image.
- **Solid** background (white or green, always the same), **no floor shadow**, no scenery,
  no text, no signature.
- If an image drifts (hair color, clothes, proportions, style), regenerate it starting from
  the idle image, not from the previous image.
- One image per message.

## Step 4: at the end

When all 10 images are approved, remind the user:

1. to download each image at full size into a single folder, under the name given;
2. to run `scripts\make_skin.py` (the command at the top of this file);
3. to open `apercu.png` in the skin folder to check the cut-out.
