# Créer un skin pour la mascotte

Un **skin** change l'apparence de la mascotte. Il ne touche à rien d'autre : la mascotte
passe par les mêmes états, au même moment. Ce guide explique comment créer le tien avec un
générateur d'images (Nano Banana ou un autre), en partant d'images brutes. Il couvre ensuite
l'import, le partage et le format du manifeste.

En bref :

1. Génère **une image par état** (10 au plus ; seul le repos est obligatoire).
2. Range-les dans un dossier, nommées d'après l'état (`repos.png`, `ecoute.png`...).
3. Lance `scripts/make_skin.py`. Il retire le fond (uni ou damier), recadre, redimensionne,
   adoucit les bords et écrit le manifeste.
4. Choisis ton skin dans **Réglages > Skins** : le changement est immédiat.

## 1. Les dix états

| État | Fichier (au choix) | Quand | Ce que le personnage exprime | Animation par défaut d'une image fixe |
| --- | --- | --- | --- | --- |
| `idle` | `repos`, `idle` | MIKATA attend « MIKATA » | **il dort** : yeux fermés, paisible | respiration lente |
| `wake` | `activation`, `wake` | « MIKATA » vient d'être entendu (0,9 s) | réveil en sursaut, surprise | petit rebond + lueur jaune |
| `listening` | `ecoute`, `listening` | il t'écoute | attentif, tourné vers toi | respiration, clignement, lueur bleue |
| `thinking` | `reflexion`, `thinking` | transcription, réflexion du LLM | il réfléchit | balancement, clignement |
| `acting` | `action`, `acting` | un outil travaille (ouvre une appli, cherche...) | affairé, concentré | rebond + balancement |
| `speaking` | `parole`, `speaking` | il répond à voix haute | il parle | petits rebonds + lueur verte |
| `success` | `fait`, `success` | « C'est fait » (1,6 s) | content, réussite | rebond + lueur verte |
| `error` | `erreur`, `error` | une erreur (2,4 s) | oups, désolé | balancement rapide + lueur rouge |
| `follow_up` | `suivi`, `follow_up` | 6 s pour enchaîner sans redire « MIKATA » | en attente de la suite | respiration + lueur turquoise |
| `mic_off` | `micro_coupe`, `mic_off` | micro en pause ou occupé, MIKATA pas lancé | muet, en retrait | respiration très lente |

Les noms de fichiers sont acceptés avec ou sans accents, en minuscules ou en majuscules, et
avec `_` ou `-` (« Réflexion.png » convient).

**Un état absent** reprend l'image de repos. Un avertissement est écrit dans
`data\logs\ui.log`, sans plantage. Tu peux donc commencer avec 2 ou 3 images et compléter
plus tard.

**Images en plus, facultatives** :

- `ecoute_blink.png`, `repos_blink.png`... : la même image, **yeux fermés**. Elle est montrée
  un instant pour un vrai clignement des yeux. Sans elle, le clignement est un léger
  écrasement vertical de l'image.
- `parole_01.png`, `parole_02.png`... : plusieurs images numérotées pour un même état. Elles
  sont jouées image par image (8 images/s, réglable avec `--fps`). Un GIF ou un APNG animé
  convient aussi.

## 2. Taille et cadrage

- **Format** : carré (1:1). Nano Banana sort du 1024 × 1024, c'est parfait. Plus petit que
  512 px, le résultat sera flou.
- **Personnage** : en pied, de face, **centré**, environ 75 % de la hauteur. Laisse de la
  marge tout autour : un bras qui sort du cadre est coupé.
- **Même cadrage pour tous les états** : même taille de personnage, même position, même
  angle. `make_skin.py` applique la même échelle à tous les états, pour que le personnage ne
  grandisse pas d'un état à l'autre. Il les pose au même niveau, en bas du cadre.
- **Fond** :
  - idéalement **uni et sans dégradé**, d'une couleur absente du personnage (blanc pur, ou
    vert vif #00FF00 si le personnage a beaucoup de blanc) ;
  - le **faux damier** de transparence est aussi reconnu ;
  - **pas d'ombre au sol** si possible. Sinon, `make_skin.py` la transforme en ombre noire
    translucide (`--shadow clean`), ou la retire (`--shadow remove`).
- **Résultat** : images de 384 px (`--size`). La mascotte s'affiche en 96, 128 ou 192 px,
  et 384 reste net même avec la mise à l'échelle de Windows à 150 ou 200 %.

## 3. Garder le même personnage d'un état à l'autre

C'est le point difficile avec un générateur d'images. Ce qui marche :

1. **Une fiche personnage fixe** : un paragraphe qui décrit le personnage, recopié **mot pour
   mot** au début de chaque prompt (voir l'exemple plus bas). Ne la reformule jamais :
   chaque mot changé peut changer un détail du personnage.
2. **Une image de référence** : génère d'abord le **repos**, ou une pose neutre si le repos
   (yeux fermés) cache trop le visage. Pour chaque autre état :
   - donne cette image à Nano Banana avec le prompt : « même personnage, mêmes vêtements,
     mêmes proportions, même style ; change seulement l'expression et la pose » ;
   - Nano Banana sait **modifier** une image : c'est bien plus fidèle qu'une nouvelle
     génération à partir de rien.
3. **Une seule conversation** pour tout le skin : le générateur garde le contexte. Si un
   état dérive (coiffure, couleurs), reprends l'image de référence plutôt que l'image
   précédente.
4. **Même fond, même cadrage, même lumière** dans chaque prompt (c'est déjà dans la fiche).
5. **Vérifie côte à côte** : `make_skin.py` écrit `apercu.png`, une planche de tous les
   états. Régénère celui qui détonne.

## 4. Exemple : un personnage chibi de style anime

### La fiche personnage

À recopier telle quelle au début de **chaque** prompt. Les prompts sont en anglais : les
générateurs d'images les suivent en général plus fidèlement. Tu peux les traduire en
français ; reste alors cohérent d'un prompt à l'autre.

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

### Un prompt par état

Le premier sert de référence. Pour les suivants, joins l'image du repos (ou de l'écoute) et
commence par : *« Same character as the reference image, same outfit, same proportions,
same art style, same framing and background. Only change the pose and expression: »*,
suivi de la fiche, puis de la ligne de l'état.

| Fichier | Prompt de l'état (après la fiche) |
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

Astuces :

- Si le fond n'est pas parfaitement uni, ajoute « plain flat background, no gradient,
  no texture ».
- Si le personnage a beaucoup de blanc, demande un fond « pure green (#00FF00) » au lieu du
  blanc.
- Si Nano Banana rend un damier gris et blanc au lieu d'un fond : ce n'est pas grave,
  `make_skin.py` le reconnaît.

## 5. Fabriquer le skin

```powershell
uv run python scripts\make_skin.py C:\Users\moi\Images\mika_chibi --name "Mika chibi" --author "Moi" --license "usage personnel" --preview
```

- Le skin est écrit dans `assets\mascot\mika-chibi\`. Il contient :
  - les images détourées ;
  - `skin.json`, avec les effets par défaut ;
  - `apercu.png` : tous les états sur un damier. **Regarde-le** pour vérifier le détourage.
- `--preview` ouvre directement la mascotte avec ce skin, et une petite fenêtre pour passer
  d'un état à l'autre, ou les faire défiler. Rien n'est enregistré dans `config.yaml`.
  Depuis un skin déjà installé : `uv run python -m mikata.ui --preview mika-chibi`.
- Pour l'utiliser ensuite : **Réglages > Skins**, clic sur le skin, et c'est immédiat. Le
  choix est gardé dans `config.yaml` (`ui.skin`).

Réglages utiles :

| Option | Effet | Quand l'utiliser |
| --- | --- | --- |
| `--tolerance 38` | écart de couleur toléré pour le fond | liseré de fond resté autour → plus haut (50) ; morceaux du personnage retirés → plus bas (25) |
| `--holes auto\|on\|off` | trous fermés (entre un bras et le corps) | fond uni visible entre les bras → `on` ; le blanc des yeux disparaît → `off` |
| `--shadow clean\|remove\|keep` | ombre au sol | `clean` : ombre noire translucide ; `remove` : retirée |
| `--size 384` | côté des images produites | 512 pour un écran 4K |
| `--margin 0.06` | marge autour du personnage | |
| `--align bottom\|center` | posé en bas (par défaut) ou centré | `center` pour un personnage qui flotte |
| `--fps 8` | vitesse des images numérotées | |
| `--replace` | remplace un skin du même nom (l'ancien part à la corbeille) | après avoir régénéré des images |

Une image **déjà transparente** (PNG avec canal alpha) est gardée telle quelle, sans
détourage.

## 6. Retoucher l'animation : `skin.json`

`make_skin.py` écrit un manifeste complet, que tu peux modifier avec n'importe quel éditeur
de texte. Après une modification, reclique sur le skin dans **Réglages > Skins** (ou
**Actualiser**). Exemple :

```json
{
  "name": "Mika chibi",
  "author": "Moi",
  "license": "usage personnel",
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

- `name`, `author`, `license` : affichés dans **Réglages > Skins**. Mets-y ce que tu veux ;
  rien d'autre n'en dépend.
- Par état :
  - `file` : l'image, dans le dossier du skin (sous-dossiers permis). Formats : PNG, JPEG,
    WebP, GIF, APNG.
  - `frames` et `columns` : bande d'images (spritesheet) découpée en grille.
  - `fps`, ou les durées propres à un GIF/APNG : vitesse de l'animation image par image.
  - `loop` : `false` joue l'animation une fois et garde la dernière image. C'est le défaut
    pour `wake`, `success` et `error`.
  - `blink_file` : image yeux fermés, utilisée par l'effet `blink`.
  - `effects` : voir le tableau ci-dessous. `{}` = aucun effet. Si `effects` est absent :
    - une image fixe reçoit les effets par défaut de son état (tableau de la section 1) ;
    - une animation image par image n'en reçoit aucun.

| Effet | Ce qu'il fait | `amplitude` | `speed` |
| --- | --- | --- | --- |
| `breathe` | respiration : l'image s'étire un peu en hauteur et se tasse | fraction de la taille, 0 à 0,2 (0,02 = 2 %) | cycles par seconde, 0 à 10 |
| `bounce` | petit rebond | hauteur du saut, 0 à 0,3 de la taille | rebonds par seconde |
| `blink` | clignement : `blink_file` montré un instant, sinon écrasement vertical | écrasement, 0 à 0,5 (0 avec `blink_file`) | clignements par seconde (0,25 = toutes les 4 s) |
| `glow` | lueur colorée derrière le personnage, qui pulse | intensité, 0 à 1 | pulsations par seconde ; `color` : « #rrggbb », par défaut la couleur de l'état |
| `sway` | balancement autour des pieds | angle en degrés, 0 à 30 | allers-retours par seconde |

Une valeur hors limites est ramenée dans les limites, et un effet inconnu est ignoré. Dans
les deux cas, un avertissement est écrit dans `data\logs\ui.log`.

## 7. Importer, partager, supprimer

- **Partager un skin** : compresse son dossier en .zip. Clic droit sur le dossier >
  **Compresser dans un fichier ZIP** (Windows 11).
- **Importer** : **Réglages > Skins > Importer un .zip…**. Le skin est vérifié **avant**
  d'écrire quoi que ce soit. Il est refusé, avec la raison affichée, dans ces cas :
  - aucun ou plusieurs `skin.json`, ou un `skin.json` trop profond (il doit être à la racine
    ou dans un seul dossier) ;
  - un chemin qui sort du dossier (`..`) ;
  - un fichier autre qu'une image ou du texte (`.json`, `.txt`, `.md`) : un `.exe` ou un
    script est refusé ;
  - plus de 300 fichiers, plus de 64 Mo au total ou plus de 16 Mo pour un seul fichier ;
  - une image de repos absente ou illisible ;
  - un skin du même nom déjà installé.

  Une fois importé, le skin est appliqué tout de suite.
- **Supprimer** : **Réglages > Skins**, choisis le skin, puis **Supprimer**. Il part à la
  **corbeille** (récupérable). Mika, le skin par défaut, ne peut pas être supprimée. Si
  c'était le skin en cours, Mika revient.
- Tes skins ne sont pas suivis par git (`.gitignore`) : ils restent sur ton PC.

## 8. En cas de souci

| Problème | Solution |
| --- | --- |
| Liseré blanc ou gris autour du personnage | `--tolerance 50`, ou régénère sur un fond plus contrasté |
| Le blanc des yeux ou un vêtement clair est devenu transparent | `--holes off`, ou un fond vert (#00FF00) au lieu du blanc |
| Fond resté visible entre les bras | `--holes on` |
| Damier resté sous les pieds | `--shadow remove` |
| Le personnage change de taille d'un état à l'autre | cadrage différent à la génération : régénère l'état en rappelant « same framing » |
| Un état affiche l'image de repos | fichier absent ou mal nommé : voir les « Ignoré » et « absent » affichés par `make_skin.py`, et `data\logs\ui.log` |
| Skin en rouge dans la liste | inutilisable (raison dans l'infobulle et dans `data\logs\ui.log`) |
