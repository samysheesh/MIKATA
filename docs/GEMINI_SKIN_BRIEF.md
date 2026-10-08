# Brief pour Gemini : créer les images d'un skin MIKATA

> **Mode d'emploi (pour toi)**
>
> 1. Ouvre une **nouvelle conversation** sur gemini.google.com.
> 2. Joins ce fichier (bouton « + » > Importer un fichier), puis écris : **« Suis ce brief. »**
> 3. Gemini te pose quelques questions sur le personnage, te montre l'image de repos, puis
>    génère les autres états un par un. Réponds **« ok »** pour passer au suivant, ou dis ce
>    qu'il faut corriger.
> 4. Télécharge chaque image validée en taille réelle (clic sur l'image > Télécharger) dans
>    un même dossier, sous le nom que Gemini t'indique (`repos.png`, `ecoute.png`...).
> 5. Fabrique le skin sur ton PC :
>
>    ```powershell
>    uv run python scripts\make_skin.py C:\Users\<toi>\Images\mon_perso --name "Mon perso" --author "Moi" --license "usage personnel" --preview
>    ```
>
> 6. Réglages de MIKATA > Skins > Actualiser, puis clique sur ton skin.
>
> Tout ce qui suit s'adresse à Gemini.

---

## Ton rôle, Gemini

Tu aides l'utilisateur à créer **le même personnage dans 10 poses**, pour la mascotte de bureau
de son assistant vocal MIKATA. Chaque image montre le personnage dans un état de l'assistant.
Un programme découpera ensuite automatiquement le fond : le cadrage et le fond doivent donc
être **identiques** d'une image à l'autre.

Parle en français avec l'utilisateur. Écris tes prompts d'image en anglais.

## Étape 1 : le personnage

Pose ces questions en **un seul message**, puis attends les réponses :

1. Quel personnage ? (par défaut : « Mika », une petite fille chibi style anime, cheveux
   turquoise en carré, grands yeux ambrés, sweat crème trop grand, gros casque audio blanc)
2. Quel style ? (par défaut : chibi anime, contours épais, aplats de couleurs, ombrage doux)
3. Des couleurs ou accessoires à éviter ou à ajouter ?

Si l'utilisateur répond « par défaut », prends les valeurs ci-dessus.

Ensuite, écris une **fiche personnage** en anglais (5 à 8 lignes) et montre-la à
l'utilisateur. Elle doit contenir :

- apparence : coiffure, couleur des yeux, vêtements, accessoires, proportions (2,5 têtes de
  haut pour un chibi) ;
- style de dessin ;
- et toujours ce bloc, mot pour mot :
  `Full body, front view, centered, the character fills about 75% of the image height with
  margin on all sides. Plain pure white background (#FFFFFF), flat, no gradient, no texture,
  no ground, no floor shadow, no text, no watermark. Square 1:1 image.`

Si le personnage contient beaucoup de blanc, remplace dans ce bloc le fond blanc par
`Plain pure green background (#00FF00)`.

**Recopie cette fiche mot pour mot au début de chaque prompt d'image. Ne la reformule
jamais.**

## Étape 2 : l'image de référence (repos)

Génère `repos.png` : fiche + *« Asleep while standing, eyes gently closed, peaceful little
smile, head slightly tilted, relaxed arms, three small floating "z" letters near the
head. »*

Montre l'image et demande : « Ce personnage te plaît ? ok / à corriger ». Régénère jusqu'à
ce que l'utilisateur dise ok. **Cette image devient la référence de toutes les suivantes.**

## Étape 3 : les autres états, un par un

Pour chaque état ci-dessous, dans cet ordre :

1. Repars de l'**image de repos validée** comme référence (modification d'image, pas une
   nouvelle génération à partir de rien). Ton prompt commence par : *« Same character as
   the reference image, same outfit, same proportions, same art style, same framing, same
   size in the frame, same background. Only change the pose and expression: »*, suivi de la
   fiche, puis de la description de l'état.
2. Montre l'image avec son **nom de fichier** en gras, et demande « ok / à corriger ».
3. Passe à l'état suivant seulement après « ok ».

| Fichier | Description de l'état |
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

Pour `ecoute_blink.png`, la référence est `ecoute.png`, pas le repos.

## Règles pour toutes les images

- Format **carré 1:1**, le personnage **entier**, centré, **même taille et même position**
  dans toutes les images.
- Fond **uni** (blanc ou vert, toujours le même), **sans ombre au sol**, sans décor, sans
  texte, sans signature.
- Si une image dérive (couleur de cheveux, vêtements, proportions, style), régénère-la en
  repartant de l'image de repos, pas de l'image précédente.
- Une seule image par message.

## Étape 4 : à la fin

Quand les 10 images sont validées, rappelle à l'utilisateur :

1. de télécharger chaque image en taille réelle dans un même dossier, avec le nom indiqué ;
2. de lancer `scripts\make_skin.py` (commande en haut de ce fichier) ;
3. d'ouvrir `apercu.png` dans le dossier du skin pour vérifier le détourage.
