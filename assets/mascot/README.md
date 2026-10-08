# Thèmes de la mascotte

Chaque dossier de `assets/mascot/` est un thème. Le thème `default` est **dessiné en code**
(`mikata/ui/qt/painter.py`). Il ne contient aucune image et n'est soumis à aucun droit de
tiers. Pour changer de thème : onglet **Mascotte** des réglages, ou `ui.theme` dans
`config.yaml`. Le changement s'applique tout de suite.

## Créer un thème

### Le plus simple : partir de la mascotte par défaut

Exporte-la en images, puis retouche-les ou remplace-les :

```powershell
uv run python scripts\export_mascot_theme.py mon_theme
```

Ça crée `assets/mascot/mon_theme/theme.json` et une bande d'images PNG par état
(`idle.png`, `wake.png`...).

### Le manifeste `theme.json`

```json
{
  "name": "Mon thème",
  "author": "Moi",
  "license": "CC-BY-4.0",
  "states": {
    "idle":      { "file": "idle.png", "frames": 26, "fps": 10 },
    "wake":      { "file": "wake.gif", "loop": false },
    "listening": { "file": "listening.png", "frames": 12, "columns": 6, "fps": 12 },
    "thinking":  { "file": "thinking.apng" },
    "speaking":  { "file": "speaking.png", "frames": 14, "fps": 10 },
    "success":   { "file": "success.png", "frames": 16, "fps": 10, "loop": false },
    "error":     { "file": "error.png", "frames": 16, "fps": 10, "loop": false },
    "follow_up": { "file": "follow_up.png", "frames": 20, "fps": 10 },
    "mic_off":   { "file": "mic_off.png" }
  }
}
```

### Les états

| État | Quand |
| --- | --- |
| `idle` | repos : MIKATA attend son mot d'activation (la mascotte dort). **Seul état obligatoire.** |
| `wake` | « MIKATA » vient d'être détecté (joué une fois) |
| `listening` | il t'écoute |
| `thinking` | transcription et réflexion du LLM |
| `acting` | un outil travaille (ouvre une appli, cherche sur le web...) |
| `speaking` | il parle |
| `success` | « C'est fait » : action réussie (joué une fois) |
| `error` | erreur (joué une fois) |
| `follow_up` | fenêtre de suivi : 6 s pour enchaîner sans redire « MIKATA » |
| `mic_off` | micro en pause, occupé ou débranché, ou MIKATA pas lancé |

Un état absent reprend l'animation d'un état proche, jusqu'à `idle` :
- `acting` → `thinking` ;
- `follow_up` → `listening` ;
- `wake` → `listening` ;
- `success` → `speaking` ;
- `error`, `mic_off`, `listening`, `thinking`, `speaking` → `idle`.

### Les images

- **Bande d'images (spritesheet) PNG** :
  - `frames` donne le nombre d'images, `columns` le nombre de colonnes (par défaut toutes
    sur une ligne) ;
  - toutes les images ont la même taille, et la largeur et la hauteur du PNG doivent tomber
    juste ;
  - `fps` donne la vitesse (8 par défaut).
- **GIF ou APNG animé** : les images et leurs durées sont lues dans le fichier. `fps`, s'il
  est donné, impose une vitesse fixe.
- **PNG fixe** : `frames` absent ou 1.
- `loop` :
  - par défaut `true`, sauf pour `wake`, `success` et `error`, joués une fois avant de garder
    la dernière image ;
  - les états `wake`, `success` et `error` durent 0,9 s, 1,6 s et 2,4 s.
- **Taille** : l'image est mise à l'échelle de la mascotte (96, 128 ou 192 px, au choix du
  clic droit), proportions gardées. Des images carrées de 256 px restent nettes partout.
- **Transparence** : tout ce qui est transparent laisse voir le bureau. Les pixels
  totalement transparents laissent passer les clics.
- **Limites** : les fichiers doivent être dans le dossier du thème, 240 images au plus par
  état, 4096 px de côté au plus.

### Droits

N'utilise que des images que tu as le droit de redistribuer si tu publies le thème, et
indique `author` et `license` dans le manifeste. Le thème par défaut est sous la licence du
projet (Apache-2.0).

Un thème invalide est ignoré : la mascotte par défaut s'affiche, et la raison est écrite dans
`data/logs/ui.log`.
