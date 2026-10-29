# Entraîner le mot d'activation « MIKATA »

Sans modèle entraîné, MIKATA réagit à **« Hey Jarvis »** (modèle openWakeWord installé par
`download_models.py`) et au raccourci **Ctrl+Alt+M**. Ce guide produit deux fichiers, propres
à ta voix et à ton micro :

- `models/mikata.onnx` : le détecteur du mot, qui donne un score toutes les 80 ms ;
- `models/mikata_verifier.pkl` : le vérificateur, qui donne la probabilité que ce soit **toi**
  qui parles. Un « MIKATA » dit par une vidéo ou par quelqu'un d'autre est refusé.

Compte environ 4 h de collecte passive (sur plusieurs jours), 20 min d'enregistrement et 2 à
3 h d'entraînement sur Google Colab (GPU gratuit). Toutes les commandes se lancent dans
PowerShell, depuis le dossier du dépôt.

## 1. Collecter ta vie courante (au moins 4 h)

C'est la donnée la plus précieuse : elle apprend au modèle ce qui n'est **pas** « MIKATA ».
Dans `config.yaml`, mets `wake.collect_negatives: true`, lance MIKATA et vis normalement
(jeux, musique, vidéos, appels, pièce calme). Ne dis pas « MIKATA » pendant la collecte.
Elle s'arrête seule après `wake.negatives_max_hours` (12 h, ~115 Mo par heure).

```powershell
uv run python scripts\record_wakeword_samples.py plan   # ce qui est fait, ce qui manque
```

Remets `collect_negatives: false` à la fin.

## 2. Enregistrer tes « MIKATA » (≈ 20 min)

Avec ton micro habituel, à ta place, comme tu le diras tous les jours :

```powershell
uv run python scripts\record_wakeword_samples.py record --set train   # 100 prises
uv run python scripts\record_wakeword_samples.py record --set hard    # 36 mots proches x 3
```

- `train` couvre 9 situations : voix normale, basse, forte, fatiguée, souriante, de loin,
  tête tournée, avec du bruit, et dans une phrase (« MIKATA, ouvre Spotify »).
- `hard` enregistre des mots proches (mikado, Nikita, Akira, « ma carte »...) à refuser.
- Entrée garde la prise, `r` la refait, `q` quitte. Relancer reprend où tu t'étais arrêté ;
  `--only loin` refait une seule situation.

## 3. Un autre jour : 30 prises d'évaluation

```powershell
uv run python scripts\record_wakeword_samples.py record --set eval
uv run python scripts\record_wakeword_samples.py plan
```

Ces prises ne servent jamais à l'entraînement : elles mesurent le modèle sur un « autre jour ».
`plan` doit dire « tout est prêt ».

## 4. Entraîner `mikata.onnx` sur Google Colab (≈ 2 à 3 h)

```powershell
uv run python scripts\record_wakeword_samples.py pack   # --negative-hours 2 pour alléger
```

Cela crée `data\wakeword\mikata_wakeword_colab.zip` (~500 Mo). Ensuite :

1. Sur <https://colab.research.google.com>, crée un notebook, puis **Exécution > Modifier le
   type d'exécution > GPU T4**.
2. Dépose l'archive dans ton Google Drive (`Mon Drive`), puis lance ces cellules une par une :

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !cp /content/drive/MyDrive/mikata_wakeword_colab.zip /content/
   ```

   ```python
   !unzip -q -o /content/mikata_wakeword_colab.zip -d /content/mikata
   !python /content/mikata/train_wakeword.py setup   # ~15 min, ~17 Go de données publiques
   !python /content/mikata/train_wakeword.py train
   ```

   `train` finit par `[mikata] modèle prêt : /content/mikata/out/mikata.onnx` (une erreur
   TensorFlow ou tflite après cette ligne est sans effet). Si Colab se déconnecte, relance les
   mêmes cellules : les étapes déjà faites sont sautées.
3. Récupère le modèle et copie-le dans `models\` :

   ```python
   from google.colab import files
   files.download("/content/mikata/out/mikata.onnx")
   ```

## 5. Entraîner le vérificateur de ta voix (≈ 1 min, sur ton PC)

```powershell
uv run python scripts\train_verifier.py
```

Il apprend ta voix qui dit « MIKATA » (prises `train`) contre ta voix qui dit autre chose
(mots proches, et tes phrases de `data\stt_eval` si tu as utilisé `eval_stt.py`).

## 6. Mesurer et choisir les seuils (≈ 10 min)

```powershell
uv run python scripts\eval_wakeword.py --model models\mikata.onnx --verifier models\mikata_verifier.pkl
```

Le script rejoue tes prises d'évaluation dans 2 h de vie courante et affiche, pour chaque
seuil : faux déclenchements par heure, ratés et délai. Il termine par un **réglage conseillé**
à copier dans `config.yaml`. Objectif : moins de 0,2 faux déclenchement par heure et moins de
5 % de ratés. Sinon, ajoute des prises ou de la vie courante et réentraîne
(`train --user-copies 50`, `--steps 80000`...).

## 7. Passer en production

Dans `config.yaml` :

```yaml
wake:
  model: models/mikata.onnx
  verifier: models/mikata_verifier.pkl
  threshold: 0.50              # <- réglage conseillé
  consecutive_frames: 1        # <- réglage conseillé
  verifier_threshold: 0.70     # <- réglage conseillé
```

Tu peux retirer `"jarvis"` de `stt.wake_words`, puis :

```powershell
uv run python -m mikata --check-config
uv run python -m mikata
```

Un « MIKATA » raté ? Appuie sur Ctrl+Alt+M juste après : les 4 dernières secondes sont gardées
dans `data\wakeword\misses\` (`wake.save_misses`), utiles pour un prochain entraînement.
