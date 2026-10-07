# MIKATA

Assistant vocal local pour PC Windows 11 : on dit « MIKATA », il écoute, transcrit,
interroge un LLM cloud, agit sur le PC et répond à voix haute dans le casque.

- Architecture (fait foi) : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Instructions pour Claude Code : [`CLAUDE.md`](CLAUDE.md)
- Lancement : `uv run python -m mikata` (options : `--check-config`, `--config CHEMIN`, `--no-tray`, `--no-audio`, `--no-llm`, `--ask "question"`)
- Réglages : `config.yaml` ; secrets : `.env` (copier `.env.example`, jamais commité)
- Logs : `data/logs/mikata.log` (rotation 5 Mo × 5, anciennes archives à la corbeille)
- Modèles : `uv run python scripts/download_models.py` (dans `models/`, non commités ; Whisper ~1,6 Go, voix ~420 Mo)

## Avancement

| Phase | Contenu | État |
| --- | --- | --- |
| 0 | Squelette : config, logs, événements, icône, CI | fusionnée |
| 1 | Audio + mot d'activation (`hey_jarvis`), bip, raccourci, priorité basse | fusionnée |
| 2 | VAD Silero + Whisper GPU, transcription spéculative, retrait du mot d'activation | fusionnée |
| 3 | LLM en streaming + voix (Piper / Kokoro), commandes en parallèle | livrée, à valider sur le PC |
| 4 à 7 | Voir `docs/ARCHITECTURE.md` §11 | à faire |

## Développement

```powershell
uv sync                       # crée .venv avec Python 3.11 et les dépendances de uv.lock
uv run ruff check .           # lint
uv run ruff format --check .  # format
uv run pytest                 # tests (audio, GPU et API Windows simulés)
```

## Phase 0 — vérification sur le PC Windows

Critère d'acceptation : `python -m mikata` démarre, l'icône apparaît, la config est
validée, quitter est propre.

Toutes les commandes se lancent dans **PowerShell**, depuis le dossier du dépôt.

### 1. Prérequis (une seule fois)

```powershell
# uv (gestionnaire Python) ; rouvrir PowerShell ensuite
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version

# Récupérer le dépôt (si ce n'est pas déjà fait)
git clone https://github.com/samysheesh/MIKATA.git
cd MIKATA
```

### 2. Récupérer la branche et installer

```powershell
git fetch origin
git checkout phase-0
git pull
uv sync                                # télécharge Python 3.11 si besoin
Copy-Item .env.example .env            # secrets vides pour l'instant (utiles en phase 3)
```

### 3. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Attendu : `All checks passed!`, `... files already formatted`, puis tous les tests en `passed`.

### 4. La config est validée

```powershell
uv run python -m mikata --check-config
echo $LASTEXITCODE
```

Attendu : un résumé de la config, la ligne `SUCCESS | ... Configuration valide : ...\config.yaml`,
puis `0`. L'avertissement « LLM non renseigné » est normal jusqu'à la phase 3.

Contre-épreuve avec une config volontairement fausse :

```powershell
uv run python -m mikata --config tests\fixtures\config_invalide.yaml
echo $LASTEXITCODE
```

Attendu : `ERROR | ... Configuration invalide`, avec `wake.threshold` et `wake.treshold`
cités, puis `2`.

### 5. Choisir les périphériques audio

```powershell
uv run python scripts\list_audio_devices.py
```

Le tableau liste les périphériques WASAPI (`--all` pour toutes les API). Mets dans
`config.yaml` une sous-chaîne de ton micro (`audio.input_device`) et de ton casque
(`audio.output_device`), puis relance la commande : les deux dernières lignes doivent
commencer par `[OK]` (code de sortie 0). `[!!] ... aucun` ou `[!!] ... ambigu` = sous-chaîne
à corriger. (L'audio n'est ouvert qu'à partir de la phase 1.)

### 6. Démarrage, icône et arrêt propre

```powershell
uv run python -m mikata
```

Attendu :

1. la console affiche `MIKATA démarré, en attente.` ;
2. une icône ronde grise marquée « M » apparaît dans la zone de notification (sous Windows 11,
   elle peut être rangée derrière la flèche `^` ; info-bulle « MIKATA — repos ») ;
3. clic droit sur l'icône → **Quitter** : l'icône disparaît, la console affiche
   `Arrêt demandé (menu Quitter).` puis `MIKATA arrêté proprement.` et rend la main.

```powershell
echo $LASTEXITCODE                                   # 0
Get-Content data\logs\mikata.log -Tail 5             # mêmes lignes dans le fichier de log
```

Arrêt par Ctrl+C : relancer `uv run python -m mikata`, puis `Ctrl+C` dans la console →
`MIKATA arrêté proprement.`, `Interruption clavier (Ctrl+C).`, l'icône disparaît.

### 7. Sans console (mode du futur démarrage automatique)

```powershell
$p = Start-Process .\.venv\Scripts\pythonw.exe -ArgumentList "-m", "mikata" -PassThru
# -> l'icône apparaît ; clic droit -> Quitter
$p.WaitForExit(); $p.ExitCode                        # 0
Get-Process pythonw -ErrorAction SilentlyContinue    # rien : aucun processus restant
Get-Content data\logs\mikata.log -Tail 3             # se termine par « MIKATA arrêté proprement. »
```

## Phase 1 — vérification sur le PC Windows

Critères d'acceptation : bip < 150 ms après « hey jarvis », CPU du processus < 3 % au
repos, 0 déclenchement sur 30 min de vidéo en fond.

### 1. Récupérer la branche, installer, télécharger les modèles

```powershell
git fetch origin
git checkout phase-1
git pull
uv sync
uv run python scripts\download_models.py      # 3 lignes [OK] dans models\
```

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Attendu : tout en `passed` (le test sur le vrai modèle `hey_jarvis` tourne maintenant que
les modèles sont téléchargés).

### 3. Micro et casque

```powershell
uv run python scripts\list_audio_devices.py   # deux lignes [OK]
```

Si ton micro n'est branché que sur l'entrée 1 de l'interface, mets `input_channel: 1` dans la
section `audio` de `config.yaml` (sinon les deux canaux sont moyennés et le niveau baisse
de moitié).

### 4. Lancer MIKATA

```powershell
uv run python -m mikata
```

Attendu dans la console : `Priorité du processus : inférieure à la normale.`,
`Mot d'activation chargé : hey_jarvis`, `Sortie : « ... » (Windows WASAPI, ...)`,
`Micro : « ... »`, `Raccourci clavier actif : <ctrl>+<alt>+m`, `MIKATA démarré, en attente.`

### 5. Critère « bip < 150 ms »

Dis « hey jarvis » 10 fois, en laissant au moins 3 s entre deux essais. Chaque fois : un bip
court dans le casque et une ligne :

```
Mot d'activation détecté (score 0.97) : bip ≈ 130 ms après la fin du mot (détection 110 + traitement 5 + sortie 12 ms).
```

Dans un autre PowerShell, pendant que MIKATA tourne :

```powershell
Select-String -Path data\logs\mikata.log -Pattern "bip ≈" | Select-Object -Last 10
```

La fin du mot est estimée sur les 2 s d'audio enregistrées dans `data\wake_logs\` (un `.wav`
par détection, que tu peux écouter ou ouvrir dans Audacity). `traitement` + `sortie` est la
part de MIKATA (≈ 20 ms) ; `détection` est le temps que met le modèle à réagir après la fin du
mot. Si la médiane dépasse 150 ms, essaie `consecutive_frames: 1` dans la section `wake` de
`config.yaml` (gain de 80 ms), puis refais ce test et le test des 30 min (étape 7).

`Ctrl+Alt+M` doit aussi faire biper (ligne `Raccourci clavier : bip lancé`).

### 6. Critère « CPU < 3 % au repos »

Laisse MIKATA tourner 5 min sans parler, puis :

```powershell
Select-String -Path data\logs\mikata.log -Pattern "CPU du processus" | Select-Object -Last 5
Get-Process python | Select-Object Id, PriorityClass, CPU
```

Attendu : des lignes `CPU du processus : x % d'un cœur (y % du PC)` avec `y` < 3 (c'est la
valeur que montre le Gestionnaire des tâches ; `x` est la même mesure rapportée à un seul
cœur), `trames perdues : 0`, et `PriorityClass` = `BelowNormal`.

### 7. Critère « 0 déclenchement sur 30 min de vidéo »

```powershell
$debut = Get-Date
```

Lance 30 min de vidéo parlée (podcast, YouTube) sur les **haut-parleurs** (pas le casque, pour
que le micro l'entende), sans dire « hey jarvis ». Puis :

```powershell
(Get-ChildItem data\wake_logs -Filter *.wav | Where-Object LastWriteTime -gt $debut).Count   # 0
Select-String -Path data\logs\mikata.log -Pattern "Mot d'activation détecté" | Where-Object { $_.Line -ge $debut.ToString("yyyy-MM-dd HH:mm") }
```

Attendu : `0` et aucune ligne. Arrête MIKATA avec « Quitter » dans l'icône.

### 8. (Facultatif) Mode `collect_negatives`

Mets `collect_negatives: true` dans la section `wake` de `config.yaml`, lance MIKATA 1 min,
quitte, puis :

```powershell
Get-ChildItem data\negatives -Recurse -Filter *.wav | Select-Object -Last 5 FullName, Length
```

Attendu : des fichiers d'environ 320 Ko (10 s à 16 kHz). Remets `collect_negatives: false`.

## Phase 2 — vérification sur le PC Windows

Critères d'acceptation : transcription prête ≤ 100 ms après la fin de parole détectée
(médiane), VRAM ≤ 2 Go, début de phrase intact quand on enchaîne sans pause.

Comment ça marche : à la détection, l'enregistrement repart du **début du pré-roll** (les 2 s
d'avant), donc « hey jarvis ouvre Spotify » dit d'une traite arrive entier à Whisper ; le mot
d'activation et tout ce qui le précède sont ensuite retirés du texte. Le bip part en parallèle,
la capture ne s'arrête jamais.

### 1. Récupérer la branche, installer, télécharger les modèles

```powershell
git fetch origin
git checkout phase-2
git pull
uv sync                                   # installe aussi les DLL CUDA (~1 Go, une fois)
uv run python scripts\download_models.py  # + silero_vad.onnx et Whisper large-v3-turbo (~1,6 Go)
```

Attendu : des lignes `[OK]`, dont `[OK] .../models/silero_vad.onnx` et
`[OK] Whisper large-v3-turbo : ...\models\whisper\...`.

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Attendu : tout en `passed` (les tests sur les vrais modèles `hey_jarvis` et Silero tournent
maintenant qu'ils sont téléchargés).

### 3. GPU et DLL CUDA

```powershell
uv run python -c "from mikata.stt.whisper_local import add_cuda_dll_directories as a; print(a()); import ctranslate2; print('GPU CUDA :', ctranslate2.get_cuda_device_count())"
```

Attendu : deux dossiers (`...\nvidia\cublas\bin` et `...\nvidia\cudnn\bin`) puis `GPU CUDA : 1`.

Contre-épreuve du message d'erreur sans GPU (le GPU est masqué le temps d'une commande) :

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"; uv run python -m mikata --no-tray; echo $LASTEXITCODE; Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Attendu : `ERROR | ... Aucun GPU NVIDIA utilisable par Whisper ... nvidia-smi ...`, puis `3`.

### 4. Lancer MIKATA et critère « VRAM ≤ 2 Go »

```powershell
nvidia-smi --query-gpu=memory.used --format=csv     # avant : note la valeur
uv run python -m mikata
```

Attendu dans la console : `Chargement de Whisper (large-v3-turbo)...` puis
`Whisper large-v3-turbo prêt sur cuda (float16) en x s, VRAM +NNNN Mo.` avec NNNN ≤ 2048,
puis `MIKATA démarré, en attente.` Dans un autre PowerShell, pendant que MIKATA tourne :

```powershell
nvidia-smi --query-gpu=memory.used --format=csv     # après : l'écart avec « avant » doit être ≤ 2 Go
```

Si ça dépasse : `compute_type: int8_float16` dans la section `stt` de `config.yaml`.

### 5. Critère « début de phrase intact »

Dis, **sans pause** après le mot d'activation :

1. « hey jarvis ouvre Spotify »
2. « hey jarvis mets le volume à trente »
3. « hey jarvis pause » (commande finie avant même le bip)

Puis, **avec pause** : « hey jarvis » … (bip) … « quelle heure est-il ».
Et enfin « hey jarvis » sans rien dire : bip descendant au bout de ~4 s
(`Écoute terminée : aucune parole.`).

```powershell
Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "Commande :" | Select-Object -Last 4
```

Attendu : `Commande : « ouvre Spotify » (texte brut : « Hey Jarvis, ouvre Spotify. »)` etc. —
le texte brut commence bien par le mot d'activation, la commande est entière.

### 6. Critère « transcription prête ≤ 100 ms après la fin de parole »

Fais 20 commandes variées (« hey jarvis … »), puis :

```powershell
$v = Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "stt_pr.t \+(\d+) ms" | Select-Object -Last 20 | ForEach-Object { [int]$_.Matches[0].Groups[1].Value }
$v -join ", "
"Médiane : " + ($v | Sort-Object)[[math]::Floor($v.Count / 2)] + " ms"
```

Attendu : médiane ≤ 100 ms. Chaque ligne `Mesures :` dit aussi si le résultat spéculatif a
servi (`spéculatif`) ou s'il a fallu un second passage (`direct`).

### 7. Benchmark STT (`scripts/eval_stt.py`)

Enregistre une fois tes 50 commandes (phrases de `scripts\stt_phrases_fr.txt`, à adapter) :
Entrée, tu lis la phrase, l'enregistrement s'arrête quand tu te tais.

```powershell
uv run python scripts\eval_stt.py record
uv run python scripts\eval_stt.py run --compute-type float16 int8_float16 --beam-size 1 2
```

Attendu : un tableau WER / latence médiane / p95 pour les 4 réglages, puis les pires phrases.
Garde dans `config.yaml` le réglage qui donne le meilleur WER sans dépasser la VRAM.

## Phase 3 — vérification sur le PC Windows

Critères d'acceptation : premier son < 2 s après ta dernière syllabe (médiane sur 20 essais) ;
choix Piper ou Kokoro tranché.

Comment ça marche : la commande transcrite part telle quelle au LLM, qui corrige lui-même les
erreurs de transcription évidentes (« lance Team » → Steam). La réponse arrive en flux,
découpée en fragments (le premier dès une virgule après 6 mots) ; chaque fragment est
synthétisé pendant que le précédent est joué. Pas encore d'outils (phase 4) : pour une action,
MIKATA répond qu'il ne peut pas encore la faire.

**Commandes en parallèle** : « hey jarvis » est accepté dès que le micro est libre, même si une
commande précédente est encore en train de réfléchir ou de parler. Les deux avancent en même
temps ; seules les réponses parlées passent l'une après l'autre (deux voix ne se superposent
pas). Le bip de la nouvelle commande se mélange à la voix sans la couper.

### 1. Récupérer la branche, installer, télécharger les voix

```powershell
git fetch origin
git checkout phase-3
git pull
uv sync
uv run python scripts\download_models.py   # + voix Piper fr_FR-siwis-medium et Kokoro (~420 Mo)
```

Attendu : des lignes `[OK]`, dont `fr_FR-siwis-medium.onnx`, `fr_FR-siwis-medium.onnx.json`,
`kokoro-v1.0.onnx` et `voices-v1.0.bin`.

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Attendu : tout en `passed` (les tests sur les vraies voix tournent maintenant qu'elles sont
téléchargées).

### 3. Sans clé LLM : refus propre

Tant que `LLM_API_KEY` est vide dans `.env` :

```powershell
uv run python -m mikata
echo $LASTEXITCODE
```

Attendu : `ERROR | ... LLM non configuré, démarrage impossible :` avec la liste de ce qui manque
(`LLM_API_KEY absente ...`, et `llm.base_url / llm.model non renseignés ...` si c'est le cas),
puis `2`. Le micro n'est pas ouvert. Pour continuer à utiliser les phases 1 et 2 sans LLM :
`uv run python -m mikata --no-llm`.

### 4. Choisir la voix : Piper ou Kokoro

```powershell
uv run python scripts\eval_tts.py --play
```

Attendu : un tableau (chargement, synthèse du premier fragment de 8 mots, facteur temps réel)
puis les 5 phrases de chaque voix jouées dans le casque. La synthèse du premier fragment doit
rester sous 200 ms (colonne `OK`). Mesures en cloud (4 vCPU, plus lent que ton 5900X) : Piper
~90 à 190 ms, Kokoro ~700 à 1 100 ms. Choisis à l'oreille parmi les voix `OK` et mets
`backend: piper` ou `backend: kokoro` dans la section `tts` de `config.yaml`.

### 5. Brancher le LLM

Dans `.env` : `LLM_API_KEY=ta-cle`. Dans `config.yaml`, section `llm` : `base_url` (endpoint
compatible OpenAI de CommandCode, qui finit en général par `/v1`) et `model` (nom exact du modèle
« flash »). Puis, sans micro :

```powershell
uv run python -m mikata --ask "Dis-moi bonjour en une phrase."
```

Attendu : la réponse dite dans le casque et affichée, puis une ligne
`--ask : premier token NNN ms, premier son NNN ms, issue : ok.` Si `issue : server` : clé,
`base_url` ou `model` incorrects (le détail est dans `data\logs\mikata.log`).

### 6. Critère « premier son < 2 s »

```powershell
uv run python -m mikata
```

Attendu au démarrage : `Chargement de Whisper ...`, `Chargement de la voix (piper)...`,
`MIKATA démarré, en attente.`

Fais 20 commandes variées (« hey jarvis, quelle heure est-il », « hey jarvis, quelle est la
capitale du Canada », « hey jarvis, raconte-moi une blague courte »...). Chaque réponse
écrit une ligne :

```
Réponse #3 : premier son ≈ 1360 ms après la dernière syllabe (depuis fin_parole : stt_prêt +0 ms, llm_premier_token +650 ms, premier_son +870 ms, total +4100 ms) ; issue : ok.
```

Dans un autre PowerShell :

```powershell
$v = Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "premier son . (\d+) ms apr" | Select-Object -Last 20 | ForEach-Object { [int]$_.Matches[0].Groups[1].Value }
$v -join ", "
"Médiane : " + ($v | Sort-Object)[[math]::Floor($v.Count / 2)] + " ms"
```

Attendu : médiane < 2000 ms. La « dernière syllabe » est la dernière trame que le VAD juge
parlée : la vraie fin du mot peut être ~80 ms plus tard, garde donc un peu de marge
(< 1 900 ms). La ligne montre où va le temps : `llm_premier_token` est la part du fournisseur.

### 7. Commandes en parallèle, corrections, son de réflexion

1. Dis « hey jarvis, explique-moi comment marche un moteur électrique », puis, **dès ta phrase
   finie** (pendant qu'il réfléchit ou commence à répondre), « hey jarvis, combien font 17 fois
   23 ». Attendu : deux bips, la ligne `Nouvelle commande pendant 1 autre(s) en
   cours.`, puis les deux réponses l'une après l'autre (`Réponse #1` et `Réponse #2` dans le log).
2. Dis « hey jarvis, lance Team ». Attendu : la réponse parle de Steam (le LLM a corrigé la
   transcription ; il dit qu'il ne peut pas encore lancer d'application, c'est la phase 4).
3. Pose une question longue : si rien n'est dit au bout de 1,2 s après ta fin de phrase, un
   petit son de réflexion doit être joué.
4. Dis « hey jarvis, nouvelle conversation ». Attendu : « D'accord, nouvelle conversation. »
