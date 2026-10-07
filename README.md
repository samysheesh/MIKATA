# MIKATA

Assistant vocal local pour PC Windows 11 : on dit « MIKATA », il écoute, transcrit,
interroge un LLM cloud, agit sur le PC et répond à voix haute dans le casque.

- Architecture (fait foi) : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Instructions pour Claude Code : [`CLAUDE.md`](CLAUDE.md)
- Lancement : `uv run python -m mikata` (options : `--check-config`, `--config CHEMIN`, `--no-tray`, `--no-audio`, `--no-llm`, `--ask "question"`)
- Réglages : `config.yaml` ; secrets : `.env` (copier `.env.example`, jamais commité) : `LLM_API_KEY`, `SEARCH_API_KEY` (Tavily, voir phase 4)
- Logs : `data/logs/mikata.log` (rotation 5 Mo × 5, anciennes archives à la corbeille)
- Modèles : `uv run python scripts/download_models.py` (dans `models/`, non commités ; Whisper ~1,6 Go, voix ~420 Mo)

## Avancement

| Phase | Contenu | État |
| --- | --- | --- |
| 0 | Squelette : config, logs, événements, icône, CI | fusionnée |
| 1 | Audio + mot d'activation (`hey_jarvis`), bip, raccourci, priorité basse | fusionnée |
| 2 | VAD Silero + Whisper GPU, transcription spéculative, retrait du mot d'activation | fusionnée |
| 3 | LLM en streaming + voix (Piper / Kokoro), commandes en parallèle | fusionnée |
| 4 | Outils (system, apps, web, media, ui, memory), réflexion choisie à la voix, coût par commande | fusionnée |
| 5 | Outils puissants (clavier/souris, fichiers, PowerShell, extinction) + garde-fou | fusionnée |
| 6 | Confort : fenêtre de suivi, barge-in, annulation, mémoire long terme | livrée, à valider sur le PC |
| 7 | Voir `docs/ARCHITECTURE.md` §11 | à faire |

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

## Phase 4 — vérification sur le PC Windows

Critères d'acceptation : « ouvre Spotify », « volume à 30 », « pause » répondent « C'est fait »
en **un seul appel LLM** ; « cherche la météo à Paris » répond juste. En plus : deux commandes
enchaînées en parallèle (« ouvre Spotify » puis « ouvre Brave ») sans refus.

Comment ça marche :

- **Outils** (22, un module par famille dans `mikata/tools/`) : `get_time`, `system_status`,
  `set_timer`, `set_reminder`, `lock` ; `open_app`, `close_app`, `focus_window`,
  `list_windows` ; `open_url`, `web_search`, `fetch_page`, `get_weather` ; `media_control`,
  `set_volume`, `mute`, `set_output_device` ; `show_text`, `clipboard_read`, `clipboard_write` ;
  `remember`, `forget`. Les appels d'une même réponse s'exécutent en parallèle.
- **Chemin court-circuité** : si tous les outils appelés sont terminaux (ouvrir, volume,
  pause...) et réussissent, MIKATA joue « C'est fait » (en cache) sans rappeler le LLM.
  Sinon les résultats repartent au LLM (8 étapes au plus).
- **Réflexion choisie à la voix** : par défaut `reasoning_effort: none` (premier mot le plus
  tôt). Un mot-clé en **début ou fin** de commande la règle pour cette commande : « réfléchis »
  → low, « réfléchis bien » → medium, « réfléchis à fond » → high. Il est détecté par le code
  (variantes « réfléchi », « réfléchit »... acceptées), retiré du texte avant l'appel, et
  configurable (`llm.reasoning_keywords`). Filet : si un outil échoue, ou au-delà de
  `llm.reasoning_fallback_steps` (2) étapes d'outils, la suite de la commande passe à `low`.
- **Coût** : une ligne `Tokens #n : entrée ... (dont cache ...), sortie ..., coût ≈ ... $ ;
  cumul ... $ sur 5.00 $ de crédit (... %, ≈ N commandes restantes)` par commande, cumul gardé
  dans `data\llm_usage.json` (tarifs dans la section `llm` de `config.yaml`).
- Aucune suppression de fichier : `show_text` écrit dans `data\show_text\`, `remember` dans
  `data\memory.json` (30 faits au plus, réécrit en place).
- Pas dans cette phase : `shutdown` / `restart` / `sleep` (phase 5, avec les 5 s pour
  annuler et le garde-fou). Minuteurs et rappels sont en mémoire : perdus si MIKATA
  redémarre. Lecture et pause sont la même touche média (bascule).

### 1. Clé de recherche web : Tavily (recommandé)

Tavily plutôt que Brave : 1 000 recherches gratuites par mois **sans carte bancaire**, et une
réponse résumée pensée pour les LLM. Brave Search n'a plus de palier gratuit sans carte
(5 $ de crédit mensuel, carte obligatoire). La météo, elle, passe par Open-Meteo, sans clé.

1. Crée un compte sur https://app.tavily.com (Google, GitHub ou e-mail).
2. Dans le tableau de bord, section **API Keys**, copie la clé (elle commence par `tvly-`).
3. Dans `.env` : `SEARCH_API_KEY=tvly-...` (la section `web` de `config.yaml` est déjà sur
   `search_provider: tavily`). Pour Brave : clé créée sur
   https://api-dashboard.search.brave.com (carte demandée) et `search_provider: brave`.

### 2. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-4
git pull
uv sync
```

Attendu : installation de `pycaw`, `comtypes`, `win11toast`, `rapidfuzz`, `trafilatura`,
`pyperclip`.

### 3. Lint, tests, config

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m mikata --check-config
```

Attendu : tout en `passed` ; puis `LLM : deepseek-v4.1-flash @ https://opencode.ai/zen/v1
(réflexion par défaut : none)`, `Recherche web : tavily`, `SEARCH_API_KEY : définie` et
`Configuration valide`.

### 4. Les quatre commandes du critère, sans micro

`--ask` passe par la même boucle que la voix (préchauffage compris) et dit la réponse dans le
casque. Lance Spotify et mets un morceau avant la 3e commande.

```powershell
uv run python -m mikata --ask "ouvre Spotify"
uv run python -m mikata --ask "volume à 30"
uv run python -m mikata --ask "pause"
uv run python -m mikata --ask "cherche la météo à Paris"
```

Attendu pour les trois premières : l'action faite, « C'est fait » dans le casque, et dans la
console :

```
Outil open_app(name='Spotify') -> ok en 15 ms : Spotify ouvert.
Tokens --ask : entrée 2650 (dont cache 2560), sortie 18 (dont réflexion 0), 1 appel(s) ; coût ≈ 0.00010 $ ; cumul ...
--ask : premier token 900 ms, premier son 1000 ms, réflexion none ; chemin : court-circuit (outil terminal, 1 appel(s) LLM) ; issue : ok.
```

`1 appel(s) LLM` est le critère. Pour la météo : `Outil get_weather(city='Paris') -> ok`,
`chemin : 1 étape(s) d'outils, 2 appel(s) LLM`, et une réponse à comparer avec
https://meteofrance.com (température actuelle, temps, min / max du jour).

### 5. Les mêmes à la voix, et en parallèle

```powershell
uv run python -m mikata
```

1. Dis « hey jarvis, ouvre Spotify », « hey jarvis, volume à 30 », « hey jarvis, pause »,
   « hey jarvis, cherche la météo à Paris ». Puis, dans un autre PowerShell :

   ```powershell
   Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "chemin : " | Select-Object -Last 4
   ```

   Attendu : `court-circuit (outil terminal, 1 appel(s) LLM)` pour les trois actions,
   `1 étape(s) d'outils, 2 appel(s) LLM` pour la météo, chacune avec `réflexion none` juste
   après le `premier son`.
2. Dis « hey jarvis, ouvre Spotify » puis, **dès ta phrase finie**, « hey jarvis, ouvre
   Brave ». Attendu : deux bips, `Nouvelle commande pendant 1 autre(s) en cours.`, Spotify et
   Brave ouverts, deux « C'est fait ».

### 6. Réflexion choisie à la voix

1. « hey jarvis, un train part à 17 h 05 et roule 76 minutes, à quelle heure arrive-t-il ? »
   (sans mot-clé : `réflexion none`, réponse peut-être fausse) ;
2. la même en ajoutant « réfléchis bien » à la fin (attendu : `Réflexion demandée : medium
   (mot-clé « réfléchis bien »).`, réponse 18 h 21) ;
3. « hey jarvis, réfléchis à fond, combien font 17 fois 23 ? » (attendu : `high`, 391) ;
4. « hey jarvis, ouvre Photoshop » (pas installé) : l'outil échoue, attendu `Réflexion
   relevée : none -> low (outil en échec).` et une phrase qui dit que l'appli est introuvable.

Niveau et latence du premier son de chaque réponse :

```powershell
Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "premier son . (\d+) ms apr.s la derni.re syllabe, r.flexion (\S+)" | Select-Object -Last 10 | ForEach-Object { $_.Matches[0].Groups[2].Value + " : " + $_.Matches[0].Groups[1].Value + " ms" }
```

Les lignes `Appel LLM n (réflexion ...) : premier élément ... ms, réflexion du modèle ... ms`
détaillent chaque appel (temps passé à réfléchir côté fournisseur).

### 7. Tour des autres outils

| Dis « hey jarvis, ... » | Attendu |
| --- | --- |
| « mets un minuteur de 10 secondes » | « C'est fait », puis 10 s après : 3 notes, « Le minuteur de 10 secondes est terminé. » et une notification Windows |
| « rappelle-moi dans 1 minute de boire de l'eau » | 1 min après : « Rappel : boire de l'eau. » + notification |
| « donne-moi une recette de crêpes et affiche-la » | résumé oral + la recette dans le Bloc-notes (`data\show_text\`) |
| « copie bonjour à tous dans le presse-papiers » | Ctrl+V colle « bonjour à tous » |
| « qu'est-ce que j'ai copié ? » | il lit le presse-papiers |
| « retiens que mon chien s'appelle Rex », puis « nouvelle conversation », puis « comment s'appelle mon chien ? » | « Rex » (fait dans `data\memory.json`) |
| « oublie le nom de mon chien » | le fait disparaît de `data\memory.json` |
| « comment va le PC ? » | processeur, mémoire, carte graphique (charge, température) |
| « quelles fenêtres sont ouvertes ? » puis « mets Spotify au premier plan » | liste, puis Spotify devant |
| « ferme Brave » | toutes les fenêtres de Brave fermées |
| « mets le son sur les haut-parleurs » / « remets le casque » | sortie par défaut de Windows changée (la voix de MIKATA reste sur `audio.output_device`) |
| « monte le son » / « coupe le son » / « remets le son » | volume +10, sourdine, son remis |
| « ouvre YouTube » | youtube.com dans le navigateur |
| « cherche les résultats de la Ligue 1 » | réponse tirée de `web_search` (Tavily) |
| « verrouille le PC » (à garder pour la fin) | écran de verrouillage |

### 8. Coût

```powershell
Select-String -Path data\logs\mikata.log -Encoding utf8 -Pattern "^.*Tokens #" | Select-Object -Last 5
Get-Content data\llm_usage.json
```

Attendu : une ligne par commande avec les tokens et le coût, et le cumul (`cost_usd`,
`commands`, depuis `since`). Une action simple coûte de l'ordre de 0,0001 $ quand le début de
la requête (prompt et outils, identiques à chaque appel) est facturé au tarif « cache », et
environ 0,001 $ sinon : la part `dont cache` de la ligne `Tokens` le montre.

## Phase 5 — vérification sur le PC Windows

Critère d'acceptation : des tests automatisés prouvent qu'aucune commande de la liste noire ne
s'exécute et que `delete_file` finit toujours dans la corbeille.

Comment ça marche :

- **Nouveaux outils** : `type_text`, `press_keys`, `mouse` (clavier et souris, accents
  compris) ; `find_files`, `read_file`, `write_file`, `move_file`, `delete_file` ;
  `run_powershell`, `run_script` (scripts de `scripts\user\`) ; `shutdown`, `restart`, `sleep`
  (reportés de la phase 4).
- **Garde-fou** (`mikata\agent\guard.py`), appliqué par le code, jamais par le LLM :
  - **Supprimer = corbeille.** `delete_file` passe par `send2trash`, et rien d'autre. Refus
    plutôt que risque : élément sur un lecteur sans corbeille (clé USB, réseau), plus gros que
    `guard.trash_max_mb` (Windows le supprimerait définitivement), ou corbeille sûre
    indisponible (`pywin32`, réinstallé pour ça). Jamais de repli vers une vraie suppression.
  - **Liste noire** : `Remove-Item` et ses alias (`rm`, `ri`, `del`, `erase`, `rd`, `rmdir`),
    `Clear-RecycleBin`, `Format-Volume`, `diskpart`, `cipher /w`, `sdelete`,
    `vssadmin delete`, `.Delete()`, `git clean`, `shutil.rmtree`, élévation administrateur,
    commandes encodées... Refusées dans PowerShell, dans les scripts, en argument, et même
    tapées au clavier ; Maj+Suppr aussi.
  - **Confirmation orale** pour les commandes risquées : arrêter un programme, toucher au
    registre, aux services, installer, écraser un fichier sans sauvegarde... MIKATA demande
    « Tu confirmes ? », bipe, et écoute sans mot d'activation (`guard.confirm_timeout_s`,
    6 s). Seul un oui clair lance la commande. `guard.confirm_risky: false` coupe ces
    questions, pas la liste noire.
  - **Écrasement = sauvegarde** : `write_file` et `move_file` copient d'abord l'ancien fichier
    dans `data\backups\`, mis à la corbeille au bout de 14 jours.
  - **Extinction / redémarrage / veille** : « Extinction dans 5 secondes. Dis stop pour
    annuler. », bip, et « stop » (ou « annule ») l'arrête.
  - **Journal** : chaque appel d'outil, avec son issue (OK, REFUSÉ, ANNULÉ, ÉCHEC), dans
    `data\actions.log`.
  - Les fichiers ne sont touchés que dans `guard.allowed_dirs` (ton dossier perso).
- **Correctif phase 4** : `open_app` ne choisit plus un nom qui ressemble seulement
  (« ouvre Photoshop » ouvrait Photos) : nom identique, mots du nom, sigle (« vs code »), ou
  « application introuvable : X ».
- **Prompt système retravaillé** : MIKATA devine l'intention au lieu de lire la transcription
  au pied de la lettre (« stop process name notepad », « arrête notepad » et « ferme le
  bloc-notes » ferment tous le Bloc-notes). Il corrige les noms mal entendus à l'aide de la
  liste des applications installées, désormais dans le prompt. Il agit sur l'interprétation la
  plus probable au lieu de poser une question, préfère les outils simples à PowerShell,
  répond court et ne parle jamais de PowerShell ni de syntaxe. Le prompt donne 5 exemples.
  Nouvel outil `close_window` : une seule fenêtre, la fenêtre active par défaut (« ferme
  cette fenêtre »).
- Sans micro (`--ask`), une commande risquée est refusée (personne pour dire oui), et une
  extinction a lieu après 5 s : **ne lance pas `--ask "éteins le PC"`** sans le vouloir.

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-5
git pull
uv sync
uv run python -c "import send2trash.win.modern; print('corbeille sûre : OK')"
```

Attendu : installation de `pywin32`, puis `corbeille sûre : OK`.

### 2. Le critère : tests automatisés

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_guard_criteria.py -v
```

Attendu : tout en `passed`. Le dernier fichier contient le critère :
- `test_blacklisted_powershell_never_runs` (et ses variantes pour les scripts, les arguments et
  le texte tapé) : une soixantaine de commandes de la liste noire, aucune n'est lancée ;
- `test_delete_file_always_ends_in_trash` : fichier, dossier, sous-dossier finissent dans la
  corbeille, avec toute fonction de suppression définitive piégée pendant le test ;
- `test_no_permanent_deletion_anywhere_in_mikata` : aucun appel de suppression définitive dans
  le code de MIKATA.

### 3. Correctif open_app

```powershell
uv run python -m mikata --ask "ouvre Photoshop"
uv run python -m mikata --ask "ouvre visual studio code"
```

Attendu : pour Photoshop (non installé), `Outil open_app(name='Photoshop') -> ÉCHEC ... :
application introuvable : Photoshop`, Photos ne s'ouvre pas et MIKATA le dit. Pour
VS Code : il s'ouvre (« C'est fait »).

### 4. Fichiers, corbeille et sauvegardes (sans micro)

```powershell
uv run python -m mikata --ask "crée sur le bureau un fichier essai-mikata.txt qui contient bonjour"
uv run python -m mikata --ask "remplace le contenu de essai-mikata.txt sur le bureau par au revoir"
Get-ChildItem -Recurse data\backups | Select-Object FullName
uv run python -m mikata --ask "mets essai-mikata.txt du bureau à la corbeille"
explorer shell:RecycleBinFolder
```

Attendu : le fichier apparaît sur le bureau ; après le remplacement, l'ancienne version
(« bonjour ») est dans `data\backups\<date>\essai-mikata.txt` ; après la suppression, le fichier
n'est plus sur le bureau mais **dans la Corbeille** (fenêtre ouverte par la dernière commande),
d'où « Restaurer » le remet en place.

### 5. Liste noire en vrai

```powershell
New-Item $HOME\Desktop\garde-fou.txt -ItemType File
uv run python -m mikata --ask "lance la commande PowerShell Remove-Item $HOME\Desktop\garde-fou.txt"
Test-Path $HOME\Desktop\garde-fou.txt
Get-Content data\actions.log -Tail 5
```

Attendu : MIKATA répond que c'est refusé (et propose la corbeille), `Test-Path` répond `True`,
et `actions.log` contient une ligne `REFUSÉ | run_powershell(command='Remove-Item ...') |
refusé par le garde-fou : suppression définitive interdite ...`.

### 6. Confirmation orale, « stop », clavier (avec le micro)

```powershell
notepad
uv run python -m mikata
```

1. « hey jarvis, lance la commande PowerShell Stop-Process -Name notepad ». Attendu : « Cette
   commande va arrêter un programme. Tu confirmes ? », un bip ; réponds « non » : le Bloc-notes
   reste ouvert (`ANNULÉ` dans `data\actions.log`). Recommence et réponds « oui » : il se ferme.
2. Rouvre le Bloc-notes et clique dedans. « hey jarvis, tape bonjour à tous, ça va ? » :
   le texte s'écrit avec les accents. « hey jarvis, appuie sur ctrl+a » : tout est sélectionné.
   « hey jarvis, appuie sur maj+suppr » : refusé.
3. « hey jarvis, mets le PC en veille ». Attendu : « Mise en veille dans 5 secondes. Dis stop
   pour annuler. », un bip ; dis « stop » : « Mise en veille annulée. ». Pour le vrai test,
   recommence sans rien dire : le PC se met en veille au bout de 5 s. (Même chose avec
   « redémarre le PC » et « éteins le PC », après avoir tout enregistré.)

### 7. Scripts à toi

```powershell
Set-Content scripts\user\bonjour.ps1 'Write-Output "Bonjour $env:USERNAME, il est $(Get-Date -Format HH:mm)"'
uv run python -m mikata --ask "lance mon script bonjour"
```

Attendu : MIKATA dit le message du script. Un script qui contient une commande de la liste
noire (par exemple `Remove-Item`) est refusé avant d'être lancé.

### 8. Le prompt sur 20 phrases approximatives

```powershell
uv run python scripts\eval_prompt.py
```

Chaque phrase (« stop process name notepad », « ouvre spotifaille », « lance Team », « mets
le son à trente », « fais contrôle S »...) passe par le vrai LLM et la vraie boucle agent.
**Rien n'est fait sur le PC** : les actions sont simulées et seulement affichées. Une phrase
réussit si le bon outil est appelé avec les bons arguments, sans PowerShell, sans question
de précision, et avec une réponse courte qui ne parle ni de PowerShell ni de syntaxe.

Attendu, pour chaque phrase :

```
 1. OK  « stop process name notepad »
      appels : close_app(name='Bloc-notes')
      dit : « C'est fait. » (1 appel(s) LLM, 950 ms)
```

puis `Réussies : 20/20 ; en un seul appel LLM : 19 ; coût ≈ 0.01 $` (la météo demande deux
appels). Une phrase en échec affiche `problème : ...`. Pour en rejouer quelques-unes :
`--only 1 2 3`. Pour garder le détail : `--json data\eval_prompt.json`.

## Phase 6 — vérification sur le PC Windows

Critère d'acceptation : dire MIKATA pendant une réponse la coupe en moins de 300 ms ; une
question de suivi passe sans redire le mot d'activation.

Comment ça marche :

- **Fenêtre de suivi** : après une réponse, si plus rien n'est en cours, MIKATA écoute
  `agent.follow_up_s` secondes (6 s) sans mot d'activation (icône en état « suivi »). Ce que
  tu dis devient une nouvelle commande, avec l'historique (« et à Tokyo ? »). Si tu ne dis
  rien : bip descendant, retour au repos. Pas de fenêtre tant qu'une autre commande tourne :
  c'est la dernière réponse qui l'ouvre. `follow_up_s: 0` la désactive.
- **Barge-in** : « MIKATA » (ou Ctrl+Alt+M) pendant qu'une réponse est jouée coupe **le son de
  cette réponse** tout de suite, et MIKATA t'écoute. Les autres commandes en parallèle
  continuent (décision de la phase 3) ; leurs réponses attendent que tu aies fini de parler au
  lieu de parler par-dessus toi. Le log donne la latence de coupure (`Barge-in : réponse #N
  coupée ≈ X ms après la fin du mot`).
- **Annulation** : une commande qui n'est que « stop », « annule », « laisse tomber »,
  « tais-toi », « c'est bon »... (`agent.cancel_phrases`) ne part pas au LLM. Elle annule la
  commande dont tu viens de couper la réponse si elle tourne encore, sinon la dernière commande
  en cours (LLM, recherche web...), avec un bip de fin. Dans la fenêtre de suivi, elle la
  ferme. Au repos, sans rien à annuler, « MIKATA, stop » reste une commande normale (pause de
  la musique). Une phrase plus longue (« annule le minuteur ») va toujours au LLM.
- **Garde-fou** : si tu coupes une annonce d'extinction, de redémarrage ou de mise en veille,
  l'action est **annulée** (tu ne l'as pas entendue) ; une question de confirmation coupée vaut
  « non ».
- **Mémoire long terme** (faite en phase 4, inchangée) : « retiens que... », « oublie que... »,
  jusqu'à 30 faits dans `data\memory.json`, donnés au LLM à chaque commande.
- Points restés ouverts en phase 5 :
  1. **Whisper** : le prompt initial (`stt.initial_prompt`, avec « MIKATA », Bloc-notes,
     PowerShell...) est complété par les noms des applications installées (menu Démarrer,
     les noms courts d'abord, `stt.prompt_max_chars` = 500 caractères au plus, Whisper n'en
     gardant que ~220 tokens). `stt.prompt_apps: false` revient au prompt fixe. Mesure :
     section 4.
  2. **« Appel LLM en échec (1/2) : pas de réponse en 20 s »** : la relance existait, mais
     elle n'arrivait qu'après 20 s d'attente, et sur la même connexion HTTP/2. Cause
     probable : une connexion restée ouverte pendant l'inactivité mais coupée en route
     (box, serveur). Désormais, si le serveur n'envoie rien en `llm.first_chunk_timeout_s`
     (8 s ; réflexion none ou low seulement, « réfléchis bien » garde 20 s), l'appel échoue
     tout de suite, la connexion est **recréée** et la relance part dessus. Le log le dit :
     `Connexion LLM recréée (...)`, puis `Appel LLM réussi à la tentative 2/2.`.
  3. **Bloc-notes de Windows 11** : `close_app`, `close_window`, `focus_window` et `open_app`
     connaissent les noms français des applis Windows et ceux de leurs processus (Bloc-notes
     ↔ notepad, Calculatrice ↔ CalculatorApp, Paramètres ↔ SystemSettings, Word ↔ WINWORD,
     Paint ↔ mspaint...). Le nom du processus est relu autrement quand psutil n'y arrive pas,
     et une appli du Store affichée par `ApplicationFrameHost` prend le nom de son vrai
     processus. Si aucune fenêtre ne correspond, l'erreur donne au LLM la liste des fenêtres
     ouvertes pour qu'il réessaie avec le bon nom.
  4. **« lance Team »** : le prompt dit maintenant que seules comptent les applications
     installées : si une seule ressemble au nom entendu (Steam), MIKATA l'ouvre sans demander
     « Teams ou Steam ? ».
  5. **`type_text`** : ton correctif (collage par le presse-papiers) est gardé tel quel.
- **Proposition, à valider** : garder la conversation sur disque. Aujourd'hui l'historique
  est oublié après 10 min sans commande, et au redémarrage. Avec `agent.history_persist:
  true`, il est écrit dans `data\history.json` à chaque échange, relu au démarrage, et gardé
  `agent.history_persist_h` heures sans commande (24 h) au lieu de 10 min. Même plafond
  (`history_max_tokens`, les plus anciens échanges sautent) ; « nouvelle conversation » le
  vide. **Désactivé par défaut** en attendant ton accord : essai en section 8.

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-6
git pull
uv sync
```

### 2. Lint et tests (dont le critère)

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_app_phase6.py -v
```

Attendu : tout en `passed`. `test_app_phase6.py` rejoue le critère avec micro, casque et LLM
simulés :
- `test_barge_in_cuts_the_answer_in_under_300_ms` : la réponse est coupée en moins de 300 ms,
  la suivante est dite ;
- `test_follow_up_question_without_wake_word` : « et à Tokyo » sans mot d'activation part au
  LLM avec l'historique, puis bip de fin après le silence ;
- `test_barge_in_does_not_cancel_other_commands`, `test_cancel_command_cancels_the_running_command`,
  `test_stop_after_barge_in_does_not_reach_the_llm`,
  `test_barge_in_on_a_shutdown_announcement_cancels_it`...

### 3. Bloc-notes de Windows 11 (sans micro)

```powershell
notepad
uv run python -c "from mikata.tools.desktop import WindowsDesktop; [print(w) for w in WindowsDesktop().windows()]"
uv run python -m mikata --ask "stop process name notepad"
notepad
uv run python -m mikata --ask "ferme le bloc-notes"
```

Attendu : la première commande liste une ligne `WindowInfo(handle=..., title='Sans titre -
Bloc-notes', process='Notepad')` (un nom de processus non vide). Les deux `--ask` ferment le
Bloc-notes (« C'est fait »), avec `Outil close_app(name='Bloc-notes') -> ok` dans la sortie (le LLM peut aussi choisir `close_window`).
Si le Bloc-notes demande d'enregistrer, c'est la croix normale : rien n'est tué.

### 4. Prompt de Whisper : mesure

Avec le jeu de test enregistré en phase 2 (`data\stt_eval`) :

```powershell
uv run python scripts\eval_stt.py run --compute-type int8_float16 --prompt none config apps
```

Attendu : les trois prompts affichés (le dernier avec `Applications : ...`), puis un tableau
avec une ligne par prompt (WER, latence médiane et p95) et les pires phrases de chacun.
`apps` devrait avoir un WER inférieur ou égal à `config` sur les noms d'applis, pour une
latence à peu près égale. Si `apps` est moins bon, mets `stt.prompt_apps: false` dans
`config.yaml` (et dis-le moi avec le tableau).

### 5. Fenêtre de suivi (avec le micro)

```powershell
uv run python -m mikata
```

1. « hey jarvis, quelle heure est-il ? ». Après la réponse, **sans** « hey jarvis » : « et à
   Tokyo ? ». Attendu : l'heure de Tokyo. Puis ne dis rien : bip descendant après 6 s.
2. « hey jarvis, ouvre Spotify ». Après « C'est fait », dans les 6 s : « mets le son à
   trente ». Attendu : le volume change.
3. « hey jarvis, quelle heure est-il ? », puis dans la fenêtre : « stop ». Attendu : un bip,
   rien d'autre (pas d'appel LLM : aucune ligne `Appel LLM` dans la console).

### 6. Barge-in et annulation (avec le micro)

1. « hey jarvis, raconte-moi une longue histoire ». Pendant la réponse : « hey jarvis ».
   Attendu : la voix se tait aussitôt, bip, MIKATA écoute ; « quelle heure est-il ? » est
   répondu. Dans un autre PowerShell :

   ```powershell
   Select-String -Path data\logs\mikata.log -Pattern "Barge-in" | Select-Object -Last 3
   ```

   Attendu : `Barge-in : réponse #N coupée ≈ X ms après la fin du mot (détection ... +
   traitement ... + sortie ... ms)` avec **X < 300**.
2. Lance de la musique dans Spotify. « hey jarvis, raconte-moi une histoire », puis pendant la
   réponse « hey jarvis, stop ». Attendu : la voix se tait, un bip, et la musique continue
   (« stop » n'est pas parti au LLM). Au repos, « hey jarvis, stop » met bien la musique en
   pause.
3. « hey jarvis, cherche les dernières nouvelles sur la conquête spatiale », puis tout de
   suite « hey jarvis, annule ». Attendu : bip, pas de réponse, et dans le log
   `« annule » -> commande #N annulée`.
4. En parallèle : « hey jarvis, cherche la météo à Tokyo cette semaine », et pendant que
   MIKATA cherche, « hey jarvis, quelle heure est-il ? ». Puis pendant que l'une des réponses
   est dite, « hey jarvis » : seule la réponse en cours est coupée, l'autre commande est dite
   ensuite.
5. Garde-fou : « hey jarvis, mets le PC en veille », et pendant « Mise en veille dans 5
   secondes... » : « hey jarvis, quelle heure est-il ? ». Attendu : l'heure, et le PC ne se met
   **pas** en veille (`Réponse coupée avant l'écoute : vaut « stop »` dans le log).

### 7. Relance du LLM

Rien à provoquer : si l'avertissement revient, le log montre maintenant la suite.

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Appel LLM en échec|Connexion LLM recréée|réussi à la tentative" | Select-Object -Last 10
```

Attendu : après `Appel LLM en échec (1/2) : pas de réponse en 8 s`, une ligne `Connexion LLM
recréée (aucune réponse)` puis `Appel LLM réussi à la tentative 2/2.`, le tout en moins de
10 s au lieu de 20.

### 8. Mémoire long terme, et la proposition « conversation gardée sur disque »

```powershell
uv run python -m mikata --ask "retiens que mon chat s'appelle Mochi"
uv run python -m mikata --ask "comment s'appelle mon chat ?"
Get-Content data\memory.json
```

Attendu : « Mochi » (le fait est relu à chaque démarrage), et le fait dans `memory.json`.

Pour essayer la conversation gardée sur disque, mets `history_persist: true` dans la section
`agent:` de `config.yaml`, puis :

```powershell
uv run python -m mikata --ask "quelle est la capitale du Japon ?"
uv run python -m mikata --ask "et combien d'habitants ?"
Get-Content data\history.json
```

Attendu : la seconde réponse parle de Tokyo alors que MIKATA a redémarré entre les deux. Dis-moi
si tu veux garder ce réglage activé par défaut, et pour combien de temps (`history_persist_h`).

### 9. Non-régression du prompt

```powershell
uv run python scripts\eval_prompt.py
```

Attendu : `Réussies : 20/20` ; « lance Team » ouvre Steam sans poser de question.

