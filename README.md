# MIKATA

Assistant vocal local pour PC Windows 11 : on dit « MIKATA », il écoute, transcrit,
interroge un LLM cloud, agit sur le PC et répond à voix haute dans le casque.

- Architecture (fait foi) : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Instructions pour Claude Code : [`CLAUDE.md`](CLAUDE.md)
- Lancement : `uv run python -m mikata` (options : `--check-config`, `--config CHEMIN`, `--no-tray`, `--no-audio`, `--no-llm`, `--ask "question"`)
- Réglages : `config.yaml` ; secrets : `.env` (copier `.env.example`, jamais commité) : `LLM_API_KEY`, `SEARCH_API_KEY` (Tavily, voir phase 4)
- Logs : `data/logs/mikata.log` (rotation 5 Mo × 5, anciennes archives à la corbeille)
- Interface (phase 8) : mascotte et réglages, lancées par MIKATA (`uv run python -m mikata.ui [--settings]`) ; captures dans [`docs/ui/`](docs/ui/), skins : [`docs/MASCOT_SKIN_GUIDE.md`](docs/MASCOT_SKIN_GUIDE.md)
- Licence : Apache-2.0 ([`LICENSE`](LICENSE)) ; composants tiers et leurs licences : [`THIRD_PARTY.md`](THIRD_PARTY.md)
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
| 6 | Confort : fenêtre de suivi, barge-in, annulation, mémoire long terme | fusionnée |
| 6b | Arrêt vocal sans mot d'activation, Whisper sur les suites courtes | fusionnée |
| 7 | Mot d'activation « MIKATA » (enregistrement, entraînement Colab, vérificateur de voix, benchmark), démarrage automatique, purge | fusionnée |
| 8 | Interface : mascotte de bureau (skins), fenêtre de réglages ; licence Apache-2.0, contrôle de secrets en CI | fusionnée |
| 9 | Finitions : ratés du mot d'activation notés au raccourci, tests de parallélisme déterministes | livrée, à valider sur le PC |

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
- **Fenêtre de suivi dans le bruit** (retour de test) : ce qui est entendu sans « MIKATA »
  passe deux filtres, et une phrase écartée ne reçoit **aucune réponse** : juste le bip de
  fin, puis le repos. MIKATA ne dit jamais « j'ai pas bien entendu » dans la fenêtre de
  suivi.
  1. **Transcription** : hésitation seule (« euh », « hmm »), texte de moins de 2 lettres ou
     incohérent (« la la la la »), ou confiance de Whisper sous
     `agent.follow_up_min_confidence` (0,2). La confiance va de 0 à 1 : probabilité moyenne
     des mots × probabilité qu'il y ait de la parole.
  2. **Jugement du LLM** (`agent.follow_up_use_llm_gate: true`) : on lui dit que la phrase a
     été entendue sans mot d'activation. Si elle ne s'adresse pas clairement à MIKATA
     (conversation avec tes amis, vidéo), il répond le marqueur `[IGNORER]`. Ce marqueur
     n'est jamais dit, et rien n'est fait ni gardé dans l'historique.
  La décision est toujours écrite dans le log : `Fenêtre de suivi : « ... » retenue
  (confiance 0.82 : ...)` ou `... ignorée (...)`.
- **Heures et unités lues correctement** (retour de test) : « 3 h 40 » est lu « trois heures
  quarante », « 14h05 » « quatorze heures cinq », « 9 h » « neuf heures », « 18:45 »
  « dix-huit heures quarante-cinq ». Après un nombre : km/h, km, min, s, ms, h (durée), °C,
  %, kg, Go, Mo, € et $ en toutes lettres, au singulier ou au pluriel (« 1 min » -> « une
  minute »).
- **Réponses longues dites, pas écrites** (retour de test : l'histoire avait été collée dans
  le Bloc-notes) : histoire, blague, explication sont racontées à voix haute, en version
  courte de 3 à 5 phrases. MIKATA n'ouvre une appli et n'écrit dans une fenêtre ou un
  fichier que si tu le demandes explicitement (« écris-la dans un fichier »). Trois exemples
  ont été ajoutés au prompt, et trois phrases à `scripts\eval_prompt.py` (23 en tout).
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
- **Conversation gardée sur disque** (activé, comme demandé) :
  - `agent.history_persist: true` : l'historique est écrit dans `data\history.json` à chaque
    échange et relu au démarrage.
  - Il est gardé `agent.history_persist_h` = 168 heures (7 jours) sans commande, au lieu de
    10 min. La durée se change aussi à la voix (« garde la conversation un mois », outil
    `set_history_duration`) ; ce réglage est gardé dans `history.json` et prime sur
    `config.yaml`.
  - Le plafond `history_max_tokens` (6 000) reste. Quand il est atteint, les échanges les
    plus anciens ne sont plus supprimés : le LLM les **résume** en arrière-plan
    (`agent.history_summarize`), et ce résumé ouvre l'historique envoyé.
  - « nouvelle conversation » vide tout : échanges et résumé (pas la durée choisie).

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

### 8. Mémoire long terme et conversation gardée sur disque

```powershell
uv run python -m mikata --ask "retiens que mon chat s'appelle Mochi"
uv run python -m mikata --ask "comment s'appelle mon chat ?"
Get-Content data\memory.json
```

Attendu : « Mochi » (le fait est relu à chaque démarrage), et le fait dans `memory.json`.

La conversation est maintenant gardée sur disque par défaut :

```powershell
uv run python -m mikata --ask "nouvelle conversation"
uv run python -m mikata --ask "quelle est la capitale du Japon ?"
uv run python -m mikata --ask "et combien d'habitants ?"
uv run python -m mikata --ask "garde la conversation un mois"
Get-Content data\history.json
```

Attendu :
- la troisième réponse parle de Tokyo, alors que MIKATA a redémarré entre les deux ;
- la quatrième confirme « un mois » ;
- `history.json` contient `"keep_s": 2592000.0` et les échanges.

Le résumé se voit dans le log quand le plafond est atteint, après une longue conversation :
`Historique : N ancien(s) échange(s) résumé(s)`, puis un champ `"summary"` dans
`history.json`.

### 9. Heures et unités dites à voix haute

```powershell
uv run python -m mikata --ask "dis exactement : il est 3 h 40, rendez-vous à 14h05, 90 km/h pendant 1 min et il fait -5 °C"
uv run python -m mikata --ask "quelle heure est-il ?"
```

Attendu, dans le casque : « il est trois heures quarante, rendez-vous à quatorze heures cinq,
90 kilomètres heure pendant une minute et il fait moins 5 degrés ». L'heure actuelle est
dite « ... heures ... », jamais « h ».

### 10. Fenêtre de suivi dans le bruit (avec le micro)

```powershell
uv run python -m mikata
```

1. « hey jarvis, quelle heure est-il ? ». Pendant la fenêtre de suivi, tousse, tape sur le
   bureau ou fais « euh... ». Attendu : bip de fin, aucune réponse. Dans le log :
   `Fenêtre de suivi : « ... » ignorée (...)`, ou `Rien compris`.
2. « hey jarvis, quelle heure est-il ? ». Pendant la fenêtre, parle à quelqu'un d'autre
   (« tu viens manger ce soir ? »). Attendu : bip de fin, aucune réponse, et dans le log
   `Fenêtre de suivi : « tu viens manger ce soir » ignorée (le LLM juge que la phrase ne
   s'adresse pas à MIKATA)`.
3. « hey jarvis, quelle heure est-il ? », puis « et à Tokyo ? ». Attendu : l'heure de Tokyo,
   et `Fenêtre de suivi : « Et à Tokyo ? » retenue (confiance 0.xx : ...)` dans le log.

Pour régler : relève les confiances des phrases retenues et ignorées
(`Select-String -Path data\logs\mikata.log -Pattern "Fenêtre de suivi" | Select-Object -Last
20`). Puis place `agent.follow_up_min_confidence` entre les deux. Une valeur plus haute
écarte plus de bruit, mais aussi plus de vraies questions. `agent.follow_up_use_llm_gate:
false` coupe le jugement du LLM.

### 11. Réponses longues dites à voix haute

```powershell
uv run python -m mikata --ask "raconte-moi une longue histoire"
uv run python -m mikata --ask "explique-moi comment marche un frigo"
```

Attendu : 3 à 5 phrases dites dans le casque, aucune ligne `Outil ...` dans la sortie (le
Bloc-notes ne s'ouvre pas). Avec « raconte-moi une histoire et écris-la dans un fichier
histoire.txt sur le bureau », le fichier est créé (`Outil write_file(...) -> ok`).

### 12. Non-régression du prompt

```powershell
uv run python scripts\eval_prompt.py
```

Attendu : `Réussies : 23/23`.
- « lance Team » ouvre Steam sans poser de question.
- « raconte-moi une longue histoire », « raconte-moi une blague » et « explique-moi comment
  marche un frigo » n'appellent aucun outil et sont dites en 3 à 5 phrases (de 1 à 5 pour la
  blague).

## Phase 6b — vérification sur le PC Windows

Critères (tes deux demandes avant la phase 7) :

1. Pendant que MIKATA parle, « stop », « arrête », « arrête-toi », « tais-toi », « ça suffit »
   ou « chut », **sans** « MIKATA », coupent la réponse en moins de 300 ms ; la latence est
   écrite dans le log. Toute autre parole est ignorée : « arrête de dire stop » et
   « non merci » ne coupent rien.
2. « et à Tokyo ? » n'est plus lu « Yeah, Tokyo ».

Comment ça marche :

- **Arrêt vocal** (`mikata/stt/voice_stop.py`) :
  - Actif seulement pendant qu'une réponse ou une annonce (minuteur, rappel) est jouée, et si
    MIKATA ne t'écoute pas déjà. Au repos, le micro ne sert qu'au mot d'activation, comme
    avant.
  - Le VAD Silero découpe ta parole en phrases. Une phrase de plus de
    `agent.voice_stop_max_s` (2 s) n'est même pas transcrite : ce n'est pas un « stop ».
  - Une phrase courte est transcrite par Whisper dès la première trame de silence (80 ms).
    Après `agent.voice_stop_silence_ms` (160 ms) de silence, si la phrase **entière** est
    dans `agent.voice_stop_phrases`, le son est coupé.
  - La comparaison ignore la casse, les accents, la ponctuation, les répétitions
    (« stop stop »), « euh » et « merci ».
  - Couper, c'est comme « MIKATA, stop » : la réponse se tait, et sa commande s'arrête si
    elle travaillait encore (outil, LLM). Il n'y a ni bip ni fenêtre de suivi. Les autres
    commandes en parallèle continuent, et leurs réponses sont dites.
  - Garde-fou de la phase 6 : une annonce d'extinction, de redémarrage ou de veille coupée
    par « stop » annule l'action.
  - `agent.cancel_phrases` ne change pas : elle sert toujours après « MIKATA ».
    `voice_stop_phrases` est plus courte, car un « non » ou un « c'est bon » dit à quelqu'un
    d'autre ne doit pas couper MIKATA.
  - Réglages dans `config.yaml`, section `agent` :
    - `voice_stop` (`false` le désactive) ;
    - `voice_stop_phrases` ;
    - `voice_stop_silence_ms` (une valeur plus haute coupe un peu plus tard mais se trompe
      moins) ;
    - `voice_stop_max_s`.
  - Log : `Arrêt vocal : « Stop ! » -> réponse #N coupée ≈ X ms après la fin de parole
    (silence 160 + attente transcription ... + sortie ... ms ; Whisper ... ms ; objectif
    < 300 ms)`. Si X ≥ 300, la ligne passe en avertissement (`WARNING`).
  - X part de la dernière trame de 80 ms où le VAD entend ta voix. Les quelques dizaines de
    ms que le VAD met à voir la fin du mot ne sont pas comptées.
- **Whisper et les phrases courtes** :
  - La langue est **forcée** : `language="fr"` dans `mikata/stt/whisper_local.py`, sans
    détection automatique. Il n'existe pas de clé `stt.language`. C'est vérifié par le test
    `test_load_and_transcribe`, et `scripts\eval_stt.py` passe par le même code.
  - `stt.initial_prompt` finit maintenant par des amorces de suites courtes :
    `"MIKATA, Bloc-notes, Spotify, Discord, Steam, Chrome, VS Code, PowerShell. Et à Tokyo ?
    Et demain ?"`.
  - **Changement du code de la phase 6** : les noms des applications installées passent
    désormais **avant** ce texte (`Applications : .... MIKATA, ... Et demain ?`). La fin du
    prompt est ce que Whisper lit juste avant ta phrase. C'est aussi ce qu'il garde si le
    prompt est trop long, car il coupe le début.
  - Risque à mesurer : sur du bruit, Whisper recopie parfois son prompt (« Et demain ? »
    inventé). `eval_stt.py --noise` compte ces cas (section 4).

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-6b
git pull
uv sync
```

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_voice_stop.py tests\test_app_voice_stop.py -v
```

Attendu : tout en `passed`. Ce que vérifient ces tests :
- `test_stop_without_wake_word_cuts_the_answer` : « stop » et « tais-toi » coupent la
  réponse simulée en moins de 300 ms, sans appel LLM, sans bip ni fenêtre de suivi.
- `test_other_speech_does_not_cut` : « arrête de dire stop », « non merci » et « arrête la
  musique » sont transcrits puis ignorés, et la réponse va au bout.
- `test_pause_inside_a_phrase_discards_the_early_transcription` : « arrête… de dire stop »,
  avec une courte pause, ne coupe pas non plus.
- `test_stop_cuts_only_the_answer_being_played`, `test_stop_cuts_an_announcement`,
  `test_voice_stop_can_be_disabled`...

### 3. Arrêt vocal (avec le micro)

```powershell
uv run python -m mikata
```

1. « hey jarvis, raconte-moi une longue histoire ». Pendant la réponse, **sans** « hey
   jarvis » : « stop ». Attendu : la voix se tait aussitôt, sans bip. Recommence avec
   « tais-toi », « ça suffit », « arrête-toi » et « chut ». Dans un autre PowerShell :

   ```powershell
   Select-String -Path data\logs\mikata.log -Pattern "Arrêt vocal" | Select-Object -Last 5
   ```

   Attendu : `Arrêt vocal : « Stop ! » -> réponse #N coupée ≈ X ms après la fin de parole
   (...)` avec **X < 300**.
2. Même histoire. Pendant la réponse, dis « arrête de dire stop », puis « non merci », puis
   une phrase quelconque à voix normale. Attendu : la réponse continue jusqu'au bout.
3. Lance de la musique dans Spotify. « hey jarvis, raconte-moi une histoire », puis
   « stop » pendant la réponse. Attendu : la voix se tait et la musique continue. Au repos,
   « stop » tout seul ne fait rien, car MIKATA n'écoute pas sans le mot d'activation.
4. Garde-fou : « hey jarvis, mets le PC en veille ». Pendant « Mise en veille dans 5
   secondes... », dis « stop ». Attendu : le PC ne se met **pas** en veille.
5. Mets `voice_stop: false` dans `config.yaml` (section `agent`) et relance MIKATA. Attendu :
   « stop » ne coupe plus, seul « hey jarvis » coupe. Remets ensuite `true`.

Si « stop » ne coupe jamais, passe `logging.level: DEBUG` et cherche `Arrêt vocal : parole
de plus de 25 trames ignorée`. Ce message veut dire que le micro entend la voix de MIKATA
dans les AirPods. Baisse leur volume ou rapproche-toi du micro. Si c'est
`Arrêt vocal : « ... » ignoré`, envoie-moi ce que Whisper a entendu.

### 4. Whisper sur les phrases courtes

Enregistre d'abord le jeu de phrases courtes (une fois, 18 phrases). Dis chaque phrase comme
une suite, en deux ou trois mots, d'un ton naturel et à ta distance habituelle du micro :

```powershell
uv run python scripts\eval_stt.py --dir data\stt_eval_courtes record --phrases scripts\stt_phrases_courtes_fr.txt
```

Puis mesure le nouveau prompt, sans prompt et avec les applis :

```powershell
uv run python scripts\eval_stt.py --dir data\stt_eval_courtes run --compute-type int8_float16 --prompt none config apps --noise data\negatives
```

Et l'ancien prompt de la phase 6, pour comparer :

```powershell
uv run python scripts\eval_stt.py --dir data\stt_eval_courtes run --compute-type int8_float16 --prompt config apps --initial-prompt "MIKATA, Bloc-notes, Spotify, Discord, Steam, Chrome, VS Code, PowerShell" --noise data\negatives
```

`--noise` prend des extraits sans commande. `data\negatives` existe si tu as utilisé
`wake.collect_negatives` (phase 1) ; sinon, prends `data\wake_logs`, ou enlève l'option.

Ce qu'on attend :
- avec le nouveau prompt, « et à Tokyo ? » devient `Et à Tokyo ?`, et le WER de la ligne
  `apps` est inférieur ou égal à celui de l'ancien prompt ;
- « et à Paris ? », « et ce soir ? »... ne sont pas dans le prompt : ils disent si l'effet
  se généralise ;
- la ligne `Bruit ... : 0 extrait(s) sur N où le prompt est recopié` (ou presque 0).

Si le nouveau prompt invente « Et demain ? » sur du bruit ou fait monter le WER, envoie-moi
les deux tableaux.

Non-régression sur les commandes normales (jeu de la phase 2), nouveau puis ancien prompt :

```powershell
uv run python scripts\eval_stt.py run --compute-type int8_float16 --prompt apps
uv run python scripts\eval_stt.py run --compute-type int8_float16 --prompt apps --initial-prompt "MIKATA, Bloc-notes, Spotify, Discord, Steam, Chrome, VS Code, PowerShell"
```

Attendu : un WER à peu près identique.

### 5. Suites courtes en vrai (avec le micro)

« hey jarvis, quelle heure est-il ? », puis dans la fenêtre de suivi : « et à Tokyo ? ».
Recommence 5 fois, avec aussi « et demain ? » et « et à Paris ? ».

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Fenêtre de suivi" | Select-Object -Last 10
```

Attendu : `Fenêtre de suivi : « Et à Tokyo ? » retenue (...)`, jamais `Yeah`.

## Phase 7 — vérification sur le PC Windows

Critères (§11) : moins de 0,2 faux déclenchement par heure et moins de 5 % de ratés sur le jeu
d'évaluation, puis une semaine d'usage réel confortable.

Le parcours, dans l'ordre :

1. Collecter ta vie courante (au moins 4 h).
2. Enregistrer tes « MIKATA ».
3. Entraîner `mikata.onnx` sur Colab.
4. Entraîner le vérificateur de ta voix sur ton PC.
5. Mesurer avec le benchmark et choisir les seuils.
6. Passer en production, puis installer le démarrage automatique.

Comment ça marche :

- **Deux étages** (§4) :
  1. `models/mikata.onnx` (openWakeWord) donne un score toutes les 80 ms.
  2. Quand ce score atteint `wake.threshold`, le **vérificateur de ta voix**
     (`models/mikata_verifier.pkl`) donne la probabilité que ce soit toi. Sous
     `wake.verifier_threshold`, la détection est refusée : un « MIKATA » dit dans une vidéo ou
     par quelqu'un d'autre ne déclenche pas.
  - Le vérificateur est un petit modèle scikit-learn : coût et délai quasi nuls.
  - Le log de détection affiche les deux scores (`score 0.92, voix 0.88`). Un refus est
    noté `Mot d'activation refusé par le vérificateur : modèle 0.81, voix 0.12`.
- **Entraînement du modèle** (`scripts/train_wakeword.py`, sur Google Colab, GPU gratuit) :
  - Positifs :
    - tes « MIKATA », recopiés 30 fois, avec chaque fois un bruit, un écho et un égaliseur
      différents ;
    - ~17 000 « Mikata » synthétiques, dits par plus de 1 100 voix Piper : 130 françaises
      (accent français) et des anglaises, avec les variantes « mi kata » et « mee kah tah ».
  - Négatifs :
    - tes mots proches et une trentaine d'autres, en voix synthétiques ;
    - ~2 000 h de sons publics (ACAV100M) ;
    - ta vie courante, qui sert aussi de bruit de fond mélangé aux positifs.
  - Le jeu d'évaluation n'est jamais envoyé à Colab.
- **Benchmark** (`scripts/eval_wakeword.py`) :
  - Tes 30 prises « eval » sont chacune placées dans 1,5 s de vie courante. Les 2 h de vie
    courante figées sont passées d'un bloc, comme en vrai.
  - Les règles de déclenchement de MIKATA sont rejouées pour chaque seuil, 1 ou 2 trames,
    avec et sans vérificateur.
  - Résultat : la courbe faux déclenchements par heure, ratés et délai médian, puis le
    **réglage conseillé** à copier dans `config.yaml`.
  - Choix du réglage : le moins de faux déclenchements, puis le moins de ratés, un délai
    ≤ 150 ms, puis le **milieu** de la plage qui tient les cibles (de la marge des deux
    côtés).
- **Démarrage automatique** (`scripts/install_autostart.ps1`) :
  - Une tâche planifiée « MIKATA » à l'ouverture de ta session, 30 s après.
  - Elle lance `.venv\Scripts\pythonw.exe -m mikata --supervise` : pas de console, avec tes
    droits normaux. Ce n'est pas un service Windows (§10).
  - `--supervise` relance MIKATA s'il plante, au plus 5 fois par heure, et pas après
    « Quitter » ni après une config invalide. L'option « redémarrer en cas d'échec » du
    Planificateur de tâches ne couvre qu'un échec au lancement.
  - Journal des relances : `data\logs\supervise.log`.
- **Purge** :
  - Au démarrage puis toutes les 24 h, partent à la corbeille :
    - les audios de détection de plus de 7 jours (`data\wake_logs`) ;
    - les sauvegardes de plus de 14 jours (`data\backups`).
  - Les logs tournent déjà : 5 Mo × 5.
  - La collecte de vie courante s'arrête seule après `wake.negatives_max_hours` (12 h,
    ~115 Mo par heure).

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-7
git pull
uv sync
uv run python scripts\download_models.py --no-whisper
```

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_eval_wakeword.py tests\test_record_wakeword.py tests\test_train_wakeword.py tests\test_supervise.py tests\test_maintenance.py -v
```

Attendu : tout en `passed`.

### 3. Collecter ta vie courante (au moins 4 h, sur plusieurs jours)

C'est la donnée la plus précieuse (§4) : 2 h sont figées pour le benchmark, le reste sert à
l'entraînement. Dans `config.yaml`, section `wake` :

```yaml
  collect_negatives: true
```

Lance MIKATA (`uv run python -m mikata`) et vis normalement : Discord, jeux, musique, vidéos,
la pièce calme, toi qui parles. La collecte s'arrête seule à 12 h. Elle peut s'étaler sur
plusieurs jours, et plusieurs situations valent mieux qu'une seule longue session.

Ne dis **pas** « MIKATA » pendant la collecte (« hey jarvis » ne gêne pas).

Pour suivre où tu en es :

```powershell
uv run python scripts\record_wakeword_samples.py plan
```

### 4. Enregistrer tes « MIKATA » (≈ 20 min)

Avec ton micro RØDE NT-USB+ habituel, AirPods sur les oreilles, à ta place. Dis « MIKATA »
comme tu le diras tous les jours, sans le surjouer.

```powershell
uv run python scripts\record_wakeword_samples.py record --set train
uv run python scripts\record_wakeword_samples.py record --set hard
```

Ce que le script te demande :

- **`train` : 100 « MIKATA »** pour l'entraînement.

  | Situation | Prises | Consigne |
  | --- | --- | --- |
  | normal | 20 | voix normale, assis à ta place (~30 cm du micro) |
  | bas | 10 | voix basse, presque chuchotée (comme la nuit) |
  | fort | 10 | voix forte, comme pour appeler quelqu'un dans la pièce |
  | fatigue | 10 | voix fatiguée, lente, traînante |
  | sourire | 10 | en souriant, vite, avec énergie |
  | loin | 10 | debout à environ 2 m du micro |
  | cote | 10 | tête tournée sur le côté, ou dos au micro |
  | bruit | 10 | en tapant au clavier, ou avec un ventilateur ou une vidéo en fond |
  | phrase | 10 | d'une traite avec une commande : « MIKATA, ouvre Spotify »... (varie) |

- **`hard` : 36 mots proches**, 3 prises chacun : mikado, Nikita, Akira, « il m'a quitté »,
  mi-carré, Mika, « ma carte », « mi casa », Takata, chiquita, Michaela, Nakata.

Chaque prise se déroule ainsi :
1. Appuie sur Entrée, puis parle. L'enregistrement s'arrête quand tu te tais.
2. Le script signale une prise vide, trop longue ou saturée.
3. Entrée garde la prise ; `r` la refait (l'ancienne part à la corbeille) ; `q` quitte.

Relancer reprend où tu t'étais arrêté. `--only loin` refait une seule situation.

### 5. Un autre jour : 30 prises d'évaluation

Un autre jour, pour que le benchmark mesure un vrai « autre jour » (voix, pièce, humeur) :

```powershell
uv run python scripts\record_wakeword_samples.py record --set eval
uv run python scripts\record_wakeword_samples.py plan
```

C'est la même répartition, en plus petit : 6 « normal » et 3 de chaque autre situation. Ces
prises ne servent **jamais** à l'entraînement. Attendu : `plan` dit « tout est prêt ».

### 6. Entraîner `mikata.onnx` sur Google Colab (≈ 2 à 3 h)

```powershell
uv run python scripts\record_wakeword_samples.py pack
```

Ça crée `data\wakeword\mikata_wakeword_colab.zip` (~500 Mo avec 4 h de vie courante ;
`--negative-hours 2` pour une archive plus petite). Ensuite :

1. Ouvre <https://colab.research.google.com>, puis **Nouveau notebook**.
2. **Exécution > Modifier le type d'exécution > GPU T4**.
3. Envoie l'archive. Le plus fiable est de la déposer dans ton Google Drive (dossier
   `Mon Drive`), puis de lancer dans une première cellule :

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   !cp /content/drive/MyDrive/mikata_wakeword_colab.zip /content/
   ```

   Ou glisse-la dans le panneau **Fichiers** de Colab, mais c'est lent pour 500 Mo.
4. Lance les cellules suivantes, une par une :

   ```python
   !unzip -q -o /content/mikata_wakeword_colab.zip -d /content/mikata
   !python /content/mikata/train_wakeword.py setup
   !python /content/mikata/train_wakeword.py train
   ```

   - `setup` (~15 min) installe un Python 3.11 à part avec des versions épinglées et
     télécharge ~17 Go de données publiques ainsi que les voix Piper.
   - `train` prépare les clips, puis entraîne. Il finit par
     `[mikata] modèle prêt : /content/mikata/out/mikata.onnx`. Un message d'erreur sur
     TensorFlow ou tflite **après** cette ligne est normal : seul le `.onnx` sert.
   - Si Colab se déconnecte, relance les mêmes cellules : chaque étape déjà faite est sautée.
5. Récupère le modèle :

   ```python
   from google.colab import files

   files.download("/content/mikata/out/mikata.onnx")
   ```

6. Copie `mikata.onnx` dans `models\` (à côté de `hey_jarvis.onnx`).

Ce script n'a pas pu tourner en vrai dans la session cloud (ni GPU, ni accès à Hugging
Face). Si une étape échoue, envoie-moi les dernières lignes de la cellule.

### 7. Vérificateur de ta voix (≈ 1 min, sur ton PC)

```powershell
uv run python scripts\train_verifier.py
```

Il apprend à partir de tes 100 prises `train` (ta voix qui dit « MIKATA ») et de ta voix qui
dit autre chose : les 36 mots proches, et tes phrases de `data\stt_eval` sauf celles qui
contiennent « MIKATA ». Attendu : `Vérificateur : ...\models\mikata_verifier.pkl`.

### 8. Le critère : benchmark (≈ 10 min pour 2 h)

```powershell
uv run python scripts\eval_wakeword.py --model models\mikata.onnx --verifier models\mikata_verifier.pkl
```

Attendu, pour chaque réglage (sans vérificateur, puis voix ≥ 0,3 / 0,5 / 0,7 / 0,9) : un
tableau seuil, faux déclenchements par heure (FD/h), ratés et délai médian. Les cases
marquées `*` tiennent les cibles. Puis la ligne `Réglage conseillé` avec les lignes à copier.

**Critère validé** si un réglage tient **< 0,2 FD/h et < 5 % de ratés**. Sinon, le
script donne le meilleur compromis et les ratés. Envoie-moi le tableau : on ajoute des prises
ou de la vie courante et on réentraîne (`train --user-copies 50`, `--steps 80000`...).

Pour comparer avec le modèle seul :

```powershell
uv run python scripts\eval_wakeword.py --model models\mikata.onnx --no-verifier
```

### 9. Passer en production

Dans `config.yaml`, section `wake`, avec les valeurs du réglage conseillé :

```yaml
wake:
  model: models/mikata.onnx
  verifier: models/mikata_verifier.pkl
  threshold: 0.50              # <- réglage conseillé
  consecutive_frames: 1        # <- réglage conseillé
  verifier_threshold: 0.70     # <- réglage conseillé
  collect_negatives: false
```

Tu peux aussi retirer `"jarvis"` de `stt.wake_words`. Ensuite :

```powershell
uv run python -m mikata --check-config
uv run python -m mikata
```

Dis 10 fois « MIKATA, quelle heure est-il ? » à ta place, puis 5 fois de loin. Attendu : un
bip à chaque fois. Puis fais dire « MIKATA » par une vidéo ou quelqu'un d'autre : rien, ou
une ligne `refusé par le vérificateur`. Dans un autre PowerShell :

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Mot d'activation" | Select-Object -Last 20
```

Attendu : `Mot d'activation détecté (score 0.xx, voix 0.xx) : bip ≈ X ms après la fin du
mot`, avec X < 150 en général.

### 10. Démarrage automatique

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -StartNow
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Status
```

Attendu : `Tâche « MIKATA » installée`, l'icône apparaît, et `-Status` affiche `Running`.
Puis ferme ta session et rouvre-la : l'icône revient ~30 s après.

Pour tester la relance après un plantage :

```powershell
Get-Process pythonw | Sort-Object StartTime | Select-Object -Last 1 | Stop-Process -Force
Get-Content data\logs\supervise.log -Tail 5
```

Attendu : `MIKATA arrêté sur une erreur (code ..., après ... s) : relance dans 10 s`, puis
l'icône revient. Avec **Quitter** dans le menu de l'icône, MIKATA ne revient pas. Pour retirer la
tâche :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Uninstall
```

Si la création est refusée, relance une fois le script dans un PowerShell « Exécuter en tant
qu'administrateur » : la tâche tournera quand même avec tes droits normaux.

### 11. Purge (rien à faire)

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Purge" | Select-Object -Last 5
```

Une ligne `Purge : N dans data/wake_logs...` apparaît quand il y a quelque chose de trop
vieux.

### 12. Une semaine d'usage réel

Laisse tourner une semaine. Chaque soir :

```powershell
(Select-String -Path data\logs\mikata.log -Pattern "Mot d'activation détecté").Count
(Select-String -Path data\logs\mikata.log -Pattern "refusé par le vérificateur").Count
```

Compare au nombre de fois où tu as vraiment dit « MIKATA ». Chaque détection a 2 s d'audio
dans `data\wake_logs\` (le nom contient le score) : écoute celles que tu n'as pas
provoquées. Si les faux déclenchements dépassent ~1 par jour, envoie-moi ces audios : on les
ajoute aux négatifs pour le prochain entraînement.

### 13. Noter les « MIKATA » ratés (ajout phase 9)

Quand tu as dit « MIKATA » et qu'il n'a pas réagi, appuie sur **Ctrl+Alt+M** (le raccourci
habituel). En plus de lancer l'écoute :

- tu entends le bip d'écoute **suivi de deux petits tics aigus** : le raté est noté ;
- les **4 dernières secondes** d'audio (16 kHz mono) sont enregistrées dans
  `data\wakeword\misses\`, un .wav par raté (`<date>_rate.wav`) ;
- une ligne `Raté du mot d'activation noté` apparaît dans le journal.

Les ratés de plus de 30 jours partent à la corbeille (`wake.miss_days`). Pour couper la
collecte : `wake.save_misses: false` dans `config.yaml`, puis redémarre.

```powershell
Get-ChildItem data\wakeword\misses
(Select-String -Path data\logs\mikata.log -Pattern "Raté du mot d'activation noté").Count
```

Appuie vite après le « MIKATA » raté : au-delà de 4 s, il n'est plus dans le tampon.
Écoute quelques fichiers : le « MIKATA » doit s'y entendre. Ces ratés sont des positifs de
plus pour le prochain entraînement. `train_wakeword.py` ne les lit pas encore tout seul : on
les branchera au prochain réentraînement.

## Phase 8 — vérification sur le PC Windows

Deux composants, dans un **processus à part** (`python -m mikata.ui`), lancé par MIKATA :

1. **La mascotte « Mika »** :
   - une petite fenêtre sans bordure, transparente, toujours au-dessus, qui ne prend jamais le
     focus ;
   - tu la glisses où tu veux sur tes deux écrans ; la position et l'écran sont gardés dans
     `data\ui_state.json` ;
   - **clic droit** : taille, bulle de texte, pause de l'écoute, réglages, masquer. Pour
     cliquer sur ce qu'il y a dessous, on la déplace : il n'y a pas de mode « clics à
     travers » (une fois activé, le clic droit ne l'atteindrait plus) ;
   - **double-clic** : réglages ;
   - une animation et une couleur par état (captures : [`docs/ui/`](docs/ui/)) :

     | État | Mika |
     | --- | --- |
     | repos | grise, elle dort (zzz) |
     | mot d'activation | jaune, « ! » (0,9 s) |
     | écoute | bleue, ondes |
     | réflexion | orange, bulles de pensée |
     | action (outil) | orange, engrenage |
     | parole | verte, bouche qui bouge, ondes sonores |
     | c'est fait | verte, coche (1,6 s) |
     | erreur | rouge, yeux en croix (2,4 s) |
     | fenêtre de suivi | turquoise, anneau qui se vide en 6 s |
     | micro coupé ou occupé, MIKATA pas lancé | anthracite, micro barré |

   - la **bulle** (désactivable) montre ce que MIKATA a compris (« Tu : ... ») et ce qu'il
     dit ;
   - **skins** : Mika est dessinée en code ; tes propres personnages s'installent dans
     `assets\mascot\<nom>\` (voir « Skins » plus bas).
2. **La fenêtre de réglages** : icône MIKATA (zone de notification) > **Réglages…**
   (ou clic gauche sur l'icône). Onglets :
   - Audio : micro et sortie, avec test ;
   - Mot d'activation : seuils du modèle et du vérificateur ;
   - Conversation : réflexion par défaut, fenêtre de suivi, durée de l'historique ;
   - Coût et clés : coût LLM cumulé, crédit restant, clés masquées ;
   - Mascotte : durée de la bulle, mascotte au démarrage ;
   - Skins : les skins installés avec un aperçu ; un clic change de skin tout de suite (choix
     gardé), import d'un .zip, suppression (à la corbeille, jamais Mika) ;
   - Journal : en direct, filtré par niveau.

Comment ça marche :

- **Aucun impact sur MIKATA** :
  - MIKATA publie ses événements sur un canal local : TCP sur 127.0.0.1, port choisi au
    démarrage, et un jeton secret écrit dans `data\ui_channel.json`.
  - Publier ne fait qu'ajouter à une file bornée. Aucun travail n'est fait dans le callback
    audio.
  - Une interface plantée, lente ou absente ne change rien : MIKATA continue, et l'interface
    se reconnecte seule.
  - L'interface tourne en priorité basse et écrit son propre journal,
    `data\logs\ui.log`.
- **Réglages** :
  - Ils sont écrits dans `config.yaml`, **seulement sur la ligne modifiée** : commentaires et
    ordre restent intacts.
  - La config est validée avant l'écriture ; l'ancienne est sauvegardée dans
    `data\backups\`.
  - Chaque réglage porte un badge :
    - **tout de suite** : seuils, réflexion, fenêtre de suivi, historique, skin, bulle ;
    - **au redémarrage** : micro, sortie audio, mascotte au démarrage, clés.
  - Les clés restent dans `.env` et ne s'affichent jamais en clair : 8 points, puis les 4
    derniers caractères.
- **Section `ui:` de `config.yaml`** :
  - `enabled: false` coupe tout (MIKATA seul, comme avant) ;
  - `mascot: false` : pas de mascotte au démarrage, mais les réglages restent accessibles
    depuis l'icône.
- **Skins** :
  - un dossier par skin dans `assets\mascot\` : un manifeste `skin.json` (nom, auteur,
    licence) et une image par état ;
  - **une seule image par état suffit** : le programme l'anime (respiration, petit rebond,
    clignement, lueur, balancement), avec des réglages par état dans le manifeste. Des
    images multiples (PNG numérotés, GIF, APNG) sont jouées image par image ;
  - un état absent reprend l'image de repos, avec un avertissement dans `data\logs\ui.log` ;
  - `scripts\make_skin.py` fabrique un skin à partir d'images générées (Nano Banana...) :
    fond uni ou damier retiré, recadrage, taille, bords propres, manifeste ;
  - guide complet, avec des prompts d'exemple : [`docs/MASCOT_SKIN_GUIDE.md`](docs/MASCOT_SKIN_GUIDE.md) ;
  - tes skins ne sont pas suivis par git.

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-8-ui
git pull
uv sync
```

`uv sync` installe PySide6-Essentials (Qt 6, ~230 Mo), ruamel.yaml et python-dotenv
(scipy, utilisé par `make_skin.py`, était déjà installé).

**Tu as déjà la branche et `config.yaml` modifié en local** (seuils du mot d'activation...) :
garde tes modifications pendant la mise à jour (`ui.theme` devient `ui.skin`) :

```powershell
git stash
git pull
git stash pop
Select-String -Path config.yaml -Pattern "skin:"
```

Attendu : `skin: default` dans la section `ui:` et tes autres réglages intacts.

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_ui_channel.py tests\test_ui_model.py tests\test_ui_skins.py tests\test_ui_skins_tab.py tests\test_skin_maker.py tests\test_ui_settings_store.py tests\test_ui_qt.py tests\test_app_ui.py -v
```

Attendu : tout en `passed`. Les tests de l'interface tournent hors écran : aucune fenêtre
n'apparaît.

### 3. Captures sur ton PC (facultatif)

```powershell
uv run python scripts\ui_screenshots.py --out data\captures --gif
explorer data\captures
```

Compare avec `docs\ui\` (rendu dans la session cloud, sous Linux).

### 4. Lancer MIKATA : la mascotte apparaît

```powershell
uv run python -m mikata
```

Attendu :

- la mascotte apparaît en bas à droite de l'écran principal, grise, endormie ;
- le jeu ou la fenêtre active garde le focus quand tu cliques ailleurs.

Vérification du canal :

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Canal de l'interface|Interface connectée" | Select-Object -Last 2
Get-Content data\logs\ui.log -Tail 5
```

### 5. Les états, à la voix

| Ce que tu fais | Mika |
| --- | --- |
| « MIKATA » | jaune « ! », puis bleue (écoute) |
| « quelle heure est-il ? » | orange (réflexion), puis verte qui parle ; bulle « Tu : quelle heure est-il » + la réponse |
| « MIKATA, ouvre le bloc-notes » | engrenage, puis coche verte « c'est fait » |
| juste après une réponse | anneau turquoise qui se vide en 6 s (fenêtre de suivi) |
| Wi-Fi coupé, puis « MIKATA, bonjour » | rouge, yeux en croix ; bulle « Je n'arrive pas à joindre le serveur. » |
| clic droit > Mettre en pause l'écoute | micro barré ; « MIKATA » ne fait plus rien. Reprendre : même menu, ou icône > Pause / reprise |
| micro USB débranché | micro barré après 2 s ; ligne `Micro : aucune trame` dans le journal |

### 6. Glisser sur les deux écrans, position gardée

1. Glisse Mika sur le second écran.
2. Quitte MIKATA (icône > Quitter) : la mascotte se ferme avec lui.
3. Relance `uv run python -m mikata` : Mika revient au même endroit, sur le même écran.

```powershell
Get-Content data\ui_state.json
```

Si l'écran mémorisé est débranché, Mika revient en bas à droite de l'écran principal.

### 7. Menu du clic droit

- **Taille** : petite, moyenne, grande (96, 128, 192 px). Les pieds restent en place.
- **Bulle de texte** : on/off.
- Plus de « Mode traversable » : pour cliquer dessous, glisse Mika ailleurs. Si tu l'avais
  activé avant cette mise à jour, il est ignoré au prochain lancement (`data\ui_state.json`
  n'est pas à modifier).
- **Masquer** : pour la faire revenir, icône MIKATA > **Afficher la mascotte**.

### 8. Fenêtre de réglages

1. Icône MIKATA > **Réglages…**.
2. **Audio** : clique « Tester le micro » et parle : la barre bouge. « Tester la sortie »
   joue un bip dans le casque.
3. **Conversation** : passe la fenêtre de suivi à 4, puis **Enregistrer**. Attendu :
   - le statut affiche « Tout de suite : Fenêtre de suivi (s) » ;
   - après une réponse, l'anneau dure 4 s.
4. Le fichier ne change que sur cette ligne, commentaire compris :

   ```powershell
   git diff config.yaml
   ```

5. Remets 6 et **Enregistrer** (ou `git checkout config.yaml`).
6. **Mot d'activation** : n'y touche pas pendant l'entraînement de `mikata.onnx`. Les
   seuils s'appliquent tout de suite quand tu les changes.
7. **Coût et clés** : coût cumulé et crédit restant (`llm.credit_usd`), clés masquées.
   « Remplacer… » écrit la nouvelle clé dans `.env`, prise en compte au redémarrage.
8. **Journal** : les lignes arrivent en direct ; le filtre « Avertissements et erreurs » ne
   garde que les problèmes.
9. Une valeur refusée par MIKATA s'affiche en bas de la fenêtre, et `config.yaml` n'est pas
   modifié.

### 9. MIKATA survit à l'interface

Tue l'interface :

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object CommandLine -like '*mikata.ui*' | ForEach-Object { Stop-Process -Id $_.ProcessId }
```

Attendu :

- MIKATA répond toujours à « MIKATA, quelle heure est-il ? » ;
- icône > **Afficher la mascotte** relance l'interface ;
- `uv run python -m mikata.ui --settings`, lancé une deuxième fois, ouvre les réglages dans
  l'instance déjà lancée (une seule mascotte).

Latence inchangée : compare les délais du bip avec et sans interface (`ui.enabled: false`
dans `config.yaml`, puis redémarrage).

```powershell
Select-String -Path data\logs\mikata.log -Pattern "bip ≈" | Select-Object -Last 10
```

### 10. CPU de l'interface

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object CommandLine -like '*mikata.ui*' | ForEach-Object { Get-Process -Id $_.ProcessId } | Select-Object Id, CPU, WorkingSet64, PriorityClass
```

Relance la commande après une minute au repos : `CPU` (en secondes) doit à peine bouger.
L'animation tourne à ~12 images/s au repos et à ~30 pendant une action. `PriorityClass`
doit valoir `BelowNormal`.

### 11. Skins

**Changer de skin** :

1. Icône MIKATA > **Réglages… > Skins**.
2. Clique sur un skin : la mascotte change tout de suite, sans redémarrer, et le choix est
   gardé (`ui.skin` dans `config.yaml`).
3. Ferme et relance MIKATA : le skin choisi est toujours là.

**Créer le tien** (détail et prompts : [`docs/MASCOT_SKIN_GUIDE.md`](docs/MASCOT_SKIN_GUIDE.md)) :

1. Génère une image par état avec Nano Banana (Gemini). Le plus simple : joins
   `docs\GEMINI_SKIN_BRIEF.md` dans une nouvelle conversation Gemini et écris « Suis ce
   brief. ». Les prompts d'exemple d'un personnage chibi sont aussi dans le guide, section 4.
2. Range-les dans un dossier, par exemple `C:\Users\<toi>\Images\mika_chibi`, nommées
   `repos.png`, `activation.png`, `ecoute.png`, `reflexion.png`, `action.png`,
   `parole.png`, `fait.png`, `erreur.png`, `suivi.png`, `micro_coupe.png`. Seul
   `repos.png` est obligatoire.
3. Fabrique le skin et ouvre l'aperçu dans la mascotte :

   ```powershell
   uv run python scripts\make_skin.py C:\Users\<toi>\Images\mika_chibi --name "Mika chibi" --author "Moi" --license "usage personnel" --preview
   start assets\mascot\mika-chibi\apercu.png
   ```

   - Attendu : le fond (uni ou damier) a disparu, les bords sont nets, sans liseré clair sur
     un fond sombre.
   - Dans la petite fenêtre d'aperçu, chaque état s'affiche ; « Défilement automatique »
     les enchaîne.
   - En cas de liseré ou de morceaux manquants : voir les options `--tolerance`, `--holes`
     et `--shadow` dans le guide, puis relance avec `--replace`.
4. Réglages > Skins > **Actualiser**, puis clique sur « Mika chibi ».
5. Un état manquant reprend l'image de repos. Vérifie l'avertissement dans le journal :

   ```powershell
   Select-String -Path data\logs\ui.log -Pattern "image de repos utilisée" | Select-Object -Last 5
   ```

**Importer et supprimer** :

1. Fais un .zip de ton skin : clic droit sur `assets\mascot\mika-chibi` > **Compresser dans
   un fichier ZIP**. Déplace ce .zip ailleurs, par exemple sur le Bureau.
2. Réglages > Skins : sélectionne « Mika chibi » > **Supprimer** > Oui. Le dossier part à la
   **corbeille** et Mika revient si c'était le skin en cours.
3. **Importer un .zip…** > choisis le .zip : le skin revient et s'applique.
4. Importe un fichier qui n'est pas un skin (un .zip quelconque) : il est refusé, et la
   raison s'affiche en bas de la fenêtre. Rien n'est écrit dans `assets\mascot\`.
5. Mika : le bouton **Supprimer** reste grisé.

### 12. Publication (licence, secrets)

- Licence : **Apache-2.0** (`LICENSE`, `NOTICE`).
- `THIRD_PARTY.md` liste les composants tiers et leurs licences.
  - Attention : le modèle `hey_jarvis` est **non commercial** (CC BY-NC-SA 4.0).
  - Attention : piper-tts est sous GPL-3.0.
- La CI lance **gitleaks** sur tout l'historique. Pour le lancer chez toi :

  ```powershell
  winget install Gitleaks.Gitleaks
  gitleaks git . --redact
  ```

- `.gitignore` ignore `.commandcode/` et tes scripts de `scripts\user\`. `bonjour.ps1`
  reste dans le dépôt comme exemple.

## Phase 9 — vérification sur le PC Windows

Deux finitions :

1. **Ratés du mot d'activation** : Ctrl+Alt+M après un « MIKATA » non entendu garde les
   4 dernières secondes dans `data\wakeword\misses\` (voir « Phase 7 », étape 13).
2. **Tests de parallélisme déterministes** : trois tests échouaient au hasard en CI.
   - Les tests concernés : `test_parallel_turns_do_not_block_each_other`,
     `test_two_commands_run_in_parallel` et `test_open_spotify_then_brave_in_parallel`.
   - Ils ne mesurent plus le temps et ne dépendent plus de l'ordre d'exécution des tâches.
     Chaque faux appel au LLM attend que l'autre ait commencé (`tests\rendezvous.py`).
   - Des commandes traitées l'une après l'autre ne se rencontreraient jamais : le test
     échoue alors avec un message clair. Le défaut protégé reste donc vérifié.

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-9-polish
git pull
uv sync
```

Si tu as modifié `config.yaml` en local, garde tes réglages pendant la mise à jour (deux
lignes sont ajoutées dans `wake:`) :

```powershell
git stash
git pull
git stash pop
Select-String -Path config.yaml -Pattern "save_misses|miss_days"
```

### 2. Lint et tests

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_wake_misses.py tests\test_maintenance.py tests\test_agent.py tests\test_app_conversation.py tests\test_app_tools.py -v
```

Attendu : tout en `passed`. Les trois tests de parallélisme passent à chaque fois. Pour le
vérifier, relance-les 10 fois :

```powershell
1..10 | ForEach-Object { uv run pytest -q tests\test_agent.py::test_parallel_turns_do_not_block_each_other tests\test_app_conversation.py::test_two_commands_run_in_parallel tests\test_app_tools.py::test_open_spotify_then_brave_in_parallel | Select-Object -Last 1 }
```

### 3. Un raté, en vrai

1. Lance MIKATA (`uv run python -m mikata`).
2. Dis quelque chose, puis appuie tout de suite sur **Ctrl+Alt+M**. Attendu : bip d'écoute
   puis deux tics, puis MIKATA t'écoute comme d'habitude.
3. Vérifie le fichier et la ligne du journal (« Phase 7 », étape 13).


## Phase 11 — vérification sur le PC Windows

Voix ElevenLabs et profils de personnage :

1. **Voix ElevenLabs** (`tts.elevenlabs`, modèle `eleven_v4_turbo`, français, PCM 24 kHz en
   flux).
   - Synthèse fragment par fragment, comme Piper et Kokoro, avec le même découpage.
   - Cache disque `data\tts_cache\elevenlabs\` : une phrase déjà dite n'est jamais refacturée.
     La clé est voix + modèle + texte normalisé (espaces et apostrophes unifiés).
2. **Retour automatique sur Kokoro**. MIKATA dit une fois « Je passe sur ma voix de base. »
   et l'interface l'affiche. Les causes :
   - crédit du mois sous `min_remaining_credits` ;
   - plafond `monthly_credit_cap` atteint (30 000 par défaut) ;
   - connexion impossible ;
   - erreur de l'API (401, 429, quota épuisé, 5xx).

   Retour sur ElevenLabs :
   - après une panne réseau : au bout de `retry_after_s` ;
   - pour un crédit épuisé : au renouvellement ;
   - pour une clé refusée : à la relecture des réglages.

   Le solde est relu au démarrage, puis toutes les `subscription_refresh_every` synthèses. Entre
   deux relevés, il est déduit localement.
3. **Coût réel mesuré**, jamais supposé. Au démarrage (au plus une fois par
   `calibration_hours`), une courte phrase est synthétisée entre deux relevés du solde. Le
   coût par caractère est ensuite remesuré à chaque relevé. Il s'affiche dans le journal et
   dans l'onglet « Coût et clés ».
4. **Profils** (`ui.profiles`) : un skin + une voix.
   - `default` (Mika) : Kokoro.
   - `gars` : Théo.
   - `femme` : Lucie.

   Choisir un skin dans l'onglet Skins change aussi la voix, tout de suite.
5. **`scripts\eleven_voices.py`** : lister les voix du compte, chercher dans la bibliothèque,
   ajouter une voix, l'essayer, l'attribuer à un skin.

Écarts avec `docs/ARCHITECTURE.md` :

- §8 écartait le TTS cloud. Tu as mesuré qu'ElevenLabs v4 Turbo donne le premier son en
  ~165 ms, sous l'objectif de 200 ms, avec une voix bien meilleure que les voix locales. Kokoro
  reste la voix de base et le secours hors ligne.
- « Relire le solde toutes les N réponses » est fait toutes les N **synthèses** : une réponse
  compte plusieurs fragments.
- Tout sort maintenant à 24 kHz : Piper (22 050 Hz) est rééchantillonné. Ses phrases en cache
  sont donc regénérées une fois.
- Rien n'a changé pour le mot d'activation ni pour `mikata.onnx`.

### 1. Récupérer la branche, installer

```powershell
git fetch origin
git checkout phase-11-voix-profils
git pull
uv sync
uv run python scripts/download_models.py     # Kokoro, voix de base et de secours
```

`config.yaml` reçoit deux blocs : `tts.elevenlabs` et `ui.profiles`. Si tu l'as modifié en
local, garde tes réglages :

```powershell
git stash
git pull
git stash pop
Select-String -Path config.yaml -Pattern "eleven_v4_turbo|profiles:|gars:|femme:"
```

Ajoute la clé dans `.env`. Elle n'est jamais commitée ni affichée (ligne `ELEVENLABS_API_KEY=`,
voir `.env.example`) :

```powershell
notepad .env
```

Tes skins doivent être dans `assets\mascot\gars\` et `assets\mascot\femme\`, comme avant.
`.gitignore` les tient hors du dépôt.

### 2. Lint et tests (API simulée, aucun appel réel)

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests\test_elevenlabs.py tests\test_voice_profiles.py tests\test_eleven_voices.py -v
```

Attendu : tout en `passed`. Les tests couvrent :

- le retour sur Kokoro pour chaque cause : crédit, plafond, réseau, 401, 429, quota, 5xx, clé
  absente ;
- le retour sur ElevenLabs ;
- le cache facturé une seule fois ;
- la voix de secours ;
- le changement de profil à chaud.

### 3. Tes voix

```powershell
uv run python scripts/eleven_voices.py list
```

Attendu : Lucie, Théo et Émilie avec leur identifiant. Les profils qui les utilisent sont
indiqués (« <- profil(s) gars »). Pour en changer, sans code :

```powershell
uv run python scripts/eleven_voices.py search --gender male --query "chaleureuse"
uv run python scripts/eleven_voices.py add <owner> <voice_id> --name "Hugo"
uv run python scripts/eleven_voices.py try <voice_id>        # quelques crédits ; écoute data\eleven_essai_<voice_id>.wav
uv run python scripts/eleven_voices.py use gars <voice_id> --name "Hugo"
uv run python scripts/eleven_voices.py use default --kokoro
```

### 4. Démarrage : solde et coût réel

Choisis le skin `femme`, lance MIKATA, puis regarde le journal :

```powershell
uv run python -m mikata
Select-String -Path data\logs\mikata.log -Pattern "ElevenLabs|Voix :" | Select-Object -Last 6
```

Attendu, en plus de « Voix : ElevenLabs « Lucie ». » :

- « ElevenLabs : coût mesuré 0.0xxx crédit par caractère » ;
- une ligne « … crédits utilisés sur 40000 (plafond 30000), … utilisables, coût mesuré …,
  environ N réponses ; renouvellement le jj/mm ».

Le coût vient du tarif en cours (promotion jusqu'au 12 octobre). Il est remesuré tout seul
après la fin de la promotion.

### 5. La voix et sa latence

Dis « MIKATA, quelle heure est-il ? » : tu entends Lucie. Pour la latence, sans micro :

```powershell
uv run python -m mikata --ask "Raconte-moi une blague courte."
```

Attendu : « premier son » proche de celui de Kokoro, à environ 200 ms près. Repose la même
question ou dis « c'est fait » plusieurs fois. Les phrases déjà dites viennent du cache :

```powershell
(Get-ChildItem data\tts_cache\elevenlabs).Count
Get-Content data\elevenlabs_usage.json
```

`cache_hits` augmente, `requests` non.

### 6. Changer de personnage, à chaud

Fais clic droit sur la mascotte, puis Réglages et l'onglet **Skins** :

- clique `gars` : le statut dit « Skin « … » appliqué (voix ElevenLabs « Théo ») » et la
  réponse suivante est dite par Théo ;
- clique `Mika (par défaut)` : retour à Kokoro.

Tout cela sans redémarrer. Le journal le note (« Voix : … (profil « gars ») »).

### 7. Compteur dans « Coût et clés »

L'onglet montre :

- les crédits utilisés ce mois, le plafond choisi et la date de renouvellement ;
- les crédits restants utilisables et une estimation du nombre de réponses ;
- le coût mesuré et la voix en cours.

La clé `ELEVENLABS_API_KEY` y est masquée (••••••••xxxx). Le compteur se met à jour tout seul
toutes les 5 s. S'il reste sur « pas encore de relevé » et que le journal dit « la clé n'a pas
le droit de lire le solde », ajoute la permission « User » en lecture à la clé sur
elevenlabs.io. Sans elle, la voix marche mais le solde et le plafond ne sont pas suivis.

### 9. Redémarrer MIKATA depuis l'interface

Le skin par défaut est maintenant **femme** (voix Lucie). Un bouton **Redémarrer MIKATA** est dans
la fenêtre de réglages (en bas, à côté de « Fermer ») et dans le menu du clic droit de la mascotte.
MIKATA s'arrête proprement (code 75) puis revient seul, mascotte comprise :
- lancé par le démarrage automatique (`--supervise`), le superviseur le relance tout de suite, sans
  compter ça comme un plantage (`data\logs\supervise.log` : « Redémarrage demandé ») ;
- lancé à la main, il se relance lui-même dans un processus détaché.

### 8. Retour sur Kokoro, en vrai

Pour chaque cas, édite `config.yaml`, relance MIKATA, pose une question, puis remets la valeur.

- **Plafond atteint** : `monthly_credit_cap: 1` dans `tts.elevenlabs`. Attendu :
  « Je passe sur ma voix de base. » suivi de la réponse en Kokoro. L'onglet « Coût et clés »
  affiche « Voix de base (Kokoro) en ce moment : plafond mensuel atteint… ». Le journal note
  « Voix de base (kokoro) : plafond mensuel atteint ».
- **Connexion** : `base_url: https://api.elevenlabs.invalid`. Attendu : même annonce, une seule
  fois.
- **Clé refusée** : mets une fausse clé dans `.env`. Attendu : même annonce, journal « clé
  ElevenLabs refusée ».
- **Retour** : remets la valeur, puis choisis un autre skin et reviens sur `femme` dans
  l'onglet Skins. MIKATA relit ses réglages et reparle avec Lucie, sans redémarrer.

```powershell
Select-String -Path data\logs\mikata.log -Pattern "Voix de base|de nouveau disponible|en pause" | Select-Object -Last 6
```

Le passage sur Kokoro au renouvellement du crédit et la reprise après un 429 sont vérifiés par
les tests (horloge simulée).
