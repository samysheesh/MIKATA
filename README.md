# MIKATA

Assistant vocal local pour PC Windows 11 : on dit « MIKATA », il écoute, transcrit,
interroge un LLM cloud, agit sur le PC et répond à voix haute dans le casque.

- Architecture (fait foi) : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Instructions pour Claude Code : [`CLAUDE.md`](CLAUDE.md)
- Lancement : `uv run python -m mikata` (options : `--check-config`, `--config CHEMIN`, `--no-tray`)
- Réglages : `config.yaml` ; secrets : `.env` (copier `.env.example`, jamais commité)
- Logs : `data/logs/mikata.log` (rotation 5 Mo × 5)

## Avancement

| Phase | Contenu | État |
| --- | --- | --- |
| 0 | Squelette : config, logs, événements, icône, CI | livrée, à valider sur le PC |
| 1 à 7 | Voir `docs/ARCHITECTURE.md` §11 | à faire |

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
