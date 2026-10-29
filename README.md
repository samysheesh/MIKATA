<p align="center">
  <img src="assets/mascot/femme/apercu.png" alt="La mascotte de MIKATA dans ses dix états : repos, réveil, écoute, réflexion, action, parole, succès, erreur, suivi, micro coupé" width="720">
</p>

# MIKATA

Assistant vocal pour PC Windows, en français. Tu dis « MIKATA », tu parles naturellement,
elle agit sur ton PC et te répond à voix haute dans le casque.

L'écoute, la transcription et la voix tournent en local : ce n'est pas ton audio qui part sur
Internet, mais la phrase transcrite, envoyée à un LLM cloud qui décide quoi répondre et quoi
faire.

> **Statut** : projet personnel, développement en pause. Le code est fourni tel quel.

## Ce qu'elle fait

- **Écoute en continu, presque sans consommer** : mot d'activation détecté sur le CPU
  (openWakeWord), avec un vérificateur qui reconnaît ta voix. Raccourci Ctrl+Alt+M en secours.
- **Transcrit en local** avec Whisper large-v3-turbo sur le GPU, dès ta première pause, sans
  attendre d'être sûre que tu as fini de parler.
- **Répond vite** : la voix démarre pendant que le modèle écrit encore (Kokoro ou Piper en
  local, ElevenLabs en option).
- **Agit sur le PC** avec plus de 60 outils : applications et fenêtres, volume et médias,
  minuteurs et rappels, fichiers, presse-papiers, clavier et souris, PowerShell, mise en
  veille et extinction.
- **Pilote les applis et le web en arrière-plan** : lecture et clics par l'accessibilité de
  Windows (UI Automation), navigateur Brave dédié et invisible, recherche web et météo.
- **Apprend** : mémoire à long terme, et les tâches réussies deviennent des compétences
  rejouées ensuite sans appel au modèle.
- **Reste prudente** : aucune suppression définitive (tout part à la corbeille), sauvegarde
  avant d'écraser un fichier, confirmation orale des actions risquées.
- **Converse naturellement** : on enchaîne sans redire « MIKATA », on la coupe en parlant, on
  l'arrête avec « stop ».
- **S'affiche en mascotte de bureau** animée (skins personnalisables), avec une fenêtre de
  réglages.

<p align="center">
  <img src="docs/ui/reglages_accueil.png" alt="Fenêtre de réglages de MIKATA" width="560">
</p>

## Prérequis

- Windows 11, un micro et un casque.
- Une carte graphique NVIDIA (Whisper sur CUDA, environ 1,6 Go de mémoire vidéo). Sans GPU :
  `stt.device: cpu` et `stt.compute_type: int8` dans `config.yaml`, en plus lent.
- Une clé d'API pour un LLM compatible OpenAI. Par défaut : DeepSeek V4.1 Flash via
  [OpenCode Zen](https://opencode.ai/zen), moins d'un centime par commande.
- Facultatif : [Brave](https://brave.com) pour la navigation pilotée, une clé
  [Tavily](https://app.tavily.com) (gratuite) pour la recherche web, une clé ElevenLabs pour
  une voix plus naturelle.

Un port macOS (Apple Silicon) est inclus, mais les fonctions les plus récentes (pilotage des
applis, navigateur) visent Windows.

## Installation

Dans PowerShell :

```powershell
# 1. uv, le gestionnaire Python (rouvrir PowerShell ensuite)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Le code et ses dépendances (Python 3.11 est installé au besoin)
git clone https://github.com/samysheesh/MIKATA.git
cd MIKATA
uv sync

# 3. Les modèles locaux, environ 2 Go (Whisper, voix, détection de parole, mot d'activation)
uv run python scripts\download_models.py

# 4. Les clés : remplir au moins LLM_API_KEY
Copy-Item .env.example .env
notepad .env
```

Puis choisis ton micro et ton casque, et vérifie la configuration :

```powershell
uv run python scripts\list_audio_devices.py   # noms à mettre dans audio.input_device / output_device
uv run python -m mikata --check-config        # « Configuration valide » attendu
```

Tous les réglages sont dans [`config.yaml`](config.yaml), commentés ; les principaux se
changent aussi depuis la fenêtre de réglages.

## Utilisation

```powershell
uv run python -m mikata
```

L'icône apparaît dans la zone de notification, avec la mascotte. Dis **« Hey Jarvis »** (ou
appuie sur Ctrl+Alt+M), puis ta demande :

- « quelle heure est-il à Tokyo ? »
- « ouvre Spotify », « mets le volume à 30 », « pause »
- « lance le dernier opening de One Piece »
- « mets un minuteur de 10 minutes »

Tant que tu n'as pas entraîné le mot « MIKATA » sur ta voix, c'est « Hey Jarvis » qui sert
de mot d'activation. Pour l'entraîner (environ une demi-journée, surtout passive) : voir
[`docs/MOT_ACTIVATION.md`](docs/MOT_ACTIVATION.md). S'il se déclenche sans raison, monte
`wake.threshold` dans `config.yaml`.

Autres options utiles :

```powershell
uv run python -m mikata --ask "quel temps fait-il à Paris ?"   # une question, sans micro
uv run python -m mikata --help                                 # toutes les options

powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -StartNow   # démarrage avec Windows
powershell -ExecutionPolicy Bypass -File scripts\install_desktop_shortcut.ps1      # raccourci sur le Bureau
```

Facultatif : `scripts\install_usn_helper.ps1`, lancé une fois en administrateur, garde à jour
l'index des fichiers pour des recherches instantanées (« où est mon CV ? »).

## Comment ça marche

```text
micro ─► mot d'activation ─► fin de phrase ─► transcription ─► LLM + outils ─► voix en flux ─► casque
         openWakeWord (CPU)   Silero VAD       Whisper (GPU)     cloud            Kokoro / Piper
```

Un seul processus Python, organisé autour d'une boucle `asyncio` ; les calculs lourds tournent
dans des threads dédiés, et l'interface (PySide6) dans un processus à part.

| Dossier | Rôle |
| --- | --- |
| `mikata/audio`, `wake`, `vad`, `stt` | capture du micro, mot d'activation, fin de parole, transcription |
| `mikata/llm`, `agent`, `action` | client du LLM, boucle de l'agent et garde-fou, exécution des programmes et politiques d'action |
| `mikata/tools`, `mcp` | les outils (un outil = une classe enregistrée avec `@register_tool`), et des outils externes par MCP |
| `mikata/uia`, `browser`, `files` | accessibilité de Windows, navigateur piloté, index des fichiers |
| `mikata/learn` | compétences apprises et leçons tirées des tours précédents |
| `mikata/tts`, `ui` | synthèse vocale ; mascotte, réglages et icône |
| `scripts/`, `mikata/eval` | téléchargement des modèles, réglage audio, entraînement du mot, bancs d'essai |

Pour aller plus loin :

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) : choix techniques et budget de latence ;
- [`docs/AGENT_CORE.md`](docs/AGENT_CORE.md) : conception du cœur de l'agent (programmes,
  politiques d'action, apprentissage), avec ses sources ;
- [`docs/MASCOT_SKIN_GUIDE.md`](docs/MASCOT_SKIN_GUIDE.md) : créer un skin pour la mascotte.

## Développement

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest                              # audio, GPU et API Windows simulés : passe aussi sous Linux
uv run python scripts\eval_bench.py        # banc d'essai : 50 tâches sur un PC simulé, avec le vrai LLM
```

## Licence

Projet de [samysheesh](https://github.com/samysheesh), avec le port macOS de
[DevYatsu](https://github.com/DevYatsu). Code sous licence [Apache-2.0](LICENSE) (voir aussi
[`NOTICE`](NOTICE)).

Les skins « gars » et « femme » ne sont pas sous licence libre (tous droits réservés). Les
modèles téléchargés gardent leur propre licence, dont le modèle « hey_jarvis », non
commercial : voir [`THIRD_PARTY.md`](THIRD_PARTY.md).
