# MIKATA — instructions pour Claude Code

MIKATA est un assistant vocal qui tourne en continu sur un PC Windows 11 (Ryzen 9 5900X, 32 Go RAM, RTX 3080 10 Go).
On l'active en disant « MIKATA », il écoute, transcrit, demande à un LLM cloud, agit sur le PC et répond à voix haute dans le casque.

## Source de vérité

- Lis `docs/ARCHITECTURE.md` en entier avant toute modification. Il fait foi.
- Le tableau « Tes décisions » de la section 1 prime sur tout le reste.
- Respecte l'arborescence et les interfaces de la section 9, et les noms de clés de config de la section 10.
- Si tu dois t'écarter du document, explique pourquoi dans le message de commit et dans ta réponse finale.

## Cible technique

- Windows 11, Python 3.11, environnement géré avec `uv` (`pyproject.toml`, `uv.lock` commité).
- Lancement : `python -m mikata`.
- Langue de l'interface, des prompts et des messages vocaux : français.

## Tu travailles dans une session cloud sous Linux

- Tu n'as ni micro, ni GPU, ni Windows. Tout le code spécifique (audio WASAPI, CUDA, pywin32, pycaw, pystray, notifications, raccourcis clavier) est isolé derrière une interface et **mocké dans les tests**.
- Les imports Windows-only se font de façon paresseuse (dans les fonctions ou derrière `sys.platform == "win32"`) pour que `pytest` passe sous Linux.
- Les dépendances Windows-only sont marquées dans `pyproject.toml` avec `; sys_platform == 'win32'`.

## Règles de code

- Typage partout, `ruff` (lint + format) et `pytest` doivent passer avant de rendre la main.
- Aucune dépendance hors `pyproject.toml`, aucun secret dans le code (`.env`, jamais commité, `.env.example` fourni).
- Tout ce qui est réglable passe par `mikata/config.py` (pydantic-settings, `config.yaml` + `.env`).
- Un outil = une classe enregistrée via `@register_tool` dans `mikata/tools/registry.py`.
- Jamais de calcul dans le callback audio PortAudio.
- Aucune suppression définitive de fichier nulle part : corbeille via `send2trash` uniquement.

## Façon de travailler

- Une phase de la section 11 par session, sur une branche `phase-N`.
- Ne modifie pas le code des autres phases sauf nécessité, et dis pourquoi.
- À la fin de chaque phase, mets à jour `README.md` avec la liste exacte des commandes que l'utilisateur doit lancer sur son PC Windows pour vérifier les critères d'acceptation de la phase.
