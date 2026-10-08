# Composants tiers

MIKATA est sous licence Apache-2.0 (`LICENSE`, `NOTICE`). Il utilise les composants
ci-dessous, chacun sous sa propre licence. Les **modèles** ne sont pas dans le dépôt
(`models/` est ignoré par git) : `scripts/download_models.py` les télécharge depuis leur
source officielle, et leur licence s'applique à leur usage.

Les licences ont été relevées le 2026-10-08, dans les métadonnées des paquets installés
(`uv.lock`) et sur les pages des projets. En cas de doute, la licence publiée par le projet
fait foi.

## Modèles téléchargés

| Composant | Fichier(s) | Licence | Usage dans MIKATA | Remarques |
| --- | --- | --- | --- | --- |
| [openWakeWord](https://github.com/dscripka/openWakeWord) : modèle `hey_jarvis` v0.1 | `models/hey_jarvis.onnx` | **CC BY-NC-SA 4.0 (non commercial)** | mot d'activation provisoire (phases 1 à 6), tests sur les vrais modèles | Les modèles pré-entraînés d'openWakeWord sont sous licence non commerciale. **Pas d'usage commercial**, et un travail dérivé doit garder la même licence. Remplacé par `mikata.onnx` (phase 7). |
| openWakeWord : extraction des caractéristiques | `models/melspectrogram.onnx`, `models/embedding_model.onnx` | Apache-2.0 (code d'openWakeWord). Le modèle d'embedding dérive de *speech_embedding* de Google (Apache-2.0). | calcul des caractéristiques audio avant `mikata.onnx` / `hey_jarvis.onnx` | Le README d'openWakeWord place ses modèles pré-entraînés sous CC BY-NC-SA 4.0 : à vérifier avant tout usage commercial. |
| `mikata.onnx`, `mikata_verifier.pkl` | `models/` | les tiens | mot d'activation « MIKATA » et vérificateur de ta voix (phase 7) | Entraînés sur ton PC, jamais commités. Avant de publier `mikata.onnx`, vérifie les licences des données d'entraînement (voix de synthèse, jeux de négatifs du notebook openWakeWord). Certaines sont non commerciales. |
| [Silero VAD](https://github.com/snakers4/silero-vad) v5.1.2 | `models/silero_vad.onnx` | MIT | détection de parole (fin de commande) | |
| [Whisper](https://github.com/openai/whisper) (OpenAI), via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | téléchargé par faster-whisper (cache Hugging Face) | MIT (poids et code) | transcription | Conversion CTranslate2 publiée par Systran (MIT). |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) : voix `fr_FR-siwis-medium` | `models/fr_FR-siwis-medium.onnx` (+ `.json`) | voix : CC BY 4.0 (jeu de données SIWIS, d'après la fiche du modèle) | voix de MIKATA par défaut | Attribution requise si tu rediffuses la voix ou des enregistrements. |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), fichiers ONNX de [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | `models/kokoro-v1.0.onnx`, `models/voices-v1.0.bin` | Apache-2.0 (poids), MIT (kokoro-onnx) | voix alternative (`tts.backend: kokoro`) | |

## Fichiers du dépôt

| Fichier | Origine | Licence |
| --- | --- | --- |
| `assets/sounds/*.wav` (wake, end, error, thinking, timer) | **Sons originaux générés par MIKATA** (`mikata/audio/earcons.py`, `synthesize()`). Vérifié le 2026-10-08 : chaque fichier est identique, à 1 bit de quantification près, à ce que produit ce code. Aucun son tiers. | Apache-2.0 (licence du projet) |
| `assets/mascot/default/` | Skin par défaut « Mika », **dessinée en code** (`mikata/ui/qt/painter.py`), sans image ni police tierce | Apache-2.0 |
| `assets/mascot/<skin>/` (tes skins) | Images que tu génères (Nano Banana...) ou importes, mises en forme par `scripts/make_skin.py`. **Pas dans le dépôt** (`.gitignore`) : elles restent sur ton PC. Auteur et licence de chaque skin : champs `author` et `license` de son `skin.json`. | celle que tu indiques |
| `tests/fixtures/hey_jarvis_ouvre_spotify.wav` | **Voix de synthèse** (confirmé : ce n'est pas ta voix), ajoutée à la phase 2 (commit a478e7f) pour les tests sur les vrais modèles. 16 kHz mono, 1,8 s, pic normalisé à 0,5, écrit par le code du projet (`mikata/audio/wavfile.py`). **Voix exacte non retrouvée** : aucun script de génération n'est dans le dépôt ni dans son historique, et la session de travail de la phase 2 n'est plus accessible. L'analyse du son (hauteur, bande passante) ne suffit pas à identifier la voix. | **Inconnue** tant que la voix n'est pas identifiée. Avant une publication : régénérer ce fichier avec une voix de licence connue (par exemple Kokoro, Apache-2.0), puis revérifier les tests sur les vrais modèles sur le PC. |
| `docs/ui/*.png`, `docs/ui/*.gif` | Captures de l'interface, rendues par `scripts/ui_screenshots.py` (données fictives ; le skin de démonstration « Mika recolorée » est Mika, recolorée puis passée par `make_skin.py`) | Apache-2.0 |

## Bibliothèques Python (dépendances de `pyproject.toml`)

| Paquet | Licence | Usage |
| --- | --- | --- |
| ctranslate2 | MIT | moteur d'inférence de faster-whisper |
| faster-whisper | MIT | transcription |
| openwakeword | Apache-2.0 (code) | mot d'activation |
| onnxruntime | MIT | modèles ONNX (mot d'activation, VAD, Kokoro, Piper) |
| piper-tts | **GPL-3.0-or-later** | synthèse vocale Piper. MIKATA l'importe comme bibliothèque : une distribution qui l'embarque (exécutable tout-en-un, par exemple) relève de la GPL-3.0. Le dépôt ne contient que le code de MIKATA (Apache-2.0, compatible GPL-3.0) ; chacun installe piper-tts avec `uv sync`. |
| kokoro-onnx | MIT | synthèse vocale Kokoro |
| openai | Apache-2.0 | client LLM (API compatible OpenAI) |
| httpx2, h2 | BSD-3-Clause, MIT | HTTP/2 persistant |
| pydantic, pydantic-settings | MIT | configuration |
| pyyaml | MIT | lecture de config.yaml |
| ruamel.yaml | MIT | réécriture de config.yaml sans perdre les commentaires (réglages, phase 8) |
| python-dotenv | BSD-3-Clause | lecture et écriture de .env (réglages, phase 8) |
| **PySide6-Essentials**, shiboken6 (Qt 6) | **LGPL-3.0** (ou GPL-2.0/3.0) | interface : mascotte et réglages (phase 8). Liée dynamiquement, installée par pip/uv, non modifiée : les obligations de la LGPL sont respectées tant qu'on ne la fige pas dans un exécutable sans permettre de la remplacer. |
| numpy | BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0 pour des parties) | calcul audio |
| scipy | BSD-3-Clause | détourage des images de skins (`scripts/make_skin.py`) ; déjà installé par openwakeword |
| sounddevice | MIT | audio (PortAudio, MIT) |
| soxr | LGPL-2.1-or-later | rééchantillonnage audio |
| pillow | MIT-CMU (HPND) | images (icône, thèmes de la mascotte) |
| loguru | MIT | journal |
| psutil | BSD-3-Clause | charge CPU, processus |
| nvidia-ml-py | BSD-3-Clause | charge GPU |
| send2trash | BSD-3-Clause | corbeille (aucune suppression définitive) |
| rapidfuzz | MIT | recherche approximative (applis, phrases) |
| trafilatura | Apache-2.0 | extraction de texte des pages web |
| pyperclip | BSD-3-Clause | presse-papiers |

Windows uniquement :

| Paquet | Licence | Usage |
| --- | --- | --- |
| nvidia-cublas-cu12, nvidia-cudnn-cu12 | licence NVIDIA (redistribuable via PyPI) | CUDA pour Whisper |
| pystray | LGPL-3.0 | icône de la zone de notification |
| pynput | LGPL-3.0 | raccourci clavier global |
| pywin32 | PSF-2.0 | API Windows (corbeille sûre, fenêtres) |
| pycaw, comtypes | MIT | volume Windows |
| win11toast | MIT | notifications |

Outils de développement : pytest (MIT), ruff (MIT). En CI : [gitleaks](https://github.com/gitleaks/gitleaks)
(MIT), qui cherche des secrets dans l'historique.
