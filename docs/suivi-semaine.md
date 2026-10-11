# Suivi de la semaine — retours MIKATA (phase 12e en test)

Fichier de travail (non versionné). Chaque retour de test vient s'ajouter ici ; au reset du
quota, on reprend tout dans l'ordre.

## À surveiller cette semaine

- [ ] **Rejeu local** : « où est X » doit répondre en < 1 s et logguer
      `rejoué en local (…, 0 appel LLM)`. Si ça repasse par le plan LLM :
      vérifier que le savoir-faire a bien `slots` (les anciens appris avant la 12e n'en
      ont pas → « oublie comment … » puis réapprendre).
- [ ] **Index** : au redémarrage, log `Index des fichiers : N entrées (base existante)`
      (pas de « se construit »). Nouveau fichier créé → visible dans find_path en ~10 s
      (helper USN actif : tâche planifiée « MIKATA Indexer USN »).
- [ ] **Transcription Whisper** : si le rejeu local ne se déclenche pas à cause d'une
      mauvaise transcription (« OUH, LE GOLF LEGENDE » au lieu de « où est league of
      legends ») → amélioration à prévoir : pattern tolérant au flou.
- [ ] **Vrai double lancement** : si deux mascottes s'affichent vraiment (pas le duo
      lanceur uv + python réel, qui est normal), le signaler.
- [ ] **Procédures pourries apprises** : si une commande dégradée (scan PowerShell) est
      mémorisée, « oublie comment … » la purge.

## Améliorations à planifier (après reset du quota)

1. Étendre le rejeu local : liaisons `outputs` pour d'autres outils (apps, media, web),
   plus de payloads structurés.
2. Dédupliquer les savoir-faire à la distillation (3 « ouvrir un dossier » appris en
   doublon : étapes identiques → une seule procédure).
3. Pattern de slots tolérant aux erreurs de transcription (fuzzy sur les mots du
   pattern, pas seulement les accents).
4. Ne pas auto-apprendre les commandes qui contiennent des replis lents (run_powershell
   de recherche disque, etc.).
5. Latence des commandes nouvelles : mesurer le cache de prompt / le premier token, voir
   si on peut gratter sur les 2 appels LLM (hors rejeu).
6. Merger `phase-12e-replay` après validation de la semaine.

## Retours de test (à remplir)

| Date | Ce qui ne marche pas / pas bien | Logs |
| --- | --- | --- |
| 10/10 | Démo : « lance le dernier opening de One Piece » → ouvre juste YouTube sans cliquer ; « chante une musique » → ouvre une recherche YouTube | browser_open échoue : « Brave est ouvert sans le port de débogage » → repli open_url |
| | | |

## Cause racine démo du 10/10 (diagnostiquée)

Brave était **déjà ouvert sans le port CDP 9222** pendant la démo. Conception 12b : MIKATA ne
ferme jamais une appli ouverte → `browser_open` refuse poliment, le LLM se rabat sur
`open_url` (ouvre la page, mais aucun clic possible).

**Corrigée par la phase 12f** (`phase-12f-browser`) : MIKATA navigue dans son propre profil
Brave (`--user-data-dir=data/brave_profile`, fenêtre minimisée en arrière-plan) — ton Brave
n'est jamais touché, et `browser_open` ne refuse plus jamais. Reste à faire : merger et
tester à la voix après le reset.

Correctifs de repli (si la 12f ne suffit pas sur un cas) :
1. `scripts/brave_debugging_port.ps1` : patcher le raccourci de Brave + son auto-démarrage
   avec `--remote-debugging-port=9222` (réversible, comme `discord_accessibility.ps1`).
2. Prompt : quand `browser_open` échoue malgré tout, dire clairement quoi faire au lieu
   de dériver vers open_url.
3. Prompt : « chante une musique » = répondre en chantant (TTS), pas ouvrir YouTube —
   sauf demande explicite d'une chanson précise à lancer.
