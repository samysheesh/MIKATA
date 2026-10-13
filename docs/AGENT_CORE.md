# MIKATA — le cœur agent : refonte à partir de l'appel LLM

13 octobre 2026 · validé le soir même (décisions en section 6) · implémentation par phases

Ce document propose de reconstruire sur des bases saines tout ce qui se passe **après la
transcription** : comprendre la demande, agir sur le PC, vérifier, apprendre. La capture audio,
le mot d'activation, la transcription, la voix et l'interface (phases 0 à 11) ne changent pas.

Une fois validé, il remplace les sections 6 et 7 de `docs/ARCHITECTURE.md` et le plan
`AGENT_V2.md` (qui n'existe que sur la branche `agent-v2-plan`).

## 1. Constat mesuré

Mesures relevées le 13/10/2026 sur le PC, dans `data/logs/mikata.log` et `data/agent2/traces/`.

| | Boucle d'outils (phases 4 à 12) | Agent v2 (phase 13) |
| --- | --- | --- |
| Commandes mesurées | 230, variées | 15, dont 13 fois la même demande |
| Réglées en un seul appel LLM | 61 %, premier son médian 2,1 à 2,4 s | aucune |
| Tâches multi-étapes | 31 commandes, 15,5 s en médiane | 41,7 s en médiane (25 à 81 s), 8,1 appels |
| Réussite | non vérifiée (aucun contrôle du résultat) | 9 sur 15 |
| Apprentissage | rejeu local vu une fois dans les logs | 0 compétence (8 compilations vides, 1 refus) |

Autres faits établis :

- **Le code généré tourne dans le processus principal.** `agent2/sandbox.py` l'exécute par
  `exec` ; un `while True: pass` gèle MIKATA malgré `timeout_s=1` (vérifié).
- **Les prompts de la v2 sont écrits pour une seule demande** (adresse `youtube.com/watch`,
  « le dernier, pas un film »).
- **La batterie à 16/20 tourne sur un simulateur** ajusté en même temps que les cas.
- **La CI est rouge sur `main`** depuis le 11/10 (28 pushes), dont deux tests du garde-fou.
- **56 outils, environ 5 600 tokens de schémas** partent à chaque appel LLM.

Conclusion : la boucle d'origine est rapide sur le cas courant mais ne vérifie rien et
n'apprend presque pas ; la v2 a les bonnes idées (code comme action, vérification,
compétences) dans une structure qui multiplie les appels et n'a pas fait ses preuves.

**Mesure de référence du banc d'essai (phase 14a).** Sur les 49 tâches du banc, PC simulé et
vrai LLM, la boucle d'origine réussit 47 tâches (96 %), en 2,2 s de médiane et 2 appels LLM ;
les enchaînements longs prennent 4,6 s et 2 à 5 appels. Les deux échecs venaient du banc.
Le nouveau cœur ne se justifie donc pas par ces demandes-là, qu'il doit seulement ne pas
dégrader, mais par ce que la boucle d'origine ne garantit pas : la sécurité par construction
(elle n'a résisté aux contenus piégés que parce que le modèle a été prudent), la vérification
du résultat, l'apprentissage, et la tenue sur le vrai web et les vraies interfaces.

## 2. Ce que dit la recherche

Chaque principe retenu s'appuie sur un résultat publié. Références en section 8.

| # | Principe | Résultat qui le fonde |
| --- | --- | --- |
| 1 | **Une seule boucle, dirigée par le modèle** | La planification pèse 50 à 75 % de la latence et les agents font 2,7 à 4,3 fois trop d'étapes (OSWorld-Human). « Chercher la solution la plus simple, n'ajouter de la complexité que si elle améliore les résultats » (Anthropic). |
| 2 | **Le code comme action, pour tout ce qui enchaîne** | Jusqu'à 20 points de réussite en plus et 30 % d'actions en moins que des appels JSON (CodeAct). Appels modèle 10,2 → 1,8 et latence 237 → 118 s (ActionEngine). Pas pour une action seule (Anthropic). |
| 3 | **Un espace d'action petit et soigné** | Peu d'outils de haut niveau, erreurs exploitables, exemples d'appel : 72 → 90 % sur les paramètres complexes. Charger 3 à 5 outils et différer le reste : 49 → 74 % (Anthropic). |
| 4 | **API d'abord, interface en secours** | Couche unifiée API + interface : +6 à +8 points. 62 % des échecs viennent de la détection des contrôles. Plusieurs actions validées par appel : jusqu'à 51 % d'étapes en moins (UFO²). |
| 5 | **Vérifier par des post-conditions bon marché** | Contrôles déterministes après chaque action, réflexion coûteuse seulement après un échec (synthèse Efficient GUI Agents). Noter l'état final, pas le chemin (Anthropic). |
| 6 | **Des compétences = programmes vérifiés, dans l'espace d'action** | Programme vérifié appelable 40,1 % contre texte non vérifié 32,6 % (ASI). Paquet révisable après échec : +10,5 points sur OSWorld ; sans critique isolé −8,6 points (EvoSkill-GUI). |
| 7 | **La sécurité par construction** | « Une fois du contenu non fiable lu, il doit être impossible qu'il déclenche une action conséquente » (Design Patterns). 0 attaque réussie sur 949, 77 % des tâches résolues contre 84 % sans défense (CaMeL). |
| 8 | **Rapide et lent en parallèle** | Répondeur rapide + raisonneur lent (Talker-Reasoner). Une phrase d'attente masque le délai mais coupe la parole dans 48 % des cas si elle est systématique (Full-Duplex-Bench-v3). |
| 9 | **Le contexte est une ressource rare** | Préfixe stable en cache, récupération au moment utile, exemples canoniques plutôt que règles (Anthropic). |
| 10 | **Mesurer avant d'optimiser** | 20 à 50 tâches réelles suffisent pour démarrer ; correcteurs déterministes ; `pass^k` pour la constance (Anthropic). Rapporter le coût avec la réussite (synthèse). |
| 11 | **Apprendre à plusieurs niveaux, du plus sûr au plus risqué** | Carnet de leçons mis à jour par petites touches : 42,4 → 59,4 % sur AppWorld, avec un modèle DeepSeek sans réflexion (ACE). Bibliothèque de code réutilisée sur des tâches nouvelles (Voyager). Outils créés à la volée : 75 % sur GAIA (Alita). |
| 12 | **Borner l'auto-évolution** | Une mémoire qui s'accumule sans garde fait tomber le taux de refus de 99,4 à 54,4 % ; 56 à 76 % des outils créés ou réutilisés librement sont dangereux (*Your Agent May Misevolve*). |

Deux mises en garde tirées des mêmes sources :

- **CaMeL a un prix** : 2,7 à 2,8 fois plus de tokens qu'un appel d'outils natif, et le
  planificateur ne peut pas décider d'après un contenu qu'il n'a pas le droit de lire.
- **Les compétences peuvent nuire** : transfert négatif sur les tâches courtes et directes, et
  aucun vérificateur n'empêche une révision de dégrader une compétence (EvoSkill-GUI). Seuls
  15,6 % des essais d'induction passent la vérification (ASI).

## 3. Architecture cible

### 3.1 Vue d'ensemble

```text
commande (texte transcrit)
   │
   ├─ 0. rejeu local d'une compétence active ───────────────► actions        0 appel LLM
   │
   └─ 1. appel LLM (modèle rapide, en flux)
          ├─ texte ──────────────────────────────────────────► parole         1 appel
          ├─ appel direct d'un outil ────────────────────────► action         1 appel
          └─ script ─► interpréteur ─► SDK ─► politiques ─► outils ─► PC      1 appel + extractions
                            │                     │
                       provenance          confirmation ou refus
   │
   └─ trace ─► apprentissage en arrière-plan (induction, vérification, révision)
```

La frontière avec le reste de MIKATA ne bouge pas : `Agent.respond(command, speech, on_phase,
listen, follow_up) -> TurnResult`. `app.py`, l'audio et l'interface ne voient aucune
différence.

### 3.2 La boucle unique

Un tour a trois issues, choisies par le modèle en un appel :

1. **Texte** : question, explication, bavardage. Dit en flux, comme aujourd'hui.
2. **Appel direct** : une action simple (ouvrir, volume, pause, minuteur). L'outil s'exécute,
   « C'est fait » sort du cache. C'est le chemin des 61 % de commandes actuelles : il ne doit
   pas ralentir.
3. **Script** : dès que la demande enchaîne, dépend d'une information à trouver, ou boucle.
   Le modèle écrit un court programme contre le SDK ; l'interpréteur l'exécute d'une traite.

Il n'y a plus d'étapes « intention », « inconnues », « canal », « plan » séparées : ce sont
des lignes du script. Exemple, pour « lance le dernier opening de One Piece » :

```python
resultats = web_search("dernier opening One Piece titre")
info = extract(resultats, {"titre": str},
               "Quel est le titre du dernier opening de One Piece ?")
expect(info.titre, "un titre a été trouvé")
browser_open("https://www.youtube.com/results?search_query=" + info.titre)
browser_click(info.titre)
return f"C'est lancé : {info.titre}."
```

Deux appels LLM (le script, l'extraction) au lieu de huit. Si le script échoue, l'erreur
revient au modèle avec ce qui a déjà été fait, et il corrige ; après deux échecs la
réflexion monte, après trois le tour se ferme aux outils et MIKATA explique.

Ce que le script rend décide de la suite : une phrase courte est dite telle quelle (un seul
appel LLM) ; une donnée, ou ce qu'il a affiché par `print`, revient au modèle qui formule.

### 3.3 L'espace d'action

- **Les outils existants restent l'implémentation.** Un outil = une classe `@register_tool`
  (règle de `CLAUDE.md` inchangée). Le SDK est une façade typée générée depuis le registre.
- **Chaque outil déclare son effet** : `lire`, `agir` (réversible), `conséquent` (sortant ou
  difficile à défaire : shell, écriture, envoi, saisie clavier), `destructif` (interdit).
- **Chaque outil renvoie un résultat structuré** (`texte` pour le modèle, `données` pour le
  script) : le script enchaîne sur des valeurs, pas sur du texte à relire. Cela remplace le
  canal latéral `publish_data`.
- **Dans un programme, un outil s'appelle par son nom d'outil**, avec les mêmes arguments :
  `set_volume(level=30)`, `move_file(source, destination)`. Le modèle n'a rien de nouveau à
  apprendre, et la notice des programmes ne liste que les outils qui rendent un objet à
  champs. Les noms du SDK (`audio.volume`, `files.move`) restent disponibles : ce sont ceux
  des compétences (phase E).

  Écart mesuré en phase 14d par rapport à la première version de ce document, qui prévoyait
  de remplacer les schémas JSON par une référence du SDK : avec deux noms pour la même
  action, le modèle se trompait d'arguments et ses programmes échouaient. Garder les schémas
  natifs et les mêmes noms a supprimé ces échecs ; réduire le nombre de schémas envoyés
  reste une optimisation possible, à faire sous contrôle du banc.
- **Ordre des canaux**, du plus fiable au moins fiable : outil dédié, navigateur (DOM),
  arbre d'accessibilité, clavier et souris, puis vision (plus tard, en secours).
- **Observations filtrées** : contrôles actionnables seulement, bornées, et différences
  depuis la dernière observation plutôt que l'écran entier.

### 3.4 L'interpréteur de scripts

Le code généré n'est plus passé à `exec`. Un interpréteur maison évalue un sous-ensemble de
Python (affectations, conditions, boucles, compréhensions, appels au SDK), nœud par nœud :

- **budget d'exécution** en nombre d'opérations et en durée : une boucle infinie s'arrête ;
- **aucun accès** en dehors des fonctions du SDK : pas d'import, pas d'attribut caché ;
- **provenance** attachée à chaque valeur et propagée par chaque opération (section 3.5) ;
- **pause et reprise** à chaque appel du SDK : c'est là que passent les confirmations orales ;
- **erreurs expurgées** : un message qui dépend d'un contenu non fiable n'est pas renvoyé au
  modèle tel quel.

Alternative étudiée : Monty (interpréteur Python en Rust de Pydantic, février 2026). Il est
rapide et isole bien, mais il est jeune, sans audit indépendant connu, et ne suit pas la
provenance des valeurs. À garder comme moteur de rechange possible.

### 3.5 Sécurité : provenance et politiques

Trois classes de sources :

| Classe | Exemples |
| --- | --- |
| Fiable | La parole de l'utilisateur, la config, les faits qu'il a dictés |
| Locale | État du système : fenêtres, volume, noms de fichiers de l'index |
| Non fiable | Pages web, résultats de recherche, contenu de fichiers, presse-papiers, texte lu dans une appli |

Règles :

1. **Le planificateur n'a pas besoin de lire le contenu non fiable.** Le script le confie à
   `extract(...)` : un appel LLM en quarantaine, sans outil, qui rend un objet au schéma fixé.
   La valeur extraite reste marquée non fiable.
2. **Les politiques se vérifient à l'appel de l'outil**, sur l'effet de l'outil et la
   provenance de ses arguments. Exemples : une commande shell construite à partir d'une
   valeur non fiable est refusée ; une adresse non fiable ne s'ouvre qu'en `http(s)` ; un
   message dont le destinataire vient d'une page demande un « oui ».
3. **Si le modèle doit lire du contenu non fiable pour décider** (explorer une interface
   inconnue, par exemple), le tour devient « teinté » : les actions conséquentes demandent
   alors une confirmation orale.
4. **Les actions destructives restent interdites**, toujours (corbeille uniquement).

`agent/guard.py` reste la bibliothèque de règles (liste noire, dossiers autorisés,
corbeille, sauvegardes) et ses tests de la phase 5 restent le critère. Il devient un filet
derrière les politiques, plus la seule barrière.

Ce que cette conception n'arrête pas : un résumé faussé par une page (rien n'est exécuté, mais
ce qui est dit peut être faux), et l'hameçonnage de l'utilisateur lui-même.

### 3.6 Compétences

Une compétence est un **paquet lisible** dans `data/skills/<nom>/` :

| Fichier | Rôle |
| --- | --- |
| `SKILL.md` | Métadonnées : intention, appli, plateformes, paramètres, mots-clés, statut, version, compteurs |
| `skill.py` | Une fonction `run(sdk, **params)` dans le sous-ensemble de l'interpréteur |
| `recover.md` | Règles de reprise connues (fenêtre surgissante, contrôle déplacé) |
| `failures/` | Au plus trois cas d'échec, pour la révision |

- **Induction** : seulement depuis une trace dont les post-conditions ont réussi. Les valeurs
  venues de la demande deviennent des paramètres.
- **Vérification avant adoption** : la demande d'origine est rejouée avec la compétence ; il
  faut que la tâche réussisse, que la compétence soit appelée, et que chaque appel ait un
  effet. Pour une compétence aux effets conséquents (un envoi), le rejeu se fait sur les
  résultats enregistrés, et la première utilisation réelle compte comme essai surveillé.
- **Dans l'espace d'action** : `skills.<nom>(...)` s'appelle depuis un script ou directement.
- **Récupération par métadonnées**, avec un seuil ; rien n'est proposé pour une action seule.
- **Rejeu local** : si la phrase colle au motif d'une compétence active et que ses paramètres
  se lient, elle s'exécute sans appel LLM (l'idée de la phase 12e, conservée).
- **Révision** : après un échec, un critique isolé (il ne voit que la demande, les
  observations et les actions) diagnostique ; la révision produit une nouvelle version,
  l'ancienne est gardée. Cycle : proposée, à l'essai, active, rétrogradée.
- **Format compatible Agent Skills** (`SKILL.md`, standard ouvert de décembre 2025) pour le
  partage. Une compétence importée arrive « proposée » et se relit avant usage : 341 des
  2 857 compétences auditées sur la place de marché d'OpenClaw étaient malveillantes.

### 3.7 Mémoire

Fichiers simples, pas de cadre externe : les scores des cadres de mémoire sont mesurés par
leurs propres éditeurs, et une comparaison de 2026 rapporte pour un même système 94 %
annoncés contre 49 % relevés par un tiers (écart non vérifié à la source).

- **Faits** dictés par l'utilisateur (l'existant, `data/memory.json`).
- **Préférences** apprises, avec confiance (« musique → Spotify »).
- **Épisodes** : résumés de traces récentes.
- Chaque entrée porte sa **provenance** : rien de ce qui vient d'un contenu non fiable ne
  devient un fait sans passer par l'utilisateur.

### 3.8 Latence : quatre chemins, un budget chacun

La latence est la priorité n° 1 du projet. Chaque demande prend le chemin le plus court qui
suffit, et chaque chemin a un budget mesuré par le banc d'essai.

| Chemin | Quand | Appels LLM | Budget visé |
| --- | --- | --- | --- |
| **0. Rejeu local** | La phrase colle à une compétence active | 0 | moins d'1 s |
| **1. Direct** | Question, bavardage, action simple | 1 | premier son sous 2 s |
| **2. Script** | La demande enchaîne ou dépend d'une information | 1 + extractions | moins de 10 s en médiane |
| **3. Exploration** | Interface inconnue, à observer pas à pas | 8 pas au plus | moins de 30 s, en tâche de fond |

Ce qui tient ces budgets :

- **Un seul appel pour choisir le chemin** : pas d'étage de compréhension avant d'agir.
- **Un script regroupe les actions** : un appel au lieu d'un par étape, et les lectures
  indépendantes partent en parallèle (`parallel(...)`).
- **Post-conditions déterministes** : aucune vérification par un second appel LLM sur le
  chemin normal. Le coût de toute vérification est compté dans le budget.
- **En exploration, plusieurs actions validées par pas** (jusqu'à 51 % d'étapes en moins dans
  UFO²).
- **Préfixe de contexte stable** pour le cache de prompt, observations filtrées, extraction
  confiée au modèle rapide.
- **Tout l'apprentissage hors du chemin critique** : induction, vérification et révision
  tournent en arrière-plan, jamais pendant la réponse.
- **Ce qui existe est conservé** : connexion préchauffée au mot d'activation, transcription
  spéculative, phrases en cache, son de réflexion.

Tâches longues (chemins 2 et 3 au-delà de quelques secondes) :

- elles deviennent des **tâches de fond** avec un état : on peut demander où elles en sont ou
  les annuler pendant qu'une autre commande passe ;
- **l'accusé de réception n'est dit que si l'attente prévue dépasse un seuil**, et il part en
  même temps que l'action, jamais avant (une phrase d'attente systématique coupe la parole
  de l'utilisateur dans 48 % des cas, Full-Duplex-Bench-v3). Chaque outil déclare sa durée
  typique.

Règle de non-régression : une phase qui ralentit le chemin 1 n'est pas fusionnée.

Piste gardée pour plus tard : exécuter d'avance, avec le modèle rapide, l'action la plus
probable (jusqu'à 20 % de latence en moins, *Speculative Actions*).

### 3.9 Modèles et contexte

- **Deux rôles configurables**, `llm.fast` et `llm.strong`, sur le client compatible OpenAI
  existant. Rapide pour le chemin court et l'extraction ; fort pour un script inédit après
  échec et pour l'induction. Les deux peuvent pointer vers le même fournisseur.
- **Préfixe stable** en tête (prompt système, outils directs, référence du SDK), puis mémoire,
  compétences candidates, état du PC, historique, commande.
- **Prompts au bon niveau** : principes et quelques exemples canoniques, aucune consigne
  écrite pour un cas précis.

### 3.10 Évaluation

Un banc d'essai vit dans le dépôt (`mikata/eval/`, tâches dans `tests/eval/`) :

- **Une tâche** : la demande, l'état de départ, des correcteurs d'**état final**
  déterministes (volume à 30, fichier présent, onglet sur telle adresse), des balises.
- **Deux modes** : à blanc sur un registre simulé (CI, non-régression de la tuyauterie) et
  réel sur le PC (la vraie mesure).
- **Mesures** : réussite sur trois essais (`pass^3`), premier son, durée, appels LLM, tokens,
  coût.
- **Jeu équilibré** : des cas où MIKATA doit agir, et des cas où il ne doit pas (pages
  piégées servies en local : zéro action conséquente attendue).
- **Départ** : 30 à 40 tâches tirées des commandes réelles et des demandes visées.

### 3.11 Auto-amélioration : quatre niveaux, du plus sûr au plus risqué

Principe gardé de `AGENT_V2.md` : **noyau stable, périphérie apprise**. Le code de `mikata/`
est écrit et relu par des humains ; ce que MIKATA apprend vit dans `data/`, lisible,
versionné, désactivable.

| Niveau | Ce qui s'améliore | Méthode | Place dans MIKATA |
| --- | --- | --- | --- |
| **1. Carnet de leçons** | Des notes courtes par appli ou domaine (« dans Discord, le bouton du micro s'appelle Rendre muet ») | ACE : petites entrées avec compteurs « utile / nuisible », ajoutées et retouchées une à une, jamais réécrites en bloc | `data/playbook/`, phase E |
| **2. Compétences** | Des programmes vérifiés, réutilisés et révisés | ASI, EvoSkill-GUI, Voyager (section 3.6) | `data/skills/`, phase E |
| **3. Nouvelles capacités** | Un outil qui manquait | Alita : l'agent écrit lui-même l'outil | Une compétence composée de primitives du SDK suffit dans la plupart des cas ; une vraie primitive nouvelle (code avec accès au système, serveur MCP) reste désactivée jusqu'à validation, phase F |
| **4. Le code du noyau** | Les prompts et le code de `mikata/` | Darwin Gödel Machine : 20 → 50 % sur SWE-bench en 80 itérations, sous bac à sable et supervision humaine | Jamais à chaud. Hors ligne seulement : le banc d'essai sert de juge, le résultat est une proposition relue par un humain |

Pourquoi réécrire le carnet par petites touches : dans ACE, une réécriture complète a fait
passer un contexte de 18 282 à 122 tokens en une étape, et la réussite sous le niveau sans
apprentissage.

Garde-fous de l'auto-évolution, tirés de *Your Agent May Misevolve* :

1. **Ce qui est appris ne touche jamais aux politiques ni au garde-fou.** Ils sont dans le
   code, hors de la périphérie apprise.
2. **Une compétence tourne dans l'interpréteur, sous les mêmes politiques** qu'un script : elle
   ne peut rien faire que les primitives n'autorisent pas.
3. **Le signal de réussite vient d'une post-condition ou d'une correction de l'utilisateur**,
   jamais de l'agent qui se note lui-même. Avec un signal peu fiable, l'apprentissage dégrade
   (−3,4 points dans ACE).
4. **Les leçons entrent dans le contexte comme des références, pas comme des règles** (ce seul
   changement réduit les comportements dangereux dans l'étude).
5. **Rien n'est promu « actif » sans repasser le banc d'essai** ; tout est versionné et
   réversible.
6. **Tout code venu de l'extérieur se relit avant usage** : face à des outils malveillants, les
   modèles testés n'en refusent que 0,3 à 7 %.

### 3.12 Tâches jamais vues

Aucune architecture ne garantit « toutes les tâches » : la couverture dépend d'abord du modèle
et de ce qu'il perçoit. Ce que la recherche montre de plus efficace, dans l'ordre où MIKATA
l'applique :

1. **Des primitives générales et du code** plutôt qu'un catalogue de cas. Shell, fichiers,
   navigateur, arbre d'accessibilité, clavier et souris, recherche web : tout ce qui se fait
   par commande, par API ou par l'interface s'écrit avec ces briques (CodeAct ; Alita :
   « prédéfinir le minimum »).
2. **Chercher le savoir au moment du besoin.** Devant une appli inconnue, le script cherche
   d'abord « comment faire X dans Y » sur le web, puis agit ; la leçon va dans le carnet
   (UFO² s'appuie de la même façon sur l'aide des applications).
3. **Explorer en observant** quand il n'y a pas de recette : observer, agir, contrôler, dans
   une limite de pas (chemin 3).
4. **Monter en puissance** : après un échec, le tour passe au modèle fort. Les gains du code
   comme action grandissent avec la capacité du modèle, et sur OSWorld l'écart entre un petit
   modèle (24 à 47 %) et les meilleurs (environ 85 %) dépasse tout ce qu'apporte la structure.
5. **La vision en secours** pour les applis sans accessibilité (62 % des échecs d'UFO²
   viennent de contrôles non détectés), phase F.
6. **Demander ou dire non** : une question courte si la demande est ambiguë, et un refus
   honnête plutôt qu'une action au hasard.
7. **Apprendre de la réussite** : la tâche inédite devient une compétence et des leçons, donc
   la suivante du même genre prend le chemin court (OSExpert : environ 80 % de l'écart
   d'efficacité avec un humain comblé par des compétences apprises).

## 4. Ce qui part, ce qui reste

| Élément | Décision | Raison |
| --- | --- | --- |
| `mikata/agent2/` (23 fichiers, 3 483 lignes), ses 20 fichiers de tests, `tests/agent2_eval/`, `scripts/eval_agent2.py`, section `agent2` de la config, passe d'arrière-plan | **Supprimé** | Section 1 |
| `mikata/skills/` (497 lignes) et `data/skills/skills.json` | **Remplacé** par les paquets de la section 3.6 | Doublons, procédures dégradées apprises, rejeu rare |
| `mikata/agent/loop.py` (884 lignes) | **Réécrit** en modules ; comportements validés gardés (suivi, barge-in, mots-clés de réflexion, résumé) | Une classe mêle huit responsabilités |
| `mikata/llm/prompts.py` | **Réécrit** | Nouveau contrat (texte, direct, script) |
| `mikata/agent/guard.py` | **Gardé**, placé derrière les politiques | Critère de la phase 5 |
| `mikata/llm/client.py`, `usage.py`, `agent/memory.py` | **Gardés** | Sains ; ajout des deux rôles de modèle |
| `mikata/tools/` (56 outils, 3 968 lignes) | **Gardés**, enrichis (effet, résultat structuré, durée) | Implémentation testée |
| `mikata/browser/`, `uia/`, `files/` (3 041 lignes) | **À auditer** en phase A | Écrits en partie pendant la période fragile ; pas encore relus |
| Audio, mot d'activation, STT, TTS, interface | **Inchangés** | Hors périmètre |

Les fichiers supprimés partent par `git rm` sur une branche ; l'historique les garde.

## 5. Phases et critères

Une phase par session, sur sa branche, validée sur le PC avant la suivante.

| Phase | Contenu | Critère d'acceptation |
| --- | --- | --- |
| **A. Socle** | Suppression de la v2, CI verte, audit de `browser/` `uia/` `files/`, banc d'essai, mesure de la boucle actuelle | CI verte sur les deux OS ; 30 tâches réelles rejouables ; chiffres de référence publiés dans le README |
| **B. Espace d'action** | Effets et résultats structurés sur les outils, référence du SDK générée, observations filtrées | Chaque outil a un effet déclaré ; aucun test existant cassé ; tokens de contexte mesurés avant et après |
| **C. Interpréteur et politiques** | Sous-ensemble Python, budget, provenance, pause ; extraction en quarantaine ; suite de pages piégées | Boucle infinie arrêtée en moins d'1 s ; zéro action conséquente sur la suite d'injection ; tests de la phase 5 verts |
| **D. Boucle unique** | Texte, direct, script, exploration bornée ; deux rôles de modèle ; tâches de fond et accusés | Budgets de la section 3.8 tenus sur le banc ; le chemin direct ne ralentit pas |
| **E. Apprentissage** | Compétences (paquets, induction vérifiée, rejeu local, critique isolé, cycle de vie) et carnet de leçons | Une demande voisine d'une tâche apprise passe en rejeu local en moins d'1 s ; une compétence cassée est rétrogradée ; le banc ne régresse pas après apprentissage |
| **F. Ouverture** | Client MCP, import et export `SKILL.md`, primitives nouvelles validées, vision en secours | Un outil externe utilisable depuis un script ; une compétence importée reste inactive avant relecture |

Cibles proposées, à confirmer une fois la référence de la phase A connue : les budgets de la
section 3.8, trois appels LLM au plus en médiane sur les tâches multi-étapes, et 80 % de
réussite sur trois essais.

## 6. Décisions prises (13/10/2026)

1. **Interpréteur maison**, avec budget d'exécution et provenance. Monty reste un moteur de
   rechange possible.
2. **Autonomie réglable** : une option « accès complet » dans les paramètres
   (`guard.full_access`). Désactivée, MIKATA demande un « oui » avant une action conséquente
   quand le tour a lu du contenu non fiable ; activée, il agit sans demander. Dans les deux
   cas, les actions destructives restent interdites et une commande shell construite à
   partir d'un contenu non fiable est refusée.
3. **Deux rôles de modèle**, `llm.fast` et `llm.strong`.
4. **`mikata/tools/` reste l'implémentation**, enrichie d'un effet déclaré, d'un niveau de
   confiance de la sortie et d'un résultat structuré.

## 7. Limites

- La généralité vient d'abord du modèle : avec un modèle rapide sans réflexion, les tâches
  inédites resteront plus fragiles qu'avec un modèle de pointe.
- Les cibles de la section 5 sont des propositions, pas des mesures.
- La lecture des papiers s'appuie sur leurs résumés et leurs sections principales ; les
  chiffres cités sont ceux des auteurs, sur leurs bancs, pas sur MIKATA.
- La vision (applis sans accessibilité) est repoussée en phase F.

## 8. Références

- CodeAct, *Executable Code Actions Elicit Better LLM Agents* : <https://arxiv.org/pdf/2402.01030>
- CoAct-1 : <https://arxiv.org/html/2508.03923v3>
- OSWorld-Human : <https://arxiv.org/abs/2506.16042>
- *Efficient GUI Agents: A Systems Survey* (sept. 2026) : <https://arxiv.org/pdf/2609.02309>
- UFO², *The Desktop AgentOS* : <https://arxiv.org/html/2504.14603>
- ASI, *Inducing Programmatic Skills for Agentic Tasks* : <https://arxiv.org/html/2504.06821v2>
- Agent Workflow Memory : <https://proceedings.mlr.press/v267/wang25bx.html>
- EvoSkill-GUI, *Reflect, Revise, Reuse* (sept. 2026) : <https://arxiv.org/html/2609.17653>
- CaMeL, *Defeating Prompt Injections by Design* : <https://arxiv.org/html/2503.18813>
- *Design Patterns for Securing LLM Agents against Prompt Injections* : <https://arxiv.org/pdf/2506.08837>
- ACE, *Agentic Context Engineering* : <https://arxiv.org/html/2510.04618v2>
- Voyager : <https://arxiv.org/abs/2305.16291>
- Alita : <https://arxiv.org/abs/2505.20286> ; Alita-G : <https://arxiv.org/html/2510.23601v1>
- Darwin Gödel Machine : <https://arxiv.org/html/2505.22954v3>
- *Your Agent May Misevolve* : <https://arxiv.org/html/2509.26354>
- SEAgent : <https://arxiv.org/html/2508.04700v1> ; OSExpert : <https://arxiv.org/abs/2603.07978>
- *Speculative Actions* : <https://arxiv.org/pdf/2510.04371>
- Talker-Reasoner : <https://arxiv.org/pdf/2410.08328>
- Full-Duplex-Bench-v3 : <https://arxiv.org/pdf/2604.04847>
- Anthropic, *Building effective agents* : <https://www.anthropic.com/engineering/building-effective-agents>
- Anthropic, *Advanced tool use* : <https://www.anthropic.com/engineering/advanced-tool-use>
- Anthropic, *Writing effective tools for agents* : <https://www.anthropic.com/engineering/writing-tools-for-agents>
- Anthropic, *Effective context engineering for AI agents* : <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
- Anthropic, *Demystifying evals for AI agents* : <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>
- Pydantic Monty : <https://pydantic.dev/articles/pydantic-monty>
- Agent Skills, standard ouvert : <https://agentskills.io>
- Sécurité d'OpenClaw (compétences malveillantes) : <https://www.bitdoze.com/openclaw-security-guide/>
- Mémoire des agents en 2026, scores annoncés : <https://mem0.ai/blog/state-of-ai-agent-memory-2026> ;
  comparaison des cadres : <https://atlan.com/know/best-ai-agent-memory-frameworks-2026/>
