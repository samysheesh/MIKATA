"""Compétences (docs/AGENT_CORE.md §3.6) : des programmes vérifiés, réutilisés et révisés.

Une compétence est un **paquet lisible** dans `data/skills/<nom>/`, au format des Agent
Skills (standard ouvert) :

- `SKILL.md` : en-tête YAML (`name`, `description`, puis l'état propre à MIKATA sous
  `metadata`) et une description pour un humain ;
- `skill.py` : `def run(<paramètres>):` dont le corps est un programme de l'interpréteur
  (`mikata.action.interp`), exécuté sous les mêmes politiques que les scripts du modèle.

Cycle de vie (décidé avec l'utilisateur le 14/10) :

1. **Induction** : après un tour réussi dont un programme a enchaîné au moins deux appels,
   un appel LLM en arrière-plan en tire une compétence paramétrée (motifs de phrases avec
   `{paramètres}`). Jamais d'un programme écrit après la lecture d'un contenu non fiable.
2. **Vérification** (code, sans LLM) : la phrase d'origine doit coller à un motif et en
   redonner les valeurs ; le programme n'appelle que des fonctions que l'original appelait ;
   chaque paramètre sert. Sinon, rien n'est gardé.
3. **À l'essai** : quand une demande colle, la compétence est proposée au modèle, qui
   l'appelle (`skills.<nom>(...)`) dans un programme ; chaque réussite ainsi surveillée
   compte. Une correction de l'utilisateur juste après compte comme un échec.
4. **Active** après `skills.promote_after` réussites (3) : une phrase qui colle à un motif
   la rejoue **en local, sans appel LLM**. Un échec la renvoie à l'essai, compteur à zéro ;
   après `skills.max_failures` échecs de suite, elle est désactivée.
5. **Révision** (phase 16b) : quand une compétence échoue et que le modèle réussit la
   demande juste après, un critique isolé (il ne voit que la demande, le programme, l'erreur
   et ce qui a marché) écrit une nouvelle version du programme, vérifiée dans le code ;
   l'ancienne est archivée (`versions/`), les derniers cas d'échec gardés
   (`failures.json`). La nouvelle version repart à l'essai.
6. **Aiguillage** (phase 15j) : pour toute autre formulation, dans toute langue, un appel
   minuscule part en même temps que l'appel normal et reconnaît la compétence active
   demandée, avec ses valeurs (vérifiées : chacune doit figurer dans la phrase). S'il
   répond avant que MIKATA ait commencé à parler, l'appel normal est coupé et la
   compétence s'exécute ; sinon il est ignoré. Aucune attente ajoutée.

Les compétences de la phase 12c (`data/skills/skills.json`) ne sont pas reprises.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from mikata.llm.base import Done, LLMClient, TextDelta, Usage

SKILLS_DIR = "data/skills"
FAILURES_FILE = "failures.json"  # phase 16b : derniers cas d'échec, pour la révision
MAX_FAILURE_CASES = 3
STATUSES = ("trial", "active", "disabled", "proposed")
STATUS_LABELS = {
    "trial": "à l'essai",
    "active": "active",
    "disabled": "désactivée",
    "proposed": "proposée",
}
RESERVED = {"list", "remember", "forget"}  # fonctions skills.* du SDK
NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+){0,5}$")
PARAM_RE = re.compile(r"\{([a-z_][a-z0-9_]{0,30})\}")
MAX_PARAMS = 4
MAX_PATTERNS = 4
MIN_CALLS = 2  # une action seule passe par l'appel direct, pas par une compétence
RELATED_MIN_WORDS = 2
_STOP = {
    "les", "des", "une", "est", "que", "qui", "pour", "dans", "sur", "avec", "mon", "mes",
    "ton", "tes", "son", "ses", "moi", "toi", "the", "and", "pas", "plus", "tout", "tous",
}  # fmt: skip


# -- texte ----------------------------------------------------------------------------------


def fold(text: str) -> str:
    """Minuscules sans accents ni ponctuation, **même longueur** que `text` : une position
    dans le texte replié est la même dans l'original (les valeurs sont prises dans
    l'original, avec leurs majuscules et accents)."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFD", ch.casefold())[:1] or " "
        out.append(base if base.isalnum() else " ")
    return "".join(out)


def words(text: str) -> set[str]:
    return {w for w in fold(text).split() if len(w) >= 3 and w not in _STOP}


# Phase 15i : la politesse et les mots d'appel autour de la demande ne comptent pas
# (« est-ce que tu peux me lancer ... s'il te plaît ? »). Formes repliées (voir `fold`).
_LEADING = (
    "est ce que tu peux", "est ce que tu pourrais", "est ce que vous pouvez",
    "est ce que vous pourriez", "est ce que", "tu peux", "tu pourrais", "peux tu",
    "pourrais tu", "vous pouvez", "pouvez vous", "tu veux bien", "s il te plait",
    "s il vous plait", "stp", "svp", "vas y", "allez", "alors", "bon", "ok", "okay", "euh",
    "dis moi", "dis", "hey", "please", "can you", "could you", "me", "m", "moi",
)  # fmt: skip
_TRAILING = (
    "s il te plait", "s il vous plait", "stp", "svp", "steuplait", "merci", "please",
    "pour moi", "maintenant", "tout de suite",
)  # fmt: skip
_LEADING_RE = re.compile(r"^\s*(?:" + "|".join(map(re.escape, _LEADING)) + r")\b")
_TRAILING_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _TRAILING)) + r")\s*$")
_CLITIC = r"(?:\s*\b(?:moi|me|m|nous)\b)?"  # « lance-moi », « mets-moi »


def _blank(text: str, regex: re.Pattern[str]) -> str:
    match = regex.search(text)
    if match is None or match.end() == match.start():
        return text
    return text[: match.start()] + " " * (match.end() - match.start()) + text[match.end() :]


def strip_politeness(folded: str) -> str:
    """Efface (par des blancs, même longueur) la politesse en tête et en fin de phrase."""
    for regex in (_LEADING_RE, _TRAILING_RE):
        while True:
            blanked = _blank(folded, regex)
            if blanked == folded:
                break
            folded = blanked
    return folded


def _verb(word: str) -> str:
    """Premier mot d'un motif, le plus souvent un verbe à l'impératif : toutes ses formes
    courantes (« lance » : lancer, lancez ; « mets » : mettre, mettez), suivi ou non de
    « moi »."""
    stem = word
    while len(stem) > 3 and stem[-1] in "eszrt" and len(word) - len(stem) < 2:
        stem = stem[:-1]
    if len(word) < 4:
        return re.escape(word) + _CLITIC
    return re.escape(stem) + r"\w{0,4}" + _CLITIC


def compile_pattern(pattern: str) -> re.Pattern[str] | None:
    """« mets l'opening {numero} de {anime} » -> expression sur le texte replié. Les mots
    sont séparés par n'importe quels blancs ou ponctuations ; le premier mot accepte ses
    autres formes (phase 15i) ; None si le motif est vide."""
    parts: list[str] = []
    position = 0
    for match in PARAM_RE.finditer(pattern):
        parts.extend(_literal(pattern[position : match.start()]))
        parts.append(f"(?P<{match.group(1)}>.+?)")
        position = match.end()
    parts.extend(_literal(pattern[position:]))
    if not parts:
        return None
    if not parts[0].startswith("(?P<"):
        parts[0] = _verb(fold(pattern).split()[0])
    body = r"\s*".join(parts)
    try:
        return re.compile(rf"^\s*{body}\s*$")
    except re.error:  # paramètre répété, etc.
        return None


def _literal(text: str) -> list[str]:
    return [re.escape(word) for word in fold(text).split()]


# Phase 15k : une valeur qui contient une seconde demande (« lance l'opening de Bleach et
# monte le son ») n'en est pas une : la phrase va au modèle, qui fera les deux.
_SECOND_ACTION = re.compile(r"\b(?:et|puis|ensuite|apres|and|then|also|aussi)\b")


def bind(pattern: str, command: str) -> dict[str, str] | None:
    """Valeurs des paramètres si `command` colle au motif, sinon None."""
    regex = compile_pattern(pattern)
    if regex is None:
        return None
    match = regex.match(strip_politeness(fold(command)))
    if match is None:
        return None
    values = {
        name: command[match.start(name) : match.end(name)].strip(" ,.;:!?'\"«»")
        for name in regex.groupindex
    }
    if any(not value or _SECOND_ACTION.search(fold(value)) for value in values.values()):
        return None  # vide, ou « Bleach et monte le son » : une seconde demande, pas une valeur
    return values


# -- programmes -----------------------------------------------------------------------------


def called_functions(code: str) -> list[str]:
    """Noms des fonctions appelées par un programme (« browser.open », « say »...), dans
    l'ordre du texte. [] si le programme ne se lit pas."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        try:  # un programme de l'interpréteur a le droit à `return` hors fonction
            tree = ast.parse("def _():\n" + textwrap.indent(textwrap.dedent(code), "    "))
        except SyntaxError:
            return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                names.append(name)
    return names


def action_calls(code: str, actions: set[str] | None = None) -> list[str]:
    """Les appels qui agissent ou lisent (fonctions du SDK et outils), sans les fonctions
    de l'interpréteur (`len`, `int`, `print`...) ni les méthodes des valeurs (`page.get`).
    `actions` : noms connus du SDK et des outils ; sans eux, tout ce qui n'est pas une
    fonction de l'interpréteur compte."""
    from mikata.action.interp import BUILTINS, INTERP_FUNCTIONS

    ignored = set(BUILTINS) | set(INTERP_FUNCTIONS) | {"say", "ask", "extract", "pick"}
    calls = called_functions(code)
    if actions is not None:
        return [name for name in calls if name in actions or name.startswith("skills.")]
    return [name for name in calls if name not in ignored]


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    return ""


def _names_used(code: str) -> set[str]:
    try:
        tree = ast.parse("def _():\n" + textwrap.indent(textwrap.dedent(code), "    "))
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def render_program(description: str, params: list[str], code: str) -> str:
    """Contenu de `skill.py` : un vrai fichier Python, lisible et vérifiable par un humain."""
    doc = description.replace('"""', "'''")
    body = textwrap.indent(textwrap.dedent(code).strip() or "return None", "    ")
    return f'"""{doc}"""\n\n\ndef run({", ".join(params)}):\n{body}\n'


def parse_program(source: str) -> tuple[list[str], str]:
    """(paramètres, corps) d'un `skill.py`. Lève ValueError si le fichier est illisible."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"skill.py illisible : {exc}") from exc
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run is None or not run.body:
        raise ValueError("skill.py sans fonction run")
    params = [arg.arg for arg in run.args.args]
    body = run.body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and len(body) > 1:
        body = body[1:]  # docstring
    lines = source.splitlines()
    start = body[0].lineno - 1
    end = body[-1].end_lineno or len(lines)
    return params, textwrap.dedent("\n".join(lines[start:end]))


def call_code(skill: Skill, values: dict[str, str]) -> str:
    """Le programme d'une ligne qui appelle la compétence (rejeu local)."""
    args = ", ".join(f"{name}={values[name]!r}" for name in skill.params)
    return f"return skills.{skill.call_name}({args})"


# -- compétence -----------------------------------------------------------------------------


@dataclass
class Skill:
    name: str  # « lancer-opening » (dossier, en-tête) ; appelée skills.lancer_opening
    description: str
    patterns: list[str]
    params: list[str]
    code: str  # corps de run(), dans le sous-ensemble de l'interpréteur
    status: str = "trial"
    version: int = 1
    successes: int = 0  # réussites depuis la dernière promotion ou le dernier échec
    failures: int = 0  # échecs de suite
    uses: int = 0
    source: str = ""  # demande d'origine
    created: str = ""
    last_error: str = ""
    path: Path | None = field(default=None, compare=False)

    @property
    def call_name(self) -> str:
        return self.name.replace("-", "_")

    @property
    def signature(self) -> str:
        return f"skills.{self.call_name}({', '.join(self.params)})"

    def match(self, command: str) -> dict[str, str] | None:
        for pattern in self.patterns:
            values = bind(pattern, command)
            if values is not None and set(values) == set(self.params):
                return values
        return None

    def to_markdown(self) -> str:
        meta = {
            "mikata-status": self.status,
            "mikata-version": str(self.version),
            "mikata-params": ", ".join(self.params),
            "mikata-patterns": " | ".join(self.patterns),
            "mikata-successes": str(self.successes),
            "mikata-failures": str(self.failures),
            "mikata-uses": str(self.uses),
            "mikata-source": self.source,
            "mikata-created": self.created,
        }
        if self.last_error:
            meta["mikata-last-error"] = self.last_error[:300]
        head = yaml.safe_dump(
            {"name": self.name, "description": self.description, "metadata": meta},
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        )
        examples = "\n".join(f"- « {p} »" for p in self.patterns)
        return (
            f"---\n{head}---\n\n# {self.name}\n\n{self.description}\n\n"
            f"Phrases reconnues :\n\n{examples}\n\n"
            f"Programme : `skill.py` (appelé `{self.signature}`). Appris de « {self.source} ».\n"
        )


def _read_skill(folder: Path) -> Skill:
    text = (folder / "SKILL.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("SKILL.md sans en-tête")
    _, head, _ = text.split("---", 2)
    data = yaml.safe_load(head) or {}
    meta = {str(k): str(v) for k, v in (data.get("metadata") or {}).items()}
    params, code = parse_program((folder / "skill.py").read_text(encoding="utf-8"))
    name = str(data.get("name", ""))
    if not NAME_RE.match(name) or name != folder.name:
        raise ValueError(f"nom invalide : {name!r}")
    status = meta.get("mikata-status", "proposed")
    return Skill(
        name=name,
        description=str(data.get("description", "")),
        patterns=[p.strip() for p in meta.get("mikata-patterns", "").split("|") if p.strip()],
        params=params,
        code=code,
        status=status if status in STATUSES else "proposed",
        version=int(meta.get("mikata-version", "1") or 1),
        successes=int(meta.get("mikata-successes", "0") or 0),
        failures=int(meta.get("mikata-failures", "0") or 0),
        uses=int(meta.get("mikata-uses", "0") or 0),
        source=meta.get("mikata-source", ""),
        created=meta.get("mikata-created", ""),
        last_error=meta.get("mikata-last-error", ""),
        path=folder,
    )


class SkillStore:
    """Les compétences de `data/skills/`, lues au démarrage, réécrites une à une."""

    def __init__(
        self,
        root: Path,
        *,
        promote_after: int = 3,
        max_failures: int = 3,
        max_skills: int = 40,
    ) -> None:
        self.root = root
        self.promote_after = promote_after
        self.max_failures = max_failures
        self.max_skills = max_skills
        self.skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            return
        if (self.root / "skills.json").is_file():
            logger.info("Savoir-faire de la phase 12c (skills.json) : ancien format, ignoré.")
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not (folder / "SKILL.md").is_file():
                continue
            try:
                skill = _read_skill(folder)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                logger.warning("Compétence {} illisible, ignorée : {}", folder.name, exc)
                continue
            self.skills[skill.call_name] = skill

    def all(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda s: s.name)

    def usable(self) -> list[Skill]:
        """Celles que MIKATA peut appeler : à l'essai ou actives (pas les proposées, à
        relire, ni les désactivées)."""
        return [s for s in self.all() if s.status in ("trial", "active")]

    def get(self, call_name: str) -> Skill | None:
        return self.skills.get(call_name)

    def save(self, skill: Skill) -> None:
        folder = self.root / skill.name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "skill.py").write_text(
            render_program(skill.description, skill.params, skill.code), encoding="utf-8"
        )
        (folder / "SKILL.md").write_text(skill.to_markdown(), encoding="utf-8")
        skill.path = folder
        self.skills[skill.call_name] = skill

    # -- recherche ------------------------------------------------------------------------

    def match(self, command: str) -> tuple[Skill, dict[str, str]] | None:
        """La compétence utilisable dont un motif colle à la phrase entière, avec ses
        valeurs ; les actives d'abord."""
        for skill in sorted(self.usable(), key=lambda s: s.status != "active"):
            values = skill.match(command)
            if values is not None:
                return skill, values
        return None

    def related(self, command: str, limit: int = 3) -> list[Skill]:
        """Compétences proches (mots communs avec la description et les motifs)."""
        wanted = words(command)
        scored = []
        for skill in self.usable():
            common = wanted & words(" ".join([skill.description, *skill.patterns]))
            if len(common) >= RELATED_MIN_WORDS:
                scored.append((len(common), skill.name, skill))
        return [skill for _, _, skill in sorted(scored, reverse=True)[:limit]]

    # -- cycle de vie ---------------------------------------------------------------------

    def record(self, call_name: str, ok: bool, error: str = "", *, local: bool = False) -> None:
        """Bilan d'une utilisation (surveillée par le modèle, ou rejeu local)."""
        skill = self.skills.get(call_name)
        if skill is None:
            return
        skill.uses += 1
        if ok:
            skill.successes += 1
            skill.failures = 0
            if skill.status == "trial" and skill.successes >= self.promote_after:
                skill.status = "active"
                logger.info(
                    "Compétence « {} » active après {} réussites : rejeu local sans LLM.",
                    skill.name,
                    skill.successes,
                )
        else:
            self._fail(skill, error, local=local)
        self.save(skill)

    def correct(self, call_names: list[str]) -> None:
        """L'utilisateur a corrigé le tour qui les a utilisées : un échec chacune."""
        for name in call_names:
            skill = self.skills.get(name)
            if skill is not None:
                self._fail(skill, "corrigée par l'utilisateur", local=False)
                self.save(skill)

    def _fail(self, skill: Skill, error: str, *, local: bool) -> None:
        skill.successes = 0
        skill.failures += 1
        skill.last_error = error
        if skill.status == "active":
            skill.status = "trial"
            logger.info("Compétence « {} » renvoyée à l'essai : {}", skill.name, error)
        if skill.failures >= self.max_failures:
            skill.status = "disabled"
            logger.info("Compétence « {} » désactivée ({} échecs).", skill.name, skill.failures)

    def note_failure(self, call_name: str, request: str, error: str) -> None:
        """Garde le cas d'échec (les `MAX_FAILURE_CASES` derniers) pour la révision."""
        skill = self.skills.get(call_name)
        if skill is None or skill.path is None:
            return
        cases = self.failure_cases(call_name)
        cases.append(
            {
                "le": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "version": skill.version,
                "demande": request,
                "erreur": error[:300],
            }
        )
        (skill.path / FAILURES_FILE).write_text(
            json.dumps(cases[-MAX_FAILURE_CASES:], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def failure_cases(self, call_name: str) -> list[dict[str, Any]]:
        skill = self.skills.get(call_name)
        if skill is None or skill.path is None:
            return []
        try:
            data = json.loads((skill.path / FAILURES_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def revise(self, skill: Skill, code: str, diagnostic: str) -> Skill:
        """Nouvelle version du programme : l'ancienne est archivée (`versions/vN.py`), la
        compétence repart à l'essai, compteurs à zéro."""
        folder = skill.path or self.root / skill.name
        versions = folder / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        (versions / f"v{skill.version}.py").write_text(
            render_program(skill.description, skill.params, skill.code), encoding="utf-8"
        )
        skill.code = code
        skill.version += 1
        skill.status = "trial"
        skill.successes = skill.failures = 0
        skill.last_error = ""
        self.save(skill)
        logger.info(
            "Compétence « {} » révisée (version {}) : {}", skill.name, skill.version, diagnostic
        )
        return skill

    def adopt(self, skill: Skill) -> Skill:
        """Range une compétence vérifiée (nom unique), à l'essai."""
        base, index = skill.name, 2
        while skill.call_name in self.skills or skill.call_name in RESERVED:
            skill.name = f"{base}-{index}"
            index += 1
        if len(self.skills) >= self.max_skills:
            raise ValueError(f"déjà {self.max_skills} compétences (en oublier une)")
        skill.status = "trial"
        skill.created = skill.created or time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save(skill)
        logger.info("Compétence « {} » apprise, à l'essai : {}", skill.name, skill.signature)
        return skill

    def forget(self, query: str) -> Skill | None:
        """Met à la corbeille la compétence la plus proche de `query` (nom ou description)."""
        from send2trash import send2trash

        wanted = words(query)
        best: Skill | None = None
        best_score = 0
        for skill in self.all():
            if fold(query).strip() in (fold(skill.name), fold(skill.call_name)):
                best = skill
                break
            score = len(wanted & words(" ".join([skill.name, skill.description, *skill.patterns])))
            if score > best_score:
                best, best_score = skill, score
        if best is None:
            return None
        if best.path is not None and best.path.exists():
            send2trash(str(best.path))
        self.skills.pop(best.call_name, None)
        logger.info("Compétence « {} » oubliée (corbeille).", best.name)
        return best


# -- induction et vérification -------------------------------------------------------------

INDUCE_SYSTEM = """\
Tu aides MIKATA, un assistant vocal qui agit sur un PC Windows, à apprendre des \
compétences. On te donne une demande de l'utilisateur et le programme qui l'a réussie. \
Écris la version réutilisable de ce programme, pour toutes les demandes du même genre.

Règles :
- Les valeurs qui viennent de la demande (un titre, un nom, un nombre, une ville) \
deviennent des paramètres ; tout le reste du programme est gardé tel quel.
- Un paramètre est un mot ou un groupe de mots dit dans la demande, et le programme \
l'utilise tel quel. Si le programme a besoin d'une autre forme (« texte » devient « .txt »), \
cette partie reste fixe : ce n'est pas un paramètre.
- Une valeur écrite ... a été choisie en lisant le résultat d'un appel précédent (une \
page, une recherche) : le programme doit la retrouver lui-même dans ce qu'il lit. Pour \
choisir parmi une liste (les liens d'une page, des fichiers) : choix = pick(liste, \
"question") rend l'élément tel quel (page = browser_read() ; choix = pick(page.links, \
"Quel lien est le plus probablement ... ?") ; browser_click(choix)). Pour une autre \
information : extract(contenu, {"champ": str}, "question").
- Au plus 4 paramètres, en minuscules_avec_soulignés.
- "motifs" : 2 à 4 façons courantes de formuler ce genre de demande à l'oral, en \
minuscules, chaque paramètre écrit {nom} ; la demande d'origine doit coller à l'un d'eux \
mot pour mot, hormis les valeurs.
- "nom" : 2 à 4 mots en minuscules reliés par des tirets (ex. lancer-video-youtube).
- "description" : une phrase qui dit ce que fait la compétence.
- "exemple" : la valeur de chaque paramètre dans la demande d'origine, telle qu'elle y est.
- "programme" : le corps du programme, qui utilise les paramètres comme des variables \
déjà définies, avec les mêmes fonctions que l'original (plus pick ou extract si besoin) \
et rien de plus.

Réponds UNIQUEMENT par un objet JSON : {"nom": "...", "description": "...", \
"motifs": ["..."], "parametres": ["..."], "exemple": {"...": "..."}, "programme": "..."}. \
Si ce programme n'a rien de réutilisable, réponds {"nom": ""}."""


async def induce(
    llm: LLMClient, command: str, code: str, effort: str | None = "low"
) -> tuple[dict[str, Any], Usage | None]:
    """Candidat (non vérifié) tiré d'un programme réussi ; {} s'il n'y a rien à garder."""
    messages = [
        {"role": "system", "content": INDUCE_SYSTEM},
        {"role": "user", "content": f"Demande : « {command} »\n\nProgramme :\n{code}"},
    ]
    chunks: list[str] = []
    usage: Usage | None = None
    async for event in llm.stream(messages, [], reasoning_effort=effort):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, Done):
            usage = event.usage
    return parse_candidate("".join(chunks)), usage


def parse_candidate(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) and data.get("nom") else {}


def check_program(
    code: str, params: list[str], original_code: str, actions: set[str] | None = None
) -> str:
    """Raison du refus d'un programme de compétence, ou "" s'il tient debout : au moins
    deux actions, aucune absente du programme d'origine, pas d'appel de compétence, chaque
    paramètre utilisé."""
    calls = action_calls(code, actions)
    if len(calls) < MIN_CALLS:
        return "moins de deux appels : une action simple n'a pas besoin de compétence"
    extra = sorted(set(calls) - set(action_calls(original_code, actions)))
    if extra:
        return f"fonctions absentes du programme d'origine : {', '.join(extra)}"
    if any(call.startswith("skills.") for call in calls):
        return "une compétence n'en appelle pas d'autre"
    unused = [p for p in params if p not in _names_used(code)]
    if unused:
        return f"paramètres inutilisés : {', '.join(unused)}"
    return ""


def verify(
    candidate: dict[str, Any],
    command: str,
    original_code: str,
    actions: set[str] | None = None,
) -> tuple[Skill | None, str]:
    """(compétence, "") si le candidat tient debout, sinon (None, raison). Tout se vérifie
    dans le code : rien de ce que le modèle affirme n'est cru sur parole. `actions` : noms
    du SDK et des outils (voir `action_calls`)."""
    name = str(candidate.get("nom", "")).strip().lower()
    if not NAME_RE.match(name) or name.replace("-", "_") in RESERVED:
        return None, f"nom invalide : {name!r}"
    params = [str(p) for p in candidate.get("parametres") or []]
    if len(params) > MAX_PARAMS or len(set(params)) != len(params):
        return None, "paramètres en trop ou répétés"
    if any(not re.fullmatch(r"[a-z_][a-z0-9_]{0,30}", p) for p in params):
        return None, "nom de paramètre invalide"
    patterns = [str(p).strip() for p in candidate.get("motifs") or [] if str(p).strip()]
    patterns = patterns[:MAX_PATTERNS]
    if not patterns:
        return None, "aucun motif"
    for pattern in patterns:
        if set(PARAM_RE.findall(pattern)) != set(params):
            return None, f"motif sans tous les paramètres : « {pattern} »"
        if len(_literal(PARAM_RE.sub(" ", pattern))) < 2:
            return None, f"motif trop vague : « {pattern} »"
    code = textwrap.dedent(str(candidate.get("programme", ""))).strip()
    reason = check_program(code, params, original_code, actions)
    if reason:
        return None, reason
    skill = Skill(
        name=name,
        description=str(candidate.get("description", "")).strip()[:300] or name,
        patterns=patterns,
        params=params,
        code=code,
        source=command,
    )
    values = skill.match(command)
    if values is None:
        return None, "la demande d'origine ne colle à aucun motif"
    example = {str(k): str(v) for k, v in (candidate.get("exemple") or {}).items()}
    for param in params:
        if fold(values[param]).split() != fold(example.get(param, "")).split():
            heard = example.get(param)
            return None, f"{param} : « {values[param]} » dans la demande, pas « {heard} »"
    return skill, ""


# -- révision (phase 16b) --------------------------------------------------------------------

REVISE_SYSTEM = """\
Tu répares une compétence de MIKATA, un assistant vocal qui agit sur un PC Windows. Une \
compétence est un programme réutilisable. Celle-ci vient d'échouer, puis la demande a \
réussi autrement, juste après. On te donne ce que doit faire la compétence, ses \
paramètres, son programme, l'erreur, les échecs précédents et le programme qui a réussi.

Écris le programme corrigé :
- il fait ce que fait le programme qui a réussi, mais en général : les valeurs propres à \
cette demande viennent des paramètres, utilisés comme des variables déjà définies ;
- mêmes fonctions que les programmes donnés, rien de plus ; pour choisir parmi une liste \
(les liens d'une page) : pick(liste, "question") ; pour une autre information : \
extract(contenu, {"champ": str}, "question") ;
- une valeur écrite ... a été choisie en lisant une page : le programme doit la retrouver \
lui-même, jamais la recopier.

Réponds UNIQUEMENT par un objet JSON : {"diagnostic": "une phrase sur la cause", \
"programme": "..."}. Si tu ne sais pas corriger, réponds {"programme": ""}."""


async def revise(
    llm: LLMClient,
    skill: Skill,
    request: str,
    error: str,
    recovery: str,
    calling_code: str = "",
    failures: list[dict[str, Any]] | None = None,
    effort: str | None = "low",
) -> tuple[dict[str, Any], Usage | None]:
    """Un appel LLM isolé : le programme corrigé et son diagnostic ({} s'il n'y en a pas)."""
    previous = "\n".join(
        f"- « {case.get('demande', '')} » : {case.get('erreur', '')}" for case in failures or []
    )
    parts = [
        f"Compétence : {skill.signature} — {skill.description}",
        f"Programme actuel :\n{skill.code}",
        f"Demande : « {request} »",
    ]
    if calling_code:
        parts.append(f"Appel de la compétence dans cette demande :\n{calling_code}")
    parts += [f"Erreur : {error}", f"Échecs précédents :\n{previous or '(aucun)'}"]
    parts.append(f"Programme qui a réussi juste après :\n{recovery}")
    messages = [
        {"role": "system", "content": REVISE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    chunks: list[str] = []
    usage: Usage | None = None
    async for event in llm.stream(messages, [], reasoning_effort=effort):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, Done):
            usage = event.usage
    text = "".join(chunks)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}, usage
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}, usage
    if not isinstance(data, dict) or not str(data.get("programme", "")).strip():
        return {}, usage
    return data, usage


def verify_revision(
    candidate: dict[str, Any], skill: Skill, recovery: str, actions: set[str] | None = None
) -> tuple[str, str]:
    """(programme, "") si la révision tient debout, sinon ("", raison). Ses fonctions
    doivent venir du programme qui a réussi ou de l'ancienne version."""
    code = textwrap.dedent(str(candidate.get("programme", ""))).strip()
    if not code:
        return "", "programme vide"
    reason = check_program(code, skill.params, f"{recovery}\n{skill.code}", actions)
    return ("", reason) if reason else (code, "")


# -- aiguillage (phase 15j) ------------------------------------------------------------------

ROUTE_SYSTEM = """\
Tu aiguilles les demandes faites à MIKATA, un assistant vocal. Ses compétences apprises :
{skills}

La demande vient de la reconnaissance vocale, dans n'importe quelle langue et n'importe \
quelle formulation. Dis si elle demande exactement l'une de ces compétences, et lis la \
valeur de chaque paramètre dans la demande, recopiée telle qu'elle y est dite.

Réponds 0 si l'action demandée est une autre (fermer au lieu de lancer, chercher au lieu \
de jouer...), si la demande en contient plusieurs, si elle est niée ou au conditionnel \
(« ne lance pas », « je voudrais savoir si »), ou s'il manque une valeur.

Réponds UNIQUEMENT par un objet JSON : {{"competence": numéro ou 0, "valeurs": \
{{"paramètre": "valeur"}}}}."""


def route_prompt(skills: list[Skill]) -> str:
    """Prompt de l'aiguillage : ne change que quand les compétences actives changent (il
    reste dans le cache de prompt du fournisseur)."""
    lines = [f"{i}. {s.signature} : {s.description}" for i, s in enumerate(skills, 1)]
    return ROUTE_SYSTEM.format(skills="\n".join(lines))


def locate(command: str, value: str) -> str | None:
    """La valeur telle qu'elle est écrite dans la demande (casse, accents), ou None si elle
    n'y figure pas : une valeur que le modèle aurait inventée n'est jamais utilisée."""
    wanted = fold(value).split()
    if not wanted:
        return None
    regex = re.compile(r"\b" + r"\s+".join(map(re.escape, wanted)) + r"\b")
    match = regex.search(fold(command))
    if match is None:
        return None
    return command[match.start() : match.end()]


def pick_route(text: str, skills: list[Skill], command: str) -> tuple[Skill, dict[str, str]] | None:
    """Lit la réponse de l'aiguillage et la vérifie dans le code : numéro valide, tous les
    paramètres et rien d'autre, chaque valeur présente dans la demande."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    number = data.get("competence", data.get("compétence", 0))
    try:
        index = int(number)
    except (TypeError, ValueError):
        return None
    if not 1 <= index <= len(skills):
        return None
    skill = skills[index - 1]
    raw = data.get("valeurs") or {}
    if not isinstance(raw, dict) or set(raw) != set(skill.params):
        return None
    values: dict[str, str] = {}
    for name in skill.params:
        found = locate(command, str(raw[name]))
        if found is None:
            return None
        values[name] = found
    return skill, values


async def route(
    llm: LLMClient, command: str, skills: list[Skill], effort: str | None = "none"
) -> tuple[tuple[Skill, dict[str, str]] | None, Usage | None]:
    """Un appel minuscule, sans outils : la compétence active que demande `command` (et
    ses valeurs), ou None."""
    messages = [
        {"role": "system", "content": route_prompt(skills)},
        {"role": "user", "content": f"Demande : « {command} »"},
    ]
    chunks: list[str] = []
    usage: Usage | None = None
    async for event in llm.stream(messages, [], reasoning_effort=effort):
        if isinstance(event, TextDelta):
            chunks.append(event.text)
        elif isinstance(event, Done):
            usage = event.usage
    return pick_route("".join(chunks), skills, command), usage
