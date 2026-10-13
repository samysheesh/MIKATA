"""Garde-fou (§7) : appliqué par le code, jamais par le LLM.

- **Supprimer = corbeille.** `Guard.trash` est le seul chemin de suppression de MIKATA :
  `send2trash`, après des vérifications qui refusent plutôt que de risquer une suppression
  définitive (lecteur sans corbeille, élément trop gros, corbeille sûre indisponible).
- **Commandes filtrées.** `check_command` refuse les commandes de suppression ou
  d'effacement (liste noire) et classe « à confirmer » les commandes risquées (liste
  grise) : arrêter un programme, toucher au registre, aux services, installer...
  C'est un filet de sécurité, pas une garantie absolue (une commande obscurcie passe).
- **Confirmation orale** (`confirm`) : MIKATA pose la question et écoute la réponse,
  sans mot d'activation ; pas de réponse claire = non.
- **Écrasement = sauvegarde** (`backup`) dans `data/backups/`, purgée (à la corbeille)
  après `guard.backup_days` jours.
- **Extinction / redémarrage / veille** (`countdown`) : annonce, puis `guard.shutdown_cancel_s`
  secondes pour dire « stop ».
- **Contrôles d'interface** (`check_ui_action`, phase 12a) : un clic UIA sur un contrôle
  destructif (suppression définitive, formatage...) est refusé ; les contrôles sensibles
  (supprimer, désinstaller, payer) demandent un « oui » oral.
- **Journal** : chaque appel d'outil est écrit dans `data/actions.log` (`log_action`).
- Droits utilisateur normaux : toute tentative d'élévation (RunAs) est refusée.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.config import Config
from mikata.tools.base import GuardDenied, ToolError, TurnIO

BACKUP_DIR = "data/backups"
# Noms dits pour les dossiers personnels -> clé de `folders` (tools.desktop.known_folders).
FOLDER_NAMES = {
    "desktop": "desktop", "bureau": "desktop",
    "documents": "documents", "mes documents": "documents",
    "downloads": "downloads", "telechargements": "downloads", "téléchargements": "downloads",
    "pictures": "pictures", "images": "pictures",
    "music": "music", "musique": "music",
    "videos": "videos", "vidéos": "videos",
}  # fmt: skip
SIZE_SCAN_BUDGET_S = 2.0  # mesure de la taille d'un dossier avant la corbeille


class Verdict(StrEnum):
    ALLOW = "autorisé"
    CONFIRM = "à confirmer"
    DENY = "refusé"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""  # en français, lisible à voix haute (« modifier le registre »)
    match: str = ""  # extrait de la commande qui a déclenché la règle


def _rules(*pairs: tuple[str, str]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in pairs]


# Un mot de commande isolé : pas collé à une lettre, un $, un tiret, ni pris dans un chemin
# ou un nom de fichier (« rd.txt », « C:\\temp\\rd »).
_CMD = r"(?<![\w$.\-\\/:]){}(?![\w\-.\\/:])"
# Alias courts de Remove-Item : n'importe où dans une commande, mais seulement en position
# de commande dans un texte tapé au clavier (« j'ai ri » n'est pas une commande).
_ALIASES = r"(rm|ri|del|erase|rd|rmdir|rmd)"
_COMMAND_START = r"(^|[;|&\n\r({]\s*|/c\s+['\"]?)"
DELETE_HINT = "pour supprimer, utilise delete_file (corbeille)"
AFTER_WEB = "agir juste après la lecture d'une page web"  # « Cette commande va ... »

ALIAS_RULE = _rules((_CMD.format(_ALIASES), f"suppression définitive ; {DELETE_HINT}"))
TYPED_ALIAS_RULE = _rules(
    (_COMMAND_START + _ALIASES + r"(?![\w\-.\\/:])", f"suppression définitive ; {DELETE_HINT}")
)

# Liste noire (§7) : suppression ou effacement définitif, contournements du filtre.
BLACKLIST_CORE = _rules(
    (_CMD.format(r"remove-item"), f"suppression définitive interdite ; {DELETE_HINT}"),
    (r"(\.|::)\s*delete(file|directory)?\s*\(", f"suppression .NET définitive ; {DELETE_HINT}"),
    (r"(%|\bforeach(-object)?)\s+(-membername\s+)?['\"]?delete\b",
     f"suppression .NET définitive ; {DELETE_HINT}"),
    (r"\bgit\s+clean\b", "git clean supprime définitivement des fichiers"),
    (r"\bos\.(remove|unlink|rmdir|removedirs)\s*\(|\bshutil\.rmtree\s*\(",
     f"suppression définitive depuis Python ; {DELETE_HINT}"),
    (_CMD.format(r"clear-recyclebin"), "vider la corbeille est interdit"),
    (_CMD.format(r"(clear-content|clear-item|clc|cli)"), "effacement du contenu interdit"),
    (_CMD.format(r"format-volume") + r"|" + _CMD.format(r"format(\.com)?") + r"\s+\w:",
     "formatage interdit"),
    (_CMD.format(r"(clear-disk|initialize-disk|remove-partition|diskpart)"),
     "opération de disque destructive interdite"),
    (_CMD.format(r"cipher(\.exe)?") + r".*\s/w", "effacement sécurisé interdit"),
    (_CMD.format(r"sdelete(64)?(\.exe)?"), "effacement sécurisé interdit"),
    (r"vssadmin.*\bdelete|wmic.*shadowcopy.*\bdelete|wbadmin.*\bdelete",
     "suppression des sauvegardes Windows interdite"),
    (r"fsutil.*\b(setzerodata|deletejournal)", "effacement de données interdit"),
    (_CMD.format(r"robocopy(\.exe)?") + r".*\s/(mir|purge)\b",
     "robocopy /mir ou /purge supprime des fichiers"),
    (r"-verb\s+['\"]?runas|" + _CMD.format(r"runas(\.exe)?"),
     "élévation administrateur interdite (droits normaux uniquement)"),
    (r"\s-(e|ec|en|enc|enco\w*)\s+[A-Za-z0-9+/=]{16,}|frombase64string",
     "commande encodée refusée : le garde-fou doit pouvoir la lire"),
    (_CMD.format(r"(stop-computer|restart-computer|shutdown(\.exe)?)"),
     "pour éteindre ou redémarrer, utilise shutdown, restart ou sleep (délai d'annulation)"),
)  # fmt: skip
BLACKLIST = ALIAS_RULE + BLACKLIST_CORE

# Liste noire macOS : s'ajoute à la liste PowerShell (filtrée dans les deux cas, car
# run_shell et les scripts Python peuvent aussi lancer ces commandes).
DARWIN_BLACKLIST = _rules(
    (r"(^|[;|&\n\r({]\s*)(sudo\s+)?rm\s+.*\s(-rf?|--recursive|--force)\b",
     "suppression massive interdite ; pour supprimer, utilise delete_file (corbeille)"),
    (_CMD.format(r"diskutil") + r"\s+(erase[dD]isk|erase[vV]olume|partition[dD]isk|zero[dD]isk)",
     "effacement de disque interdit"),
    (r"~?/Library/Application Support/com\.apple\.TCC|tcc\.db|\bsqlite3\b.*TCC",
     "la base des autorisations macOS (TCC) est intouchable"),
    (_CMD.format(r"(nvram|csrutil|bless)"), "réglage du démarrage macOS interdit"),
)  # fmt: skip


def _blacklist_for(platform: str) -> list[tuple[re.Pattern[str], str]]:
    """Liste noire de la plateforme : PowerShell partout, plus les commandes du shell
    natif sur macOS. Les deux listes restent des constantes distinctes : modifier
    `BLACKLIST` (Windows) ne doit pas changer silencieusement la sémantique macOS."""
    if platform == "darwin":
        return BLACKLIST + DARWIN_BLACKLIST
    return BLACKLIST


# Liste grise : permis, mais seulement après un « oui » de l'utilisateur.
GRAYLIST = _rules(
    (_CMD.format(r"(stop-process|kill|spps|taskkill(\.exe)?)"), "arrêter un programme"),
    (_CMD.format(r"(set|new|remove|rename|clear)-itemproperty")
     + r"|" + _CMD.format(r"reg(\.exe)?") + r"\s+(add|delete|import|copy|restore|load|unload)\b"
     + r"|(new-item|set-item)\b.*\bhk(lm|cu|cr|u|cc):", "modifier le registre"),
    (_CMD.format(r"(stop|start|restart|set|new|suspend)-service")
     + r"|" + _CMD.format(r"sc(\.exe)?") + r"\s+(stop|start|config|create|delete)\b"
     + r"|" + _CMD.format(r"net") + r"\s+(stop|start)\b", "modifier un service Windows"),
    (_CMD.format(r"(winget|choco|msiexec)(\.exe)?") + r"|"
     + _CMD.format(r"(install|uninstall)-(package|module|script)") + r"|"
     + _CMD.format(r"(add|remove)-appxpackage"), "installer ou désinstaller un logiciel"),
    (_CMD.format(r"(set-executionpolicy|bcdedit|netsh|powercfg)")
     + r"|" + _CMD.format(r"(disable|enable|set|rename|restart)-netadapter")
     + r"|" + _CMD.format(r"(set|add|remove)-mppreference")
     + r"|" + _CMD.format(r"(new|set|remove|disable|enable)-netfirewallrule"),
     "changer un réglage du système"),
    (_CMD.format(r"((un)?register-scheduledtask|schtasks(\.exe)?)"),
     "créer ou changer une tâche planifiée"),
    (_CMD.format(r"net") + r"\s+(user|localgroup)\b|"
     + _CMD.format(r"(new|remove|set|disable|enable)-localuser|add-localgroupmember"),
     "modifier un compte utilisateur"),
    (_CMD.format(r"(takeown|icacls|cacls)(\.exe)?|set-acl"), "changer des droits d'accès"),
    (_CMD.format(r"(invoke-expression|iex)"), "exécuter du code généré ou téléchargé"),
    (_CMD.format(r"(set-content|sc|out-file|add-content|ac)")
     + r"|(?<![\d>])>(?!>|&|\s*\$null)",
     "écrire dans un fichier sans sauvegarde (write_file en fait une)"),
    (_CMD.format(r"(move-item|mv|mi|move|ren|rename-item|rni)"),
     "déplacer ou renommer des fichiers sans sauvegarde (move_file en fait une)"),
    (_CMD.format(r"(new-item|ni|copy-item|copy|cp|cpi)") + r".*\s-force\b",
     "écraser un fichier existant sans sauvegarde"),
    (r"\bgit\s+(reset\s+--hard|checkout\s+--|restore\b|stash\s+(drop|clear))",
     "effacer des modifications non enregistrées dans git"),
)  # fmt: skip

# Scripts Python de l'utilisateur : appels de suppression définitive.
PYTHON_BLACKLIST = _rules(
    (r"\bos\.(remove|unlink|rmdir|removedirs)\s*\(|\bshutil\.rmtree\s*\(|\.(unlink|rmdir)\s*\(",
     f"le script supprime définitivement des fichiers ; {DELETE_HINT}"),
)  # fmt: skip

# Contrôles d'interface (phase 12a, outils UIA) : le nom du contrôle est le seul signal
# fiable pour un clic. Refusés : destructifs ou irréversibles ; à confirmer : les autres
# actions sensibles (les noms sont en français, casse et accents ignorés).
UI_DENY = _rules(
    (r"supprimer\s+d[ée]finitivement", f"suppression définitive ; {DELETE_HINT}"),
    (r"vider\s+la\s+corbeille", "vider la corbeille est interdit"),
    (r"formater|r[ée]initialiser\s+(ce\s+)?pc|effacer\s+le\s+disque",
     "action destructive sur le système"),
)  # fmt: skip
UI_CONFIRM = _rules(
    (r"\bsupprimer\b", "supprimer quelque chose"),
    (r"d[ée]sinstaller", "désinstaller un logiciel"),
    (r"\b(acheter|payer|commander|abonner|abonnement)\b", "dépenser de l'argent"),
)  # fmt: skip

# Navigation du navigateur (phase 12b) : seuls http et https, jamais les schémas locaux
# ou exécutables. Une page peut contenir n'importe quoi : on ne laisse passer que le web.
_NAVIGATION_SCHEMES = ("http://", "https://")
DOWNLOAD_EXTENSIONS = (
    ".exe", ".msi", ".zip", ".7z", ".rar", ".dmg", ".apk", ".appx", ".iso", ".torrent",
)  # fmt: skip


def _words(text: str) -> list[str]:
    """Mots sans casse ni accents ; « d'accord » donne « d » et « accord »."""
    text = unicodedata.normalize("NFKD", text.casefold().replace("'", " ").replace("’", " "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


# Réponses orales (comparées sans accents). Dans le doute, c'est non.
YES_WORDS = frozenset(
    _words("oui ouais ouaip yes ok okay accord vas confirme confirmé confirmer go exactement "
           "absolument évidemment carrément allez fais lance")
)  # fmt: skip
NO_WORDS = frozenset(
    _words("non nan no pas annule annuler stop stoppe arrête arrêter attends attend surtout "
           "jamais laisse tombe pause")
)  # fmt: skip
CANCEL_WORDS = frozenset(
    _words("stop stoppe annule annuler arrête arrêter non nan attends attend pause")
)  # fmt: skip


def is_yes(answer: str | None) -> bool:
    """Oui clair : un mot d'accord et aucun mot de refus (« oui... non attends » = non)."""
    words = set(_words(answer or ""))
    return bool(words & YES_WORDS) and not words & NO_WORDS


def is_cancel(answer: str | None) -> bool:
    return bool(set(_words(answer or "")) & CANCEL_WORDS)


def check_patterns(text: str, rules: Iterable[tuple[re.Pattern[str], str]]) -> Decision | None:
    for pattern, reason in rules:
        match = pattern.search(text)
        if match is not None:
            return Decision(Verdict.DENY, reason, match.group(0).strip())
    return None


def _flatten(command: str) -> str:
    """Accents graves PowerShell (`r`m) et guillemets collés retirés, avant le filtre."""
    return command.replace("`", "")


def log_action(tool: str, arguments: dict[str, Any], status: str, detail: str = "") -> None:
    """Une ligne dans data/actions.log (sink dédié, cf. logging_setup)."""
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    if len(args) > 300:
        args = args[:300] + "…"
    detail = " ".join(detail.split())[:200]
    logger.bind(action=True).info("{} | {}({}) | {}", status.upper(), tool, args, detail)


class Guard:
    def __init__(
        self,
        config: Config,
        home: Path | None = None,
        trash: Callable[[str], None] | None = None,
        clock: Callable[[], datetime] = datetime.now,
        folders: dict[str, Path] | None = None,
    ) -> None:
        self.config = config
        self.cfg = config.guard
        self.home = (home or Path.home()).expanduser().resolve()
        # Vrais dossiers personnels (Windows : Bureau dans OneDrive...), sinon home/<Nom>.
        self.folders = {name: Path(path) for name, path in (folders or {}).items()}
        self.allowed = [self._absolute(Path(d)) for d in self.cfg.allowed_dirs]
        self.backup_dir = config.resolve(BACKUP_DIR)
        self._trash = trash
        self._clock = clock

    def _absolute(self, path: Path) -> Path:
        text = os.path.expandvars(str(path))
        if text == "~":
            text = ""
        elif text.startswith(("~/", "~\\")):
            text = text[2:]
        path = Path(text)
        if not path.is_absolute():
            path = self._personal_folder(text)
        return path.resolve(strict=False)

    def absolute(self, raw: str) -> Path:
        """Chemin absolu de `raw`, dossiers personnels compris (« Bureau/x » -> vrai Bureau),
        sans contrôle des dossiers autorisés : pour les outils qui ne font que chercher."""
        return self._absolute(Path(raw.strip().strip('"').strip("'")))

    def _personal_folder(self, relative: str) -> Path:
        """« Bureau/x.txt » ou « Desktop\\x.txt » -> vrai dossier Bureau (OneDrive compris)."""
        parts = [part for part in re.split(r"[\\/]+", relative) if part]
        if parts:
            key = FOLDER_NAMES.get(parts[0].casefold())
            if key is not None and key in self.folders:
                return self.folders[key].joinpath(*parts[1:])
        return self.home / relative

    # -- chemins ----------------------------------------------------------------------

    def resolve_path(self, raw: str) -> Path:
        """Chemin absolu dans un dossier autorisé (`guard.allowed_dirs`), sinon refus."""
        text = raw.strip().strip('"').strip("'")
        if not text:
            raise ToolError("chemin vide")
        path = self._absolute(Path(text))
        if not any(path == root or path.is_relative_to(root) for root in self.allowed):
            roots = ", ".join(str(root) for root in self.allowed)
            raise GuardDenied(f"chemin hors des dossiers autorisés ({roots}) : {path}")
        return path

    def check_removable(self, path: Path) -> None:
        """Refuse de mettre à la corbeille (ou de déplacer) un dossier racine autorisé, le
        dossier perso, ou le dossier de MIKATA lui-même."""
        if path in self.allowed or path == self.home or path.parent == path:
            raise GuardDenied(
                f"refusé : {path} est un dossier racine, il ne part pas à la corbeille"
            )
        project = self.config.base_dir.resolve()
        if project == path or project.is_relative_to(path):
            raise GuardDenied(f"refusé : {path} contient MIKATA lui-même")

    # -- commandes --------------------------------------------------------------------

    def check_command(self, command: str, *, after_web: bool = False) -> Decision:
        """Commande PowerShell, cmd ou shell : refusée, à confirmer, ou autorisée."""
        text = _flatten(command)
        denied = check_patterns(text, _blacklist_for(sys.platform))
        if denied is not None:
            return denied
        risky = check_patterns(text, GRAYLIST)
        if risky is not None and self.cfg.confirm_risky:
            return Decision(Verdict.CONFIRM, risky.reason, risky.match)
        if after_web and self.cfg.confirm_after_web:
            return Decision(Verdict.CONFIRM, AFTER_WEB, "")
        return Decision(Verdict.ALLOW)

    def check_script(self, path: Path) -> Decision:
        """Contenu d'un script de scripts/user/ : mêmes listes que les commandes."""
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise ToolError(f"script illisible : {exc}") from exc
        if path.suffix.lower() == ".py":
            denied = check_patterns(content, PYTHON_BLACKLIST)
            if denied is not None:
                return denied
            # Commandes lancées depuis Python (subprocess) : seulement la liste noire.
            return check_patterns(_flatten(content), _blacklist_for(sys.platform)) or Decision(
                Verdict.ALLOW
            )
        return self.check_command(content)

    def check_typed_text(self, text: str) -> Decision:
        """Texte tapé au clavier : une commande de suppression tapée dans un terminal
        contournerait le filtre (liste noire seulement ; alias courts en position de
        commande, pour laisser passer « j'ai ri »)."""
        flat = _flatten(text)
        found = check_patterns(flat, BLACKLIST_CORE) or check_patterns(flat, TYPED_ALIAS_RULE)
        return found or Decision(Verdict.ALLOW)

    @staticmethod
    def check_keys(keys: list[str]) -> Decision:
        """Maj+Suppr supprime sans passer par la corbeille : refusé."""
        if "shift" in keys and "delete" in keys:
            return Decision(Verdict.DENY, f"Maj+Suppr supprime définitivement ; {DELETE_HINT}")
        return Decision(Verdict.ALLOW)

    def check_ui_action(self, control_name: str) -> Decision:
        """Cliquer (ou remplir) un contrôle d'interface par son nom (outils UIA) : refusé,
        à confirmer, ou autorisé. Filet de sécurité, pas une garantie : un bouton sans nom
        lisible passe toujours — mais une vraie action destructive est presque toujours
        nommée, et chaque clic est journalisé dans actions.log."""
        denied = check_patterns(control_name, UI_DENY)
        if denied is not None:
            return denied
        risky = check_patterns(control_name, UI_CONFIRM)
        if risky is not None and self.cfg.confirm_risky:
            return Decision(Verdict.CONFIRM, risky.reason, risky.match)
        return Decision(Verdict.ALLOW)

    @staticmethod
    def check_navigation(url: str) -> Decision:
        """URL ouverte par le navigateur (phase 12b) : http et https seulement."""
        text = url.strip()
        scheme = text.split(":", 1)[0] + ":" if ":" in text else ""
        if text.casefold().startswith(_NAVIGATION_SCHEMES):
            return Decision(Verdict.ALLOW)
        return Decision(
            Verdict.DENY,
            "navigation refusée : seuls http:// et https:// sont ouverts, "
            f"pas « {scheme or 'rien'} »",
        )

    @staticmethod
    def check_download(url: str) -> Decision:
        """Lien cliqué dans le navigateur qui ressemble à un téléchargement : confirmé."""
        path = url.split("?", 1)[0].split("#", 1)[0].casefold()
        if path.endswith(DOWNLOAD_EXTENSIONS):
            return Decision(Verdict.CONFIRM, "télécharger un fichier")
        return Decision(Verdict.ALLOW)

    # -- corbeille --------------------------------------------------------------------

    def trash(self, path: Path) -> None:
        """Seul chemin de suppression : la corbeille, ou rien.

        Refuse (`GuardDenied`) si l'élément risquerait d'être supprimé définitivement :
        lecteur sans corbeille, taille au-delà de `guard.trash_max_mb`, ou (Windows
        seulement) module de corbeille sûre absent. Sous macOS, send2trash natif va
        au Finder : pas de porte pywin32/IFileOperation, donc pas de vérification.
        Aucune autre méthode n'est tentée en cas d'échec.
        """
        if not path.exists() and not path.is_symlink():
            raise ToolError(f"introuvable : {path}")
        self.check_removable(path)
        if sys.platform == "win32":
            _check_windows_trash(path)
        limit = self.cfg.trash_max_mb
        size_mb = _size_mb(path, limit)
        if size_mb > limit:
            raise GuardDenied(
                f"refusé : plus de {limit:.0f} Mo, la corbeille de Windows le supprimerait "
                "définitivement"
            )
        trash = self._trash or _send2trash
        try:
            trash(str(path))
        except Exception as exc:
            raise ToolError(
                f"mise à la corbeille impossible, rien n'a été supprimé : {exc}"
            ) from exc
        if path.exists():
            raise ToolError(f"{path} est toujours là après la mise à la corbeille")

    # -- sauvegardes ------------------------------------------------------------------

    def backup(self, path: Path) -> Path:
        """Copie de `path` dans data/backups/<horodatage>/ avant de l'écraser."""
        stamp = self._clock().strftime("%Y%m%d-%H%M%S-%f")
        target = self.backup_dir / stamp / path.name
        counter = 1
        while target.exists():  # jamais une sauvegarde écrasée par une autre
            target = self.backup_dir / f"{stamp}-{counter}" / path.name
            counter += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
        logger.info("Sauvegarde avant écrasement : {} -> {}", path, target)
        self.purge_backups()
        return target

    def purge_backups(self) -> int:
        """Sauvegardes de plus de `guard.backup_days` jours : à la corbeille."""
        if not self.backup_dir.is_dir():
            return 0
        limit = time.time() - self.cfg.backup_days * 86_400
        count = 0
        trash = self._trash or _send2trash
        for entry in sorted(self.backup_dir.iterdir()):
            try:
                if entry.stat().st_mtime < limit:
                    trash(str(entry))
                    count += 1
            except Exception as exc:  # une sauvegarde gardée trop longtemps n'est pas grave
                logger.warning("Sauvegarde {} non purgée : {}", entry, exc)
        if count:
            logger.info("{} ancienne(s) sauvegarde(s) mise(s) à la corbeille.", count)
        return count

    # -- dialogue ---------------------------------------------------------------------

    async def confirm(self, question: str, turn: TurnIO | None) -> bool:
        """Pose `question` à voix haute et écoute ; vrai seulement sur un « oui » clair."""
        if turn is None or turn.listen is None:
            logger.warning("Confirmation impossible sans micro : refus ({}).", question)
            return False
        answer = await turn.ask(question, self.cfg.confirm_timeout_s)
        accepted = is_yes(answer)
        logger.info("Confirmation « {} » -> « {} » : {}.", question, answer, accepted)
        return accepted

    async def countdown(
        self,
        announce: str,
        cancelled: str,
        turn: TurnIO | None,
        sleep: Callable[[float], Awaitable[Any]],
    ) -> bool:
        """« <announce> dans 5 secondes. Dis stop pour annuler. » ; faux si annulé."""
        seconds = self.cfg.shutdown_cancel_s
        if seconds <= 0:
            return True
        if turn is None:
            await sleep(seconds)
            return True
        turn.speak(f"{announce} dans {seconds:.0f} secondes. Dis stop pour annuler.")
        await turn.drain()
        if turn.listen is None:
            await sleep(seconds)
            return True
        answer = await turn.listen(seconds)
        if is_cancel(answer):
            logger.info("{} annulé(e) : « {} ».", announce, answer)
            turn.speak(cancelled)
            return False
        return True


def _send2trash(path: str) -> None:
    from send2trash import send2trash

    send2trash(path)


def _check_windows_trash(path: Path) -> None:
    """Windows : la corbeille sûre de send2trash (IFileOperation, qui refuse un élément
    non recyclable) exige pywin32 ; sans lui, send2trash supprime définitivement en
    silence ce qui ne tient pas dans la corbeille. Les lecteurs amovibles ou réseau n'ont
    pas de corbeille."""
    try:
        import send2trash.win.modern  # noqa: F401
    except ImportError as exc:
        raise GuardDenied("corbeille sûre indisponible (pywin32 absent) : rien supprimé") from exc
    import ctypes

    drive = Path(path).anchor
    drive_fixed = 3
    kind = ctypes.windll.kernel32.GetDriveTypeW(drive)  # type: ignore[attr-defined,unused-ignore]
    if kind != drive_fixed:
        raise GuardDenied(f"refusé : le lecteur {drive} n'a pas de corbeille (amovible ou réseau)")


def _size_mb(path: Path, limit_mb: float) -> float:
    """Taille en Mo ; s'arrête dès que `limit_mb` est dépassé, ou après le budget de temps
    (on renvoie alors « au-delà de la limite » : dans le doute, refuser)."""
    if not path.is_dir():
        return path.lstat().st_size / 1e6
    total = 0
    deadline = time.monotonic() + SIZE_SCAN_BUDGET_S
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
        if total / 1e6 > limit_mb or time.monotonic() > deadline:
            return max(total / 1e6, limit_mb + 1)
    return total / 1e6
