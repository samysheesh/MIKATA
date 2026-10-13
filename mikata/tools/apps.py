"""Outils applications et fenêtres (§7) : open_app ★, close_app ★, focus_window ★,
list_windows.

Correspondance des noms **stricte, par mots** (corrigée en phase 5 : « ouvre Photoshop »
ouvrait Photos). Le LLM corrige déjà les erreurs de transcription ; ici, un nom n'est
retenu que si :
- il est identique (sans casse, accents ni ponctuation), ou collé (« vscode ») ;
- chaque mot demandé est un mot du nom (« chrome » -> Google Chrome), à une faute de
  frappe près pour les mots de 5 lettres et plus (« spotfy »), ou un sigle de mots qui se
  suivent (« vs code » -> Visual Studio Code, « cs2 » -> Counter-Strike 2) ;
- ou l'inverse : chaque mot du nom est dans la demande (« Spotify Music » -> Spotify) ;
- ou les deux textes sont presque identiques (ratio ≥ 90).
Sinon : aucun candidat, « application introuvable : X ».

Ajout phase 6 : noms français des applis Windows et noms de leurs processus
(`APP_ALIASES`) : « notepad » trouve le Bloc-notes (fenêtre ou menu Démarrer), « Bloc-notes »
trouve le processus Notepad de Windows 11, et ainsi de suite. Si aucune fenêtre ne
correspond, le message d'erreur liste les fenêtres ouvertes : le LLM peut réessayer avec le
bon nom au lieu de conclure « processus introuvable ».
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, Field

from mikata.action.effects import Effect
from mikata.tools.base import AppEntry, BaseTool, NoParams, ToolError, WindowInfo
from mikata.tools.registry import register_tool

# Mots de liaison dits autour d'un nom (« ouvre l'application Spotify »).
FILLERS = frozenset(
    {"l", "la", "le", "les", "de", "du", "des", "d", "app", "appli", "application", "logiciel",
     "programme", "jeu", "fenetre"}
)  # fmt: skip
# Mots en trop tolérés dans la demande quand le nom est plus court (« Spotify Music »,
# « Microsoft Word ») ; tout autre mot en trop (« Steam VR ») est une autre appli.
GENERIC = frozenset(
    {"microsoft", "adobe", "google", "music", "musique", "browser", "navigateur", "desktop",
     "pc", "windows", "for", "pour", "player", "lecteur", "client", "launcher", "officiel"}
)  # fmt: skip
WORD_TYPO_MIN_LEN = 5
WORD_TYPO_RATIO = 85.0
WHOLE_RATIO = 90.0
TITLE_SCORE = 80.0
ALIAS_PENALTY = 1.0  # à score égal, le nom demandé passe devant son alias
OPEN_WINDOWS_IN_ERROR = 12

# Nom affiché (français) -> noms de processus ou noms anglais de la même appli. Utilisé dans
# les deux sens ; comparé avec `fold` (sans casse, accents ni ponctuation).
APP_ALIASES: dict[str, tuple[str, ...]] = {
    "Bloc-notes": ("notepad",),
    "Calculatrice": ("calculator", "calculatorapp", "calc"),
    "Paint": ("mspaint",),
    "Microsoft Edge": ("msedge", "edge"),
    "Word": ("winword",),
    "PowerPoint": ("powerpnt",),
    "Microsoft Teams": ("ms-teams", "teams"),
    "Visual Studio Code": ("code",),
    "Paramètres": ("systemsettings", "settings", "parametres windows"),
    "Terminal": ("windowsterminal", "windows terminal"),
    "Invite de commandes": ("cmd", "command prompt"),
    "Gestionnaire des tâches": ("taskmgr", "task manager"),
    "Explorateur de fichiers": ("explorer", "file explorer"),
    "Lecteur multimédia": ("media player",),
    "Outil Capture d'écran": ("snippingtool", "snipping tool", "capture d ecran"),
    "Horloge": ("clock", "alarmes"),
    "Pense-bête": ("stickynotes", "sticky notes"),
}

# Noms macOS (noms d'applis et de processus courants) : s'ajoutent aux alias Windows,
# jamais à la place (comparés avec `fold`, dans les deux sens comme les autres).
DARWIN_ALIASES: dict[str, tuple[str, ...]] = {
    "Finder": ("finder",),
    "Safari": ("safari",),
    "Aperçu": ("preview", "apercu"),
    "Réglages Système": ("system settings", "systemsettings", "preferences systeme"),
    "App Store": ("app store",),
    "Messages": ("messages",),
    "Mail": ("mail",),
    "Plans": ("maps",),
    "Notes": ("notes",),
    "Rappels": ("reminders",),
    "Musique": ("music",),
    "Photos": ("photos",),
    "Calendrier": ("calendar", "calendrier"),
    "Terminal macOS": ("terminal",),
}
APP_ALIASES.update(DARWIN_ALIASES)


def fold(text: str) -> str:
    """Forme de comparaison : minuscules, sans accents, sans « .exe » ni ponctuation."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\.exe\b", "", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def _query_words(text: str) -> list[str]:
    return [word for word in fold(text).split() if word not in FILLERS]


def _same_word(asked: str, word: str) -> bool:
    from rapidfuzz import fuzz

    if asked == word:
        return True
    shortest = min(len(asked), len(word))
    return shortest >= WORD_TYPO_MIN_LEN and fuzz.ratio(asked, word) >= WORD_TYPO_RATIO


def _covered(asked: Sequence[str], words: Sequence[str]) -> int | None:
    """Nombre de mots de `words` utilisés si chaque mot de `asked` y trouve sa place (mot
    identique, faute de frappe, ou sigle de mots consécutifs), sinon None."""
    used: set[int] = set()
    for item in asked:
        for i, word in enumerate(words):
            if i not in used and _same_word(item, word):
                used.add(i)
                break
        else:
            span = _acronym_span(item, words, used)
            if span is None:
                return None
            used.update(span)
    return len(used)


def _acronym_span(item: str, words: Sequence[str], used: set[int]) -> range | None:
    """Mots consécutifs dont les initiales forment `item` (« vs » -> visual studio), ou
    initiales suivies d'un mot entier (« vscode » -> visual studio code)."""
    if len(item) < 2:
        return None
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            span = range(start, end)
            if used & set(span):
                break
            initials = "".join(words[i][0] for i in span)
            last_full = "".join(words[i][0] for i in span[:-1]) + words[end - 1]
            if len(span) >= 2 and item in (initials, last_full):
                return span
    return None


def name_score(query: str, name: str) -> float | None:
    """Score (plus haut = meilleur) si `name` correspond vraiment à `query`, sinon None."""
    from rapidfuzz import fuzz

    asked, words = _query_words(query), fold(name).split()
    if not asked or not words:
        return None
    if asked == words or "".join(asked) == "".join(words):
        return 100.0
    used = _covered(asked, words)
    if used is not None:  # la demande est dans le nom ; moins de mots en trop = mieux
        return 95.0 - 2 * (len(words) - used)
    if _covered(words, asked) is not None:  # le nom est dans la demande (« Spotify Music »)
        extra = [w for w in asked if not any(_same_word(w, word) for word in words)]
        if all(w in GENERIC for w in extra):
            return 85.0 - 2 * len(extra)
    ratio = fuzz.ratio(" ".join(asked), " ".join(words))
    return ratio - 10.0 if ratio >= WHOLE_RATIO else None


def query_variants(query: str) -> list[str]:
    """`query`, puis ses alias (APP_ALIASES) : « notepad » -> [notepad, Bloc-notes]."""
    variants = [query]
    folded = fold(query)
    words = [w for w in folded.split() if w not in FILLERS]
    if not words:
        return variants
    asked = " ".join(words)
    for name, aliases in APP_ALIASES.items():
        group = [name, *aliases]
        if any(asked in (fold(item), fold(item).replace(" ", "")) for item in group):
            variants.extend(item for item in group if fold(item) != asked)
    return variants


def _best_score(query: str, name: str) -> float | None:
    """Meilleur score de `name` pour `query` ou l'un de ses alias (alias un peu pénalisé)."""
    scores = [
        score - (ALIAS_PENALTY if i else 0.0)
        for i, variant in enumerate(query_variants(query))
        if (score := name_score(variant, name)) is not None
    ]
    return max(scores) if scores else None


def find_app(query: str, apps: Sequence[AppEntry]) -> AppEntry | None:
    """Application qui correspond à `query` (ou à un alias : « notepad » -> Bloc-notes) ;
    à score égal, le nom le plus court."""
    scored = [
        (score, -len(app.name), -i, app)
        for i, app in enumerate(apps)
        if (score := _best_score(query, app.name)) is not None
    ]
    return max(scored, key=lambda item: item[:3])[3] if scored else None


def find_windows(query: str, windows: Sequence[WindowInfo]) -> list[WindowInfo]:
    """Fenêtres de l'appli qui correspond à `query` (ou à un alias) : par nom de processus,
    sinon par le nom d'appli en fin de titre (« Bloc-notes » pour notepad)."""
    best: tuple[float, list[WindowInfo]] | None = None
    by_process: dict[str, list[WindowInfo]] = {}
    for window in windows:
        by_process.setdefault(window.process, []).append(window)
    for process, group in by_process.items():
        score = _best_score(query, process) if process else None
        if score is not None and (best is None or score > best[0]):
            best = (score, group)
    if best is None or best[0] < TITLE_SCORE:
        for variant in query_variants(query):
            titled = _title_matches(variant, windows)
            if titled:
                best = (TITLE_SCORE, titled)
                break
    return best[1] if best is not None else []


def no_window_error(name: str, windows: Sequence[WindowInfo]) -> ToolError:
    """« aucune fenêtre ouverte pour X », avec la liste des fenêtres pour que le LLM
    réessaie avec le bon nom."""
    if not windows:
        return ToolError(f"aucune fenêtre ouverte pour « {name} » (aucune fenêtre ouverte)")
    listed = "; ".join(f"{w.process or '?'} : {w.title}" for w in windows[:OPEN_WINDOWS_IN_ERROR])
    return ToolError(f"aucune fenêtre ouverte pour « {name} ». Fenêtres ouvertes : {listed}")


def _title_app(title: str) -> str:
    """Dernier morceau du titre, où Windows met le nom de l'appli (« x.txt - Bloc-notes »)."""
    return re.split(r"\s+[-–—|]\s+", title.strip())[-1]


def _title_matches(query: str, windows: Sequence[WindowInfo]) -> list[WindowInfo]:
    """Fenêtres dont le nom d'appli en fin de titre correspond ; sinon celles dont le titre
    contient tous les mots demandés. Toujours d'une seule appli (« ferme le bloc-notes » ne
    ferme pas l'onglet « Bloc-notes et astuces - YouTube - Google Chrome »)."""
    found = [w for w in windows if name_score(query, _title_app(w.title)) is not None]
    if not found:
        asked = _query_words(query)
        found = [w for w in windows if asked and _covered(asked, fold(w.title).split()) is not None]
    if not found:
        return []
    process = found[0].process
    return [w for w in found if w.process == process]


def installed_app_names(apps: Sequence[AppEntry], limit: int = 400) -> list[str]:
    """Noms des applications installées, triés et dédoublonnés, pour le prompt (le LLM y
    corrige les noms mal transcrits)."""
    names: dict[str, str] = {}
    for app in apps:
        name = " ".join(app.name.split())
        if 0 < len(name) <= 60:
            names.setdefault(name.casefold(), name)
    return [names[key] for key in sorted(names)][:limit]


def open_apps(windows: Sequence[WindowInfo], limit: int = 8) -> list[str]:
    """Noms courts des applis ouvertes (contexte dynamique du LLM, §6)."""
    names: list[str] = []
    for window in windows:
        name = window.process or window.title
        if name and name.casefold() not in (n.casefold() for n in names):
            names.append(name)
    return names[:limit]


class AppName(BaseModel):
    name: str = Field(
        min_length=1,
        description="Nom de l'application tel qu'affiché par Windows (ex. « Spotify », "
        "« Brave », « Steam »), corrigé si la transcription l'a déformé",
    )


@register_tool
class OpenApp(BaseTool[AppName]):
    name = "open_app"
    sdk = "apps.open"
    effect = Effect.ACT
    expected_s = 1.0
    description = "Ouvre (ou ramène au premier plan) une application ou un jeu installé."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        app = find_app(args.name, await self.ctx.run_blocking(desktop.list_apps))
        if app is None:  # installée depuis le dernier index ?
            app = find_app(args.name, await self.ctx.run_blocking(desktop.list_apps, True))
        if app is None:
            raise ToolError(f"application introuvable : {args.name}")
        await self.ctx.run_blocking(desktop.launch, app)
        return f"{app.name} ouvert."


@register_tool
class CloseApp(BaseTool[AppName]):
    name = "close_app"
    sdk = "apps.close"
    effect = Effect.ACT
    description = "Ferme une application ouverte (toutes ses fenêtres), comme avec la croix."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        opened = await self.ctx.run_blocking(desktop.windows)
        windows = find_windows(args.name, opened)
        if not windows:
            raise no_window_error(args.name, opened)
        for window in windows:
            await self.ctx.run_blocking(desktop.close_window, window)
        label = windows[0].process or windows[0].title
        return f"{label} fermé ({len(windows)} fenêtre(s))."


class WindowParams(BaseModel):
    name: str | None = Field(
        default=None,
        description="Appli ou titre de la fenêtre à fermer ; vide : la fenêtre active",
    )


@register_tool
class CloseWindow(BaseTool[WindowParams]):
    name = "close_window"
    sdk = "apps.close_window"
    effect = Effect.ACT
    description = (
        "Ferme une seule fenêtre, comme avec la croix : la fenêtre active (« ferme cette "
        "fenêtre »), ou celle d'une appli ou d'un titre donné."
    )
    params = WindowParams
    terminal = True

    async def run(self, args: WindowParams) -> str:
        desktop = self.ctx.require_desktop()
        if args.name:
            opened = await self.ctx.run_blocking(desktop.windows)
            found = find_windows(args.name, opened)
            if not found:
                raise no_window_error(args.name, opened)
            window = found[0]
        else:
            active = await self.ctx.run_blocking(desktop.foreground_window)
            if active is None:
                raise ToolError("aucune fenêtre active")
            window = active
        await self.ctx.run_blocking(desktop.close_window, window)
        return f"Fenêtre « {window.title} » fermée."


@register_tool
class FocusWindow(BaseTool[AppName]):
    name = "focus_window"
    sdk = "apps.focus"
    effect = Effect.ACT
    description = "Met au premier plan la fenêtre d'une application déjà ouverte."
    params = AppName
    terminal = True

    async def run(self, args: AppName) -> str:
        desktop = self.ctx.require_desktop()
        opened = await self.ctx.run_blocking(desktop.windows)
        windows = find_windows(args.name, opened)
        if not windows:
            raise no_window_error(args.name, opened)
        await self.ctx.run_blocking(desktop.focus_window, windows[0])
        # Le focus peut viser le navigateur de travail de MIKATA : on le re-cache
        # aussitôt (sa fenêtre ne doit jamais apparaître à l'écran).
        if self.ctx.config.browser.own_profile:
            browser = self.ctx.browser
            keep_hidden = getattr(browser, "keep_hidden", None)
            if keep_hidden is not None:
                await self.ctx.run_blocking(keep_hidden)
        return f"« {windows[0].title} » au premier plan."


@register_tool
class ListWindows(BaseTool[NoParams]):
    name = "list_windows"
    returns = "{windows: [{app, title}]}"
    sdk = "apps.windows"
    effect = Effect.READ
    expected_s = 0.2
    description = "Liste les fenêtres ouvertes (application et titre)."
    params = NoParams

    async def run(self, args: NoParams) -> str:
        desktop = self.ctx.require_desktop()
        windows = await self.ctx.run_blocking(desktop.windows)
        if not windows:
            return "Aucune fenêtre ouverte."
        lines = [f"- {w.process or '?'} : {w.title}" for w in windows[:30]]
        self.ctx.publish_data(
            {"windows": [{"app": w.process, "title": w.title} for w in windows[:30]]}
        )
        return "Fenêtres ouvertes :\n" + "\n".join(lines)
