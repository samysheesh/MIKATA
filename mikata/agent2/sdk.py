"""Agent v2 (phase 13a) : le SDK MIKATA, façade typée sur les outils existants.

Précision (remplace « sous-processus » dans docs/AGENT_V2.md §2.5 et §6) : le SDK n'est
pas un nouvel accès au système. C'est une façade sur `ToolRegistry.execute` — le
garde-fou, les confirmations orales et les différences Windows/macOS restent donc dans
les outils, sans rien dupliquer.
"""

from __future__ import annotations

import uuid
from typing import Any

from mikata.agent2.schema import ActionRecord, Observation
from mikata.llm.base import ToolCallReq
from mikata.tools.base import current_turn

# Réponse de l'utilisateur limitée (la question posée par le SDK est courte).
_ASK_TIMEOUT_S = 8.0
_PAGE_MAX_CHARS = 1500
_PRINT_MAX_CHARS = 2000


class SdkError(Exception):
    """Un outil a échoué ; le message (français) est montré au LLM."""


class ActionLimitReached(SdkError):
    """Plus de `max_actions` actions dans un même script."""


class Recorder:
    """Garde les actions d'un script (pour la trace)."""

    def __init__(self) -> None:
        self.actions: list[ActionRecord] = []


class Sdk:
    """L'espace d'action de l'agent v2 : les outils existants, avec des noms stables.

    Chaque méthode de confort n'est qu'un appel à `self.tool(...)` ; le LLM ne connaît
    que ce qui est décrit dans `sdk_reference()`.
    """

    def __init__(
        self,
        registry: Any,
        *,
        max_actions: int,
        observe_lines: int,
        recorder: Recorder | None = None,
    ) -> None:
        self._registry = registry
        self._max_actions = max_actions
        self._observe_lines = observe_lines
        self._recorder = recorder if recorder is not None else Recorder()
        self._actions = 0
        self._counting = True  # observe() coupe comptage et enregistrement
        self._last_observation: Observation | None = None

    # -- action générique ---------------------------------------------------------------

    async def tool(self, name: str, /, *positional: Any, **args: Any) -> str:
        """Appelle l'outil `name` (registre existant). Lève SdkError si l'outil échoue ou
        n'existe pas. Compte l'action, l'enregistre dans le Recorder.

        `name` est positional-only : les outils qui ont un argument `name` (open_app,
        find_files, find_path...) restent appelables via `self.tool("open_app", name=...).
        Des arguments positionnels sont tolérés (le modèle écrit parfois
        `sdk.tool('set_volume', 50)`) : ils sont reliés aux champs du modèle, dans
        l'ordre.
        """
        if positional:
            tool = self._registry.tools.get(name)
            params = getattr(tool, "params", None)
            fields = list(params.model_fields) if params is not None else []
            # strict=False : les arguments en trop sont simplement ignorés.
            for field, value in zip(fields, positional, strict=False):
                args[field] = value
        tool = self._registry.tools.get(name)
        if tool is None:
            raise SdkError(f"outil inconnu : {name}")
        counting = self._counting
        if counting:
            if self._actions >= self._max_actions:
                raise ActionLimitReached(
                    f"limite de {self._max_actions} actions atteinte dans ce script"
                )
            self._actions += 1
        result = await self._registry.execute(
            ToolCallReq(id=uuid.uuid4().hex[:8], name=name, arguments=args)
        )
        if counting:
            self._recorder.actions.append(
                ActionRecord(
                    tool=name,
                    args=args,
                    ok=result.ok,
                    result=result.content[:500],
                    duration_s=result.duration_s,
                )
            )
        if not result.ok:
            raise SdkError(result.content)
        return result.content

    # -- observation (lecture seule, ne compte pas comme action) ------------------------

    async def observe(self) -> Observation:
        """L'état de l'écran : app active, titre, page navigateur, résumé UIA.

        Chaque lecture qui échoue est ignorée (champ vide), jamais d'exception.
        """
        obs = Observation(changed=True)
        self._counting = False
        try:
            try:
                windows = await self.tool("list_windows")
            except SdkError:
                windows = ""
            for line in windows.splitlines():
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                parts = stripped[2:].split(" : ", 1)
                obs.active_app = parts[0].strip()
                obs.title = parts[1].strip() if len(parts) > 1 else obs.active_app
                break
            try:
                page = await self.tool("browser_read")
                obs.page = page[:_PAGE_MAX_CHARS]
            except SdkError:
                obs.page = ""
            try:
                summary = await self.tool("uia_read")
            except SdkError:
                summary = ""
            if not obs.page and summary:
                # Repli : le navigateur n'est pas contrôlable (Brave sans le port CDP)
                # mais l'écran est lisible par UIA — les prédicats de page peuvent
                # vérifier ce qui est effectivement visible.
                obs.page = summary[:_PAGE_MAX_CHARS]
        finally:
            self._counting = True
        obs.summary = "\n".join(summary.splitlines()[: self._observe_lines])
        previous = self._last_observation
        obs.changed = previous is None or (
            previous.active_app != obs.active_app
            or previous.title != obs.title
            or previous.page[:200] != obs.page[:200]
            or previous.summary[:200] != obs.summary[:200]
        )
        self._last_observation = obs
        return obs

    # -- interaction avec l'utilisateur -------------------------------------------------

    async def say(self, text: str) -> str:
        """Dit une phrase (s'il y a un tour en cours), sans passer par un outil."""
        turn = current_turn()
        if turn is None:
            return ""
        turn.say(text)
        return text

    async def ask_user(self, question: str) -> str | None:
        """Une seule question courte ; la réponse de l'utilisateur, ou None sans micro."""
        turn = current_turn()
        if turn is None or turn.listen is None:
            return None
        turn.say(question)
        await turn.drain()
        answer = await turn.listen(_ASK_TIMEOUT_S)
        return answer if answer else None

    # -- méthodes de confort (chaque outil vérifié sur la branche) -----------------------

    async def open_app(self, name: str) -> str:
        return await self.tool("open_app", name=name)

    async def close_app(self, name: str) -> str:
        return await self.tool("close_app", name=name)

    async def focus(self, title: str) -> str:
        return await self.tool("focus_window", name=title)

    async def windows(self) -> str:
        return await self.tool("list_windows")

    async def open_url(self, url: str) -> str:
        return await self.tool("open_url", url=url)

    async def web_search(self, query: str) -> str:
        return await self.tool("web_search", query=query)

    async def fetch(self, url: str) -> str:
        return await self.tool("fetch_page", url=url)

    async def browser_open(self, url: str) -> str:
        return await self.tool("browser_open", url=url)

    async def browser_read(self) -> str:
        return await self.tool("browser_read")

    async def browser_click(self, text: str, nth: int = 1) -> str:
        return await self.tool("browser_click", text=text, nth=nth)

    async def browser_type(self, field: str, text: str) -> str:
        return await self.tool("browser_type", field=field, text=text)

    # Pas de browser_find : l'outil n'existe pas sur la branche (browser_read suffit).

    async def ui_read(self, window: str | None = None, control: str | None = None) -> str:
        return await self.tool("uia_read", window=window, control=control)

    async def ui_click(self, window: str | None, control: str) -> str:
        return await self.tool("uia_click", window=window, control=control)

    async def ui_type(self, window: str | None, control: str, text: str) -> str:
        return await self.tool("uia_set_text", window=window, control=control, text=text)

    async def find_files(self, query: str) -> str:
        return await self.tool("find_files", name=query)

    async def find_path(self, query: str) -> str:
        return await self.tool("find_path", name=query)

    async def list_folder(self, path: str) -> str:
        return await self.tool("list_folder", path=path)

    async def open_in_explorer(self, path: str) -> str:
        return await self.tool("open_in_explorer", path=path)

    async def show_text(self, text: str) -> str:
        return await self.tool("show_text", text=text)

    async def read_file(self, path: str) -> str:
        return await self.tool("read_file", path=path)

    async def write_file(self, path: str, text: str) -> str:
        return await self.tool("write_file", path=path, content=text)

    async def move(self, src: str, dst: str) -> str:
        return await self.tool("move_file", source=src, destination=dst)

    async def trash(self, path: str) -> str:
        return await self.tool("delete_file", path=path)

    async def type_text(self, text: str) -> str:
        return await self.tool("type_text", text=text)

    async def keys(self, *names: str) -> str:
        return await self.tool("press_keys", keys="+".join(names))

    async def media(self, action: str) -> str:
        return await self.tool("media_control", action=action)

    async def volume(self, percent: int) -> str:
        return await self.tool("set_volume", level=percent)

    async def mute(self, on: bool = True) -> str:
        return await self.tool("mute", mute=on)

    async def timer(self, seconds: int, label: str | None = None) -> str:
        return await self.tool("set_timer", seconds=seconds, label=label)

    async def reminder(
        self, text: str, at: str | None = None, in_minutes: int | None = None
    ) -> str:
        return await self.tool("set_reminder", text=text, at=at, in_minutes=in_minutes)

    async def remember(self, fact: str) -> str:
        return await self.tool("remember", fact=fact)

    # Pas de recall : l'outil n'existe pas sur la branche (memory_tools a remember/forget).


def sdk_reference() -> str:
    """La liste des méthodes du SDK, une ligne par méthode — injectée dans les prompts."""
    return """\
open_app(name) — ouvrir une application installée.
close_app(name) — fermer une application ouverte.
focus(title) — mettre une fenêtre au premier plan.
windows() — fenêtres ouvertes.
open_url(url) — ouvrir un site dans le navigateur de l'utilisateur.
web_search(query) — rechercher sur le web.
fetch(url) — texte d'une page web.
browser_open(url) — ouvrir une page dans le navigateur de travail.
browser_read() — titre, texte, liens et boutons de l'onglet actif.
browser_click(text, nth=1) — cliquer un élément par son texte.
browser_type(field, text) — écrire dans un champ de page.
ui_read(window, control) — texte d'une fenêtre ou d'un contrôle.
ui_click(window, control) — cliquer un contrôle par son nom.
ui_type(window, control, text) — écrire dans un champ d'application.
find_files(query) — chercher un fichier (parcours, jokers).
find_path(query) — chercher dans l'index de tous les fichiers (rapide).
list_folder(path) — contenu d'un dossier.
open_in_explorer(path) — montrer un fichier ou dossier dans l'Explorateur.
show_text(text) — afficher un texte à l'écran (résultat, chemin trouvé...).
read_file(path) — contenu d'un fichier texte (tronqué).
write_file(path, text) — créer ou modifier un fichier.
move(src, dst) — déplacer ou renommer un fichier/dossier.
trash(path) — mettre à la corbeille.
type_text(text) — taper du texte au clavier.
keys(*names) — combinaison de touches (keys("ctrl", "s")).
media(action) — play, pause, next, previous, stop.
volume(percent) — volume 0 à 100.
mute(on=True) — couper ou remettre le son.
timer(seconds, label) — minuteur.
reminder(text, at, in_minutes) — rappel.
remember(fact) — mémoriser un fait.
say(text) — dire une phrase à l'utilisateur.
ask_user(question) — poser une question et écouter la réponse.
observe() — état de l'écran (app active, titre, page, résumé)."""
