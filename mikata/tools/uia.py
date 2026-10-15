"""Outils de pilotage fin des applications (phase 12a) : uia_window, uia_read, uia_click,
uia_set_text, uia_state.

Ces outils lisent l'arbre d'accessibilité (Windows : UIA via `uiautomation` ; macOS :
AXUIElement via `macuiauto`, mêmes signatures ; Chromium : préférez CDP quand c'est
possible) et cliquent par nom, au lieu de taper à l'aveugle au clavier. Garde-fou : les contrôles
destructifs (suppression définitive, formatage...) sont refusés, les contrôles sensibles
(supprimer, désinstaller, payer...) demandent un « oui » oral, et le texte écrit dans un
champ passe le même filtre que type_text (une commande de suppression ne se tape pas).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mikata.action.effects import Effect, Trust
from mikata.agent.guard import Verdict
from mikata.tools.base import BaseTool, GuardDenied, ToolContext, ToolError, UiAutomation
from mikata.tools.registry import register_tool
from mikata.tools.shell import _approve


def app_launch_limit(ctx: ToolContext, own: float | None = None) -> float | None:
    """Phase 17e : délai d'un outil qui peut lancer une appli tenue cachée (la première
    demande la démarre). Le 14/10, sous charge, Spotify a mis plus de 15 s à démarrer :
    `uia_items` expirait, et le tour perdait ces 15 s. Le délai couvre le lancement, puis
    l'outil ; sans appli listée, rien ne change."""
    config = ctx.config
    if not config.uia.devtools_apps:
        return own
    return max(own or 0.0, config.tools.timeout_s + config.uia.devtools_launch_timeout_s)


def _uia(ctx: ToolContext) -> UiAutomation:
    """Le pilote UIA, sauf si la section `uia` de la config est coupée (message dédié)."""
    if not ctx.config.uia.enabled:
        raise ToolError("le pilotage des applications est désactivé (uia.enabled: false)")
    return ctx.require_uia()


class WindowParams(BaseModel):
    window: str | None = Field(
        default=None,
        description="Titre de la fenêtre, ou une partie (ex. « Discord », « Sans titre »)",
    )
    process: str | None = Field(
        default=None, description="Nom du processus (ex. « Discord », « notepad »)"
    )


class UiaWindowParams(WindowParams):
    """Sans argument : la fenêtre active."""


@register_tool
class UiaWindow(BaseTool[UiaWindowParams]):
    name = "uia_window"
    returns = "{window, controls: [{name, kind, text}]}"
    sdk = "ui.controls"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 0.8
    description = (
        "Liste les contrôles (boutons, champs, textes, éléments) d'une fenêtre "
        "d'application, par titre ou nom de processus (sans argument : fenêtre active). "
        "À utiliser avant de cliquer, pour connaître les noms exacts."
    )
    params = UiaWindowParams
    terminal = False

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx)

    async def run(self, args: UiaWindowParams) -> str:
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        controls = await self.ctx.run_blocking(uia.controls, win, self.ctx.config.uia.max_controls)
        if not controls:
            raise ToolError(f"aucun contrôle lisible dans « {win.title} »")
        lines = [f"Fenêtre « {win.title} » ({win.process}), {len(controls)} contrôles :"]
        lines.extend(
            f"- {control.name} [{control.kind}]"
            + (f" : « {control.text} »" if control.text else "")
            for control in controls
        )
        self.ctx.publish_data(
            {
                "window": win.title,
                "controls": [{"name": c.name, "kind": c.kind, "text": c.text} for c in controls],
            }
        )
        return "\n".join(lines)


class UiaReadParams(WindowParams):
    control: str | None = Field(
        default=None, description="Nom d'un contrôle précis ; sans : tout le texte de la fenêtre"
    )


@register_tool
class UiaRead(BaseTool[UiaReadParams]):
    name = "uia_read"
    sdk = "ui.read"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 0.8
    description = (
        "Lit le texte d'un contrôle ou de toute une fenêtre d'application (par titre ou "
        "processus), sans la mettre au premier plan."
    )
    params = UiaReadParams
    terminal = False

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx)

    async def run(self, args: UiaReadParams) -> str:
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        return await self.ctx.run_blocking(
            uia.read, win, args.control, self.ctx.config.uia.max_text_chars
        )


class UiaClickParams(WindowParams):
    control: str = Field(min_length=1, description="Nom du contrôle à cliquer (ex. « Lire »)")
    kind: str | None = Field(
        default=None,
        description="Type du contrôle (ex. « bouton », « lien », « élément ») pour lever "
        "l'ambiguïté ; sans : le premier contrôle cliquable de ce nom",
    )
    nth: int = Field(default=1, ge=1, le=20, description="N-ième contrôle de ce nom (1 = premier)")
    near: str | None = Field(
        default=None,
        description="Texte de la même ligne, pour choisir parmi des contrôles de même nom "
        "(ex. le bouton « Play » de la ligne où figure « Tiakola »)",
    )
    item: str | None = Field(
        default=None,
        description="Élément de liste (vu par uia_items) dans lequel cliquer le contrôle : "
        "ex. control « Play », item « Jet Lag », near « Tiakola »",
    )


@register_tool
class UiaClick(BaseTool[UiaClickParams]):
    name = "uia_click"
    sdk = "ui.click"
    effect = Effect.ACT
    expected_s = 0.8
    description = (
        "Clique un contrôle d'application par son nom (bouton, lien, élément de liste...), "
        "dans la fenêtre donnée. Le garde-fou peut refuser ou demander confirmation."
    )
    params = UiaClickParams
    terminal = True

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx)

    async def run(self, args: UiaClickParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_ui_action(args.control)
        await _approve(self.ctx, guard, decision, f"Cliquer « {args.control} »")
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        if args.item:  # phase 17a : le bouton de cet élément-là
            item, shown = split_item(args.item)
            contains = [text for text in (shown, args.near) if text]
            return await self.ctx.run_blocking(uia.click_in, win, item, [args.control], contains)
        return await self.ctx.run_blocking(
            uia.click, win, args.control, args.kind, args.nth, args.near
        )


class UiaSetTextParams(WindowParams):
    control: str = Field(min_length=1, description="Nom du champ à remplir (ex. « Recherche »)")
    text: str = Field(min_length=1, max_length=2000, description="Texte à écrire dans le champ")
    submit: bool = Field(
        default=False,
        description="true : valider ensuite comme avec Entrée (lancer la recherche, envoyer)",
    )


@register_tool
class UiaSetText(BaseTool[UiaSetTextParams]):
    name = "uia_set_text"
    sdk = "ui.set_text"
    effect = Effect.CONSEQUENTIAL
    expected_s = 0.8
    description = (
        "Écrit un texte dans un champ d'application nommé (plus sûr que taper à l'aveugle). "
        "Le garde-fou peut refuser le texte (commande de suppression)."
    )
    params = UiaSetTextParams
    terminal = True

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx)

    async def run(self, args: UiaSetTextParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_typed_text(args.text)
        if decision.verdict is Verdict.DENY:
            raise GuardDenied(f"texte non écrit : {decision.reason} (« {decision.match} »)")
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        return await self.ctx.run_blocking(uia.set_text, win, args.control, args.text, args.submit)


class UiaStateParams(BaseModel):
    """Aucun argument : l'état ambiant gardé en mémoire."""


@register_tool
class UiaState(BaseTool[UiaStateParams]):
    name = "uia_state"
    sdk = "ui.state"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 0.6  # lu à la demande : une relève de Discord prend ~0,5 s
    description = (
        "Lit l'état ambiant du PC (par ex. qui est dans le salon vocal Discord), sans rien "
        "ouvrir ni déplacer."
    )
    params = UiaStateParams
    terminal = False

    async def run(self, args: UiaStateParams) -> str:
        uia = _uia(self.ctx)
        summary = await self.ctx.run_blocking(uia.state_summary)
        if not summary:
            # Un état vide ne prouve rien (Discord fermé, illisible...) : surtout pas « tu
            # n'es pas en vocal », dont le modèle concluait qu'il n'y avait rien à faire.
            # Phase 17f : et ce vide ne dit rien de l'état des applis ; MIKATA concluait
            # alors d'après les échanges d'avant (« tu es encore connecté », 15/10).
            raise ToolError(
                "état ambiant vide : rien à lire pour l'instant, ce qui ne dit rien de l'état "
                "des applis ; pour le savoir, lis leur fenêtre maintenant (uia_window)"
            )
        return summary


# -- éléments de liste et attente (phase 17a) -----------------------------------------------


def split_item(text: str) -> tuple[str, str]:
    """(nom, ce qu'il affiche) d'un élément donné tel que uia_items l'a montré :
    « « Général » — Salon vocal • 2 membres » -> (« Général », « Salon vocal • 2 membres »).
    Le modèle recopie souvent la ligne entière ; ce qu'elle affiche départage alors deux
    éléments de même nom (le salon textuel et le salon vocal « Général »)."""
    name, _, shown = text.partition(" — ")
    shown = shown.split(" [boutons", 1)[0].split(" | ", 1)[0]
    return name.strip().strip("«»").strip(), shown.strip()


ITEMS_PATIENCE = 6  # ~3 s : des éléments cherchés pas encore affichés


class UiaItemsParams(WindowParams):
    query: str | None = Field(
        default=None, description="Mots que l'élément doit contenir (titre, artiste...)"
    )
    limit: int = Field(default=20, ge=1, le=60, description="Éléments rendus, au plus")


@register_tool
class UiaItems(BaseTool[UiaItemsParams]):
    name = "uia_items"
    sdk = "ui.items"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 0.8
    returns = "{window, rows: [{name, texts, buttons}]}"
    description = (
        "Lit les éléments de liste d'une fenêtre (résultats de recherche, salons, fichiers, "
        "titres...) avec ce que chacun affiche et ses boutons : pour choisir le bon puis "
        "cliquer dedans (uia_click avec item). Filtrable par des mots (query)."
    )
    params = UiaItemsParams
    terminal = False

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx)

    async def run(self, args: UiaItemsParams) -> str:
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        items = await self.ctx.run_blocking(uia.items, win, args.query, args.limit)
        for _ in range(ITEMS_PATIENCE):  # une vue qui vient de s'ouvrir se remplit encore
            if items or not args.query:
                break
            await self.ctx.sleep(0.5)
            items = await self.ctx.run_blocking(uia.items, win, args.query, args.limit)
        if not items:
            where = f" contenant « {args.query} »" if args.query else ""
            raise ToolError(f"aucun élément de liste{where} dans « {win.title} »")
        lines = [f"Éléments de « {win.title} » ({len(items)}) :"]
        for item in items:
            line = f"- « {item.name} »"
            if item.texts:
                line += " — " + " | ".join(item.texts)
            if item.buttons:
                line += " [boutons : " + ", ".join(item.buttons) + "]"
            lines.append(line)
        self.ctx.publish_data(
            {
                "window": win.title,
                # « rows » et non « items » : `res.items` serait la méthode du dictionnaire.
                "rows": [
                    {"name": i.name, "texts": list(i.texts), "buttons": list(i.buttons)}
                    for i in items
                ],
            }
        )
        return "\n".join(lines)


class UiaWaitParams(WindowParams):
    text: str = Field(
        min_length=1, description="Texte attendu (nom d'un élément, contenu d'un champ)"
    )
    timeout_s: float = Field(default=8.0, ge=0.5, le=20.0, description="Attente maximale, en s")


@register_tool
class UiaWait(BaseTool[UiaWaitParams]):
    name = "uia_wait"
    sdk = "ui.wait"
    effect = Effect.READ
    expected_s = 1.5
    description = (
        "Attend qu'un texte apparaisse dans une fenêtre (une page qui se charge, un champ "
        "qui montre la recherche, une appli qui démarre), sans agir. Avant de lire ou de "
        "cliquer dans une vue qui vient de s'ouvrir."
    )
    params = UiaWaitParams
    terminal = False

    def time_limit(self) -> float | None:
        return app_launch_limit(self.ctx, 25.0)

    async def run(self, args: UiaWaitParams) -> str:
        import time

        uia = _uia(self.ctx)
        deadline = time.monotonic() + args.timeout_s
        while True:
            try:
                win = await self.ctx.run_blocking(uia.window, args.window, args.process)
                found = await self.ctx.run_blocking(uia.find_text, win, args.text)
            except ToolError:  # appli pas encore ouverte, fenêtre en train de changer
                found = None
            if found is not None:
                return f"Affiché : « {found[:120]} »."
            if time.monotonic() >= deadline:
                raise ToolError(f"« {args.text} » n'est pas apparu en {args.timeout_s:g} s")
            await self.ctx.sleep(0.5)
