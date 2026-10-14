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

    async def run(self, args: UiaWindowParams) -> str:
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        controls = await self.ctx.run_blocking(uia.controls, win, self.ctx.config.uia.max_controls)
        if not controls:
            raise ToolError(f"aucun contrôle lisible dans « {win.title} »")
        lines = [f"Fenêtre « {win.title} » ({win.process}), {len(controls)} contrôles :"]
        lines.extend(f"- {control.name} [{control.kind}]" for control in controls)
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

    async def run(self, args: UiaClickParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_ui_action(args.control)
        await _approve(self.ctx, guard, decision, f"Cliquer « {args.control} »")
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        return await self.ctx.run_blocking(uia.click, win, args.control, args.kind, args.nth)


class UiaSetTextParams(WindowParams):
    control: str = Field(min_length=1, description="Nom du champ à remplir (ex. « Recherche »)")
    text: str = Field(min_length=1, max_length=2000, description="Texte à écrire dans le champ")


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

    async def run(self, args: UiaSetTextParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_typed_text(args.text)
        if decision.verdict is Verdict.DENY:
            raise GuardDenied(f"texte non écrit : {decision.reason} (« {decision.match} »)")
        uia = _uia(self.ctx)
        win = await self.ctx.run_blocking(uia.window, args.window, args.process)
        return await self.ctx.run_blocking(uia.set_text, win, args.control, args.text)


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
            raise ToolError("état ambiant vide : rien à lire pour l'instant")
        return summary
