"""Outils du navigateur (phase 12b) : browser_tabs, browser_open ★, browser_read,
browser_click ★, browser_type ★, browser_back ★.

MIKATA pilote Brave par le protocole DevTools : elle lit la page (titre, texte, liens,
boutons), clique par texte visible et écrit dans les champs — en arrière-plan, sans vision.
Garde-fou : seuls http/https s'ouvrent ; un clic qui ressemble à un téléchargement demande
confirmation ; une page lue marque le tour (confirm_after_web s'applique).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mikata.action.effects import Effect, Trust
from mikata.agent.guard import Verdict
from mikata.tools.base import BaseTool, BrowserControl, GuardDenied, ToolContext, ToolError
from mikata.tools.registry import register_tool
from mikata.tools.shell import _approve
from mikata.tools.web import _mark_web_read


def _browser(ctx: ToolContext) -> BrowserControl:
    if not ctx.config.browser.enabled:
        raise ToolError("le contrôle du navigateur est désactivé (browser.enabled: false)")
    return ctx.require_browser()


class BrowserOpenParams(BaseModel):
    url: str = Field(min_length=1, description="URL complète, ex. « https://www.youtube.com »")


@register_tool
class BrowserOpen(BaseTool[BrowserOpenParams]):
    name = "browser_open"
    sdk = "browser.open"
    effect = Effect.ACT
    expected_s = 3.0
    description = (
        "Ouvre une URL dans la fenêtre de travail de MIKATA (Brave, en arrière-plan : "
        "marche toujours, même si le Brave de l'utilisateur est ouvert) et attend la page. "
        "Ne conclut pas la demande : lis ensuite la page (browser_read) et clique "
        "(browser_click) ou écris (browser_type) pour finir l'action. Pour une simple "
        "information, web_search est bien plus rapide."
    )
    params = BrowserOpenParams
    terminal = False  # une tâche web est souvent en plusieurs étapes : on continue

    async def run(self, args: BrowserOpenParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_navigation(args.url)
        await _approve(self.ctx, guard, decision, "Cette navigation")
        return await _browser(self.ctx).open_url(args.url)


def parse_page(text: str) -> dict[str, Any]:
    """Page lue par `BrowserControl.read`, en champs : titre, adresse, texte, liens, boutons."""
    page: dict[str, Any] = {"title": "", "url": "", "text": "", "links": [], "buttons": []}
    body: list[str] = []
    in_text = False
    for line in text.splitlines():
        if line.startswith("Titre : ") and not page["title"]:
            page["title"] = line[len("Titre : ") :].strip()
        elif line.startswith("URL : ") and not page["url"]:
            page["url"] = line[len("URL : ") :].strip()
        elif line.startswith("Texte :"):
            in_text = True
        elif line.startswith("Liens : "):
            in_text = False
            page["links"] = [part.strip() for part in line[len("Liens : ") :].split(" | ")]
        elif line.startswith("Boutons : "):
            in_text = False
            page["buttons"] = [part.strip() for part in line[len("Boutons : ") :].split(" | ")]
        elif in_text:
            body.append(line)
    page["text"] = "\n".join(body).strip()
    return page


class BrowserReadParams(BaseModel):
    """Lit l'onglet actif du navigateur."""


@register_tool
class BrowserRead(BaseTool[BrowserReadParams]):
    name = "browser_read"
    returns = "{title, url, text, links, buttons}"
    sdk = "browser.read"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 1.0
    description = (
        "Lit la page du navigateur (titre, texte, liens et boutons visibles), sans la "
        "mettre au premier plan. À utiliser avant de cliquer."
    )
    params = BrowserReadParams
    terminal = False

    async def run(self, args: BrowserReadParams) -> str:
        text = await _browser(self.ctx).read(self.ctx.config.browser.page_text_max_chars)
        _mark_web_read()
        self.ctx.publish_data(parse_page(text))
        return text


class BrowserClickParams(BaseModel):
    text: str = Field(min_length=1, description="Texte visible de l'élément à cliquer")
    nth: int = Field(default=1, ge=1, le=20, description="N-ième élément de ce texte (1 = premier)")


@register_tool
class BrowserClick(BaseTool[BrowserClickParams]):
    name = "browser_click"
    sdk = "browser.click"
    effect = Effect.ACT
    output = Trust.UNTRUSTED
    expected_s = 2.0
    description = "Clique un élément de la page du navigateur par son texte visible."
    params = BrowserClickParams
    terminal = True

    async def run(self, args: BrowserClickParams) -> str:
        guard = self.ctx.require_guard()
        browser = _browser(self.ctx)
        info = await browser.find(args.text, args.nth)
        decision = guard.check_ui_action(args.text)
        if decision.verdict is Verdict.ALLOW:
            decision = guard.check_download(str(info.get("href") or ""))
        await _approve(self.ctx, guard, decision, f"Cliquer « {args.text} »")
        return await browser.click(args.text, args.nth)


class BrowserTypeParams(BaseModel):
    field: str = Field(min_length=1, description="Libellé ou nom du champ (ex. « Rechercher »)")
    text: str = Field(min_length=1, max_length=2000, description="Texte à écrire dans le champ")


@register_tool
class BrowserType(BaseTool[BrowserTypeParams]):
    name = "browser_type"
    sdk = "browser.type"
    effect = Effect.CONSEQUENTIAL
    expected_s = 1.0
    description = "Écrit un texte dans un champ de la page du navigateur (par libellé)."
    params = BrowserTypeParams
    terminal = True

    async def run(self, args: BrowserTypeParams) -> str:
        guard = self.ctx.require_guard()
        decision = guard.check_typed_text(args.text)
        if decision.verdict is Verdict.DENY:
            raise GuardDenied(f"texte non écrit : {decision.reason} (« {decision.match} »)")
        return await _browser(self.ctx).type_text(args.field, args.text)


class BrowserBackParams(BaseModel):
    """Page précédente."""


@register_tool
class BrowserBack(BaseTool[BrowserBackParams]):
    name = "browser_back"
    sdk = "browser.back"
    effect = Effect.ACT
    expected_s = 1.0
    description = "Revient à la page précédente du navigateur."
    params = BrowserBackParams
    terminal = True

    async def run(self, args: BrowserBackParams) -> str:
        return await _browser(self.ctx).back()


class BrowserTabsParams(BaseModel):
    """Onglets ouverts."""


@register_tool
class BrowserTabs(BaseTool[BrowserTabsParams]):
    name = "browser_tabs"
    sdk = "browser.tabs"
    effect = Effect.READ
    output = Trust.UNTRUSTED
    expected_s = 0.5
    description = "Liste les onglets ouverts du navigateur (titre et URL)."
    params = BrowserTabsParams
    terminal = False

    async def run(self, args: BrowserTabsParams) -> str:
        return await _browser(self.ctx).tabs()


class BrowserCloseParams(BaseModel):
    """Fermeture du navigateur de travail."""


@register_tool
class BrowserClose(BaseTool[BrowserCloseParams]):
    name = "browser_close"
    sdk = "browser.close"
    effect = Effect.ACT
    expected_s = 0.5
    description = (
        "Ferme complètement le navigateur de travail de MIKATA : sa fenêtre et sa lecture "
        "en cours s'arrêtent (jamais le navigateur de l'utilisateur)."
    )
    params = BrowserCloseParams
    terminal = True

    async def run(self, args: BrowserCloseParams) -> str:
        return await _browser(self.ctx).quit()
