"""Contrôleur du navigateur (phase 12b) : Brave piloté par le protocole DevTools.

Stratégie d'attache (jamais déranger l'utilisateur) :
1. le port DevTools répond -> on s'y attache ;
2. Brave n'est pas ouvert -> on le lance avec `--remote-debugging-port` (profil normal,
   identifiants et favoris conservés) ;
3. Brave est déjà ouvert **sans** le port -> impossible de s'attacher (Chromium n'active
   le port qu'au lancement) : erreur en français qui propose la marche à suivre, jamais de
   fermeture automatique du navigateur.

Le JavaScript envoyé au navigateur est fixe, écrit ici, jamais fourni par le LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.browser.cdp import (
    CdpClient,
    CdpClosed,
    CdpError,
    Target,
    close_target,
    list_targets,
)
from mikata.config import Config
from mikata.tools.base import Desktop, ToolError

# Placeholders des arguments dans le JavaScript fixe : des noms impossibles à retrouver
# dans un texte utilisateur (un « ARG2 » dans un champ de recherche ne doit rien casser).
ARG_TEXT = "__MIKATA_ARG_TEXT__"
ARG_NTH = "__MIKATA_ARG_NTH__"
ARG_FIELD = "__MIKATA_ARG_FIELD__"

# Lecture structurée de la page : titre, URL, texte, liens et boutons visibles. Phase 15h :
# les liens du contenu d'abord, ceux des menus (barre latérale de YouTube...) en dernier,
# sans doublon : avant, les 60 premiers liens d'une page de résultats YouTube étaient le
# menu, et aucun titre de vidéo n'y figurait.
_READ_JS = """(() => {
  const text = document.body ? document.body.innerText.slice(0, 8000) : "";
  const menu = "nav, header, footer, aside, [role=navigation], #guide, ytd-guide-renderer, "
    + "ytd-mini-guide-renderer, tp-yt-app-drawer";
  const anchors = [...document.querySelectorAll("a[href]")];
  const inMenu = a => a.closest(menu) !== null;
  const ordered = [...anchors.filter(a => !inMenu(a)), ...anchors.filter(inMenu)];
  const seen = new Set();
  const links = ordered
    .map(a => (a.innerText || a.getAttribute("title") || "").trim().replace(/\\s+/g, " "))
    .filter(t => t && t.length < 200 && !seen.has(t) && seen.add(t))
    .slice(0, 80);
  const buttons = [...document.querySelectorAll("button, [role=button], input[type=submit]")]
    .map(b => (b.innerText || b.value || b.getAttribute("aria-label") || "").trim())
    .filter(t => t && t.length < 120)
    .slice(0, 60);
  return JSON.stringify({ title: document.title, url: location.href, text, links, buttons });
})()"""

# Le n-ième élément dont le texte visible contient le texte cherché : libellé et lien,
# sans cliquer.
_FIND_JS = """(() => {
  const needle = __MIKATA_ARG_TEXT__.toLowerCase();
  const candidates = [...document.querySelectorAll(
    "a, button, [role=button], [role=link], input[type=submit], summary")];
  const labelOf = el => ((el.innerText || el.value || el.getAttribute("aria-label") || "")
    + " " + (el.title || "")).trim();
  const matches = candidates.filter(el => labelOf(el).toLowerCase().includes(needle));
  const el = matches[__MIKATA_ARG_NTH__ - 1];
  if (!el) {
    return JSON.stringify({ ok: false, found: matches.length,
      candidates: matches.slice(0, 8).map(labelOf).filter(t => t) });
  }
  return JSON.stringify({ ok: true, label: labelOf(el).slice(0, 80), tag: el.tagName,
    href: el.href || "" });
})()"""

# Clique le n-ième élément dont le texte visible contient le texte cherché (sans la souris).
_CLICK_JS = """(() => {
  const needle = __MIKATA_ARG_TEXT__.toLowerCase();
  const labelOf = el => ((el.innerText || el.value || el.getAttribute("aria-label") || "")
    + " " + (el.title || "")).trim();
  const el = [...document.querySelectorAll(
    "a, button, [role=button], [role=link], input[type=submit], summary")]
    .filter(el => labelOf(el).toLowerCase().includes(needle))[__MIKATA_ARG_NTH__ - 1];
  if (!el) return JSON.stringify({ ok: false });
  el.scrollIntoView({ block: "center" });
  el.click();
  return JSON.stringify({ ok: true });
})()"""

# Écrit le texte dans le champ dont le libellé contient le nom donné (React compris).
_TYPE_JS = """(() => {
  const needle = __MIKATA_ARG_FIELD__.toLowerCase();
  const fields = [...document.querySelectorAll(
    "input[type=text], input[type=search], input:not([type]), textarea, [role=textbox], "
    + "[contenteditable=true]")];
  const labelOf = el => (el.getAttribute("placeholder") || el.getAttribute("aria-label")
    || el.getAttribute("name") || el.id || "").trim();
  const field = fields.find(el => labelOf(el).toLowerCase().includes(needle)) || fields[0];
  if (!field) return JSON.stringify({ ok: false });
  const text = __MIKATA_ARG_TEXT__;
  const proto = field instanceof HTMLTextAreaElement
    ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value");
  if (setter && setter.set && field.tagName !== "DIV") setter.set.call(field, text);
  else field.textContent = text;
  field.dispatchEvent(new Event("input", { bubbles: true }));
  field.dispatchEvent(new Event("change", { bubbles: true }));
  return JSON.stringify({ ok: true, tag: field.tagName });
})()"""

_BACK_JS = "history.back(); true"
# Met en pause tout ce qui joue dans la page (vidéos, musiques) ; rend le nombre arrêté.
# Pause / reprise de ce qui joue dans la page (phase 15g). La pause marque ce qu'elle arrête
# (data-mikata-paused) : « Reprendre » ne relance que ça, jamais une vidéo mise en pause à
# la main. Rendent le nombre de médias touchés.
_PAUSE_JS = (
    "(() => { let n = 0; document.querySelectorAll('video, audio').forEach(m => "
    "{ if (!m.paused) { m.pause(); m.dataset.mikataPaused = '1'; n++; } }); return n; })()"
)
_RESUME_JS = (
    "(() => { let n = 0; document.querySelectorAll('video, audio').forEach(m => "
    "{ if (m.dataset.mikataPaused === '1') { delete m.dataset.mikataPaused; m.play(); n++; } "
    "}); return n; })()"
)


def _js(snippet: str, **args: str | int) -> str:
    placeholders = {
        "text": ARG_TEXT,
        "nth": ARG_NTH,
        "field": ARG_FIELD,
    }
    for name, value in args.items():
        snippet = snippet.replace(placeholders[name], json.dumps(value), 1)
    return snippet


_WS_CAPTION = 0x00C00000
_TECHNICAL_CLASSES = ("IME", "MSCTFIME UI")


def _work_windows(user32: Any, pids: set[int]) -> list[tuple[int, str]]:
    """Fenêtres des processus du profil de travail : (handle, « navigateur » | « technique »
    | « autre »). Navigateur : classe Chrome_WidgetWin_1 avec barre de titre."""
    import ctypes

    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd: int, _: Any) -> bool:
        hwnds.append(hwnd)
        return True

    found: list[tuple[int, str]] = []
    try:
        user32.EnumWindows(visit, 0)
        for hwnd in hwnds:
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value not in pids:
                continue
            name = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, name, 64)
            style = user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
            if name.value in _TECHNICAL_CLASSES:
                found.append((hwnd, "technique"))
            elif name.value == "Chrome_WidgetWin_1" and style & _WS_CAPTION == _WS_CAPTION:
                found.append((hwnd, "navigateur"))
            else:
                found.append((hwnd, "autre"))
    except OSError:
        return found
    return found


class BrowserController:
    """Implémente `tools.base.BrowserControl`. Connexion paresseuse : le port DevTools
    n'est cherché qu'à la première utilisation."""

    def __init__(self, config: Config, desktop: Desktop) -> None:
        self._config = config
        self._desktop = desktop
        self._client: CdpClient | None = None
        self._work_pids: set[int] | None = None  # cache des PID du profil de travail
        self._work_pids_at = 0.0
        self.visible = False  # phase 15a : fenêtre de travail montrée à la demande
        self.media_paused = False  # phase 15g : la lecture a été mise en pause d'ici

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- attache ----------------------------------------------------------------------

    def _brave_exe(self) -> str | None:
        roots = [
            Path(os.environ.get("LOCALAPPDATA", "")),
            Path(os.environ.get("PROGRAMFILES", "")),
            Path(os.environ.get("PROGRAMFILES(X86)", "")),
        ]
        for root in roots:
            exe = root / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe"
            if exe.is_file():
                return str(exe)
        return None

    def _browser_running(self) -> bool:
        """Un navigateur Chromium est-il déjà ouvert ? Le processus de Brave s'appelle
        « chrome » (moteur Chromium) : dans le doute on refuse poliment plutôt que de
        lancer une instance qui s'effacerait devant l'instance existante."""
        for window in self._desktop.windows():
            if window.process.casefold() in ("brave", "chrome"):
                return True
        return False

    async def _launch_brave(self) -> None:
        exe = self._brave_exe()
        if exe is None:
            raise ToolError("Brave introuvable : installe-le ou lance-le une fois")
        cfg = self._config.browser
        # Pas de « --remote-allow-origins=* » : ce drapeau laisserait n'importe quelle page
        # web ouverte sur le PC se connecter au port de contrôle. Le client de MIKATA
        # n'envoie pas d'en-tête Origin, donc Chromium l'accepte sans ce drapeau.
        argv = [exe, f"--remote-debugging-port={cfg.cdp_port}"]
        startupinfo = None
        if cfg.own_profile:
            # Phase 12f : profil de travail dédié, fenêtre cachée/minimisée en arrière-plan
            # — le navigateur de l'utilisateur n'est jamais touché. Si ce profil tourne
            # déjà, le lancement est un no-op (verrou singleton de Chromium) et le port
            # répond déjà.
            profile = self._config.resolve(cfg.profile_dir)
            profile.mkdir(parents=True, exist_ok=True)
            argv.append(f"--user-data-dir={profile}")
            # Les clics de MIKATA passent par du JavaScript (pas un « geste utilisateur ») :
            # sans ce drapeau, Chromium mettrait les vidéos en pause sur un profil neuf.
            argv.append("--autoplay-policy=no-user-gesture-required")
            # Une fenêtre cachée/occluse est mise en pause par Chromium (chargement
            # arrêté, vidéo jamais lancée) : on désactive cet étranglement.
            argv.extend(
                [
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=CalculateNativeWinOcclusion",
                ]
            )
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                # Toujours cachée à la création : Brave ignore SW_MINIMIZE et ferait un
                # flash « ouverte en grand puis minimisée ». Cachée -> minimisée (au
                # premier _minimize_work_window, selon browser.window_hidden) ne flashe
                # jamais.
                startupinfo.wShowWindow = 0
            if cfg.window_hidden:
                # Parade définitive contre le flash : même si Brave ignore SW_HIDE (ou
                # restaure la session), la fenêtre naît hors écran — physiquement
                # invisible. Seule la fenêtre du profil de travail est concernée.
                argv.append("--window-position=-32000,-32000")
            if self._work_window_pids():
                # Le profil tourne déjà : un nouveau lancement serait repris par le
                # verrou singleton de Chromium, qui RESTAURE la fenêtre existante en
                # plein écran avant qu'on la re-cache — le flash. On n'attend que le
                # port : l'instance existante va le servir.
                logger.info("Profil de travail déjà lancé : attente du port {}.", cfg.cdp_port)
                return
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=0x08000000 if sys.platform == "win32" else 0,  # sans console
        )
        self._work_pids = None  # la liste des fenêtres change au prochain appel
        label = (
            "profil de travail dédié (fenêtre minimisée)" if cfg.own_profile else "profil normal"
        )
        logger.info("Brave lancé avec le port de contrôle {} ({}).", cfg.cdp_port, label)

    @property
    def running(self) -> bool:
        """Le navigateur de travail tourne (profil dédié, même fenêtre cachée)."""
        if not self._config.browser.own_profile:
            return False
        return bool(self._work_window_pids())

    async def prewarm(self) -> None:
        """Lance le profil de travail en arrière-plan dès le démarrage de MIKATA : la
        première demande n'attend plus le démarrage à froid (~150 Mo au repos, ~0 % CPU).
        No-op si le profil tourne déjà."""
        if not self._config.browser.own_profile or self._work_window_pids():
            return
        await self._launch_brave()

    def keep_hidden(self) -> None:
        """Remet aussitôt la fenêtre de travail en arrière-plan : un focus_window du code
        généré peut la montrer — on la re-cache dans la foulée (jamais le Brave de
        l'utilisateur)."""
        self._minimize_work_window()

    def is_work_window(self, handle: int) -> bool:
        """La fenêtre `handle` appartient au profil de travail ? Elle ne doit alors
        jamais passer au premier plan (le focus serait un flash avant le re-cache)."""
        if not self._config.browser.own_profile or sys.platform != "win32" or not handle:
            return False
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return pid.value in self._work_window_pids()

    def _work_window_pids(self) -> set[int]:
        """PID des processus du profil de travail (cache court : la liste bouge peu).

        Seule la ligne de commande distingue le Brave de MIKATA de celui de l'utilisateur :
        le profil de travail porte `--user-data-dir=<profile_dir>`.
        """
        if self._work_pids is not None and time.monotonic() - self._work_pids_at < 30.0:
            return self._work_pids
        import psutil

        profile = str(self._config.resolve(self._config.browser.profile_dir)).casefold()
        pids: set[int] = set()
        try:
            for proc in psutil.process_iter(["pid"]):
                try:
                    if any(profile in arg.casefold() for arg in proc.cmdline()):
                        pids.add(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        except OSError:
            return self._work_pids or pids
        self._work_pids, self._work_pids_at = pids, time.monotonic()
        return pids

    def _minimize_work_window(self) -> None:
        """Remet la fenêtre du profil de travail en arrière-plan (best effort, Windows).

        Le drapeau de lancement n'agit qu'à la création : si la fenêtre a été montrée à la
        main (connexion à un compte...), on la re-cache (SW_HIDE) ou la re-minimise
        (selon `browser.window_hidden`) à chaque commande navigateur. Seules les fenêtres
        des processus du profil de travail sont touchées : celles du Brave de
        l'utilisateur ne le sont jamais.
        """
        if sys.platform != "win32" or self.visible:
            return  # montrée à la demande de l'utilisateur : on la laisse
        import ctypes

        pids = self._work_window_pids()
        if not pids:
            return
        show = 0 if self._config.browser.window_hidden else 6  # SW_HIDE / SW_MINIMIZE
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnds: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def visit(hwnd: int, _: Any) -> bool:
            hwnds.append(hwnd)
            return True

        try:
            user32.EnumWindows(visit, 0)
            for hwnd in hwnds:
                if not user32.IsWindowVisible(hwnd):
                    continue
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value in pids:
                    user32.ShowWindow(hwnd, show)
        except OSError:
            return  # l'état de la fenêtre est un plus, jamais une erreur

    async def _list_targets(self) -> list[Target]:
        try:
            return await list_targets(self._config.browser.cdp_port)
        except CdpError:
            return []

    def _unattached_hint(self) -> str:
        hint = "ferme complètement Brave et redemande (MIKATA le rouvrira avec le contrôle)"
        if self._config.browser.fallback_uia:
            hint += ", ou lis la fenêtre avec uia_window / uia_read"
        return hint

    async def _wait_targets(self) -> list[Target]:
        """Le port du navigateur qui démarre : jusqu'à ~10 s."""
        for _ in range(20):
            targets = await self._list_targets()
            if targets:
                return targets
            await asyncio.sleep(0.5)
        return []

    async def _ensure(self) -> CdpClient:
        if self._config.browser.own_profile:
            # À chaque commande : la fenêtre de travail reste en arrière-plan, même si
            # elle a été restaurée à la main entre deux commandes.
            self._minimize_work_window()
        if self._client is not None:
            return self._client
        cfg = self._config.browser
        last_error = ""
        for _attempt in range(2):
            targets = await self._list_targets()
            if not targets:
                if cfg.own_profile:
                    # Le port ne répond pas : lancer (ou rejoindre) le profil de travail.
                    await self._launch_brave()
                    self._minimize_work_window()  # la fenêtre vient d'apparaître
                    targets = await self._wait_targets()
                elif self._browser_running():
                    raise ToolError(
                        f"Brave est ouvert sans son port de contrôle ({cfg.cdp_port}) : "
                        + self._unattached_hint()
                    )
                else:
                    await self._launch_brave()
                    targets = await self._wait_targets()
            if not targets:
                raise ToolError(
                    f"le port de contrôle ({cfg.cdp_port}) ne répond pas : "
                    + self._unattached_hint()
                )
            # Premier onglet « page » du navigateur : limite connue, plusieurs fenêtres
            # ouvertes ne sont pas distinguées (v1 : l'onglet visé est celui de Brave).
            pages = [target for target in targets if target.kind == "page"] or targets
            client = CdpClient(pages[0].ws_url, timeout_s=cfg.click_timeout_s)
            try:
                await client.connect()
            except Exception as exc:
                # Onglet qui se ferme au lancement, navigateur en train de mourir : on
                # jette cette cible et on reprend tout depuis le début.
                last_error = str(exc)
                await client.close()
                continue
            self._client = client
            if cfg.own_profile:
                self._minimize_work_window()
            return client
        raise ToolError(f"impossible de piloter le navigateur ({last_error})")

    # -- primitives (interface BrowserControl) ----------------------------------------

    async def _lost(self, exc: Exception) -> ToolError:
        """La connexion au navigateur est morte : on jette le client (le prochain appel
        se rattachera ou relancera Brave) et on l'explique en français."""
        if isinstance(exc, CdpClosed):
            client = self._client
            self._client = None
            if client is not None:
                await client.close()
            return ToolError("le navigateur a été fermé : redemande, MIKATA le rouvrira")
        return ToolError(str(exc))

    async def _page_json(self, client: CdpClient, expression: str) -> dict[str, Any]:
        """Évalue du JavaScript fixe qui renvoie du JSON, en erreur française propre."""
        try:
            data = await client.evaluate(expression)
        except CdpError as exc:
            raise await self._lost(exc) from exc
        if not isinstance(data, str):
            raise ToolError("la page ne se laisse pas lire")
        return json.loads(data)

    async def tabs(self) -> str:
        cfg = self._config.browser
        try:
            targets = await list_targets(cfg.cdp_port)
        except CdpError as exc:
            raise await self._lost(exc) from exc
        pages = [t for t in targets if t.kind == "page"]
        if not pages:
            return "Aucun onglet ouvert."
        lines = [f"{len(pages)} onglet(s) :"]
        lines.extend(f"- {t.title[:80] or '(sans titre)'} — {t.url}" for t in pages)
        return "\n".join(lines)

    async def open_url(self, url: str) -> str:
        client = await self._ensure()
        try:
            await client.navigate(url)
        except CdpError as exc:
            raise await self._lost(exc) from exc
        return f"Page ouverte : {url}"

    async def read(self, max_chars: int) -> str:
        client = await self._ensure()
        page = await self._page_json(client, _READ_JS)
        lines = [f"Titre : {page.get('title', '')}", f"URL : {page.get('url', '')}"]
        text = str(page.get("text", "")).strip()
        if text:
            if len(text) > max_chars:
                text = text[:max_chars] + " […tronqué]"
            lines.append(f"Texte :\n{text}")
        links = page.get("links") or []
        if links:
            lines.append("Liens : " + " | ".join(links[:60]))
        buttons = page.get("buttons") or []
        if buttons:
            lines.append("Boutons : " + " | ".join(buttons[:40]))
        return "\n".join(lines)

    async def find(self, text: str, nth: int) -> dict[str, Any]:
        """Libellé et lien du n-ième élément « text », sans cliquer (vérifications avant)."""
        client = await self._ensure()
        result = await self._page_json(client, _js(_FIND_JS, text=text, nth=nth))
        if not result.get("ok"):
            found = result.get("found", 0)
            close = result.get("candidates") or []
            detail = f" (proches : {', '.join(close[:5])})" if close else ""
            raise ToolError(f"élément « {text} » introuvable ({found} ressemblant(s){detail})")
        return result

    async def click(self, text: str, nth: int) -> str:
        client = await self._ensure()
        try:
            before = await client.evaluate("location.href")
        except CdpError:
            before = None
        result = await self._page_json(client, _js(_CLICK_JS, text=text, nth=nth))
        if not result.get("ok"):
            raise ToolError(f"élément « {text} » introuvable au moment du clic")
        try:
            after = await client.evaluate("location.href")
        except CdpError:
            after = None
        if before != after:
            # Le clic a navigué : on attend la stabilité de la nouvelle page.
            await client.wait_ready(5.0)
        return f"« {text} » cliqué."

    async def type_text(self, field: str, text: str) -> str:
        client = await self._ensure()
        result = await self._page_json(client, _js(_TYPE_JS, field=field, text=text))
        if not result.get("ok"):
            raise ToolError(f"champ « {field} » introuvable dans la page")
        return f"Texte écrit dans le champ « {field} »."

    async def back(self) -> str:
        client = await self._ensure()
        try:
            await client.evaluate(_BACK_JS)
        except CdpError as exc:
            raise await self._lost(exc) from exc
        return "Page précédente."

    # -- gestion depuis l'interface (phase 15a) ---------------------------------------

    async def list_tabs(self) -> list[dict[str, str]]:
        """Onglets du navigateur de travail : identifiant, titre, adresse. [] s'il dort."""
        if not self._config.browser.own_profile or not self._work_window_pids():
            return []
        try:
            targets = await list_targets(self._config.browser.cdp_port)
        except CdpError:
            return []
        return [
            {"id": t.id, "title": t.title or t.url, "url": t.url}
            for t in targets
            if t.kind == "page" and t.id
        ]

    async def close_tab(self, tab_id: str) -> str:
        """Ferme un onglet. Le dernier n'est pas fermé mais vidé (page blanche) : le
        navigateur de travail reste lancé, prêt pour la prochaine demande."""
        cfg = self._config.browser
        tabs = await self.list_tabs()
        target = next((tab for tab in tabs if tab["id"] == tab_id), None)
        if target is None:
            raise ToolError("onglet introuvable (déjà fermé ?)")
        if len(tabs) <= 1:
            targets = await list_targets(cfg.cdp_port)
            page = next(t for t in targets if t.id == tab_id)
            client = CdpClient(page.ws_url, timeout_s=cfg.click_timeout_s)
            try:
                await client.connect()
                await client.navigate("about:blank")
            except CdpError as exc:
                raise ToolError(f"onglet impossible à vider : {exc}") from exc
            finally:
                await client.close()
            return "Onglet vidé (le navigateur reste prêt)."
        try:
            await close_target(cfg.cdp_port, tab_id)
        except CdpError as exc:
            raise ToolError(str(exc)) from exc
        if self._client is not None:  # l'onglet suivi a peut-être disparu : on se rattachera
            await self._client.close()
            self._client = None
        return "Onglet fermé."

    async def _run_in_tabs(self, script: str) -> int:
        """Exécute un JavaScript fixe dans chaque onglet ; somme des nombres rendus."""
        cfg = self._config.browser
        if not self._work_window_pids():
            return 0
        try:
            targets = await list_targets(cfg.cdp_port)
        except CdpError:
            return 0
        total = 0
        for target in (t for t in targets if t.kind == "page"):
            client = CdpClient(target.ws_url, timeout_s=cfg.click_timeout_s)
            try:
                await client.connect()
                total += int(await client.evaluate(script) or 0)
            except (CdpError, OSError, ValueError, TypeError):
                continue
            finally:
                await client.close()
        return total

    async def pause_media(self) -> int:
        """Met en pause ce qui joue dans tous les onglets ; rend le nombre de médias arrêtés."""
        paused = await self._run_in_tabs(_PAUSE_JS)
        if paused:
            self.media_paused = True
        return paused

    async def toggle_media(self) -> str:
        """Bouton Pause / Reprendre : met en pause ce qui joue, sinon relance ce qui a été
        mis en pause d'ici. Rend « pause », « lecture » ou « rien »."""
        if await self.pause_media():
            return "pause"
        resumed = await self._run_in_tabs(_RESUME_JS)
        self.media_paused = False
        return "lecture" if resumed else "rien"

    def set_visible(self, visible: bool) -> None:
        """Montre la fenêtre de travail à l'écran (pour voir ce que fait MIKATA, se
        connecter à un compte...) ou la remet en arrière-plan.

        Seules les vraies fenêtres du navigateur (classe Chrome_WidgetWin_1, avec barre de
        titre) sont montrées. Phase 15g : la version d'avant montrait aussi les fenêtres
        techniques de Chromium (« Default IME », « MSCTFIME UI »), désactivées : posées
        par-dessus la page, elles avalaient tous les clics. On les remet en place ici."""
        self.visible = visible
        if not visible:
            self._minimize_work_window()
            return
        if sys.platform != "win32":
            return
        import ctypes

        pids = self._work_window_pids()
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        for hwnd, kind in _work_windows(user32, pids):
            if kind == "technique":
                user32.ShowWindow(hwnd, 0)  # SW_HIDE : invisibles d'ordinaire
            elif kind == "navigateur":
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                # Créée hors écran (--window-position=-32000) : on la ramène devant.
                user32.SetWindowPos(hwnd, 0, 120, 80, 1280, 820, 0x0040)  # SWP_SHOWWINDOW
                user32.SetForegroundWindow(hwnd)

    async def quit(self) -> str:
        """Ferme complètement le navigateur de travail : fenêtre et lecture comprises.

        Seuls les processus du profil dédié sont arrêtés (ligne de commande
        `--user-data-dir`) — le Brave de l'utilisateur n'est jamais touché.
        """
        import psutil

        profile = str(self._config.resolve(self._config.browser.profile_dir)).casefold()
        killed = 0
        try:
            for proc in psutil.process_iter(["pid"]):
                try:
                    if any(profile in arg.casefold() for arg in proc.cmdline()):
                        proc.kill()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        except OSError:
            pass
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._work_pids = None
        logger.info("Navigateur de travail fermé ({} processus).", killed)
        return "Navigateur de travail fermé."


# Réutilisé par les tests : le JavaScript de lecture est fixe.
__all__ = ["BrowserController", "Target"]
