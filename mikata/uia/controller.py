"""Implémentation Windows/macOS de `tools.base.UiAutomation` (phase 12a).

- win32 : bibliothèque `uiautomation` (pur Python, sur comtypes), importée ici,
  paresseusement. Toutes les méthodes sont bloquantes : les outils les appellent
  via `ToolContext.run_blocking` (le thread COM dédié aux outils).
- darwin : backend AXUIElement via `macuiauto` (import paresseux), mêmes signatures.
  Rôles AX documentés dans `_AX_KINDS` (AXButton -> bouton, AXTextField -> champ...).
  Note : pour Chromium/Electron (Discord, VS Code...), préférez CDP quand c'est
  possible (arbre complet sans droits d'accessibilité) ; AX reste le repli natif.

Noms comparés sans casse ni accents (comme tools/apps.py) : « discorde » trouve « Discord »,
« envoie » trouve « Envoyer ».
"""

from __future__ import annotations

import contextlib
import re
import sys
import unicodedata
from typing import Any

from loguru import logger
from rapidfuzz import fuzz

from mikata.config import Config
from mikata.tools.base import Desktop, ToolError, UiControlInfo, UiWindowInfo
from mikata.uia.client import UiaAccessibilityClient
from mikata.uia.state import AmbientState

# Nom de contrôle UIA -> nom français donné au LLM.
_KINDS = {
    "ButtonControl": "bouton",
    "EditControl": "champ",
    "TextControl": "texte",
    "DocumentControl": "document",
    "ListItemControl": "élément",
    "ListControl": "liste",
    "CheckBoxControl": "case",
    "RadioButtonControl": "case",
    "HyperlinkControl": "lien",
    "MenuItemControl": "menu",
    "TabItemControl": "onglet",
    "ComboBoxControl": "liste déroulante",
    "TreeItemControl": "arbre",
    "HeaderControl": "en-tête",
    "TitleBarControl": "barre de titre",
    "SliderControl": "curseur",
    "SpinnerControl": "spinner",
    "ProgressBarControl": "progression",
    "ToolBarControl": "barre d'outils",
    "ImageControl": "image",
    "CustomControl": "contrôle",
}  # fmt: skip
# Conteneurs sans intérêt pour le LLM : pas listés par `controls`.
_SKIP_KINDS = {"fenêtre", "panneau", "groupe", "barre de titre", "document"}
# Un clic sans type précis ne vise que les contrôles réellement cliquables.
_CLICKABLE = {"bouton", "lien", "menu", "élément", "case", "onglet", "arbre", "image", "contrôle"}
_MAX_VISITED = 2000  # garde-fou contre les arbres énormes (navigateurs...)
_MATCH_RATIO = 80.0  # score rapidfuzz minimal pour qu'un nom de contrôle « corresponde »
_TITLE_RATIO = 55.0  # un titre de fenêtre court matche à un score plus faible
_NAME_MAX = 200  # caractères gardés d'un nom de contrôle


def _module() -> Any:
    """Bibliothèque d'automatisation, importée à la première utilisation.

    win32 : `uiautomation` (comtypes) ; darwin : `macuiauto` (AXUIElement).
    """
    if sys.platform == "darwin":
        return _ax_module()
    try:
        import uiautomation
    except ImportError as exc:
        raise ToolError(
            "le pilotage fin des applications n'est pas installé (uiautomation)"
        ) from exc
    return uiautomation


def _ax_module() -> Any:
    """Backend AXUIElement macOS (`macuiauto`), import paresseux."""
    try:
        import macuiauto  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ToolError("le pilotage fin des applications n'est pas installé (macuiauto)") from exc
    return macuiauto


# Rôle AX macOS -> nom français donné au LLM (miroir de `_KINDS` Windows).
_AX_KINDS = {
    "AXButton": "bouton",
    "AXTextField": "champ",
    "AXSecureTextField": "champ",
    "AXTextArea": "champ",
    "AXStaticText": "texte",
    "AXWebArea": "document",
    "AXRow": "élément",
    "AXList": "liste",
    "AXCheckBox": "case",
    "AXRadioButton": "case",
    "AXLink": "lien",
    "AXMenuItem": "menu",
    "AXTabGroup": "onglet",
    "AXComboBox": "liste déroulante",
    "AXOutlineRow": "arbre",
    "AXSlider": "curseur",
    "AXProgressIndicator": "progression",
    "AXToolbar": "barre d'outils",
    "AXImage": "image",
}  # fmt: skip


def _ax_kind(role: str) -> str:
    return _AX_KINDS.get(role, "contrôle")


def _ax_name(node: Any) -> str:
    for attr in ("AXTitle", "AXDescription", "AXValue"):
        try:
            value = node.get(attr)
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        return str(getattr(node, "title", "") or "").strip()
    except Exception:
        return ""


def _ax_role(node: Any) -> str:
    try:
        return str(node.get("AXRole") or "")
    except Exception:
        return str(getattr(node, "role", "") or "")


def _ax_children(node: Any) -> list[Any]:
    try:
        children = node.get("AXChildren") or []
        return list(children)
    except Exception:
        pass
    try:
        return list(node.children())
    except Exception:
        return []


def fold(text: str) -> str:
    """Minuscules sans accents : « Écouter » et « ecouter » sont égaux."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + " […tronqué]"


def _pattern(node: Any, name: str) -> Any:
    """Pattern UIA `name` du contrôle, ou None s'il ne le supporte pas (ou erreur COM)."""
    if sys.platform == "darwin":
        return None  # AX n'a pas de patterns UIA : actions via AXPress/AXSetValue
    try:
        return getattr(node, f"Get{name}Pattern")()
    except Exception:
        return None


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _is_uia_node(node: Any) -> bool:
    """Nœud `uiautomation` (Windows) plutôt que AX (`macuiauto`) ? Détection par attribut,
    pas par plateforme : les tests utilisent des sosies UIA sur n'importe quel OS."""
    try:
        return getattr(node, "ControlTypeName", None) is not None
    except Exception:
        return False


class UiController:
    """Pilotage UIA réel : fenêtres trouvées par le bureau Windows (tools.desktop),
    contrôles lus dans l'arbre d'accessibilité. L'état ambiant (présence vocale Discord)
    est rafraîchi en arrière-plan par un thread dédié, jamais loggé."""

    def __init__(self, config: Config, desktop: Desktop) -> None:
        self._desktop = desktop
        self._state = AmbientState(config.uia.state_poll_s, self._snapshot)
        self._state.start()
        self._client = UiaAccessibilityClient()
        if config.uia.accessibility_client:
            self._client.start()

    # -- état ambiant (cache, jamais bloquant) ----------------------------------------

    def pause_state(self, paused: bool) -> None:
        self._state.pause(paused)

    def state_summary(self) -> str:
        return self._state.summary()

    def aclose(self) -> None:
        self._client.stop()
        self._state.stop()

    # -- fenêtres ---------------------------------------------------------------------

    def window(self, title: str | None, process: str | None) -> UiWindowInfo:
        """Fenêtre par nom de processus, par titre, ou la fenêtre active sans argument."""
        windows = self._desktop.windows()
        if process is not None:
            wanted = fold(process)
            for win in windows:
                if wanted and wanted in fold(win.process):
                    return UiWindowInfo(win.title, win.process, win.handle)
            names = sorted({w.process for w in windows if w.process}) or ["(aucune)"]
            raise ToolError(
                f"aucune fenêtre du processus « {process} » "
                f"(processus ouverts : {', '.join(names)})"
            )
        if title is not None:
            wanted = fold(title)
            best, best_score = None, 0.0
            for win in windows:
                if not win.title:
                    continue
                score = fuzz.ratio(wanted, fold(win.title))
                if score > best_score:
                    best, best_score = win, score
            if best is not None and best_score >= _TITLE_RATIO:
                return UiWindowInfo(best.title, best.process, best.handle)
            titles = ", ".join(f"« {w.title} »" for w in windows if w.title) or "aucune"
            raise ToolError(f"aucune fenêtre ne ressemble à « {title} » (ouvertes : {titles})")
        win = self._desktop.foreground_window()
        if win is None:
            raise ToolError("aucune fenêtre au premier plan")
        return UiWindowInfo(win.title, win.process, win.handle)

    def _attach(self, window: UiWindowInfo) -> Any:
        """Contrôle racine de la fenêtre, par sa poignée native. `ControlFromHandle` +
        `GetTopLevelControl` est la seule voie fiable : `WindowControl(NativeWindowHandle=...)`
        ignore la poignée et renvoie la fenêtre au premier plan (constaté sur Discord).
        Sur darwin : racine AXUIElement de l'application (via `macuiauto`, par pid si
        la poignée porte un pid, sinon fenêtre au premier plan)."""
        if _is_darwin():
            return self._attach_ax(window)
        auto = _module()
        if not window.handle:
            raise ToolError(f"la fenêtre « {window.title} » n'a pas de poignée Windows")
        try:
            node = auto.ControlFromHandle(window.handle)
        except Exception as exc:
            raise ToolError(f"la fenêtre « {window.title} » ne se laisse pas lire : {exc}") from exc
        if node is None:
            raise ToolError(f"la fenêtre « {window.title} » ne se laisse pas lire")
        top = node.GetTopLevelControl()
        if top is None:
            raise ToolError(f"la fenêtre « {window.title} » ne se laisse pas lire")
        return top

    def _attach_ax(self, window: UiWindowInfo) -> Any:
        """Racine AX de la fenêtre (darwin, `macuiauto`, import paresseux)."""
        ax = _ax_module()
        try:
            if window.handle:
                try:
                    return ax.from_pid(int(window.handle))
                except (ValueError, TypeError):
                    pass
            return ax.focused_window()
        except Exception as exc:
            raise ToolError(f"la fenêtre « {window.title} » ne se laisse pas lire : {exc}") from exc

    def _walk(self, root: Any, max_nodes: int) -> list[Any]:
        """Parcours en profondeur, borné : une fenêtre géante ne bloque pas l'outil.
        Nœuds UIA : `GetChildren` ; nœuds AX (darwin) : `AXChildren`."""
        try:
            children = list(root.GetChildren())
        except Exception:
            children = list(_ax_children(root))
            get_children = _ax_children
        else:
            get_children = lambda node: node.GetChildren()  # noqa: E731
        nodes: list[Any] = []
        stack = children
        while stack and len(nodes) < max_nodes:
            node = stack.pop()
            nodes.append(node)
            with contextlib.suppress(Exception):
                stack.extend(get_children(node))
        return nodes

    # -- contrôles --------------------------------------------------------------------

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        root = self._attach(window)
        if not _is_uia_node(root):
            return self._controls_ax(root, max_controls)
        found: list[UiControlInfo] = []
        for node in self._walk(root, _MAX_VISITED):
            name = (node.Name or "").strip()
            if not name:
                continue
            kind = _KINDS.get(node.ControlTypeName, "contrôle")
            if kind in _SKIP_KINDS:
                continue
            found.append(UiControlInfo(name=name[:_NAME_MAX], kind=kind))
            if len(found) >= max_controls:
                break
        return found

    def _controls_ax(self, root: Any, max_controls: int) -> list[UiControlInfo]:
        found: list[UiControlInfo] = []
        for node in self._walk(root, _MAX_VISITED):
            name = _ax_name(node)
            if not name:
                continue
            kind = _ax_kind(_ax_role(node))
            if kind in _SKIP_KINDS:
                continue
            found.append(UiControlInfo(name=name[:_NAME_MAX], kind=kind))
            if len(found) >= max_controls:
                break
        return found

    def _node_name(self, node: Any) -> str:
        if not _is_uia_node(node):
            return _ax_name(node)
        return (node.Name or "").strip()

    def _node_kind(self, node: Any) -> str:
        if not _is_uia_node(node):
            return _ax_kind(_ax_role(node))
        return _KINDS.get(node.ControlTypeName, "contrôle")

    def _find(
        self,
        window: UiWindowInfo,
        control: str,
        kind: str | None,
        nth: int,
        clickable: bool = False,
    ) -> Any:
        root = self._attach(window)
        wanted = fold(control)
        candidates: list[Any] = []
        for node in self._walk(root, _MAX_VISITED):
            name = self._node_name(node)
            if not name:
                continue
            node_kind = self._node_kind(node)
            if kind is not None and fold(kind) != fold(node_kind):
                continue
            if clickable and kind is None and node_kind not in _CLICKABLE:
                continue
            if wanted == fold(name) or fuzz.ratio(wanted, fold(name)) >= _MATCH_RATIO:
                candidates.append(node)
        if not candidates:
            where = f" (type « {kind} »)" if kind else ""
            raise ToolError(f"contrôle introuvable : « {control} »{where}")
        if nth > len(candidates):
            raise ToolError(f"« {control} » : {len(candidates)} occurrence(s), pas la {nth}e")
        return candidates[nth - 1]

    # -- lecture ----------------------------------------------------------------------

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        root = self._attach(window)
        if control is not None:
            node = self._find(window, control, None, 1)
            text = _control_text(node)
        else:
            text = self._document_text(root) or self._visible_text(root)
        if not text.strip():
            raise ToolError("aucun texte lisible dans cette fenêtre")
        return _truncate(text, max_chars)

    def _document_text(self, root: Any) -> str:
        if not _is_uia_node(root):
            for node in [root, *self._walk(root, 200)]:
                if _ax_role(node) == "AXWebArea":
                    text = _ax_name(node)
                    if text:
                        return text
            return ""
        for node in [root, *self._walk(root, 200)]:
            if node.ControlTypeName == "DocumentControl":
                text = _pattern(node, "Text")
                if text is not None:
                    try:
                        return text.DocumentRange.GetText(-1)
                    except Exception:
                        continue
        return ""

    def _visible_text(self, root: Any) -> str:
        parts: list[str] = []
        ax = not _is_uia_node(root)
        for node in self._walk(root, _MAX_VISITED):
            if ax:
                role = _ax_role(node)
                if role in ("AXStaticText", "AXTextField", "AXTextArea"):
                    name = _ax_name(node)
                    if name:
                        parts.append(name)
            elif node.ControlTypeName == "TextControl" and node.Name:
                parts.append(node.Name.strip())
            elif node.ControlTypeName == "EditControl":
                value = _pattern(node, "Value")
                if value is not None:
                    try:
                        if value.Value:
                            parts.append(value.Value.strip())
                    except Exception:
                        pass
            if sum(len(part) for part in parts) > 8000:
                break
        return "\n".join(parts)

    # -- actions ----------------------------------------------------------------------

    def click(self, window: UiWindowInfo, control: str, kind: str | None, nth: int) -> str:
        node = self._find(window, control, kind, nth, clickable=True)
        name = self._node_name(node) or "(sans nom)"
        if not _is_uia_node(node):
            return self._click_ax(node, name)
        invoke = _pattern(node, "Invoke")
        if invoke is not None:
            try:
                invoke.Invoke()
                return f"« {name} » cliqué."
            except Exception as exc:
                raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
        toggle = _pattern(node, "Toggle")
        if toggle is not None:
            try:
                toggle.Toggle()
                return f"« {name} » basculé."
            except Exception as exc:
                raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
        raise ToolError(f"« {name} » ne se laisse pas cliquer (pas un bouton)")

    def _click_ax(self, node: Any, name: str) -> str:
        """Clic AX : action `AXPress` (boutons, cases, liens...)."""
        try:
            node.perform("AXPress")
            return f"« {name} » cliqué."
        except Exception as exc:
            try:
                press = node.get("AXPress")
                if press is not None:
                    return f"« {name} » cliqué."
            except Exception:
                pass
            raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc

    def set_text(self, window: UiWindowInfo, control: str, text: str) -> str:
        node = self._find(window, control, "champ", 1)
        if not _is_uia_node(node):
            try:
                node.set("AXValue", text)
            except Exception as exc:
                raise ToolError(f"« {control} » n'a pas accepté le texte : {exc}") from exc
            return f"« {control} » rempli."
        value = _pattern(node, "Value")
        if value is None:
            raise ToolError(f"« {control} » n'est pas un champ modifiable")
        try:
            value.SetValue(text)
        except Exception as exc:
            raise ToolError(f"« {control} » n'a pas accepté le texte : {exc}") from exc
        return f"« {control} » rempli."

    # -- état ambiant ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, str]:
        data: dict[str, str] = {}
        voice = self._discord_voice()
        if voice:
            data["discord"] = voice
        return data

    def _discord_voice(self) -> str:
        """Présence vocale Discord : lue dans l'arbre d'accessibilité du client. Le salon
        connecté y est un lien dont le nom commence par « <salon> (salon vocal), membre1,
        membre2, durée de l'appel... » (structure constatée sur le vrai client, fenêtre
        Discord en français ; motif anglais accepté aussi). Ne lève jamais : un état
        illisible est vide, pas une erreur."""
        windows = [w for w in self._desktop.windows() if w.process and "discord" in fold(w.process)]
        if not windows:
            return ""
        try:
            root = self._attach(self.window(None, windows[0].process))
            nodes = self._walk(root, _MAX_VISITED)
            channel, members = _discord_voice_from(nodes)
        except Exception as exc:
            logger.debug("État ambiant Discord illisible : {}", exc)
            return ""
        if channel is None:
            return ""
        text = f"Discord : salon vocal « {channel} »."
        if members:
            text += f" En vocal : {', '.join(members[:12])}."
        return text


def _control_text(node: Any) -> str:
    if not _is_uia_node(node):
        return _ax_name(node)
    value = _pattern(node, "Value")
    if value is not None:
        try:
            if value.Value:
                return value.Value
        except Exception:
            pass
    text = _pattern(node, "Text")
    if text is not None:
        try:
            return text.DocumentRange.GetText(-1)
        except Exception:
            pass
    return node.Name or ""


def _discord_voice_from(nodes: list[Any]) -> tuple[str | None, list[str]]:
    """(salon vocal connecté, membres) depuis un arbre UIA de Discord, ou (None, []).

    Le lien du salon connecté a la forme « <salon> (salon vocal), <membre>, <membre>,
    durée de l'appel... » (client français ; « voice channel » en anglais). Les salons non
    connectés sont des boutons « <nom> (salon vocal) » sans membres : ignorés.
    """
    for node in nodes:
        if _is_uia_node(node):
            is_link = node.ControlTypeName == "HyperlinkControl"
            name = (node.Name or "").strip()
        else:
            is_link = _ax_role(node) == "AXLink"
            name = _ax_name(node)
        if not is_link:
            continue
        folded = fold(name)
        if "(salon vocal)," not in folded and "(voice channel)," not in folded:
            continue
        parts = [part.strip() for part in name.split(",")]
        channel = re.sub(
            r"\s*\((salon vocal|voice channel)\)\s*$", "", parts[0], flags=re.IGNORECASE
        ).strip()
        members: list[str] = []
        for part in parts[1:]:
            flat = fold(part)
            if not part or flat.startswith("duree") or flat.startswith("duration"):
                break
            if any(fold(part) == fold(member) for member in members):
                continue
            members.append(part)
        if channel:
            return channel, members
    return None, []
