"""Implémentation Windows/macOS de `tools.base.UiAutomation` (phase 12a).

`UiController` est le point d'entrée unique (même construction partout) : il choisit une
seule fois, à l'initialisation, le moteur de pilotage — `config.uia.backend`, décidé par la
plateforme (`mikata/plat.py`) :
- `UiaBackend` : arbre UIA Windows via `uiautomation` (pur Python, sur comtypes), importé
  paresseusement, dans `attach` uniquement. Toutes les méthodes sont bloquantes : les
  outils les appellent via `ToolContext.run_blocking` (le thread COM dédié aux outils).
- `AxBackend` : arbre AXUIElement macOS via `macuiauto` (import paresseux), mêmes
  signatures. Rôles AX documentés dans `AxBackend._AX_KINDS` (AXButton -> bouton,
  AXTextField -> champ...). Note : pour Chromium/Electron (Discord, VS Code...), préférez
  CDP quand c'est possible (arbre complet sans droits d'accessibilité) ; AX reste le
  repli natif.

Noms comparés sans casse ni accents (comme tools/apps.py) : « discorde » trouve « Discord »,
« envoie » trouve « Envoyer ».
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from loguru import logger
from rapidfuzz import fuzz

from mikata.config import Config
from mikata.tools.base import Desktop, ToolError, UiControlInfo, UiWindowInfo
from mikata.uia.client import UiaAccessibilityClient
from mikata.uia.state import AmbientState

# Conteneurs sans intérêt pour le LLM : pas listés par `controls`.
_SKIP_KINDS = {"fenêtre", "panneau", "groupe", "barre de titre", "document"}
# Un clic sans type précis ne vise que les contrôles réellement cliquables.
_CLICKABLE = {"bouton", "lien", "menu", "élément", "case", "onglet", "arbre", "image", "contrôle"}
_MAX_VISITED = 2000  # garde-fou contre les arbres énormes (navigateurs...)
_MATCH_RATIO = 80.0  # score rapidfuzz minimal pour qu'un nom de contrôle « corresponde »
_TITLE_RATIO = 55.0  # un titre de fenêtre court matche à un score plus faible
_NAME_MAX = 200  # caractères gardés d'un nom de contrôle


def fold(text: str) -> str:
    """Minuscules sans accents : « Écouter » et « ecouter » sont égaux."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + " […tronqué]"


class _BackendBase(ABC):
    """Partie commune aux deux moteurs : parcours d'arbre borné et état ambiant (salon
    vocal Discord). Le bureau sert à énumérer les fenêtres pour l'état ambiant ; chaque
    moteur apporte ses opérations de nœuds (`attach`, `children`, nom, type, texte,
    clic, champ)."""

    def __init__(self, desktop: Desktop) -> None:
        self._desktop = desktop

    @abstractmethod
    def attach(self, window: UiWindowInfo) -> Any:
        """Racine de l'arbre d'accessibilité de la fenêtre, par son moteur."""

    @abstractmethod
    def children(self, node: Any) -> list[Any]:
        """Enfants d'un nœud : `GetChildren` (UIA) ou `AXChildren` (AX)."""

    def walk(self, root: Any, max_nodes: int) -> list[Any]:
        """Parcours en profondeur, borné : une fenêtre géante ne bloque pas l'outil.
        Enfants lus par le moteur : `GetChildren` (UIA) ou `AXChildren` (AX)."""
        nodes: list[Any] = []
        stack = self.children(root)
        while stack and len(nodes) < max_nodes:
            node = stack.pop()
            nodes.append(node)
            with contextlib.suppress(Exception):
                stack.extend(self.children(node))
        return nodes

    def discord_voice(self, window_for_process: Callable[[str], UiWindowInfo]) -> str:
        """Présence vocale Discord : lue dans l'arbre d'accessibilité du client. Le salon
        connecté y est un lien dont le nom commence par « <salon> (salon vocal), membre1,
        membre2, durée de l'appel... » (structure constatée sur le vrai client, fenêtre
        Discord en français ; motif anglais accepté aussi). Ne lève jamais : un état
        illisible est vide, pas une erreur."""
        windows = [w for w in self._desktop.windows() if w.process and "discord" in fold(w.process)]
        if not windows:
            return ""
        try:
            root = self.attach(window_for_process(windows[0].process))
            nodes = self.walk(root, _MAX_VISITED)
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


class UiaBackend(_BackendBase):
    """Moteur Windows : arbre UIA via `uiautomation`, importé paresseusement (`attach`)."""

    # Nom de contrôle UIA -> nom français donné au LLM.
    _KINDS: ClassVar[dict[str, str]] = {
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

    def attach(self, window: UiWindowInfo) -> Any:
        """Contrôle racine de la fenêtre, par sa poignée native. `ControlFromHandle` +
        `GetTopLevelControl` est la seule voie fiable : `WindowControl(NativeWindowHandle=...)`
        ignore la poignée et renvoie la fenêtre au premier plan (constaté sur Discord)."""
        try:
            import uiautomation as auto
        except ImportError as exc:
            raise ToolError(
                "le pilotage fin des applications n'est pas installé (uiautomation)"
            ) from exc
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

    def children(self, node: Any) -> list[Any]:
        try:
            return list(node.GetChildren())
        except Exception:
            return []

    def node_name(self, node: Any) -> str:
        return (node.Name or "").strip()

    def node_kind(self, node: Any) -> str:
        return self._KINDS.get(node.ControlTypeName, "contrôle")

    @staticmethod
    def _pattern(node: Any, name: str) -> Any:
        """Pattern UIA `name` du contrôle, ou None s'il ne le supporte pas (ou erreur COM)."""
        try:
            return getattr(node, f"Get{name}Pattern")()
        except Exception:
            return None

    def control_text(self, node: Any) -> str:
        value = self._pattern(node, "Value")
        if value is not None:
            try:
                if value.Value:
                    return value.Value
            except Exception:
                pass
        text = self._pattern(node, "Text")
        if text is not None:
            try:
                return text.DocumentRange.GetText(-1)
            except Exception:
                pass
        return node.Name or ""

    def document_text(self, root: Any) -> str:
        for node in [root, *self.walk(root, 200)]:
            if node.ControlTypeName == "DocumentControl":
                text = self._pattern(node, "Text")
                if text is not None:
                    try:
                        return text.DocumentRange.GetText(-1)
                    except Exception:
                        continue
        return ""

    def visible_text(self, root: Any) -> str:
        parts: list[str] = []
        for node in self.walk(root, _MAX_VISITED):
            if node.ControlTypeName == "TextControl" and node.Name:
                parts.append(node.Name.strip())
            elif node.ControlTypeName == "EditControl":
                value = self._pattern(node, "Value")
                if value is not None:
                    try:
                        if value.Value:
                            parts.append(value.Value.strip())
                    except Exception:
                        pass
            if sum(len(part) for part in parts) > 8000:
                break
        return "\n".join(parts)

    def click(self, node: Any, name: str) -> str:
        invoke = self._pattern(node, "Invoke")
        if invoke is not None:
            try:
                invoke.Invoke()
                return f"« {name} » cliqué."
            except Exception as exc:
                raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
        toggle = self._pattern(node, "Toggle")
        if toggle is not None:
            try:
                toggle.Toggle()
                return f"« {name} » basculé."
            except Exception as exc:
                raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
        raise ToolError(f"« {name} » ne se laisse pas cliquer (pas un bouton)")

    def set_text(self, node: Any, control: str, text: str) -> str:
        value = self._pattern(node, "Value")
        if value is None:
            raise ToolError(f"« {control} » n'est pas un champ modifiable")
        try:
            value.SetValue(text)
        except Exception as exc:
            raise ToolError(f"« {control} » n'a pas accepté le texte : {exc}") from exc
        return f"« {control} » rempli."


class AxBackend(_BackendBase):
    """Moteur macOS : arbre AXUIElement via `macuiauto`, importé paresseusement (`attach`)."""

    # Rôle AX macOS -> nom français donné au LLM (miroir de `UiaBackend._KINDS` Windows).
    _AX_KINDS: ClassVar[dict[str, str]] = {
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

    def attach(self, window: UiWindowInfo) -> Any:
        """Racine AX de la fenêtre (`macuiauto`, import paresseux) : par pid si la poignée
        porte un pid, sinon fenêtre au premier plan."""
        try:
            import macuiauto as ax  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ToolError(
                "le pilotage fin des applications n'est pas installé (macuiauto)"
            ) from exc
        try:
            if window.handle:
                try:
                    return ax.from_pid(int(window.handle))
                except (ValueError, TypeError):
                    pass
            return ax.focused_window()
        except Exception as exc:
            raise ToolError(f"la fenêtre « {window.title} » ne se laisse pas lire : {exc}") from exc

    @staticmethod
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

    @staticmethod
    def _ax_role(node: Any) -> str:
        try:
            return str(node.get("AXRole") or "")
        except Exception:
            return str(getattr(node, "role", "") or "")

    @staticmethod
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

    def _ax_kind(self, role: str) -> str:
        return self._AX_KINDS.get(role, "contrôle")

    def children(self, node: Any) -> list[Any]:
        return self._ax_children(node)

    def node_name(self, node: Any) -> str:
        return self._ax_name(node)

    def node_kind(self, node: Any) -> str:
        return self._ax_kind(self._ax_role(node))

    def control_text(self, node: Any) -> str:
        return self._ax_name(node)

    def document_text(self, root: Any) -> str:
        for node in [root, *self.walk(root, 200)]:
            if self._ax_role(node) == "AXWebArea":
                text = self._ax_name(node)
                if text:
                    return text
        return ""

    def visible_text(self, root: Any) -> str:
        parts: list[str] = []
        for node in self.walk(root, _MAX_VISITED):
            role = self._ax_role(node)
            if role in ("AXStaticText", "AXTextField", "AXTextArea"):
                name = self._ax_name(node)
                if name:
                    parts.append(name)
            if sum(len(part) for part in parts) > 8000:
                break
        return "\n".join(parts)

    def click(self, node: Any, name: str) -> str:
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

    def set_text(self, node: Any, control: str, text: str) -> str:
        try:
            node.set("AXValue", text)
        except Exception as exc:
            raise ToolError(f"« {control} » n'a pas accepté le texte : {exc}") from exc
        return f"« {control} » rempli."


class UiController:
    """Pilotage fin réel des applications (outils `uia_*`) : fenêtres trouvées par le
    bureau (`tools.desktop`), contrôles lus dans l'arbre d'accessibilité par le moteur
    choisi une fois à la construction (UIA Windows ou AX macOS). L'état ambiant (présence
    vocale Discord) est rafraîchi en arrière-plan par un thread dédié, jamais loggé."""

    def __init__(self, config: Config, desktop: Desktop) -> None:
        # Choix unique du moteur, à la construction, par la plateforme (mikata/plat.py,
        # via la config) : plus aucune branche sys.platform ni test de type de nœud
        # dans les méthodes publiques.
        self._backend = UiaBackend(desktop) if config.uia.backend == "uia" else AxBackend(desktop)
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
            for win in windows:  # le nom cherché est dans le titre, ou c'est le processus
                if wanted and (wanted in fold(win.title) or wanted == fold(win.process)):
                    return UiWindowInfo(win.title, win.process, win.handle)
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

    # -- contrôles --------------------------------------------------------------------

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        root = self._backend.attach(window)
        found: list[UiControlInfo] = []
        for node in self._backend.walk(root, _MAX_VISITED):
            name = self._backend.node_name(node)
            if not name:
                continue
            kind = self._backend.node_kind(node)
            if kind in _SKIP_KINDS:
                continue
            found.append(UiControlInfo(name=name[:_NAME_MAX], kind=kind))
            if len(found) >= max_controls:
                break
        return found

    def _find(
        self,
        window: UiWindowInfo,
        control: str,
        kind: str | None,
        nth: int,
        clickable: bool = False,
    ) -> Any:
        root = self._backend.attach(window)
        wanted = fold(control)
        candidates: list[Any] = []
        seen: dict[str, float] = {}  # noms rencontrés -> ressemblance, pour l'erreur
        for node in self._backend.walk(root, _MAX_VISITED):
            name = self._backend.node_name(node)
            if not name:
                continue
            if len(seen) < 2000:
                seen.setdefault(name[:_NAME_MAX], fuzz.token_set_ratio(wanted, fold(name)))
            node_kind = self._backend.node_kind(node)
            if kind is not None and fold(kind) != fold(node_kind):
                continue
            if clickable and kind is None and node_kind not in _CLICKABLE:
                continue
            if wanted == fold(name) or fuzz.ratio(wanted, fold(name)) >= _MATCH_RATIO:
                candidates.append(node)
        if not candidates:
            where = f" (type « {kind} »)" if kind else ""
            close = [n for n, score in sorted(seen.items(), key=lambda kv: -kv[1]) if score >= 40]
            hint = f" ; noms proches : {', '.join(f'« {n} »' for n in close[:6])}" if close else ""
            raise ToolError(f"contrôle introuvable : « {control} »{where}{hint}")
        if nth > len(candidates):
            raise ToolError(f"« {control} » : {len(candidates)} occurrence(s), pas la {nth}e")
        return candidates[nth - 1]

    # -- lecture ----------------------------------------------------------------------

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        root = self._backend.attach(window)
        if control is not None:
            node = self._find(window, control, None, 1)
            text = self._backend.control_text(node)
        else:
            text = self._backend.document_text(root) or self._backend.visible_text(root)
        if not text.strip():
            raise ToolError("aucun texte lisible dans cette fenêtre")
        return _truncate(text, max_chars)

    # -- actions ----------------------------------------------------------------------

    def click(self, window: UiWindowInfo, control: str, kind: str | None, nth: int) -> str:
        node = self._find(window, control, kind, nth, clickable=True)
        name = self._backend.node_name(node) or "(sans nom)"
        return self._backend.click(node, name)

    def set_text(self, window: UiWindowInfo, control: str, text: str) -> str:
        node = self._find(window, control, "champ", 1)
        return self._backend.set_text(node, control, text)

    # -- état ambiant ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, str]:
        data: dict[str, str] = {}
        voice = self._backend.discord_voice(lambda process: self.window(None, process))
        if voice:
            data["discord"] = voice
        return data


def _discord_voice_from(nodes: list[Any]) -> tuple[str | None, list[str]]:
    """(salon vocal connecté, membres) depuis un arbre de Discord (nœuds UIA ou AX), ou
    (None, []).

    Le lien du salon connecté a la forme « <salon> (salon vocal), <membre>, <membre>,
    durée de l'appel... » (client français ; « voice channel » en anglais). Les salons non
    connectés sont des boutons « <nom> (salon vocal) » sans membres : ignorés.
    """
    for node in nodes:
        if getattr(node, "ControlTypeName", None) is not None:
            is_link = node.ControlTypeName == "HyperlinkControl"
            name = (node.Name or "").strip()
        else:
            is_link = AxBackend._ax_role(node) == "AXLink"
            name = AxBackend._ax_name(node)
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
