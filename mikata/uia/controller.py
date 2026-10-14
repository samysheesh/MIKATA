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
import sys
import time
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from loguru import logger
from rapidfuzz import fuzz

from mikata.config import Config
from mikata.tools.base import Desktop, ToolError, UiControlInfo, UiItemInfo, UiWindowInfo
from mikata.uia.client import UiaAccessibilityClient
from mikata.uia.state import AmbientState, load_uiautomation

# Conteneurs sans intérêt pour le LLM : pas listés par `controls`.
_SKIP_KINDS = {"fenêtre", "panneau", "groupe", "barre de titre", "document"}
# Un clic sans type précis ne vise que les contrôles réellement cliquables.
_CLICKABLE = {"bouton", "lien", "menu", "élément", "case", "onglet", "arbre", "image", "contrôle"}
_MAX_VISITED = 4000  # garde-fou contre les arbres énormes (Spotify : ~2000 nœuds, 14/10)
_MATCH_RATIO = 80.0  # score rapidfuzz minimal pour qu'un nom de contrôle « corresponde »
_TITLE_RATIO = 55.0  # un titre de fenêtre court matche à un score plus faible
_NAME_MAX = 200  # caractères gardés d'un nom de contrôle
_NEAR_LEVELS = 4  # « près de » : la ligne ou le groupe, jusqu'à 4 niveaux au-dessus
_NEAR_NODES = 80  # nœuds lus dans chacun de ces groupes
_ROW_NODES = 60  # nœuds lus dans un élément de liste (click_in)
_VALUE_KINDS = {"champ", "liste déroulante"}  # contrôles dont `controls` donne le contenu
_TOGGLE_KINDS = {"bouton", "case"}  # contrôles dont `controls` donne l'état d'interrupteur
_STATE = {True: "activé", False: "désactivé"}


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

    @abstractmethod
    def parent(self, node: Any) -> Any:
        """Parent du nœud, ou None (racine, nœud disparu)."""

    @abstractmethod
    def node_name(self, node: Any) -> str:
        """Nom accessible du nœud ("" s'il n'en a pas)."""

    @abstractmethod
    def node_kind(self, node: Any) -> str:
        """Type du nœud en français (« bouton », « champ »...) ; « contrôle » sinon."""

    @abstractmethod
    def control_text(self, node: Any) -> str:
        """Texte d'un contrôle (valeur d'un champ, texte d'un document, sinon son nom)."""

    @abstractmethod
    def document_text(self, root: Any) -> str:
        """Texte du document principal de la fenêtre ("" s'il n'y en a pas)."""

    @abstractmethod
    def visible_text(self, root: Any) -> str:
        """Textes visibles de la fenêtre, bornés."""

    @abstractmethod
    def click(self, node: Any, name: str) -> str:
        """Clique le nœud ; lève ToolError s'il ne réagit pas."""

    def toggle_state(self, node: Any) -> bool | None:
        """Phase 17d : un interrupteur (micro coupé, case cochée) est-il activé ? None si
        le nœud n'en est pas un. Son nom ne change pas (« Rendre muet » reste « Rendre
        muet ») : sans son état, MIKATA recliquait pour « vérifier » et défaisait l'action."""
        return None

    @abstractmethod
    def set_text(self, node: Any, control: str, text: str) -> str:
        """Remplit un champ ; lève ToolError s'il refuse le texte."""

    def walk(self, root: Any, max_nodes: int) -> list[Any]:
        """Parcours en profondeur, borné : une fenêtre géante ne bloque pas l'outil.
        Enfants lus par le moteur : `GetChildren` (UIA) ou `AXChildren` (AX). Dans l'ordre
        de la fenêtre (phase 16g : la pile rendait les frères à l'envers, et le « premier »
        contrôle était le dernier affiché)."""
        return list(self.iter_walk(root, max_nodes))

    def iter_walk(self, root: Any, max_nodes: int) -> Iterator[Any]:
        """Le même parcours, au fil de l'eau : on s'arrête dès qu'on a trouvé."""
        count = 0
        stack = list(reversed(self.children(root)))
        while stack and count < max_nodes:
            node = stack.pop()
            count += 1
            yield node
            with contextlib.suppress(Exception):
                stack.extend(reversed(self.children(node)))

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
        "DataItemControl": "élément",  # phase 17a : lignes de grilles (résultats Spotify...)
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
            auto = load_uiautomation()
        except ModuleNotFoundError as exc:
            raise ToolError(
                "le pilotage fin des applications n'est pas installé (uiautomation)"
            ) from exc
        except ImportError as exc:  # cache comtypes illisible
            raise ToolError(f"le pilotage fin des applications a échoué : {exc}") from exc
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

    def parent(self, node: Any) -> Any:
        try:
            return node.GetParentControl()
        except Exception:
            return None

    @staticmethod
    def _prop(node: Any, name: str) -> str:
        """Propriété texte d'un nœud, "" s'il a disparu entre-temps : juste après un clic,
        l'application redessine sa fenêtre et un nœud parcouru n'existe plus (COMError
        0x80040201 relevée sur Discord) ; il est sauté, la lecture continue."""
        try:
            return str(getattr(node, name) or "")
        except Exception:
            return ""

    def node_name(self, node: Any) -> str:
        return self._prop(node, "Name").strip()

    def node_kind(self, node: Any) -> str:
        return self._KINDS.get(self._prop(node, "ControlTypeName"), "contrôle")

    @staticmethod
    def _pattern(node: Any, name: str) -> Any:
        """Pattern UIA `name` du contrôle, ou None s'il ne le supporte pas (ou erreur COM)."""
        try:
            return getattr(node, f"Get{name}Pattern")()
        except Exception:
            return None

    def toggle_state(self, node: Any) -> bool | None:
        toggle = self._pattern(node, "Toggle")
        if toggle is None:
            return None
        try:
            state = int(toggle.ToggleState)
        except Exception:
            return None
        return {0: False, 1: True}.get(state)

    def control_text(self, node: Any) -> str:
        state = self.toggle_state(node)
        if state is not None:
            return f"{self._prop(node, 'Name')} : {_STATE[state]}"
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
        return self._prop(node, "Name")

    def document_text(self, root: Any) -> str:
        for node in [root, *self.walk(root, 200)]:
            if self._prop(node, "ControlTypeName") == "DocumentControl":
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
            kind = self._prop(node, "ControlTypeName")
            name = self._prop(node, "Name").strip()
            if kind == "TextControl" and name:
                parts.append(name)
            elif kind == "EditControl":
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
            except Exception as exc:
                raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
            state = self.toggle_state(node)
            after = f" : maintenant {_STATE[state]}" if state is not None else ""
            return f"« {name} » basculé{after}."
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

    def parent(self, node: Any) -> Any:
        try:
            return node.get("AXParent")
        except Exception:
            return None

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
        """Clic AX : action `AXPress` (boutons, cases, liens...). Un échec est un échec :
        lire l'attribut `AXPress` ne clique pas."""
        try:
            node.perform("AXPress")
        except Exception as exc:
            raise ToolError(f"« {name} » n'a pas répondu au clic : {exc}") from exc
        return f"« {name} » cliqué."

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
        self._backend: _BackendBase = (
            UiaBackend(desktop) if config.uia.backend == "uia" else AxBackend(desktop)
        )
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
            text = ""
            if kind in _VALUE_KINDS:  # phase 16g : ce qui est tapé dans un champ
                value = self._backend.control_text(node).strip()
                text = value[:_NAME_MAX] if value != name else ""
            elif kind in _TOGGLE_KINDS:  # phase 17d : un interrupteur, son état
                state = self._backend.toggle_state(node)
                text = _STATE[state] if state is not None else ""
            found.append(UiControlInfo(name=name[:_NAME_MAX], kind=kind, text=text))
            if len(found) >= max_controls:
                break
        return found

    def _find(
        self,
        window: UiWindowInfo,
        control: str | list[str],
        kind: str | None,
        nth: int,
        clickable: bool = False,
        near: str | list[str] | None = None,
    ) -> Any:
        """Le contrôle `control` (un nom, approché ; ou plusieurs noms acceptables, exacts,
        par ordre de préférence), le plus proche des textes `near` s'ils sont donnés."""
        root = self._backend.attach(window)
        labels = [control] if isinstance(control, str) else list(control)
        wanted = [fold(label) for label in labels]
        shown = " / ".join(labels)
        candidates: list[Any] = []
        seen: dict[str, float] = {}  # noms rencontrés -> ressemblance, pour l'erreur
        for node in self._backend.walk(root, _MAX_VISITED):
            name = self._backend.node_name(node)
            if not name:
                continue
            if len(seen) < 2000:
                seen.setdefault(name[:_NAME_MAX], fuzz.token_set_ratio(wanted[0], fold(name)))
            node_kind = self._backend.node_kind(node)
            if kind is not None and fold(kind) != fold(node_kind):
                continue
            if clickable and kind is None and node_kind not in _CLICKABLE:
                continue
            folded = fold(name)
            if folded in wanted or (
                len(wanted) == 1 and fuzz.ratio(wanted[0], folded) >= _MATCH_RATIO
            ):
                candidates.append(node)
        if not candidates:
            where = f" (type « {kind} »)" if kind else ""
            close = [n for n, score in sorted(seen.items(), key=lambda kv: -kv[1]) if score >= 40]
            hint = f" ; noms proches : {', '.join(f'« {n} »' for n in close[:6])}" if close else ""
            raise ToolError(f"contrôle introuvable : « {shown} »{where}{hint}")
        # Phase 16g : un nom exact passe avant un nom approchant (« Play Jet Lag » et non
        # « Play Jet Lag Radio », qui ressemble assez pour être candidat).
        exact = [node for node in candidates if fold(self._backend.node_name(node)) in wanted]
        candidates = exact or candidates
        if near:  # phase 16g : le contrôle de la ligne (du groupe) qui mentionne `near`
            texts = [near] if isinstance(near, str) else list(near)
            ranked = []
            for index, node in enumerate(candidates):
                level = self._near_level(node, texts)
                if level is not None:  # le plus proche, puis le nom préféré, puis l'ordre
                    name = fold(self._backend.node_name(node))
                    rank = wanted.index(name) if name in wanted else len(wanted)
                    ranked.append((level, rank, index))
            candidates = [candidates[i] for _, _, i in sorted(ranked)]
            if not candidates:
                where = " et ".join(f"« {t} »" for t in texts)
                raise ToolError(f"aucun « {shown} » près de {where}")
        if nth > len(candidates):
            raise ToolError(f"« {shown} » : {len(candidates)} occurrence(s), pas la {nth}e")
        return candidates[nth - 1]

    def _near_level(self, node: Any, texts: list[str]) -> int | None:
        """À combien de niveaux au-dessus du nœud (1 à 4) se trouve le premier groupe qui
        mentionne tous les `texts`, ou None. Ex. le bouton « Play » de la ligne « Jet Lag »
        qui cite « Tiakola » : niveau 1 ; un « Play » dont il faut remonter à toute la page
        pour trouver ces mots passe après."""
        wanted = [fold(text).strip() for text in texts if text.strip()]
        ancestor = node
        for level in range(1, _NEAR_LEVELS + 1):
            ancestor = self._backend.parent(ancestor)
            if ancestor is None:
                return None
            seen = " | ".join(  # le nom du contrôle lui-même ne compte pas
                fold(self._backend.node_name(other))
                for other in [ancestor, *self._backend.walk(ancestor, _NEAR_NODES)]
                if other is not node
            )
            if all(text in seen for text in wanted):
                return level
        return None

    def click_in(
        self,
        window: UiWindowInfo,
        item: str,
        buttons: list[str],
        contains: list[str] | None = None,
    ) -> str:
        """Phase 16g : clique, dans l'élément nommé `item` (une ligne de liste) qui
        mentionne tous les `contains`, son bouton nommé comme l'un des `buttons` (« Play »
        vaut aussi « Play <titre> »). Ex. la ligne « Jet Lag » qui cite « Tiakola », son
        bouton « Play ». Les éléments au nom exact d'abord, puis ceux qui commencent par
        `item` (« Jet Lag (Radio Edit) »), dans l'ordre de la fenêtre."""
        root = self._backend.attach(window)
        wanted = fold(item).strip()
        names = [fold(button).strip() for button in buttons]
        needs = [fold(text).strip() for text in contains or [] if text.strip()]
        close: list[Any] = []
        for node in self._backend.iter_walk(root, _MAX_VISITED):  # s'arrête à la bonne
            name = fold(self._backend.node_name(node)).strip()
            if name == wanted:
                done = self._press_in(window, node, names, needs)
                if done is not None:
                    return done
            elif name.startswith(wanted) and not any(name.startswith(n) for n in names):
                close.append(node)
        for row in close:
            done = self._press_in(window, row, names, needs)
            if done is not None:
                return done
        mention = f" qui mentionne « {' / '.join(contains or [])} »" if needs else ""
        raise ToolError(f"aucun élément « {item} »{mention} avec un bouton {' / '.join(buttons)}")

    def items(self, window: UiWindowInfo, query: str | None, limit: int) -> list[UiItemInfo]:
        """Phase 17a : les éléments de liste de la fenêtre (les plus extérieurs : une ligne,
        pas ses cellules), dans l'ordre, avec leurs textes et leurs boutons ; ceux qui
        contiennent tous les mots de `query` s'il est donné."""
        root = self._backend.attach(window)
        words = fold(query or "").split()
        found: list[UiItemInfo] = []
        stack = list(reversed(self._backend.children(root)))
        visited = 0
        while stack and visited < _MAX_VISITED and len(found) < limit:
            node = stack.pop()
            visited += 1
            name = self._backend.node_name(node)
            if name and self._backend.node_kind(node) == "élément":
                info = self._item(node, name)
                seen = fold(" ".join([info.name, *info.texts, *info.buttons]))
                if all(word in seen for word in words):
                    found.append(info)
                continue  # ses cellules ne sont pas d'autres éléments
            with contextlib.suppress(Exception):
                stack.extend(reversed(self._backend.children(node)))
        return found

    def _item(self, node: Any, name: str) -> UiItemInfo:
        texts: list[str] = []
        buttons: list[str] = []
        for inner in self._backend.walk(node, _ROW_NODES):
            label = self._backend.node_name(inner).strip()
            if not label or label == name:
                continue
            target = buttons if self._backend.node_kind(inner) == "bouton" else texts
            if label not in target and label not in buttons:
                target.append(label[:_NAME_MAX])
        return UiItemInfo(name[:_NAME_MAX], tuple(texts[:8]), tuple(buttons[:6]))

    def find_text(self, window: UiWindowInfo, text: str) -> str | None:
        """Phase 17a : le premier contrôle dont le nom, ou le contenu d'un champ, contient
        `text` ; None s'il n'y en a pas (encore). S'arrête dès qu'il le trouve."""
        root = self._backend.attach(window)
        wanted = fold(text).strip()
        for node in self._backend.iter_walk(root, _MAX_VISITED):
            name = self._backend.node_name(node)
            if wanted in fold(name):
                return name
            if self._backend.node_kind(node) in _VALUE_KINDS:
                value = self._backend.control_text(node)
                if wanted in fold(value.replace("+", " ")):
                    return f"{name} : {value}"
        return None

    def _press_in(
        self, window: UiWindowInfo, row: Any, names: list[str], needs: list[str]
    ) -> str | None:
        """Le bouton `names` de l'élément `row` s'il mentionne `needs` : cliqué."""
        inside = self._backend.walk(row, _ROW_NODES)
        if needs:
            seen = " | ".join(fold(self._backend.node_name(n)) for n in inside)
            if not all(text in seen for text in needs):
                return None
        for node in inside:
            name = fold(self._backend.node_name(node)).strip()
            if self._backend.node_kind(node) == "bouton" and any(
                name == n or name.startswith(n + " ") for n in names
            ):
                return self._click(window, node, self._backend.node_name(node))
        return None

    # -- lecture ----------------------------------------------------------------------

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        if control is not None:
            node = self._find(window, control, None, 1)
            text = self._backend.control_text(node)
        else:
            root = self._backend.attach(window)
            text = self._backend.document_text(root) or self._backend.visible_text(root)
        if not text.strip():
            raise ToolError("aucun texte lisible dans cette fenêtre")
        return _truncate(text, max_chars)

    # -- actions ----------------------------------------------------------------------

    def click(
        self,
        window: UiWindowInfo,
        control: str | list[str],
        kind: str | None,
        nth: int,
        near: str | list[str] | None = None,
    ) -> str:
        node = self._find(window, control, kind, nth, clickable=True, near=near)
        name = self._backend.node_name(node) or "(sans nom)"
        return self._click(window, node, name)

    def _click(self, window: UiWindowInfo, node: Any, name: str) -> str:
        """Le clic, la fenêtre montrée s'il le faut (`_shown_for_click`)."""
        with _shown_for_click(window.handle):
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


@contextlib.contextmanager
def _shown_for_click(handle: int | None) -> Iterator[None]:
    """Phase 17d. Une appli réduite (MIKATA l'y met pour agir en arrière-plan) se lit et se
    pilote, mais certaines actions n'y prennent pas : Spotify ignore « Play » fenêtre réduite
    ou cachée derrière les autres, et le fait fenêtre visible (vérifié le 14/10). Le temps du
    clic, la fenêtre réduite est montrée par-dessus SANS être activée (l'utilisateur garde le
    clavier), puis réduite à nouveau : environ 0,3 s."""
    if sys.platform != "win32" or not handle:
        yield
        return
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    if not user32.IsIconic(handle):
        yield
        return
    flags = 0x0001 | 0x0002 | 0x0010 | 0x0040  # NOSIZE | NOMOVE | NOACTIVATE | SHOWWINDOW
    previous = user32.GetForegroundWindow()
    user32.ShowWindow(handle, 4)  # SW_SHOWNOACTIVATE
    user32.SetWindowPos(handle, -1, 0, 0, 0, 0, flags)  # par-dessus...
    user32.SetWindowPos(handle, -2, 0, 0, 0, 0, flags)  # ...sans y rester épinglée
    try:
        yield
    finally:
        time.sleep(SHOW_FOR_CLICK_S)
        user32.ShowWindow(handle, 7)  # SW_SHOWMINNOACTIVE
        # Remise à l'écran, une fenêtre réduite prend quand même le premier plan (constaté
        # sur Spotify) : il revient aussitôt à la fenêtre de l'utilisateur. Seulement si c'est
        # cette fenêtre-là qui l'a pris : si l'utilisateur a changé de fenêtre entre-temps,
        # on n'y touche pas.
        if previous and previous != handle and user32.GetForegroundWindow() == handle:
            user32.keybd_event(0x12, 0, 0, 0)  # Alt : lève le verrou de premier plan
            user32.keybd_event(0x12, 0, 0x0002, 0)
            user32.SetForegroundWindow(previous)


SHOW_FOR_CLICK_S = 0.3  # le temps que l'appli prenne le clic, fenêtre visible


def _discord_voice_from(nodes: list[Any]) -> tuple[str | None, list[str]]:
    """(salon vocal connecté, membres) depuis un arbre de Discord (nœuds UIA ou AX), ou
    (None, []).

    Le lien du salon connecté a la forme « <salon> (salon vocal), <membre>, <membre>,
    durée de l'appel... » (client français ; « voice channel » en anglais). Les salons non
    connectés sont des boutons « <nom> (salon vocal) » sans membres : ignorés.
    """
    for node in nodes:
        try:
            if getattr(node, "ControlTypeName", None) is not None:
                is_link = node.ControlTypeName == "HyperlinkControl"
                name = (node.Name or "").strip()
            else:
                is_link = AxBackend._ax_role(node) == "AXLink"
                name = AxBackend._ax_name(node)
        except Exception:  # nœud disparu pendant la lecture : sauté
            continue
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
