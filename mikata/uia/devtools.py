"""Moteur DevTools des outils `uia_*` (phase 17e) : une appli Chromium (Electron, CEF)
listée dans `uia.devtools_apps` est lue par son propre arbre d'accessibilité, celui que
Chromium expose aussi à UIA (mêmes noms : « Play », « What do you want to play? »), mais
demandé d'un coup par DevTools (`Accessibility.getFullAXTree`, ~0,1 s pour 2 000 à 3 000
nœuds sur Spotify) au lieu d'un parcours COM nœud par nœud. Les actions visent le nœud DOM
du nœud d'accessibilité (`backendDOMNodeId`) : un clic en JavaScript, un texte inséré comme
au clavier (`Input.insertText`). Elles prennent fenêtre cachée.

Mêmes opérations de nœuds que `UiaBackend` et `AxBackend` : toute la logique de
`UiController` (contrôles, éléments de liste, « près de », clic dans un élément...) s'applique
telle quelle, et une compétence apprise avec UIA marche ici sans changement.
"""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

from mikata.browser.apps import DevtoolsApps
from mikata.tools.base import Desktop, ToolError, UiWindowInfo
from mikata.uia.controller import _STATE, _BackendBase

# Rôle d'accessibilité Chromium -> nom français donné au LLM (comme `UiaBackend._KINDS`).
_KINDS: dict[str, str] = {
    "button": "bouton",
    "PopUpButton": "bouton",
    "textbox": "champ",
    "searchbox": "champ",
    "spinbutton": "champ",
    "StaticText": "texte",
    "heading": "texte",
    "link": "lien",
    "row": "élément",
    "listitem": "élément",
    "option": "élément",
    "treeitem": "arbre",
    "list": "liste",
    "listbox": "liste",
    "grid": "liste",
    "table": "liste",
    "tree": "liste",
    "treegrid": "liste",
    "checkbox": "case",
    "radio": "case",
    "switch": "case",
    "menuitem": "menu",
    "menuitemcheckbox": "menu",
    "menuitemradio": "menu",
    "tab": "onglet",
    "combobox": "liste déroulante",
    "slider": "curseur",
    "progressbar": "progression",
    "toolbar": "barre d'outils",
    "image": "image",
    "img": "image",
    "columnheader": "en-tête",
    "rowheader": "en-tête",
    "gridcell": "cellule",
    "cell": "cellule",
    "RootWebArea": "document",
    "WebArea": "document",
    "generic": "groupe",
    "group": "groupe",
    "section": "groupe",
    "paragraph": "groupe",
    "article": "groupe",
    "figure": "groupe",
    "region": "panneau",
    "navigation": "panneau",
    "main": "panneau",
    "banner": "panneau",
    "contentinfo": "panneau",
    "complementary": "panneau",
    "form": "panneau",
    "dialog": "fenêtre",
    "alertdialog": "fenêtre",
}
# Nœuds sans intérêt : morceaux de texte déjà portés par leur parent, sauts de ligne.
_SKIPPED_ROLES = {"InlineTextBox", "LineBreak", "ListMarker", "none"}
_TOGGLE_PROPS = ("checked", "pressed")
_TEXT_MAX = 20000  # texte du document lu par uia_read
_ROW_ROLES = {"row", "listitem", "option", "treeitem"}
# Après Entrée : résultats arrivés en ~0,65 s sur Spotify (mesuré le 14/10).
SETTLE_STEP_S = 0.35
SETTLE_QUIET_S = 1.5
SETTLE_MAX_S = 4.0

# Clic d'un élément (un vrai clic de souris n'est pas nécessaire : les applis React, comme
# Spotify, réagissent à l'événement click) ; focus et sélection du contenu d'un champ, pour
# que le texte inséré le remplace.
_CLICK_FN = "function () { this.scrollIntoView({block: 'center'}); this.click(); return true; }"
_FOCUS_FN = """function () {
  this.focus();
  if (typeof this.select === 'function') { this.select(); }
  else {
    const range = document.createRange();
    range.selectNodeContents(this);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  return document.activeElement === this || this.contains(document.activeElement);
}"""
_DOCUMENT_JS = f"document.body ? document.body.innerText.slice(0, {_TEXT_MAX}) : ''"


class AxNode:
    """Un nœud de l'arbre d'accessibilité de Chromium, avec ses liens de parenté (les
    nœuds ignorés sont transparents : leurs enfants remontent au parent visible)."""

    __slots__ = ("backend_id", "children", "name", "parent", "process", "props", "role", "value")

    def __init__(self, raw: dict[str, Any], process: str) -> None:
        self.role = _ax_value(raw.get("role"))
        self.name = _ax_value(raw.get("name")).strip()
        self.value = _ax_value(raw.get("value"))
        self.props = {
            str(prop.get("name")): _ax_value(prop.get("value"))
            for prop in raw.get("properties") or []
        }
        self.backend_id = raw.get("backendDOMNodeId")
        self.process = process
        self.children: list[AxNode] = []
        self.parent: AxNode | None = None


def _ax_value(value: Any) -> str:
    """Valeur d'un champ AXValue ({type, value}) en texte."""
    if isinstance(value, dict):
        inner = value.get("value")
        return "" if inner is None else str(inner)
    return ""


def build_tree(raw_nodes: list[dict[str, Any]], process: str) -> AxNode:
    """Racine de l'arbre (la page), nœuds ignorés et morceaux de texte retirés."""
    by_id = {str(raw.get("nodeId")): raw for raw in raw_nodes}
    if not raw_nodes:
        raise ToolError(f"{process} : arbre d'accessibilité vide")
    root_raw = next((raw for raw in raw_nodes if not raw.get("parentId")), raw_nodes[0])
    root = AxNode(root_raw, process)
    stack: list[tuple[dict[str, Any], AxNode]] = [(root_raw, root)]
    seen = {str(root_raw.get("nodeId"))}
    while stack:
        raw, owner = stack.pop()
        # Enfants dans l'ordre du document ; un enfant ignoré passe la main à ses enfants.
        pending = [str(child) for child in raw.get("childIds") or []]
        attached: list[AxNode] = []
        while pending:
            child_id = pending.pop(0)
            child = by_id.get(child_id)
            if child is None or child_id in seen:
                continue
            seen.add(child_id)
            role = _ax_value(child.get("role"))
            if child.get("ignored") or (role in ("generic", "none") and not _named(child)):
                pending[0:0] = [str(c) for c in child.get("childIds") or []]
                continue
            if role in _SKIPPED_ROLES:
                continue
            node = AxNode(child, process)
            node.parent = owner
            attached.append(node)
            stack.append((child, node))
        owner.children.extend(attached)
    return root


def _named(raw: dict[str, Any]) -> bool:
    return bool(_ax_value(raw.get("name")).strip())


class DevtoolsBackend(_BackendBase):
    """Opérations de nœuds sur l'arbre d'accessibilité de Chromium, lu par DevTools."""

    _KINDS: ClassVar[dict[str, str]] = _KINDS

    def __init__(self, desktop: Desktop | None, apps: DevtoolsApps) -> None:
        super().__init__(desktop)  # type: ignore[arg-type]
        self._apps = apps

    def attach(self, window: UiWindowInfo) -> AxNode:
        result = self._apps.call(window.process, "Accessibility.getFullAXTree")
        return build_tree(result.get("nodes") or [], window.process)

    def children(self, node: Any) -> list[Any]:
        return list(node.children)

    def parent(self, node: Any) -> Any:
        return node.parent

    def node_name(self, node: Any) -> str:
        return str(node.name)

    def node_kind(self, node: Any) -> str:
        role = node.role
        if role == "combobox" and node.props.get("editable"):
            return "champ"  # champ de recherche à suggestions (« What do you want to play? »)
        return self._KINDS.get(role, "contrôle")

    def toggle_state(self, node: Any) -> bool | None:
        for prop in _TOGGLE_PROPS:
            state = node.props.get(prop)
            if state in ("true", "True"):
                return True
            if state in ("false", "False"):
                return False
        return None

    def control_text(self, node: Any) -> str:
        state = self.toggle_state(node)
        if state is not None:
            return f"{node.name} : {_STATE[state]}"
        if node.value:
            return str(node.value)
        if node.role in ("RootWebArea", "WebArea"):
            return self.document_text(node)
        return str(node.name)

    def document_text(self, root: Any) -> str:
        text = self._apps.evaluate(root.process, _DOCUMENT_JS)
        return text if isinstance(text, str) else ""

    def visible_text(self, root: Any) -> str:
        parts: list[str] = []
        for node in self.walk(root, 4000):
            if node.role == "StaticText" and node.name:
                parts.append(node.name)
            elif self.node_kind(node) == "champ" and node.value:
                parts.append(str(node.value).strip())
            if sum(len(part) for part in parts) > 8000:
                break
        return "\n".join(parts)

    def _call_on(self, node: Any, function: str, name: str) -> Any:
        """Appelle `function` sur l'élément DOM du nœud ; ToolError s'il a disparu."""
        if not node.backend_id:
            raise ToolError(f"« {name} » ne correspond à aucun élément de la page")
        apps = self._apps
        try:
            resolved = apps.call(node.process, "DOM.resolveNode", backendNodeId=node.backend_id)
        except ToolError as exc:
            raise ToolError(f"« {name} » a disparu de la page (elle a changé)") from exc
        object_id = (resolved.get("object") or {}).get("objectId")
        if not object_id:
            raise ToolError(f"« {name} » a disparu de la page (elle a changé)")
        try:
            result = apps.call(
                node.process,
                "Runtime.callFunctionOn",
                objectId=object_id,
                functionDeclaration=function,
                returnByValue=True,
            )
        finally:
            with contextlib.suppress(ToolError):
                apps.call(node.process, "Runtime.releaseObject", objectId=object_id)
        if result.get("exceptionDetails"):
            raise ToolError(f"« {name} » n'a pas répondu")
        return (result.get("result") or {}).get("value")

    def click(self, node: Any, name: str) -> str:
        before = self.toggle_state(node)
        self._call_on(node, _CLICK_FN, name)
        if before is None:
            return f"« {name} » cliqué."
        after = self._fresh_state(node)
        state = f" : maintenant {_STATE[after]}" if after is not None else ""
        return f"« {name} » basculé{state}."

    def _fresh_state(self, node: Any) -> bool | None:
        """L'état d'un interrupteur relu après le clic (l'arbre lu avant est périmé)."""
        import time

        time.sleep(0.15)  # le temps que l'appli applique le changement
        try:
            result = self._apps.call(
                node.process,
                "Accessibility.getPartialAXTree",
                backendNodeId=node.backend_id,
                fetchRelatives=False,
            )
        except ToolError:
            return None
        nodes = result.get("nodes") or []
        return self.toggle_state(AxNode(nodes[0], node.process)) if nodes else None

    def set_text(self, node: Any, control: str, text: str) -> str:
        if not self._call_on(node, _FOCUS_FN, control):
            raise ToolError(f"« {control} » ne prend pas le focus")
        # Comme une saisie au clavier : l'appli voit le texte arriver et réagit (recherche).
        self._apps.call(node.process, "Input.insertText", text=text)
        return f"« {control} » rempli."

    def submit(self, node: Any, control: str) -> str:
        """Entrée dans le champ qui vient d'être rempli (il a gardé le focus) : la recherche
        part (Spotify : la page de résultats, avec les artistes, au lieu des suggestions).
        Puis on attend que l'appli ait réagi (`_settle`) : lue aussitôt, la vue montrait
        encore les suggestions, et la mauvaise ligne était cliquée (constaté le 14/10)."""
        before = self._rows(node.process)
        for kind in ("keyDown", "keyUp"):
            extra = {"text": "\r"} if kind == "keyDown" else {}
            self._apps.call(
                node.process,
                "Input.dispatchKeyEvent",
                type=kind,
                key="Enter",
                code="Enter",
                windowsVirtualKeyCode=13,
                nativeVirtualKeyCode=13,
                **extra,
            )
        self._settle(node.process, before)
        return f"« {control} » rempli et validé."

    def _rows(self, process: str) -> tuple[str, ...]:
        """Les éléments de liste affichés (lignes, éléments, options), par leur nom : ce qui
        change quand une recherche, un envoi, une navigation aboutit — pas le compteur de
        lecture ni une animation."""
        root = self.attach(UiWindowInfo(process, process, 0, "devtools"))
        names: list[str] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.role in _ROW_ROLES:
                names.append(node.name)
            stack.extend(reversed(node.children))
        return tuple(names)

    def _settle(self, process: str, before: tuple[str, ...]) -> None:
        """Attend que les éléments de liste aient changé puis tiennent en place (deux
        lectures égales) ; rien n'a bougé en SETTLE_QUIET_S : rien à attendre. Plafond
        SETTLE_MAX_S, jamais une erreur."""
        import time

        start = time.monotonic()
        last: tuple[str, ...] | None = None
        changed = False
        while time.monotonic() - start < SETTLE_MAX_S:
            time.sleep(SETTLE_STEP_S)
            try:
                rows = self._rows(process)
            except ToolError:
                return
            changed = changed or rows != before
            if changed and rows == last:
                return
            if not changed and time.monotonic() - start >= SETTLE_QUIET_S:
                return
            last = rows
