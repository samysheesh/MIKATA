"""Outils clavier et souris (§7) : type_text ★, press_keys ★, mouse ★.

Le doc prévoit pyautogui ; on utilise pynput (déjà installé pour le raccourci clavier) :
le texte est collé (presse-papiers), donc les accents français passent, ce que ni pyautogui ni
pynput ne garantissent sur un clavier AZERTY.

Garde-fou : Maj+Suppr (suppression définitive dans l'Explorateur) est refusé, et un texte
tapé qui contient une commande de suppression aussi (il pourrait l'être dans un terminal).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mikata.agent.guard import Verdict
from mikata.tools.base import BaseTool, GuardDenied, ToolError
from mikata.tools.registry import register_tool

MODIFIERS = ("ctrl", "alt", "shift", "win")
SPECIAL_KEYS = {
    "enter", "tab", "esc", "space", "backspace", "delete", "insert", "home", "end", "pageup",
    "pagedown", "up", "down", "left", "right", "capslock", "printscreen", "menu", "pause",
    *(f"f{i}" for i in range(1, 21)),
}  # fmt: skip
ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "contrôle": "ctrl", "controle": "ctrl",
    "windows": "win", "cmd": "win", "super": "win", "meta": "win",
    "maj": "shift", "majuscule": "shift",
    "return": "enter", "entrée": "enter", "entree": "enter",
    "escape": "esc", "echap": "esc", "échap": "esc", "échappement": "esc",
    "del": "delete", "suppr": "delete", "supprimer": "delete",
    "espace": "space", "spacebar": "space",
    "retour": "backspace", "effacer": "backspace",
    "haut": "up", "bas": "down", "gauche": "left", "droite": "right",
    "début": "home", "debut": "home", "fin": "end",
    "pgup": "pageup", "pgdn": "pagedown", "page_up": "pageup", "page_down": "pagedown",
    "tabulation": "tab", "impr": "printscreen", "prtsc": "printscreen",
}  # fmt: skip
TYPE_MAX_CHARS = 2000


def parse_keys(combo: str) -> list[str]:
    """« Ctrl+Shift+T » -> ["ctrl", "shift", "t"] ; ValueError si une touche est inconnue."""
    parts = [part.strip().casefold() for part in combo.replace(" + ", "+").split("+")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"combinaison illisible : « {combo} »")
    keys: list[str] = []
    for part in parts:
        key = ALIASES.get(part, part)
        if key not in MODIFIERS and key not in SPECIAL_KEYS and len(key) != 1:
            raise ValueError(f"touche inconnue : « {part} »")
        keys.append(key)
    if all(key in MODIFIERS for key in keys) and len(keys) > 1:
        raise ValueError(f"il manque la touche principale dans « {combo} »")
    return keys


class PynputInput:
    """Implémentation Windows de `tools.base.InputControl` (pynput, import paresseux)."""

    def type_text(self, text: str) -> None:
        """Colle `text` (presse-papiers + Ctrl+V), puis restaure le presse-papiers.

        Taper caractère par caractère (pynput, SendInput) dépend de la disposition du clavier :
        sur un AZERTY, « à » devient « ç » et la suite se corrompt. Coller est indépendant de
        la disposition et conserve tous les caractères, accents et emojis compris.
        """
        import time

        import pyperclip
        from pynput.keyboard import Controller, Key, KeyCode

        try:
            previous: str | None = pyperclip.paste()
        except pyperclip.PyperclipException:
            previous = None
        pyperclip.copy(text)
        time.sleep(0.05)
        keyboard = Controller()
        keyboard.press(Key.ctrl)
        try:
            keyboard.tap(KeyCode.from_char("v"))
        finally:
            keyboard.release(Key.ctrl)
        time.sleep(0.25)  # laisse l'application lire le presse-papiers avant de le restaurer
        if previous is not None:
            pyperclip.copy(previous)

    def press_keys(self, keys: list[str]) -> None:
        from pynput.keyboard import Controller, Key, KeyCode

        names = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "win": "cmd",
                 "enter": "enter", "esc": "esc", "pageup": "page_up", "pagedown": "page_down",
                 "printscreen": "print_screen", "capslock": "caps_lock"}  # fmt: skip

        def resolve(key: str) -> Key | KeyCode:
            if len(key) == 1:
                return KeyCode.from_char(key)
            return getattr(Key, names.get(key, key))

        keyboard = Controller()
        pressed = [resolve(key) for key in keys]
        for key in pressed[:-1]:
            keyboard.press(key)
        try:
            keyboard.tap(pressed[-1])
        finally:
            for key in reversed(pressed[:-1]):
                keyboard.release(key)

    def mouse(self, action: str, x: int | None, y: int | None, amount: int) -> None:
        from pynput.mouse import Button, Controller

        mouse = Controller()
        if x is not None and y is not None:
            mouse.position = (x, y)
        if action == "move":
            return
        if action in ("scroll_up", "scroll_down"):
            mouse.scroll(0, amount if action == "scroll_up" else -amount)
            return
        button = {"right_click": Button.right, "middle_click": Button.middle}.get(
            action, Button.left
        )
        mouse.click(button, 2 if action == "double_click" else 1)


class TypeParams(BaseModel):
    text: str = Field(min_length=1, max_length=TYPE_MAX_CHARS, description="Texte à taper")


@register_tool
class TypeText(BaseTool[TypeParams]):
    name = "type_text"
    description = "Tape un texte au clavier dans la fenêtre active (accents compris)."
    params = TypeParams
    terminal = True

    async def run(self, args: TypeParams) -> str:
        decision = self.ctx.require_guard().check_typed_text(args.text)
        if decision.verdict is Verdict.DENY:
            raise GuardDenied(f"texte non tapé : {decision.reason} (« {decision.match} »)")
        control = self.ctx.require_input()
        await self.ctx.run_blocking(control.type_text, args.text)
        return f"Texte tapé ({len(args.text)} caractères)."


class KeysParams(BaseModel):
    keys: str = Field(
        min_length=1,
        description="Touche ou combinaison, ex. « ctrl+c », « alt+tab », « win+d », « enter », "
        "« f5 »",
    )
    repeat: int = Field(default=1, ge=1, le=20, description="Nombre d'appuis")


@register_tool
class PressKeys(BaseTool[KeysParams]):
    name = "press_keys"
    description = "Appuie sur une touche ou un raccourci clavier dans la fenêtre active."
    params = KeysParams
    terminal = True

    async def run(self, args: KeysParams) -> str:
        try:
            keys = parse_keys(args.keys)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        decision = self.ctx.require_guard().check_keys(keys)
        if decision.verdict is Verdict.DENY:
            raise GuardDenied(decision.reason)
        control = self.ctx.require_input()
        for _ in range(args.repeat):
            await self.ctx.run_blocking(control.press_keys, keys)
        return f"Touches « {'+'.join(keys)} » envoyées."


class MouseParams(BaseModel):
    action: Literal[
        "click", "double_click", "right_click", "middle_click", "scroll_up", "scroll_down", "move"
    ] = Field(description="Action de la souris")
    x: int | None = Field(default=None, ge=0, description="Position horizontale en pixels")
    y: int | None = Field(default=None, ge=0, description="Position verticale en pixels")
    amount: int = Field(default=3, ge=1, le=50, description="Crans de molette")


@register_tool
class Mouse(BaseTool[MouseParams]):
    name = "mouse"
    description = "Clic, double-clic, clic droit, molette ou déplacement de la souris."
    params = MouseParams
    terminal = True

    async def run(self, args: MouseParams) -> str:
        if (args.x is None) != (args.y is None):
            raise ToolError("donner x et y ensemble, ou aucun des deux")
        if args.action == "move" and args.x is None:
            raise ToolError("move demande une position x, y")
        control = self.ctx.require_input()
        await self.ctx.run_blocking(control.mouse, args.action, args.x, args.y, args.amount)
        where = f" en ({args.x}, {args.y})" if args.x is not None else ""
        return f"Souris : {args.action}{where}."
