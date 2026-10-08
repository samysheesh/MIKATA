"""Pilotage fin des applications Windows (phase 12a) : arbre d'accessibilité (UIA).

- `controller.py` : implémentation réelle derrière `tools.base.UiAutomation` (imports
  paresseux de `uiautomation`, Windows seulement), avec l'état ambiant (présence vocale
  Discord).
- `state.py` : cache d'état ambiant en RAM, rafraîchi par un thread dédié, jamais loggé.
"""
