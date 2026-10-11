"""Agent v2 (phase 13d) : les recettes — le raccourci des demandes voisines.

Une recette relie une intention (verbe + catégorie d'objet) à un savoir-faire éprouvé.
La clé inclut les types d'inconnues ; le match ne vérifie que le verbe et la catégorie.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from mikata.agent2.schema import Intent, Trace

RECIPES_FILE = "data/agent2/recipes.json"


class Recipe(BaseModel):
    verb: str
    object_kind: str
    resolve: list[dict[str, str]] = []  # [{"name", "kind"}], l'ordre des inconnues
    capability: str
    skill: str
    skill_args: dict[str, str] = {}  # param -> "{nom_inconnue}"
    successes: int = 1
    failures: int = 0
    source_traces: list[str] = []


class RecipeStore:
    def __init__(self, path: Path, skills: object, os_name: str) -> None:
        self.path = path
        self.skills = skills
        self.os_name = os_name
        self.recipes: dict[str, Recipe] = self._load()

    def _load(self) -> dict[str, Recipe]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8")).get("recipes", {})
        except (OSError, ValueError, TypeError, AttributeError):
            return {}
        loaded: dict[str, Recipe] = {}
        for key, recipe in raw.items():
            try:
                loaded[str(key)] = Recipe.model_validate(recipe)
            except (ValueError, TypeError):
                continue
        return loaded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_suffix(".json.part")
        payload = {
            "version": 1,
            "recipes": {
                key: recipe.model_dump(mode="json") for key, recipe in self.recipes.items()
            },
        }
        part.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        part.replace(self.path)

    @staticmethod
    def key(intent: Intent) -> str:
        kinds = "+".join(sorted({unknown.kind for unknown in intent.unknowns}))
        return f"{intent.verb.value}|{intent.object_kind.strip().lower()}|{kinds}"

    def match(self, intent: Intent) -> Recipe | None:
        """La recette d'une intention voisine, si elle a fait ses preuves et que le
        savoir-faire est toujours utilisable."""
        normalized = intent.object_kind.strip().lower()
        for recipe in self.recipes.values():
            if recipe.verb != intent.verb.value or recipe.object_kind != normalized:
                continue
            if recipe.failures >= recipe.successes:
                continue
            if recipe.skill not in self.skills.usable(recipe.capability, self.os_name):  # type: ignore[attr-defined]
                continue
            return recipe
        return None

    def from_trace(self, trace: Trace, skill: str, args_map: dict[str, str]) -> None:
        """Crée (ou rafraîchit) la recette de la trace ; la trace doit être vérifiée."""
        if trace.intent is None:
            return
        key = self.key(trace.intent)
        existing = self.recipes.get(key)
        if existing is not None:
            existing.successes += 1
            existing.source_traces.append(trace.id)
            existing.skill = skill
        else:
            self.recipes[key] = Recipe(
                verb=trace.intent.verb.value,
                object_kind=trace.intent.object_kind.strip().lower(),
                resolve=[
                    {"name": unknown.name, "kind": unknown.kind}
                    for unknown in trace.intent.unknowns
                ],
                capability=self.skills.skills[skill].capability,  # type: ignore[attr-defined]
                skill=skill,
                skill_args=args_map,
                source_traces=[trace.id],
            )
        self._save()

    def add(self, recipe: Recipe) -> None:
        """Ajoute (ou rafraîchit) une recette déjà construite — adoption de nuit (13e)."""
        kinds = "+".join(sorted(item["kind"] for item in recipe.resolve))
        key = f"{recipe.verb}|{recipe.object_kind}|{kinds}"
        existing = self.recipes.get(key)
        if existing is not None:
            existing.successes += recipe.successes
            existing.source_traces.extend(recipe.source_traces)
        else:
            self.recipes[key] = recipe
        self._save()

    def record(self, key: str, success: bool) -> None:
        recipe = self.recipes.get(key)
        if recipe is None:
            return
        if success:
            recipe.successes += 1
        else:
            recipe.failures += 1
        self._save()
