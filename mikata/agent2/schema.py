"""Agent v2 (phase 13a) : tous les modèles de données, en Pydantic v2.

Un objet par étape du modèle universel d'une tâche : Intention -> Inconnues -> Canal ->
Plan (sous-buts + prédicats) -> Étapes -> Trace. Les prédicats sont une union discriminée
par leur champ `type`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Verb(StrEnum):
    PLAY = "play"
    OPEN = "open"
    FIND = "find"
    ASK = "ask"
    SEND = "send"
    CREATE = "create"
    EDIT = "edit"
    ORGANIZE = "organize"
    CONTROL = "control"
    SCHEDULE = "schedule"
    MONITOR = "monitor"
    AUTOMATE = "automate"
    CHAT = "chat"  # répondre sans agir (explication, blague, conversation)


class Unknown(BaseModel):
    name: str  # identifiant court, ex. "titre_exact", "fichier", "contact"
    question: str  # ce qu'il faut savoir, en français
    kind: Literal["web", "file", "contact", "memory", "screen", "app", "user"]


class Intent(BaseModel):
    verb: Verb
    object: str  # « dernier opening de One Piece »
    object_kind: str = ""  # catégorie générale : « opening d'anime »,
    # « musique », « facture », « message »
    entities: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)  # « sur Spotify »
    target: str | None = None  # « Paul », « écran 2 »
    unknowns: list[Unknown] = Field(default_factory=list)
    channel_hint: str | None = None  # canal imposé par l'utilisateur, ex. "spotify"
    sub_intents: list[Intent] = Field(default_factory=list)  # demande composite


class Resolved(BaseModel):
    name: str
    value: str
    source: str  # "memory" | "screen" | "files" | "web" | "app" | "user" | "intent"
    confident: bool = True


# -- prédicats de vérification (union discriminée par "type") ------------------------------


class AppFocused(BaseModel):
    type: Literal["app_focused"] = "app_focused"
    name: str  # sous-chaîne du nom de l'appli/processus


class WindowTitleMatches(BaseModel):
    type: Literal["window_title"] = "window_title"
    pattern: str  # regex, insensible à la casse


class PageShows(BaseModel):
    type: Literal["page_shows"] = "page_shows"
    pattern: str  # regex sur le texte de l'onglet actif


class FileExists(BaseModel):
    type: Literal["file_exists"] = "file_exists"
    path: str


class TextVisible(BaseModel):
    type: Literal["text_visible"] = "text_visible"
    pattern: str  # regex sur l'observation résumée


class ToolSucceeded(BaseModel):
    type: Literal["tool_ok"] = "tool_ok"
    tool: str  # dernier appel de cet outil réussi


class Judge(BaseModel):
    type: Literal["judge"] = "judge"
    question: str  # jugé par le LLM sur l'observation (moins fiable)


Predicate = Annotated[
    AppFocused | WindowTitleMatches | PageShows | FileExists | TextVisible | ToolSucceeded | Judge,
    Field(discriminator="type"),
]


class SubGoal(BaseModel):
    goal: str  # « YouTube affiche les résultats pour le titre »
    channel: str  # nom d'un canal du registre (channels.py)
    done_when: Predicate
    skill: str | None = None  # nom d'un savoir-faire à appeler (13d)
    skill_args: dict[str, str] = Field(default_factory=dict)


class Observation(BaseModel):
    active_app: str = ""
    title: str = ""
    page: str = ""  # texte résumé de l'onglet navigateur actif, "" sinon
    summary: str = ""  # éléments visibles résumés (≤ observe_max_lines lignes)
    changed: bool = True


class ActionRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    result: str  # tronqué à 500 caractères
    duration_s: float


class Step(BaseModel):
    subgoal_index: int
    attempt: int
    code: str  # script exécuté (ou "skill:<name>")
    actions: list[ActionRecord]
    error: str | None = None
    observation: Observation
    goal_reached: bool


class Trace(BaseModel):
    id: str  # uuid4 hex
    started: datetime
    request: str
    os: str  # mikata.plat.platform()
    intent: Intent | None = None
    resolved: list[Resolved] = Field(default_factory=list)
    channel: str | None = None
    channel_announced: bool = False
    plan: list[SubGoal] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    verified: bool = False
    outcome: Literal["ok", "failed", "cancelled", "chat", "error"] = "error"
    answer: str = ""  # ce que MIKATA a dit à la fin
    recipe: str | None = None  # recette utilisée (13d)
    skills_used: list[str] = Field(default_factory=list)
    feedback: str | None = None  # correction de l'utilisateur (13d)
    llm_calls: int = 0
    cost_usd: float = 0.0


Intent.model_rebuild()
