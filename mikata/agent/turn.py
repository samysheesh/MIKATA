"""Un tour de conversation : ce qui circule entre les composants de la boucle agent.

`Turn` réunit la demande, la destination des phrases, l'historique de travail (messages
envoyés au modèle, étapes gardées pour la conversation), l'état de sécurité et le résultat
mesuré. Chaque composant (`model`, `actions`, `skill_path`) reçoit le tour entier au lieu
d'une dizaine d'arguments : son interface reste étroite et stable.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from mikata.action.effects import Trust
from mikata.action.runtime import ScriptOutcome, SkillChoice
from mikata.llm.base import ToolCallReq, Usage
from mikata.llm.reasoning import at_least
from mikata.tts.cache import PHRASES

if TYPE_CHECKING:
    from mikata.tools.base import TurnIO
    from mikata.tools.registry import ToolResult

UNTRUSTED_HISTORY = "(contenu lu ailleurs, non gardé dans l'historique)"
HISTORY_TOOL_CHARS = 400  # résultat d'outil gardé dans l'historique de conversation
FOLLOW_UP_MARKER = "[IGNORER]"
FOLLOW_UP_NOTE = (
    "Fenêtre de suivi : l'utilisateur vient d'avoir un échange avec toi et a la parole "
    "quelques secondes sans dire le mot d'activation. Cette phrase est donc, le plus souvent, "
    "la suite de l'échange précédent, même courte ou elliptique (« et à Tokyo ? », « et "
    "demain ? », « pourquoi ? », « plus fort », « merci ») : réponds-y normalement en "
    "t'appuyant sur les messages précédents. Réponds "
    f"exactement {FOLLOW_UP_MARKER}, sans rien d'autre et sans outil, si la phrase n'est pas "
    "une demande ou une question qui prolonge l'échange précédent : propos adressé à une autre "
    "personne (un prénom, « tu as vu... », « on devrait... »), avis ou remarque en passant, "
    "sujet sans lien avec ce qui vient d'être dit, vidéo ou bruit. Une phrase qui parle d'un "
    "sujet nouveau sans te le demander explicitement est à ignorer. Ne demande jamais de "
    "répéter."
)
# « J'ai pas bien entendu », « tu peux répéter ? » : jamais dit dans la fenêtre de suivi.
_REPEAT_RE = re.compile(
    r"pas (?:bien |tout |tr[eè]s bien )?(?:entendu|compris|saisi|capt[ée])|r[ée]p[ée]ter|"
    r"r[ée]p[èe]te|^\W*pardon\W*$|^\W*comment\W*\?\W*$",
    re.IGNORECASE,
)


class SpeechSink(Protocol):
    """Destination des phrases d'une commande (`audio.player.SpeechStream`)."""

    def say(self, text: str) -> None: ...
    def say_audio(self, pcm: Any, sample_rate: int) -> None: ...


class CachedPhrases(Protocol):
    sample_rate: int

    def get(self, key: str) -> Any: ...


class SilentSpeech:
    """Destination muette (option --ask --no-audio) : le texte est seulement renvoyé."""

    def say(self, text: str) -> None:
        pass

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        pass

    async def drain(self) -> None:
        pass


class PhraseSpeaker:
    """Les phrases fixes (« C'est fait », « Je n'arrive pas à joindre le serveur »...) :
    l'audio en cache s'il existe, sinon le texte, synthétisé."""

    def __init__(self, phrases: CachedPhrases | None) -> None:
        self.phrases = phrases

    def __call__(self, speech: SpeechSink, key: str) -> None:
        if self.phrases is not None and (pcm := self.phrases.get(key)) is not None:
            speech.say_audio(pcm, self.phrases.sample_rate)
        else:
            speech.say(PHRASES[key])


class SpeechProbe:
    """Destination des phrases qui retient si quelque chose a été dit (phase 15j : l'appel
    normal n'est jamais coupé une fois qu'il a commencé à parler)."""

    def __init__(self, speech: SpeechSink) -> None:
        self._speech = speech
        self.said = False

    def say(self, text: str) -> None:
        self.said = True
        self._speech.say(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        self.said = True
        self._speech.say_audio(pcm, sample_rate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._speech, name)


@dataclass
class TurnState:
    """Ce qui évolue pendant un tour (docs/AGENT_CORE.md §3.5)."""

    trust: Trust = Trust.TRUSTED  # devient « non fiable » dès que le modèle lit un contenu
    no_tools: bool = False  # après un refus ou un « non » : plus aucun outil dans ce tour
    script_failures: int = 0
    strong: bool = False  # rôle « fort » du modèle pour la suite du tour


@dataclass
class Reply:
    """Réponse d'un appel au modèle."""

    text: str
    calls: list[ToolCallReq]
    reasoning: str
    ignored: bool = False  # fenêtre de suivi : phrase pas adressée à MIKATA


class FollowUpGate:
    """Fenêtre de suivi : retient le texte tant qu'il peut être `FOLLOW_UP_MARKER`, et bloque
    une demande de répétition en début de réponse. Hors suivi, tout passe tel quel."""

    def __init__(self, active: bool) -> None:
        self.active = active
        self.ignored = False
        self.reason = ""
        self._pending = ""
        self._open = not active
        self._spoken = False

    def feed(self, text: str) -> str:
        """Texte à donner au découpeur en phrases (peut être vide : retenu)."""
        if self._open or self.ignored:
            return "" if self.ignored else text
        self._pending += text
        head = self._pending.lstrip()
        if head.upper().startswith(FOLLOW_UP_MARKER):
            self._ignore("le LLM juge que la phrase ne s'adresse pas à MIKATA")
            return ""
        if FOLLOW_UP_MARKER.startswith(head.upper()):
            return ""  # peut encore être le marqueur
        self._open = True
        released, self._pending = self._pending, ""
        return released

    def finish(self) -> str:
        """Fin du flux : le texte encore retenu, ou "" si c'était le marqueur."""
        if self._open or self.ignored:
            return ""
        head = self._pending.strip().rstrip(".!").upper()
        if head and FOLLOW_UP_MARKER.startswith(head) and len(head) >= 3:
            self._ignore("le LLM juge que la phrase ne s'adresse pas à MIKATA")
            return ""
        self._open = True
        released, self._pending = self._pending, ""
        return released

    def allow(self, fragment: str) -> bool:
        """Le fragment peut-il être dit ?"""
        if self.ignored:
            return False
        if self.active and not self._spoken and _REPEAT_RE.search(fragment):
            self._ignore(f"demande de répétition jamais dite en suivi (« {fragment} »)")
            return False
        self._spoken = True
        return True

    def _ignore(self, reason: str) -> None:
        self.ignored, self.reason = True, reason


@dataclass
class TurnResult:
    answer: str = ""
    outcome: str = "ok"  # ok | reset | server | timeout | limit | ignored (suivi)
    first_token_at: float | None = None
    attempts: int = 0
    tool_calls: list[ToolCallReq] = field(default_factory=list)
    # Ajouts phase 4
    command: str = ""  # commande envoyée (sans mot-clé de réflexion)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_calls: int = 0
    rounds: int = 0  # étapes d'outils
    shortcut: bool = False  # outil terminal réussi : pas de second appel LLM
    usage: Usage | None = None
    cost_usd: float = 0.0
    effort: str | None = None  # niveau de réflexion au départ
    effort_final: str | None = None  # après le filet automatique
    effort_reasons: list[str] = field(default_factory=list)
    # Ajouts phase 14d : programmes exécutés (outil script) et confirmations demandées.
    scripts: list[ScriptOutcome] = field(default_factory=list)
    confirmations: int = 0
    # Ajouts phase 14e : résultats des outils directs (hors programmes), tour teinté.
    direct_results: list[ToolResult] = field(default_factory=list)
    tainted: bool = False
    narration: list[str] = field(default_factory=list)  # phase 15c : phrases d'attente dites
    # Ajouts phase 15e : programmes (code, écrit avant toute lecture non fiable ?), dans
    # l'ordre de `scripts` ; compétences appelées (nom, réussie, erreur) ; rejeu local.
    script_sources: list[tuple[str, bool]] = field(default_factory=list)
    skill_uses: list[tuple[str, bool, str]] = field(default_factory=list)
    skill_choices: list[SkillChoice] = field(default_factory=list)  # phase 17f : pick() retenus
    # Phase 17f : effets des actions dans les applis, relus après coup ; celles sans effet.
    effects: list[str] = field(default_factory=list)
    no_effect: list[str] = field(default_factory=list)
    # Phase 15g : appels directs (appel, arguments écrits avant toute lecture non fiable ?),
    # dans l'ordre de `direct_results`.
    direct_sources: list[tuple[ToolCallReq, bool]] = field(default_factory=list)
    local_skill: str = ""
    skill_via: str = ""  # phase 15j : « motif » (0 appel) ou « aiguillage » (1 petit appel)

    @property
    def effort_label(self) -> str:
        """« none », « medium (« réfléchis bien ») », « none -> low (outil en échec) »."""
        label = self.effort or "défaut"
        if self.effort_final != self.effort:
            label += f" -> {self.effort_final or 'défaut'}"
        return f"{label} ({', '.join(self.effort_reasons)})" if self.effort_reasons else label

    @property
    def path(self) -> str:
        """Chemin de la requête, pour le log (§10)."""
        if self.local_skill:
            via = "reconnue par l'aiguillage" if self.skill_via == "aiguillage" else "en local"
            return f"compétence « {self.local_skill} » {via}, {self.llm_calls} appel(s) LLM"
        if self.scripts:
            calls = sum(len(outcome.records) for outcome in self.scripts)
            end = "direct" if self.shortcut else "reformulé"
            return (
                f"programme ({len(self.scripts)}, {calls} appel(s) d'outil, {end}, "
                f"{self.llm_calls} appel(s) LLM)"
            )
        if self.shortcut:
            return f"court-circuit (outil terminal, {self.llm_calls} appel(s) LLM)"
        if self.rounds:
            return f"{self.rounds} étape(s) d'outils, {self.llm_calls} appel(s) LLM"
        return f"réponse directe, {self.llm_calls} appel(s) LLM"

    def raise_effort(self, level: str | None, reason: str) -> None:
        """Relève la réflexion pour la suite du tour (filet automatique, rôle fort)."""
        raised = at_least(self.effort_final, level)
        if raised != self.effort_final:
            logger.info("Réflexion relevée : {} -> {} ({}).", self.effort_final, raised, reason)
            self.effort_final = raised
            self.effort_reasons.append(reason)


@dataclass
class Turn:
    """Un tour en cours : ce que chaque composant lit et complète."""

    command: str
    speech: SpeechSink
    result: TurnResult
    io: TurnIO  # les outils parlent et écoutent par là (`tools.base.turn_context`)
    messages: list[dict[str, Any]]  # envoyés au modèle pendant ce tour
    steps: list[dict[str, Any]] = field(default_factory=list)  # gardés pour la conversation
    state: TurnState = field(default_factory=TurnState)
    on_phase: Callable[[str], None] | None = None
    follow_up: bool = False
    spoken: list[str] = field(default_factory=list)  # ce qui a été dit, pour la réponse
    # Phase 16c : ce que fait le tour en ce moment (« tu en es où ? »), depuis quand.
    activity: str = "thinking"
    started_at: float = field(default_factory=time.monotonic)

    def phase(self, name: str) -> None:
        if self.on_phase is not None:
            self.on_phase(name)

    def take_tool_speech(self) -> list[str]:
        """Phrases dites par les outils depuis le dernier relevé (« Extinction dans 5 s »)."""
        said = list(self.io.spoken)
        self.io.spoken.clear()
        return said
