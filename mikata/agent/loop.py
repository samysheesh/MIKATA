"""Boucle agent (§6).

1. Messages : prompt système + faits mémorisés + historique + contexte dynamique + commande.
   Les définitions d'outils, fixes et triées, partent avec chaque appel (cache de prompt).
2. Appel LLM en streaming ; chaque fragment de texte terminé part aussitôt au TTS.
3. Appels d'outils : exécutés **en parallèle** (`asyncio.gather`). Si tous sont terminaux
   et ont réussi : « C'est fait » en cache, sans second appel LLM (chemin court-circuité).
   Sinon les résultats (tronqués) repartent au LLM, au plus `llm.max_tool_rounds` étapes.
4. Erreur réseau / API avant le premier mot : un retry immédiat, puis « Je n'arrive pas à
   joindre le serveur » (en cache, fonctionne hors ligne). `llm.turn_timeout_s` par requête.
   Phase 6 : un serveur muet échoue après `llm.first_chunk_timeout_s` (et non plus
   `request_timeout_s`), sur une connexion neuve pour la relance (llm/client.py).
6. Fenêtre de suivi (ajout phase 6, `follow_up=True`) : la phrase a été entendue sans mot
   d'activation, peut-être dans une conversation à côté. Le LLM répond `FOLLOW_UP_MARKER`
   si elle ne s'adresse pas clairement à MIKATA : rien n'est dit ni fait, la fenêtre se
   ferme (`outcome = "ignored"`). Le début de la réponse est retenu tant qu'il peut être ce
   marqueur, et une demande de répétition (« j'ai pas bien entendu ») n'est jamais dite.
5. Niveau de réflexion (ajout phase 4) : `llm.reasoning_effort` par défaut, ou celui d'un
   mot-clé dit en début / fin de commande (`llm.reasoning_keywords`, détecté par le code).
   Filet : un outil en échec, ou plus de `llm.reasoning_fallback_steps` étapes -> au moins
   « low » pour la suite de la commande.
Plusieurs tours peuvent tourner en parallèle (une commande n'attend pas la précédente).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from mikata.action.effects import Trust
from mikata.action.policy import Verdict, confirm_question, decide
from mikata.action.runtime import ScriptOutcome, ScriptRuntime
from mikata.action.sdk import SdkCatalog
from mikata.agent.guard import log_action
from mikata.agent.memory import LongTermMemory, ShortTermMemory, is_reset_command
from mikata.agent.script_tool import (
    SCRIPT_SCHEMA,
    SCRIPT_TOOL,
    data_message,
    failure_message,
    has_data,
    script_guide,
    spoken_result,
)
from mikata.config import Config
from mikata.learn.playbook import Lesson, Playbook, render
from mikata.learn.reflect import is_correction, reflect, should_reflect
from mikata.learn.skills import Skill, SkillStore, call_code, induce, locate, route, verify
from mikata.learn.trace import TraceStore, TurnTrace, trace_from_result
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq, Usage
from mikata.llm.prompts import active_window_title, build_messages, dynamic_context
from mikata.llm.reasoning import at_least, detect_reasoning
from mikata.plat import platform
from mikata.tools.base import ToolError, TurnIO, current_turn, turn_context
from mikata.tts.cache import PHRASES
from mikata.tts.normalize import FragmentSplitter

if TYPE_CHECKING:
    from mikata.llm.usage import UsageTracker
    from mikata.tools.registry import ToolRegistry, ToolResult

UNTRUSTED_HISTORY = "(contenu lu ailleurs, non gardé dans l'historique)"
SCRIPT_FAILURES_BEFORE_ESCALATION = 2
MAX_SCRIPT_FAILURES = 3  # programmes en échec dans un même tour
# Phase 15j : l'appel normal a fini le premier sans rien dire (il demande un outil) : on
# laisse encore ce temps à l'aiguillage avant d'exécuter quoi que ce soit.
ROUTE_GRACE_S = 1.0


@dataclass
class _TurnState:
    """Ce qui évolue pendant un tour (docs/AGENT_CORE.md §3.5)."""

    trust: Trust = Trust.TRUSTED  # devient « non fiable » dès que le modèle lit un contenu
    no_tools: bool = False  # après un refus ou un « non » : plus aucun outil dans ce tour
    script_failures: int = 0
    strong: bool = False  # rôle « fort » du modèle pour la suite du tour


MAX_ATTEMPTS = 2  # un retry immédiat (§6)
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
# Applis ouvertes (contexte dynamique) : relevées au mot d'activation, pendant que
# l'utilisateur parle ; relues au moment de la commande seulement si elles datent.
OPEN_APPS_FRESH_S = 10.0
OPEN_APPS_TIMEOUT_S = 0.3  # le contexte ne doit pas retarder l'appel LLM
OPEN_APPS_PREWARM_TIMEOUT_S = 2.0


class SpeechSink(Protocol):
    """Destination des phrases d'une commande (`audio.player.SpeechStream`)."""

    def say(self, text: str) -> None: ...
    def say_audio(self, pcm: Any, sample_rate: int) -> None: ...


class CachedPhrases(Protocol):
    sample_rate: int

    def get(self, key: str) -> Any: ...


# Phase 15c : phrase d'attente (clé de PHRASES) selon l'étape en cours.
_ACTIVITY_PREFIXES = (
    ("browser_", "browser"),
    ("web_search", "search"),
    ("fetch_page", "search"),
    ("find_", "files"),
    ("list_folder", "files"),
    ("read_file", "files"),
    ("uia_", "app"),
)
_SDK_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\(")


def activity_of(calls: Sequence[ToolCallReq]) -> str:
    """L'étape lancée par ces appels : browser, search, files, app ou work."""
    names: list[str] = []
    for call in calls:
        if call.name == SCRIPT_TOOL:  # un programme : les fonctions qu'il appelle
            names.extend(_SDK_CALL.findall(str(call.arguments.get("code", ""))))
        else:
            names.append(call.name)
    for prefix, kind in _ACTIVITY_PREFIXES:
        if any(name.startswith(prefix) for name in names):
            return kind
    return "work"


class Narrator:
    """Phase 15c (réglage agent.narration « bavard ») : destination des phrases d'une
    commande qui retient quand MIKATA a parlé pour la dernière fois. `run` dit une phrase
    d'attente (en cache, sans LLM) si rien n'a été dit `after_s` secondes après le début,
    selon l'étape en cours (`activity`), puis « ça prend un peu plus de temps » après un
    nouveau silence de 2 × `after_s`. Jamais pendant une écoute (confirmation)."""

    MAX_LINES = 2
    POLL_S = 0.2

    def __init__(
        self,
        speech: SpeechSink,
        say_phrase: Callable[[SpeechSink, str], None],
        after_s: float,
    ) -> None:
        self._speech = speech
        self._say_phrase = say_phrase
        self.after_s = after_s
        self.activity = "thinking"
        self.listening = False
        self.lines: list[str] = []  # phrases d'attente dites (clés de PHRASES)
        self._started = time.monotonic()
        self._last_said: float | None = None

    def say(self, text: str) -> None:
        self._last_said = time.monotonic()
        self._speech.say(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        self._last_said = time.monotonic()
        self._speech.say_audio(pcm, sample_rate)

    def __getattr__(self, name: str) -> Any:  # drain, cancelled... de la vraie destination
        return getattr(self._speech, name)

    def due(self, now: float) -> str | None:
        """La phrase d'attente à dire maintenant, ou None."""
        if self.listening or len(self.lines) >= self.MAX_LINES:
            return None
        if self._last_said is None:
            if now - self._started >= self.after_s:
                return f"wait_{self.activity}"
            return None
        if now - self._last_said >= 2 * self.after_s and "wait_long" not in self.lines:
            return "wait_long"
        return None

    async def run(self) -> None:
        while len(self.lines) < self.MAX_LINES:
            await asyncio.sleep(self.POLL_S)
            key = self.due(time.monotonic())
            if key is None:
                continue
            key = key if key in PHRASES else "wait_work"
            logger.info("Narration : « {} » (étape : {}).", PHRASES[key], self.activity)
            self.lines.append(key)
            self._say_phrase(self, key)


class SilentSpeech:
    """Destination muette (option --ask --no-audio) : le texte est seulement renvoyé."""

    def say(self, text: str) -> None:
        pass

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        pass

    async def drain(self) -> None:
        pass


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


def _listening(
    narrator: Narrator, listen: Callable[[float], Awaitable[str]] | None
) -> Callable[[float], Awaitable[str]] | None:
    """L'écoute d'une réponse (confirmation) suspend les phrases d'attente."""
    if listen is None:
        return None

    async def wrapped(timeout_s: float) -> str:
        narrator.listening = True
        try:
            return await listen(timeout_s)
        finally:
            narrator.listening = False

    return wrapped


def _with_reasoning(messages: list[dict[str, Any]], effort: str | None) -> list[dict[str, Any]]:
    """Messages envoyés pour un appel : la réflexion d'un appel d'outil de la commande en
    cours n'est jointe que si elle existe, ou si la réflexion est demandée pour cet appel.

    DeepSeek en mode réflexion exige `reasoning_content` sur ces messages : après un passage
    de « none » à « low » (filet), elle est donc envoyée vide plutôt qu'absente.
    """
    thinking = effort in ("low", "medium", "high")
    out = []
    for message in messages:
        if "reasoning_content" in message and not (message["reasoning_content"] or thinking):
            message = {k: v for k, v in message.items() if k != "reasoning_content"}
        out.append(message)
    return out


class _SpeechProbe:
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


def _inducible(result: TurnResult) -> str:
    """Le programme de ce tour qui peut devenir une compétence, ou "" : le dernier qui a
    réussi, écrit avant toute lecture d'un contenu non fiable (un texte piégé ne doit pas
    pouvoir s'inscrire dans une compétence), avec au moins deux appels, sans compétence.
    Sans programme, l'enchaînement des appels directs du tour (phase 15g, voir
    `_steps_as_program`)."""
    for (code, trusted), outcome in zip(
        reversed(result.script_sources), reversed(result.scripts), strict=False
    ):
        if outcome.ok and trusted and len(outcome.records) >= 2 and "skills." not in code:
            return code
    if result.scripts:
        return ""
    return _steps_as_program(result)


def _steps_as_program(result: TurnResult) -> str:
    """Phase 15g : « ouvre YouTube, lis la page, clique la vidéo » fait en trois appels
    directs, écrit comme un programme pour l'apprentissage. Un argument choisi après la
    lecture d'un contenu non fiable n'est jamais recopié : il devient `...`, et la
    compétence devra le retrouver elle-même dans ce qu'elle lit."""
    steps = result.direct_sources
    if len(steps) < 2 or len(result.direct_results) != len(steps):
        return ""
    if not all(r.ok for r in result.direct_results):
        return ""
    lines = []
    for call, trusted in steps:
        if call.name.endswith("_skill") or call.name == "remember":
            return ""  # apprendre à apprendre ou à retenir : non
        args = ", ".join(
            f"{key}={value!r}" if trusted else f"{key}=..." for key, value in call.arguments.items()
        )
        lines.append(f"{call.name}({args})")
    return "\n".join(lines)


HISTORY_FILE = "data/history.json"


SUMMARY_PROMPT = """\
Tu tiens la mémoire de MIKATA, l'assistant vocal de l'utilisateur. Mets à jour le résumé de \
leur conversation avec les échanges ci-dessous. Garde l'essentiel : ce que l'utilisateur a \
dit de lui, ses demandes et préférences, les décisions, les réponses importantes, ce qui \
reste en cours. Laisse de côté les actions banales déjà faites (« C'est fait »). Au plus \
120 mots, en phrases simples, sans markdown, en français. Réponds seulement par le résumé."""


def short_term_memory(config: Config) -> ShortTermMemory:
    """Historique de conversation : en mémoire, ou sur disque si `agent.history_persist`."""
    agent = config.agent
    if agent.history_persist:
        return ShortTermMemory(
            agent.history_max_tokens,
            agent.history_persist_h * 3600,
            path=config.resolve(HISTORY_FILE),
            summarize=agent.history_summarize,
        )
    return ShortTermMemory(
        agent.history_max_tokens,
        agent.session_reset_min * 60,
        summarize=agent.history_summarize,
    )


@dataclass
class _Reply:
    text: str
    calls: list[ToolCallReq]
    reasoning: str
    ignored: bool = False  # fenêtre de suivi : phrase pas adressée à MIKATA


class _FollowUpGate:
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


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        config: Config,
        memory: ShortTermMemory | None = None,
        phrases: CachedPhrases | None = None,
        window_title: Callable[[], str | None] = active_window_title,
        clock: Callable[[], datetime] = datetime.now,
        *,
        tools: ToolRegistry | None = None,
        long_memory: LongTermMemory | None = None,
        skills: SkillStore | None = None,
        usage: UsageTracker | None = None,
        open_apps: Callable[[], Awaitable[Sequence[str]]] | None = None,
        installed_apps: Callable[[], Sequence[str]] | None = None,
        playbook: Playbook | None = None,
        traces: TraceStore | None = None,
        jev: Any = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.memory = memory if memory is not None else short_term_memory(config)
        self.phrases = phrases
        self.tools = tools
        self.long_memory = long_memory
        self.skills = skills
        self.usage = usage
        self._window_title = window_title
        self._clock = clock
        self._open_apps = open_apps
        self._installed_apps = installed_apps  # liste déjà en mémoire, jamais bloquante
        self._apps: Sequence[str] = ()
        self._apps_at = float("-inf")
        self._tools_warmed = False
        self._summary_task: asyncio.Task[None] | None = None
        self.jev = jev  # phase 15k : premier étage de l'aiguillage (mikata.learn.jev)
        # Phase 15e : compétences. Dernier programme qui peut en devenir une (demande,
        # code), compétences du dernier tour (pour une correction), apprentissage en fond.
        self._skill_task: asyncio.Task[None] | None = None
        self._last_inducible: tuple[str, str] | None = None
        self._last_skill_uses: tuple[list[str], float] | None = None
        # Phase 14e : carnet de leçons, traces, réflexion en arrière-plan.
        self.playbook = playbook
        self.traces = traces
        self._last_trace: tuple[TurnTrace, float] | None = None
        self._confirm_lock = asyncio.Lock()
        self._reflect_task: asyncio.Task[None] | None = None
        # Phase 14d : SDK des programmes, tiré du registre (une fois).
        self._catalog = SdkCatalog(tools.tools, platform()) if tools is not None else None
        self._script_guide = (
            script_guide(self._catalog.structured_returns())
            if self._catalog is not None and config.agent.scripts
            else None
        )
        if tools is not None and tools.ctx.history is None:  # outil set_history_duration
            tools.ctx.history = self.memory
        if tools is not None and skills is not None:  # outil remember_skill
            tools.ctx.skills_capture = self._capture_skill

    async def prewarm(self) -> None:
        """À chaque mot d'activation : connexion LLM préchauffée, applis ouvertes relevées ;
        une fois : index des applications."""
        jobs = []
        if self.tools is not None and not self._tools_warmed:
            self._tools_warmed = True
            jobs.append(self.tools.warm())
        if self._open_apps is not None:
            jobs.append(self._refresh_apps(OPEN_APPS_PREWARM_TIMEOUT_S))
        prewarm = getattr(self.llm, "prewarm", None)
        if prewarm is not None:
            jobs.append(prewarm())
        skills = self.skills.usable() if self.skills is not None else []
        if self.jev is not None and any(s.status == "active" for s in skills):
            jobs.append(self.jev.prewarm())  # phase 15k : aiguillage prêt
        await asyncio.gather(*jobs)

    def _say_phrase(self, speech: SpeechSink, key: str) -> None:
        if self.phrases is not None and (pcm := self.phrases.get(key)) is not None:
            speech.say_audio(pcm, self.phrases.sample_rate)
        else:
            speech.say(PHRASES[key])

    def say_phrase(self, speech: SpeechSink, key: str) -> None:
        """Alias public de `_say_phrase`."""
        self._say_phrase(speech, key)

    async def respond(
        self,
        command: str,
        speech: SpeechSink,
        *,
        label: str = "commande",
        on_phase: Callable[[str], None] | None = None,
        listen: Callable[[float], Awaitable[str]] | None = None,
        follow_up: bool = False,
    ) -> TurnResult:
        """Répond à voix haute à `command` ; renvoie le texte et les mesures du tour.

        `on_phase("acting")` / `on_phase("thinking")` : outils en cours / LLM relancé.
        `listen(délai)` (phase 5) : écoute une réponse sans mot d'activation, pour les
        confirmations orales et le « stop » avant extinction ; None sans micro.
        `follow_up` (phase 6) : phrase de la fenêtre de suivi, que le LLM peut ignorer.
        """
        llm_cfg = self.config.llm
        request = detect_reasoning(command, llm_cfg.reasoning_keywords)
        result = TurnResult(command=request.text, effort=llm_cfg.reasoning_effort)
        if request.level is not None:
            result.effort = request.level
            result.effort_reasons.append(f"« {request.keyword} »")
            logger.info("Réflexion demandée : {} (mot-clé « {} »).", request.level, request.keyword)
        result.effort_final = result.effort
        if is_reset_command(request.text):
            self.memory.clear()
            self._say_phrase(speech, "reset")
            result.answer, result.outcome = PHRASES["reset"], "reset"
            return result
        narrator: Narrator | None = None
        narrating: asyncio.Task[None] | None = None
        # Fenêtre de suivi : la phrase sera peut-être ignorée, MIKATA ne dit rien d'avance.
        if self.config.agent.narration == "bavard" and not follow_up:
            narrator = Narrator(speech, self._say_phrase, self.config.agent.narration_after_s)
            speech = narrator
            listen = _listening(narrator, listen)
            narrating = asyncio.create_task(narrator.run())
        try:
            async with asyncio.timeout(llm_cfg.turn_timeout_s):
                await self._run_turn(request.text, speech, result, on_phase, listen, follow_up)
        except TimeoutError:
            logger.warning("Requête LLM trop longue (> {} s).", llm_cfg.turn_timeout_s)
            result.outcome = "timeout"
            self._say_phrase(speech, "timeout")
        finally:
            if narrating is not None:
                narrating.cancel()
            if narrator is not None:
                result.narration = [PHRASES[key] for key in narrator.lines]
            if self.usage is not None and result.llm_calls:
                result.cost_usd = self.usage.record(label, result.usage, result.llm_calls)
        return result

    async def _refresh_apps(self, timeout: float) -> None:
        assert self._open_apps is not None
        try:
            async with asyncio.timeout(timeout):
                self._apps = await self._open_apps()
            self._apps_at = time.monotonic()
        except Exception as exc:  # contexte facultatif
            logger.debug("Applis ouvertes inconnues : {!r}", exc)

    async def _context(self) -> str:
        stale = time.monotonic() - self._apps_at > OPEN_APPS_FRESH_S
        if self._open_apps is not None and stale:
            await self._refresh_apps(OPEN_APPS_TIMEOUT_S)
        return dynamic_context(self._clock(), self._window_title(), self._apps)

    async def context(self) -> str:
        """Alias public de `_context`."""
        return await self._context()

    def _escalate(self, result: TurnResult, reason: str) -> None:
        raised = at_least(result.effort_final, "low")
        if raised != result.effort_final:
            logger.info("Réflexion relevée : {} -> {} ({}).", result.effort_final, raised, reason)
            result.effort_final = raised
            result.effort_reasons.append(reason)

    async def _run_turn(
        self,
        command: str,
        speech: SpeechSink,
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
        listen: Callable[[float], Awaitable[str]] | None = None,
        follow_up: bool = False,
    ) -> None:
        facts = self.long_memory.facts if self.long_memory is not None else ()
        drain = getattr(speech, "drain", None)
        turn = TurnIO(say=speech.say, listen=listen, **({"drain": drain} if drain else {}))
        installed = self._installed_apps() if self._installed_apps is not None else ()
        context = await self._context()
        if follow_up:
            context = f"{context} {FOLLOW_UP_NOTE}"
        lessons = self._lessons_for(command, context)
        started = time.monotonic()
        messages = build_messages(
            self.memory.messages(),
            command,
            context,
            facts,
            installed,
            script_guide=self._script_guide,
            lessons=render(lessons),
            narration=self.config.agent.narration == "bavard",
        )
        schemas = self.tools.schemas() if self.tools is not None else []
        if self._script_guide is not None:
            schemas = [*schemas, SCRIPT_SCHEMA]
        state = _TurnState()
        max_rounds = self.config.llm.max_tool_rounds
        fallback_steps = self.config.llm.reasoning_fallback_steps
        spoken: list[str] = []
        steps: list[dict[str, Any]] = []  # appels d'outils gardés dans l'historique
        # Phase 15e : une compétence active dont un motif colle est rejouée sans LLM ;
        # sinon les compétences qui correspondent sont signalées au modèle.
        replayed = await self._skill_first(
            command, speech, messages, steps, result, turn, state, on_phase
        )
        if replayed is not None:
            spoken.extend(replayed)
            if result.shortcut:
                max_rounds = -1  # tout est dit : pas d'appel au modèle
        # Phase 15j : aiguillage, en parallèle du premier appel normal.
        router = None if replayed is not None else self._start_router(command, follow_up)
        for round_no in range(max_rounds + 1):
            last = round_no == max_rounds  # dernier appel : sans outils, réponse obligée
            gate = _FollowUpGate(follow_up and round_no == 0)
            offered = [] if last or state.no_tools else schemas
            if router is not None:
                reply, routed = await self._race(
                    router, messages, offered, speech, result, gate, state
                )
                router = None
                if routed is not None:
                    skill, values = routed
                    spoken.extend(
                        await self._replay(
                            skill,
                            values,
                            "aiguillage",
                            speech,
                            messages,
                            steps,
                            result,
                            turn,
                            state,
                            on_phase,
                        )
                    )
                    if result.shortcut:
                        break
                    continue  # une donnée à formuler, ou un échec : le modèle reprend
            else:
                reply = await self._call_llm(
                    messages, offered, speech, result, gate, strong=state.strong
                )
            if reply is None:
                result.outcome = "server"
                self._say_phrase(speech, "server")
                return
            if reply.ignored:  # rien n'est dit ni fait, rien n'entre dans l'historique
                logger.info("Fenêtre de suivi : « {} » ignorée ({}).", command, gate.reason)
                result.outcome = "ignored"
                return
            if reply.text.strip():
                spoken.append(reply.text.strip())
            if not reply.calls:
                break
            if last or self.tools is None or state.no_tools:
                result.tool_calls.extend(reply.calls)
                logger.warning("Le LLM demande encore des outils : abandon ({} étapes).", round_no)
                result.outcome = "limit"
                if not spoken:
                    self._say_phrase(speech, "error")
                break
            if isinstance(speech, Narrator):
                speech.activity = activity_of(reply.calls)
            with turn_context(turn):  # les outils parlent et écoutent dans cette commande
                results = await self._run_tools(reply, messages, steps, result, on_phase, state)
            if isinstance(speech, Narrator):
                speech.activity = "thinking"
            tool_spoken = list(turn.spoken)  # ex. « Extinction dans 5 secondes... »
            spoken.extend(tool_spoken)
            turn.spoken.clear()
            earlier_failure = any(not r.ok for r in result.direct_results) or any(
                not outcome.ok for outcome in result.scripts
            )
            # Après un échec dans le tour, pas de « C'est fait » automatique : seul un outil
            # ou un programme qui a dit sa propre phrase conclut ; sinon le modèle explique
            # (une phrase du modèle avant ses outils annonce une étape, elle ne conclut pas).
            if all(r.ok and r.terminal for r in results) and (tool_spoken or not earlier_failure):
                result.shortcut = True
                if not spoken:  # une action simple : « C'est fait » en cache
                    self._say_phrase(speech, "done")
                    spoken.append(PHRASES["done"])
                break
            if on_phase is not None:
                on_phase("thinking")
            if any(not r.ok for r in results):
                self._escalate(result, "outil en échec")
            elif result.rounds >= fallback_steps:
                self._escalate(result, f"plus de {fallback_steps} étape(s)")
        result.answer = " ".join(spoken).strip()
        if result.answer or steps:
            self.memory.add(command, result.answer or PHRASES["done"], steps)
            self._schedule_summary()
        result.tainted = state.trust is Trust.UNTRUSTED
        self._after_turn(command, result, lessons, time.monotonic() - started)
        self._skills_after_turn(command, result)

    # -- apprentissage (phase 14e) -----------------------------------------------------

    def _lessons_for(self, command: str, context: str) -> list[Lesson]:
        cfg = self.config.learning
        if self.playbook is None or not cfg.enabled or cfg.lessons_in_context == 0:
            return []
        return self.playbook.select(command, context, cfg.lessons_in_context)

    def _after_turn(
        self, command: str, result: TurnResult, lessons: list[Lesson], seconds: float
    ) -> None:
        """Après un tour qui a agi : trace, bilan des leçons présentes, et réflexion en
        arrière-plan s'il y a un signal fiable (échec corrigé, ou correction de
        l'utilisateur). Jamais bloquant, jamais après un contenu non fiable."""
        cfg = self.config.learning
        if not cfg.enabled or (self.playbook is None and self.traces is None):
            return
        ids = [lesson.id for lesson in lessons]
        trace = trace_from_result(command, result, seconds, ids, tainted=result.tainted)
        previous = self._last_trace
        self._last_trace = (trace, time.monotonic()) if trace.acted else None
        if not trace.acted:
            return
        if self.traces is not None and cfg.traces:
            self.traces.save(trace)
        if self.playbook is None:
            return
        failed = (
            trace.outcome != "ok"
            or bool(trace.failures)
            or any(not script.ok for script in trace.scripts)
        )
        if ids:
            self.playbook.mark(ids, helpful=not failed)
        correction_of = None
        if (
            previous is not None
            and time.monotonic() - previous[1] <= cfg.correction_window_s
            and is_correction(command)
        ):
            correction_of = previous[0]
            if correction_of.lessons:  # elles étaient là quand MIKATA s'est trompé
                self.playbook.mark(correction_of.lessons, helpful=False)
        if not should_reflect(trace, correction_of):
            return
        if self._reflect_task is not None and not self._reflect_task.done():
            return
        self._reflect_task = asyncio.create_task(
            self._reflect(trace, correction_of), name="réflexion"
        )

    async def _reflect(self, trace: TurnTrace, correction_of: TurnTrace | None) -> None:
        assert self.playbook is not None
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                lessons, usage = await reflect(
                    self.llm,
                    trace,
                    correction_of=correction_of,
                    effort=self.config.learning.reflect_effort,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # hors ligne, réponse illisible : on n'apprend rien
            logger.warning("Réflexion impossible : {}", exc)
            return
        if self.usage is not None:
            self.usage.record("réflexion", usage, 1)
        source = "correction" if correction_of is not None else "échec corrigé"
        for scope, text in lessons:
            self.playbook.add(scope, text, source)

    async def learning_idle(self) -> None:
        """Attend la fin de la réflexion et de l'écriture de compétence en cours (tests,
        arrêt propre)."""
        tasks = [t for t in (self._reflect_task, self._skill_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- compétences (phase 15e) -------------------------------------------------------

    def _skills_on(self) -> bool:
        return self.skills is not None and self.config.skills.enabled and self.tools is not None

    def _callable_skills(self) -> dict[str, Skill]:
        if not self._skills_on():
            return {}
        assert self.skills is not None
        return {skill.call_name: skill for skill in self.skills.usable()}

    async def _skill_first(
        self,
        command: str,
        speech: SpeechSink,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        result: TurnResult,
        turn: TurnIO,
        state: _TurnState,
        on_phase: Callable[[str], None] | None,
    ) -> list[str] | None:
        """Avant le premier appel au modèle. Compétence active qui colle : rejouée en
        local (le programme d'une ligne `return skills.<nom>(...)`, par le même chemin
        qu'un programme du modèle) ; réussie avec une phrase ou rien à dire, le tour est
        fini sans LLM (`result.shortcut`) ; avec une donnée, le modèle la formule ; en
        échec, le modèle reprend la demande. Rend les phrases dites, ou None si rien n'a
        été rejoué."""
        if not self._skills_on():
            return None
        assert self.skills is not None
        found = self.skills.match(command)
        if found is not None and found[0].status == "active" and self.config.skills.local_replay:
            skill, values = found
            return await self._replay(
                skill, values, "motif", speech, messages, steps, result, turn, state, on_phase
            )
        hint = self._skill_hint(command, found)
        if hint:
            messages.insert(-1, {"role": "system", "content": hint})
        return None

    async def _replay(
        self,
        skill: Skill,
        values: dict[str, str],
        via: str,
        speech: SpeechSink,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        result: TurnResult,
        turn: TurnIO,
        state: _TurnState,
        on_phase: Callable[[str], None] | None,
    ) -> list[str]:
        """Exécute une compétence active sans le modèle (le programme d'une ligne
        `return skills.<nom>(...)`, par le même chemin qu'un programme du modèle). Réussie
        avec une phrase ou rien à dire : tour fini (`result.shortcut`) ; avec une donnée, le
        modèle la formule ; en échec, le modèle reprend la demande. Rend les phrases dites."""
        call = ToolCallReq("competence", SCRIPT_TOOL, {"code": call_code(skill, values)})
        result.local_skill, result.skill_via = skill.name, via
        with turn_context(turn):
            results = await self._run_tools(
                _Reply("", [call], ""), messages, steps, result, on_phase, state
            )
        spoken = list(turn.spoken)
        turn.spoken.clear()
        (outcome,) = results
        if outcome.ok and outcome.terminal:
            result.shortcut = True
            if not spoken:
                self._say_phrase(speech, "done")
                spoken.append(PHRASES["done"])
            logger.info("Compétence « {} » ({}) : {}.", skill.name, via, values)
            return spoken
        if not outcome.ok:
            logger.info("Compétence « {} » : échec ({}), le modèle reprend.", skill.name, via)
            messages.insert(
                len(messages) - 2 - len(results),  # avant la demande et le rejeu
                {
                    "role": "system",
                    "content": f"MIKATA a essayé la compétence {skill.signature}, qui a "
                    "échoué (voir le résultat). Fais la demande autrement, sans elle.",
                },
            )
        return spoken

    def _start_router(self, command: str, follow_up: bool) -> asyncio.Task[Any] | None:
        """Phase 15j : lance l'aiguillage s'il y a des compétences actives. Jamais dans la
        fenêtre de suivi (la phrase n'est peut-être pas pour MIKATA)."""
        if follow_up or not self._skills_on() or not self.config.skills.route:
            return None
        assert self.skills is not None
        active = [s for s in self.skills.usable() if s.status == "active"]
        if not active:
            return None
        return asyncio.create_task(self._route(command, active), name="aiguillage")

    async def _route(
        self, command: str, active: list[Skill]
    ) -> tuple[Skill, dict[str, str]] | None:
        """Aiguillage en deux étages (phase 15k). Jev (~0,4 s) décide seul quand il est sûr
        (compétence ou « aucune ») ; s'il hésite, ne répond pas ou n'est pas configuré, le
        petit appel LLM décide. Les deux partent ensemble : un doute de Jev ne coûte rien."""
        llm = asyncio.create_task(self._route_llm(command, active))
        try:
            if self.jev is not None:
                try:
                    async with asyncio.timeout(self.config.skills.jev_timeout_s):
                        decision = await self.jev.decide(command, active)
                except TimeoutError:
                    logger.info("Jev : trop lent, l'aiguillage LLM décide.")
                else:
                    if decision.kind == "aucune":
                        return None
                    if decision.kind == "compétence" and decision.skill is not None:
                        values = {k: locate(command, v) for k, v in decision.values.items()}
                        if all(values.values()):
                            return decision.skill, values  # type: ignore[return-value]
            return await llm
        finally:
            if not llm.done():
                llm.cancel()
                await asyncio.gather(llm, return_exceptions=True)

    async def _route_llm(
        self, command: str, active: list[Skill]
    ) -> tuple[Skill, dict[str, str]] | None:
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.config.skills.route_timeout_s):
                routed, usage = await route(self.llm, command, active)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # trop lent, hors ligne : l'appel normal décide seul
            logger.info("Aiguillage sans réponse : {}", str(exc) or type(exc).__name__)
            return None
        if self.usage is not None:
            self.usage.record("aiguillage", usage, 1)
        ms = (time.monotonic() - start) * 1000
        if routed is None:
            logger.info("Aiguillage : aucune compétence ({:.0f} ms).", ms)
        else:
            logger.info("Aiguillage : « {} » {} ({:.0f} ms).", routed[0].name, routed[1], ms)
        return routed

    async def _race(
        self,
        router: asyncio.Task[Any],
        messages: list[dict[str, Any]],
        offered: list[dict[str, Any]],
        speech: SpeechSink,
        result: TurnResult,
        gate: _FollowUpGate,
        state: _TurnState,
    ) -> tuple[_Reply | None, tuple[Skill, dict[str, str]] | None]:
        """Premier appel normal et aiguillage en même temps. L'aiguillage l'emporte s'il
        reconnaît une compétence avant que l'appel normal ait dit quoi que ce soit : l'appel
        normal est alors coupé, ou sa réponse (des outils pas encore exécutés) écartée.
        S'il a fini le premier sans parler, l'aiguillage a encore `ROUTE_GRACE_S`."""
        probe = _SpeechProbe(speech)
        main = asyncio.create_task(
            self._call_llm(messages, offered, probe, result, gate, strong=state.strong)
        )
        done, _ = await asyncio.wait({router, main}, return_when=asyncio.FIRST_COMPLETED)
        if main in done and not router.done():
            reply = main.result()
            if reply is not None and not reply.text.strip() and not probe.said:
                await asyncio.wait({router}, timeout=ROUTE_GRACE_S)
        if not router.done():
            router.cancel()
            await asyncio.gather(router, return_exceptions=True)
            return await main, None
        routed = router.result()
        if routed is not None:
            result.llm_calls += 1
        if routed is None or probe.said:
            return await main, None
        if main.done():
            reply = main.result()
            if reply is not None and reply.text.strip():
                return reply, None  # il a déjà répondu en mots : on garde sa réponse
            logger.info("Aiguillage : réponse de l'appel normal écartée (outils non exécutés).")
        else:
            main.cancel()
            await asyncio.gather(main, return_exceptions=True)
        return None, routed

    def _skill_hint(self, command: str, found: tuple[Skill, dict[str, str]] | None) -> str:
        """Note pour le modèle : la compétence qui colle (avec ses valeurs), sinon les
        compétences proches. Rien pour une demande sans rapport."""
        assert self.skills is not None
        if found is not None:
            skill, values = found
            args = ", ".join(f"{k}={v!r}" for k, v in values.items())
            return (
                f"Compétence apprise qui correspond : {skill.signature} — {skill.description} "
                f"Si elle convient, appelle-la dans un programme : "
                f"`return skills.{skill.call_name}({args})`."
            )
        related = self.skills.related(command)
        if not related:
            return ""
        lines = [f"- {s.signature} — {s.description}" for s in related]
        return (
            "Compétences apprises proches (appelables dans un programme, si l'une convient "
            "exactement) :\n" + "\n".join(lines)
        )

    def _skills_after_turn(self, command: str, result: TurnResult) -> None:
        """Bilan des compétences utilisées, correction de l'utilisateur, et programme à
        apprendre. Jamais bloquant : l'écriture d'une compétence part en arrière-plan."""
        if not self._skills_on():
            return
        assert self.skills is not None
        now = time.monotonic()
        previous = self._last_skill_uses
        if (
            previous is not None
            and now - previous[1] <= self.config.learning.correction_window_s
            and is_correction(command)
        ):
            self.skills.correct(previous[0])
        turn_ok = result.outcome == "ok"
        for name, ok, error in result.skill_uses:
            self.skills.record(
                name, ok and turn_ok, error or result.outcome, local=bool(result.local_skill)
            )
        names = [name for name, _, _ in result.skill_uses]
        self._last_skill_uses = (names, now) if names else None
        code = _inducible(result)
        self._last_inducible = (command, code) if code and turn_ok else None
        if (
            self._last_inducible is None
            or names
            or not self.config.skills.learn
            or self.skills.match(command) is not None
            or self.skills.related(command)  # une voisine existe : pas de doublon
            or (self._skill_task is not None and not self._skill_task.done())
        ):
            return
        self._skill_task = asyncio.create_task(
            self._learn_skill(command, self._last_inducible[1]), name="compétence"
        )

    async def _learn_skill(self, command: str, code: str) -> Skill | None:
        """Écrit la compétence (un appel LLM), la vérifie dans le code, la range à l'essai."""
        assert self.skills is not None
        try:
            async with asyncio.timeout(self.config.llm.turn_timeout_s):
                candidate, usage = await induce(
                    self.llm, command, code, self.config.skills.learn_effort
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # hors ligne, réponse illisible : rien n'est appris
            logger.warning("Compétence non écrite : {}", str(exc) or type(exc).__name__)
            return None
        if self.usage is not None:
            self.usage.record("compétence", usage, 1)
        if not candidate:
            logger.info("Rien de réutilisable dans « {} ».", command)
            return None
        actions: set[str] | None = None
        if self._catalog is not None:
            entries = self._catalog.entries.values()
            actions = {e.name for e in entries} | {e.tool for e in entries}
        skill, reason = verify(candidate, command, code, actions)
        if skill is None:
            logger.info("Compétence refusée à la vérification ({}) : {}", command, reason)
            return None
        try:
            return self.skills.adopt(skill)
        except (OSError, ValueError) as exc:
            logger.warning("Compétence non rangée : {}", exc)
            return None

    async def _capture_skill(self, description: str) -> str:
        """Outil remember_skill : « retiens comment faire ça » sur le dernier programme."""
        if self._last_inducible is None or self.skills is None:
            raise ToolError("aucun programme récent à retenir")
        command, code = self._last_inducible
        if self._skill_task is not None and not self._skill_task.done():
            await asyncio.gather(self._skill_task, return_exceptions=True)
            if self.skills.match(command) is not None:
                skill = self.skills.match(command)[0]  # type: ignore[index]
                return f"C'est déjà retenu : {skill.description}"
        skill = await self._learn_skill(command, code)
        if skill is None:
            raise ToolError("je n'ai pas réussi à en faire une compétence réutilisable")
        return (
            f"Retenu : {skill.description} Je la referai sans réfléchir après "
            f"{self.config.skills.promote_after} réussites."
        )

    def _schedule_summary(self) -> None:
        """Résume en arrière-plan les échanges sortis de l'historique (plafond atteint)."""
        if self.memory.pending and (self._summary_task is None or self._summary_task.done()):
            self._summary_task = asyncio.create_task(self._summarize(), name="résumé")

    async def _summarize(self) -> None:
        while self.memory.pending:
            generation, previous, exchanges = self.memory.summary_request()
            lines = [f"Résumé actuel : {previous or '(aucun)'}", "", "Échanges à intégrer :"]
            for user, assistant in exchanges:
                lines += [f"Utilisateur : {user}", f"MIKATA : {assistant}"]
            messages = [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ]
            parts: list[str] = []
            usage: Usage | None = None
            try:
                async with asyncio.timeout(self.config.llm.turn_timeout_s):
                    async for event in self.llm.stream(messages, [], reasoning_effort="none"):
                        if isinstance(event, TextDelta):
                            parts.append(event.text)
                        elif isinstance(event, Done):
                            usage = event.usage
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # hors ligne : nouvel essai au prochain échange
                logger.warning("Résumé de l'historique impossible : {}", exc)
                return
            if self.usage is not None:
                self.usage.record("résumé", usage, 1)
            summary = "".join(parts).strip()
            if not self.memory.apply_summary(generation, len(exchanges), summary):
                return  # conversation remise à zéro entre-temps, ou résumé vide
            logger.info(
                "Historique : {} ancien(s) échange(s) résumé(s) ({} caractères).",
                len(exchanges),
                len(self.memory.summary),
            )

    async def _run_tools(
        self,
        reply: _Reply,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        result: TurnResult,
        on_phase: Callable[[str], None] | None,
        state: _TurnState | None = None,
    ) -> list[ToolResult]:
        assert self.tools is not None
        state = state or _TurnState()
        result.rounds += 1
        calls = [
            call if call.id else ToolCallReq(f"call_{result.rounds}_{i}", call.name, call.arguments)
            for i, call in enumerate(reply.calls)
        ]
        result.tool_calls.extend(calls)
        if on_phase is not None:
            on_phase("acting")
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": reply.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ],
        }
        # La réflexion repart avec ses appels d'outils pendant la commande (exigé par
        # DeepSeek en mode réflexion, cf. `_with_reasoning`), mais pas dans l'historique des
        # commandes suivantes.
        messages.append({**assistant, "reasoning_content": reply.reasoning})
        steps.append(assistant)
        # Les outils directs partent ensemble ; un programme s'exécute seul, à son tour.
        trust_before = state.trust
        direct = [call for call in calls if call.name != SCRIPT_TOOL]
        done = dict(
            zip(
                (call.id for call in direct),
                await asyncio.gather(
                    *(self._execute_direct(call, trust_before, result) for call in direct)
                ),
                strict=True,
            )
        )
        for call in calls:
            if call.name == SCRIPT_TOOL:
                done[call.id] = await self._run_script(call, state, result)
        results = [done[call.id] for call in calls]
        result.direct_sources.extend((call, trust_before is Trust.TRUSTED) for call in direct)
        result.tool_results.extend(r for r in results if r.name != SCRIPT_TOOL)
        result.direct_results.extend(r for r in results if r.name != SCRIPT_TOOL)
        for res in results:
            messages.append({"role": "tool", "tool_call_id": res.call_id, "content": res.content})
            untrusted = res.ok and self._output_trust(res.name) is Trust.UNTRUSTED
            if untrusted:
                state.trust = Trust.UNTRUSTED  # le modèle va lire un contenu venu d'ailleurs
            if res.status in ("politique", "annulé"):
                state.no_tools = True  # un refus de politique ou un « non » ne se contourne pas
            short = UNTRUSTED_HISTORY if untrusted else res.content
            if len(short) > HISTORY_TOOL_CHARS:
                short = short[:HISTORY_TOOL_CHARS] + " […]"
            steps.append({"role": "tool", "tool_call_id": res.call_id, "content": short})
        return results

    def _output_trust(self, tool_name: str) -> Trust:
        entry = self._catalog.by_tool(tool_name) if self._catalog is not None else None
        return entry.output if entry is not None else Trust.LOCAL

    async def _execute_direct(
        self, call: ToolCallReq, turn_trust: Trust, result: TurnResult
    ) -> ToolResult:
        """Un outil direct, après la politique : si le modèle a lu du contenu non fiable
        dans ce tour, une action conséquente demande un « oui » (sauf accès complet)."""
        from mikata.tools.registry import ToolResult

        assert self.tools is not None
        entry = self._catalog.by_tool(call.name) if self._catalog is not None else None
        if entry is not None:
            decision = decide(
                entry, Trust.TRUSTED, turn_trust, full_access=self.config.guard.full_access
            )
            refusal = ""
            status = "politique"
            if decision.verdict is Verdict.DENY:
                refusal = f"refusé : {decision.reason}"
            elif decision.verdict is Verdict.CONFIRM:
                result.confirmations += 1
                guard = self.tools.ctx.guard
                async with self._confirm_lock:  # une question à la fois, jamais deux voix
                    accepted = guard is not None and await guard.confirm(
                        confirm_question(entry), current_turn()
                    )
                if not accepted:
                    refusal, status = "l'utilisateur n'a pas confirmé", "annulé"
            if refusal:
                log_action(call.name, call.arguments, status, refusal)
                return ToolResult(call.id, call.name, False, refusal, False, 0.0, None, status)
        return await self.tools.execute(call)

    async def _run_script(
        self, call: ToolCallReq, state: _TurnState, result: TurnResult
    ) -> ToolResult:
        """Exécute un programme (outil script) et prépare la suite du tour.

        Réussi avec une phrase courte en retour : elle est dite telle quelle et le tour
        s'arrête là (résultat « terminal »). Réussi avec une donnée : le modèle la formule.
        En échec : l'erreur revient au modèle, qui corrige ; après un refus ou un « non »,
        plus aucun outil n'est proposé."""
        from mikata.tools.registry import ToolResult

        assert self.tools is not None and self._catalog is not None
        start = time.monotonic()
        code = str(call.arguments.get("code", ""))
        runtime = ScriptRuntime(
            self.tools,
            self._catalog,
            llm=self.llm,
            full_access=self.config.guard.full_access,
            turn_trust=state.trust,
            extract_effort=self.config.llm.reasoning_effort,
            skills=self._callable_skills(),
            jev=self.jev,
        )
        outcome = await runtime.run(code, timeout_s=self.config.agent.script_timeout_s)
        result.scripts.append(outcome)
        result.script_sources.append((code, state.trust is Trust.TRUSTED))
        result.skill_uses.extend(runtime.skill_uses)
        result.tool_calls.extend(runtime.tool_calls)
        result.tool_results.extend(runtime.tool_results)
        result.llm_calls += runtime.llm_calls
        if runtime.usage is not None:
            result.usage = runtime.usage if result.usage is None else result.usage + runtime.usage
        result.confirmations += sum(
            r.verdict in ("à confirmer", "non confirmé") for r in outcome.records
        )
        duration = time.monotonic() - start
        logger.info(
            "Programme : {} en {:.0f} ms ({} appel(s), {} opérations){}",
            "ok" if outcome.ok else f"ÉCHEC ({outcome.kind})",
            duration * 1000,
            len(outcome.records),
            outcome.steps,
            "" if outcome.ok else f" : {outcome.error}",
        )
        log_action(SCRIPT_TOOL, {"code": code}, "ok" if outcome.ok else outcome.kind, outcome.error)
        if not outcome.ok:
            status = {"refus": "politique", "annulé": "annulé", "garde-fou": "refusé"}.get(
                outcome.kind, "échec"
            )
            if outcome.retryable:
                state.script_failures += 1
                if state.script_failures >= SCRIPT_FAILURES_BEFORE_ESCALATION:
                    state.strong = True
                    strong = at_least(result.effort_final, self.config.llm.strong_effort)
                    if strong != result.effort_final:
                        logger.info(
                            "Rôle fort : réflexion {} -> {} (programme en échec).",
                            result.effort_final,
                            strong,
                        )
                        result.effort_final = strong
                        result.effort_reasons.append("programme en échec")
                if state.script_failures >= MAX_SCRIPT_FAILURES:
                    state.no_tools = True  # on n'insiste pas : le modèle l'explique
            else:
                state.no_tools = True
            return ToolResult(
                call.id, SCRIPT_TOOL, False, failure_message(outcome), False, duration, None, status
            )
        sentence = spoken_result(outcome)
        if not has_data(outcome):
            turn = current_turn()
            if sentence is not None and turn is not None:
                turn.speak(sentence)
            content = sentence or "Programme terminé."
            return ToolResult(call.id, SCRIPT_TOOL, True, content, True, duration)
        # Une donnée à formuler : le modèle va la lire, donc le tour en prend la confiance.
        state.trust = state.trust.join(outcome.trust).join(outcome.read_trust)
        return ToolResult(call.id, SCRIPT_TOOL, True, data_message(outcome), False, duration)

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        speech: SpeechSink,
        result: TurnResult,
        gate: _FollowUpGate | None = None,
        strong: bool = False,
    ) -> _Reply | None:
        """Un appel LLM (avec son retry) ; le texte part au TTS fragment par fragment."""
        gate = gate or _FollowUpGate(active=False)

        def say(fragments: list[str]) -> None:
            for fragment in fragments:
                if gate.allow(fragment):
                    speech.say(fragment)

        splitter = FragmentSplitter(self.config.tts.first_chunk_min_words)
        effort = result.effort_final
        parts: list[str] = []
        calls: list[ToolCallReq] = []
        done: Done | None = None
        start = time.monotonic()
        first_at: float | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result.attempts += 1
            result.llm_calls += 1
            start = time.monotonic()
            try:
                request = _with_reasoning(messages, effort)
                # Rôle « fort » : un autre modèle seulement s'il est configuré (sinon l'appel
                # reste identique, la réflexion relevée fait la différence).
                extra = (
                    {"model": self.config.llm.strong_model}
                    if strong and self.config.llm.strong_model
                    else {}
                )
                async for event in self.llm.stream(
                    request, tools, reasoning_effort=effort, **extra
                ):
                    if isinstance(event, Done):
                        done = event
                        break
                    now = time.monotonic()
                    first_at = first_at or now
                    if result.first_token_at is None:
                        result.first_token_at = now
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                        say(splitter.feed(gate.feed(event.text)))
                    elif isinstance(event, ToolCallReq):
                        calls.append(event)
                if attempt > 1:
                    logger.info("Appel LLM réussi à la tentative {}/{}.", attempt, MAX_ATTEMPTS)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if parts:  # la réponse a commencé : on garde ce qui est dit
                    logger.warning("Flux LLM interrompu : {}", exc)
                    break
                logger.warning("Appel LLM en échec ({}/{}) : {}", attempt, MAX_ATTEMPTS, exc)
                calls.clear()
                if attempt == MAX_ATTEMPTS:
                    return None
        say(splitter.feed(gate.finish()))
        say(splitter.flush())
        if done is not None and done.usage is not None:
            result.usage = done.usage if result.usage is None else result.usage + done.usage
        self._log_call(result.llm_calls, effort, start, first_at, done, calls)
        return _Reply(
            "".join(parts), calls, done.reasoning if done is not None else "", gate.ignored
        )

    @staticmethod
    def _log_call(
        number: int,
        effort: str | None,
        start: float,
        first_at: float | None,
        done: Done | None,
        calls: list[ToolCallReq],
    ) -> None:
        def ms(t: float | None) -> str:
            return "—" if t is None else f"{t * 1000:.0f} ms"

        reasoning = done.reasoning_s if done is not None else None
        tokens = done.usage.reasoning if done is not None and done.usage is not None else 0
        logger.info(
            "Appel LLM {} (réflexion {}) : premier élément {}, réflexion du modèle {}{}, "
            "total {}, outils : {}.",
            number,
            effort or "défaut",
            ms(first_at - start if first_at is not None else None),
            ms(reasoning),
            f" ({tokens} tokens)" if tokens else "",
            ms(time.monotonic() - start),
            ", ".join(call.name for call in calls) or "aucun",
        )

    async def aclose(self) -> None:
        for task in (self._summary_task, self._skill_task, self._reflect_task):
            if task is not None and not task.done():
                task.cancel()
        aclose = getattr(self.llm, "aclose", None)
        if aclose is not None:
            await aclose()
        if self.jev is not None:
            await self.jev.aclose()
        if self.tools is not None:
            await self.tools.aclose()
