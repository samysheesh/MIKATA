"""Orchestrateur et machine à états (docs/ARCHITECTURE.md §3).

Phase 1 : capture continue → tampon circulaire → mot d'activation (executor dédié)
→ bip + journal de détection. Le raccourci clavier a le même effet.
Phase 2 : écoute depuis le début du pré-roll (VAD Silero) → transcription (Whisper,
spéculative) → texte sans le mot d'activation.
Phase 3 : réflexion (LLM en streaming) → parole (TTS par fragments) ; son de réflexion.
Phase 4 : outils (état ACTING), annonces des minuteurs et rappels (`announce`).
Phase 5 : écoute courte sans mot d'activation (`listen_once`) pour les confirmations
orales du garde-fou et le « stop » avant extinction.
Phase 6 :
- **barge-in** : « MIKATA » (ou le raccourci) pendant qu'une réponse est jouée coupe le son
  de **cette** réponse tout de suite (`SpeechStream.cancel`) et lance l'écoute. Les autres
  commandes continuent (décision de la phase 3) ; leurs réponses attendent que tu aies fini
  de parler (`Speaker.hold`).
- **annulation** : une commande qui n'est qu'un « stop », « annule », « laisse tomber »...
  (`agent.cancel_phrases`) n'est pas envoyée au LLM. Elle annule la commande dont la
  réponse vient d'être coupée si elle tourne encore, sinon la dernière commande en cours ;
  en fenêtre de suivi, elle la ferme. Sans rien à annuler ni réponse coupée, « stop » part
  au LLM comme une commande normale (pause de la musique).
- **fenêtre de suivi** : après une réponse, si rien d'autre n'est en cours, le micro écoute
  `agent.follow_up_s` secondes sans mot d'activation (état FOLLOW_UP) ; ce qui est dit
  devient une nouvelle commande, avec l'historique. Sinon : bip de fin. Dans le bruit ou
  une conversation à côté : une transcription peu fiable (`agent.follow_up_min_confidence`)
  est ignorée, puis le LLM juge si la phrase s'adresse à MIKATA
  (`agent.follow_up_use_llm_gate`). Phrase ignorée : bip de fin, aucune réponse.
Phase 6b : **arrêt vocal** sans mot d'activation. Pendant qu'une réponse est jouée et que
personne n'a le micro, le VAD tourne et `stt.voice_stop.VoiceStopSpotter` transcrit les
phrases courtes ; « stop », « tais-toi »... (`agent.voice_stop_phrases`, phrase entière)
coupent le son de cette réponse et arrêtent sa commande si elle travaille encore, comme
« MIKATA, stop ». Toute autre parole est ignorée. Latence de coupure dans le log.
Phase 7 : score du vérificateur de voix dans le log de détection, purge quotidienne
(`maintenance.run_maintenance`), collecte de la vie courante bornée à
`wake.negatives_max_hours`.
Phase 8 : **interface** dans un processus à part (mascotte, réglages), reliée par le canal
local `ui.bridge.UiBridge`. L'orchestrateur y publie ses changements d'état et ce qu'il
comprend et dit (`_ui`, qui ne fait qu'ajouter à une file) ; il reçoit `reload_config`
(réglages appliqués sans redémarrer, `LIVE_SETTINGS`) et `pause` (micro coupé côté
logiciel, §8). Sans interface, ou si elle plante, rien ne change. Un chien de garde signale
un micro qui ne donne plus de trames (occupé, débranché).

Commandes en parallèle : chaque commande a son propre état (`_Request.state`). Il n'y a
qu'un micro, donc une seule commande **écoute** à la fois (`_listening`) ; dès que sa fin de
parole est détectée, le micro est libre et « MIKATA » peut lancer la suivante pendant que la
première transcrit, réfléchit, agit ou parle. Rien n'est refusé ni mis en file, sauf la
parole elle-même : deux réponses ne se superposent pas (`audio.player.Speaker`).
L'état affiché (icône) est le plus « actif » des états en cours.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

from mikata.audio.base import SAMPLE_RATE
from mikata.audio.capture import CapturedFrame
from mikata.audio.ringbuffer import RingBuffer
from mikata.config import Config
from mikata.events import BusEvent, EventBus, Shutdown, WakeDetected
from mikata.maintenance import INTERVAL_S as MAINTENANCE_INTERVAL_S
from mikata.maintenance import run_maintenance
from mikata.stt.base import STT
from mikata.stt.speculative import CommandRecorder, EndReason
from mikata.stt.text import (
    clean_transcript,
    describe_confidence,
    follow_up_rejection,
    is_cancel_command,
    strip_wake_word,
    wake_word_pattern,
)
from mikata.stt.voice_stop import StopHeard, VoiceStopSpotter
from mikata.vad.base import VAD
from mikata.wake.base import WakeWordDetector
from mikata.wake.recordings import (
    MISS_CLIP_S,
    MISSES_DIR,
    NEGATIVE_CLIP_S,
    NegativeCollector,
    estimate_tail_silence_s,
    purge_older_than,
    save_detection,
    save_miss,
)
from mikata.wake.trigger import WakeTrigger

if TYPE_CHECKING:
    from mikata.agent.loop import Agent, TurnResult
    from mikata.audio.player import Speaker, SpeechStream
    from mikata.ui.bridge import UiBridge

WAKE_LOG_S = 2.0
HOTKEY_PRE_ROLL_S = 0.3  # raccourci : pas de mot d'activation à retirer, peu de pré-roll
# Le tampon garde le pré-roll + une marge : entre la détection (thread audio) et le début de
# l'écoute (orchestrateur), quelques trames peuvent arriver ; on veut le pré-roll complet
# **avant** la détection, plus tout ce qui a suivi.
RING_MARGIN_S = 1.0
FOLLOW_UP_PRE_ROLL_S = 0.3  # suivi : tu peux commencer à parler pile à la fin de la réponse
FOLLOW_UP_SOURCE = "suivi"
# Réponse orale d'une commande dont la réponse a été coupée : « non » à une confirmation,
# « stop » à un compte à rebours (garde-fou : dans le doute, rien n'est fait).
INTERRUPTED_ANSWER = "stop"
VOICE_STOP_TARGET_MS = 300  # arrêt vocal : coupure visée après la fin de parole
MIC_SILENT_S = 2.0  # phase 8 : plus aucune trame du micro depuis N s -> « micro occupé »
APP_TASK = "app:"  # phase 17e : préfixe des tâches « appli tenue en arrière-plan »


# Phase 8 : réglages appliqués sans redémarrer quand l'interface réécrit config.yaml (le
# reste demande un redémarrage). Préfixe « section. » : toute la section.
def _media_sessions() -> list[Any]:
    """Phase 17e : les sessions média de Windows (import paresseux de winrt)."""
    from mikata.tools.media_sessions import sessions

    return sessions()


LIVE_SETTINGS = frozenset(
    {
        "wake.threshold",
        "wake.verifier_threshold",
        "wake.consecutive_frames",
        "wake.refractory_s",
        "agent.follow_up_s",
        "agent.follow_up_min_confidence",
        "agent.follow_up_use_llm_gate",
        "agent.history_persist_h",
        "agent.thinking_sound_after_s",
        "audio.output_device",  # phase 15b : la sortie change sans redémarrer
        "guard.full_access",  # relu à chaque tour par la boucle
        "audio.input_device",  # phase 15g : micro rouvert sans redémarrer
        "ui.mascot",  # phase 15g : appliqué par l'interface elle-même
        "agent.narration",  # phase 15c
        "agent.narration_after_s",
        "agent.ack_after_s",  # phase 16c
        "audio.earcon_volume",
        "llm.reasoning_effort",
        "ui.skin",  # lus par l'interface elle-même ; phase 11 : la voix du profil aussi
        "ui.bubble_s",
        "ui.profiles",  # ajouts phase 11 : profils (skin + voix) et voix ElevenLabs
        "tts.backend",
        "tts.elevenlabs",
    }
)


def is_live_setting(key: str) -> bool:
    return any(key == k or (k.endswith(".") and key.startswith(k)) for k in LIVE_SETTINGS)


def changed_settings(old: Config, new: Config) -> list[str]:
    """Clés (« section.clé ») dont la valeur diffère entre deux configurations."""
    a = old.model_dump(mode="json", exclude={"secrets"})
    b = new.model_dump(mode="json", exclude={"secrets"})
    return sorted(
        f"{section}.{key}"
        for section in b
        for key in b[section]
        if a.get(section, {}).get(key) != b[section][key]
    )


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"


# Priorité d'affichage quand plusieurs commandes sont en cours (la plus haute gagne).
DISPLAY_PRIORITY = (
    State.LISTENING,
    State.SPEAKING,
    State.ACTING,
    State.THINKING,
    State.TRANSCRIBING,
    State.FOLLOW_UP,
)


class StatusIcon(Protocol):
    """Icône de la zone de notification (implémentée par `ui.tray.TrayIcon`)."""

    def start(
        self,
        on_quit: Callable[[], None],
        on_action: Callable[[str], None] | None = None,
        apps: Sequence[str] = (),  # phase 17e : « Fermer <appli> » des applis cachées
    ) -> None: ...
    def set_state(self, state: State) -> None: ...
    def stop(self) -> None: ...


class AudioSource(Protocol):
    """Capture micro (implémentée par `audio.capture.AudioCapture`)."""

    dropped_frames: int
    overflows: int

    def start(self, loop: asyncio.AbstractEventLoop) -> None: ...
    async def read(self) -> CapturedFrame: ...
    def stop(self) -> None: ...


class SoundOutput(Protocol):
    """Sortie audio (implémentée par `audio.player.AudioPlayer`)."""

    @property
    def output_latency_s(self) -> float: ...
    def start(self, loop: asyncio.AbstractEventLoop) -> None: ...
    def play_nowait(self, pcm: np.ndarray) -> None: ...
    def close(self) -> None: ...


class HotkeySource(Protocol):
    """Raccourci clavier global (implémenté par `ui.hotkeys.HotkeyListener`)."""

    def start(self, on_trigger: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...


class CpuSampler(Protocol):
    def sample(self) -> tuple[float, float]: ...


class SpeechToText(STT, Protocol):
    """STT chargé au démarrage (implémenté par `stt.whisper_local.WhisperSTT`)."""

    def load(self) -> None: ...
    def close(self) -> None: ...


class ResettableVAD(VAD, Protocol):
    def reset(self) -> None: ...


@dataclass(eq=False)
class _Request:
    """Une commande, de la détection à la fin de la réponse. Horodatages : time.monotonic."""

    id: int
    source: str
    wake_at: float  # bip
    wake_frame_at: float  # capture de la trame de détection
    recorder: CommandRecorder | None = None
    state: State = State.LISTENING
    end_at: float = 0.0  # fin de parole détectée
    reason: EndReason | None = None
    stream: SpeechStream | None = None
    marks: dict[str, float] = field(default_factory=dict)
    # Ajouts phase 6
    task: asyncio.Task[None] | None = None  # traitement (transcription -> réponse)
    barged: _Request | None = None  # commande dont la réponse a été coupée par celle-ci
    follow_up: bool = False  # réponse jouée jusqu'au bout : fenêtre de suivi possible

    @property
    def last_syllable_at(self) -> float:
        """Dernière trame de parole ; à défaut (commande finie avant le bip), la détection."""
        if self.recorder is not None and self.recorder.last_speech_at is not None:
            return self.recorder.last_speech_at
        return self.wake_frame_at


class App:
    def __init__(
        self,
        config: Config,
        tray: StatusIcon | None = None,
        bus: EventBus | None = None,
        *,
        audio: AudioSource | None = None,
        detector: WakeWordDetector | None = None,
        player: SoundOutput | None = None,
        earcons: Mapping[str, np.ndarray] | None = None,
        hotkeys: HotkeySource | None = None,
        cpu: CpuSampler | None = None,
        vad: ResettableVAD | None = None,
        stt: SpeechToText | None = None,
        agent: Agent | None = None,
        speaker: Speaker | None = None,
        ui: UiBridge | None = None,
        ui_launcher: Callable[[Sequence[str]], object] | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Ajouts phase 8 : `ui` (canal vers l'interface), `ui_launcher` (lance l'interface
        avec des options : `launch_ui`), `config_path` (relu sur `reload_config`)."""
        self.config = config
        self.bus = bus or EventBus()
        self._tray = tray
        self._audio = audio
        self._detector = detector
        self._player = player
        self._earcons = earcons or {}
        self._audio_broken = False  # phase 15f : relecture des périphériques à refaire
        self._mcp: Any = None  # phase 16f : serveurs MCP (mikata.mcp.bridge.McpBridge)
        self._hotkeys = hotkeys
        self._cpu = cpu
        self._vad = vad
        self._stt = stt
        self._agent = agent
        self._speaker = speaker
        self._ui = ui
        self._ui_launcher = ui_launcher
        self._config_path = config_path or config.base_dir / "config.yaml"
        self.paused = False  # micro coupé côté logiciel (phase 8)
        self.restart_requested = False  # « Redémarrer » dans l'interface : __main__ relance
        self._mic_status = "ok"
        self._app_tasks: list[dict[str, object]] = []  # phase 17e : applis en arrière-plan
        self._last_frame_at = 0.0
        self.state = State.IDLE
        self._wake_pattern = wake_word_pattern(config.stt.wake_words)
        self._listening: _Request | None = None  # la commande qui a le micro
        self._active: dict[int, _Request] = {}  # commandes en cours (y compris l'écoute)
        self._ids = itertools.count(1)
        self._request_tasks: set[asyncio.Task[None]] = set()
        # Dernières commandes reconnues et réponses (bornées : MIKATA tourne h24).
        self.commands: deque[str] = deque(maxlen=100)
        self.answers: deque[str] = deque(maxlen=100)

        # Assez long pour le pré-roll et pour un raté du mot d'activation (phase 9).
        self.ring = RingBuffer.for_seconds(
            max(config.audio.pre_roll_s + RING_MARGIN_S, MISS_CLIP_S + RING_MARGIN_S)
        )
        self._written = 0  # échantillons écrits dans le tampon depuis le démarrage
        self._wake_written = 0  # valeur de `_written` à la dernière détection
        wake = config.wake
        self._trigger = WakeTrigger(wake.threshold, wake.consecutive_frames, wake.refractory_s)
        self._collector = NegativeCollector() if wake.collect_negatives else None
        self._wake_logs_dir = config.resolve("data/wake_logs")
        self._misses_dir = config.resolve(MISSES_DIR)
        self.misses = 0  # ratés du mot d'activation notés au raccourci (phase 9)
        self._negatives_dir = config.resolve("data/negatives")
        self._negatives_saved = 0  # extraits de vie courante sur le disque (phase 7)
        self._wake_executor: Executor | None = None
        self._io_executor: Executor | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._wake_audio: np.ndarray | None = None
        self.detections = 0
        # Arrêt vocal (phase 6b) : actif seulement pendant qu'une réponse est jouée.
        self._voice_stop: VoiceStopSpotter | None = None
        self._voice_stop_armed = False
        agent_cfg = config.agent
        if agent_cfg.voice_stop and stt is not None and vad is not None and speaker is not None:
            self._voice_stop = VoiceStopSpotter(
                stt,
                agent_cfg.voice_stop_phrases,
                self._on_voice_stop,
                silence_ms=agent_cfg.voice_stop_silence_ms,
                max_s=agent_cfg.voice_stop_max_s,
            )

    # -- cycle de vie ---------------------------------------------------------------------

    def request_shutdown(self, reason: str) -> None:
        """Demande l'arrêt ; appelable depuis n'importe quel thread."""
        self.bus.publish_threadsafe(Shutdown(reason=reason))

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            logger.debug("État : {} -> {}", self.state, state)
        self.state = state
        if self._tray is not None:
            self._tray.set_state(state)
        self._publish_ui("state", state=state.value, follow_up_s=self.config.agent.follow_up_s)

    # -- interface (phase 8) ----------------------------------------------------------------

    def _publish_ui(self, kind: str, **fields: object) -> None:
        """Message pour l'interface : ajouté à une file, rien d'autre (jamais bloquant)."""
        if self._ui is not None:
            self._ui.publish({"type": kind, **fields})

    def _ui_snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "follow_up_s": self.config.agent.follow_up_s,
            "mic": self._mic_status,
            "tasks": self._background_tasks(),  # phase 13f : onglet « En cours »
        }

    def _on_ui_command(self, command: dict[str, object]) -> None:
        kind = command.get("type")
        if kind == "reload_config":
            self._spawn(self.reload_config(), "relecture de la config")
        elif kind == "pause":
            self.set_paused(bool(command.get("paused")))
        elif kind == "restart":
            logger.info("Redémarrage demandé depuis l'interface.")
            self.restart_requested = True
            self.request_shutdown("redémarrage demandé depuis l'interface")
        elif kind == "list_tasks":  # phase 13f ; phase 15a : avec les onglets
            self._spawn(self._publish_browser_state(), "état du navigateur de travail")
        elif kind == "close_task":  # phase 13f
            task_id = command.get("id")
            if task_id == "browser":
                self._close_work_browser()
            elif isinstance(task_id, str) and task_id.startswith(APP_TASK):  # phase 17e
                close = {"name": task_id[len(APP_TASK) :], "action": "close"}
                self._spawn(self._app_action(close), "appli en arrière-plan")
            self._publish_ui("tasks", tasks=self._background_tasks())
        elif kind == "browser":  # phase 15a : gérer le navigateur de travail
            self._spawn(self._browser_action(command), "navigateur de travail")
        elif kind == "app":  # phase 17e : une appli tenue en arrière-plan
            self._spawn(self._app_action(command), "appli en arrière-plan")

    def _on_tray_action(self, action: str) -> None:
        """Menu de l'icône (thread pystray) : renvoyé dans la boucle asyncio."""
        loop = self.bus.loop
        if loop is not None:
            loop.call_soon_threadsafe(self.ui_action, action)

    def ui_action(self, action: str) -> None:
        """« settings » / « mascot » : demandé à l'interface si elle tourne, sinon lancée ;
        « pause » : bascule le micro ; « browser_close » : ferme le navigateur de travail."""
        if action == "pause":
            self.set_paused(not self.paused)
            return
        if action == "browser_close":
            self._close_work_browser()
            return
        if action.startswith("app_close:"):  # phase 17e : « Fermer Spotify »
            close = {"name": action.split(":", 1)[1], "action": "close"}
            self._spawn(self._app_action(close), "appli en arrière-plan")
            return
        message = {"settings": "open_settings", "mascot": "show_mascot"}.get(action)
        if message is None:
            return
        if self._ui is not None and self._ui.connected:
            self._publish_ui(message)
        elif self._ui_launcher is not None:
            self._ui_launcher(["--settings"] if action == "settings" else [])

    def _close_work_browser(self) -> None:
        """Menu de l'icône : ferme le navigateur de travail (lecture comprise), jamais
        celui de l'utilisateur."""
        registry = self._agent.tools if self._agent is not None else None
        browser = getattr(registry.ctx, "browser", None) if registry is not None else None
        quit_fn = getattr(browser, "quit", None)
        if quit_fn is not None:
            self._spawn(quit_fn(), "fermeture du navigateur de MIKATA")

    def prewarm_work_browser(self) -> None:
        """Lance le navigateur de travail en arrière-plan dès le démarrage : la première
        demande web ne paie plus le démarrage à froid (no-op s'il tourne déjà)."""
        if not self.config.browser.own_profile:
            return
        registry = self._agent.tools if self._agent is not None else None
        browser = getattr(registry.ctx, "browser", None) if registry is not None else None
        prewarm = getattr(browser, "prewarm", None)
        if prewarm is not None:
            self._spawn(prewarm(), "préchauffage du navigateur de travail")

    def _work_browser(self) -> Any:
        registry = self._agent.tools if self._agent is not None else None
        return getattr(registry.ctx, "browser", None) if registry is not None else None

    async def _browser_action(self, command: dict[str, object]) -> None:
        """Afficher, masquer, mettre en pause, fermer un onglet, puis renvoyer l'état."""
        browser = self._work_browser()
        action = command.get("action")
        if browser is None:
            return
        try:
            if action in ("show", "hide"):
                await asyncio.to_thread(browser.set_visible, action == "show")
            elif action == "pause":  # phase 15g : Pause / Reprendre
                toggle = getattr(browser, "toggle_media", None)
                done = await toggle() if toggle is not None else await browser.pause_media()
                logger.info("Navigateur de travail : {}.", done)
            elif action == "close_tab" and isinstance(command.get("tab"), str):
                await browser.close_tab(str(command["tab"]))
            elif action == "start":
                await browser.prewarm()
        except Exception as exc:
            logger.warning("Navigateur de travail ({}) : {}", action, exc)
        await self._publish_browser_state()

    async def _publish_browser_state(self) -> None:
        browser = self._work_browser()
        tabs: list[dict[str, str]] = []
        if browser is not None and hasattr(browser, "list_tabs"):
            try:
                tabs = await browser.list_tabs()
            except Exception as exc:
                logger.debug("Onglets illisibles : {}", exc)
        self._app_tasks = await self._read_app_tasks()
        self._publish_ui("tasks", tasks=self._background_tasks(tabs))

    # -- applis tenues en arrière-plan (phase 17e) ---------------------------------------

    async def _read_app_tasks(self) -> list[dict[str, object]]:
        """Les applis de `uia.devtools_apps` pour l'accueil des réglages : ouvertes ou non,
        affichées ou cachées, et ce qu'elles jouent (sessions média de Windows)."""
        registry = self._agent.tools if self._agent is not None else None
        ctx = registry.ctx if registry is not None else None
        uia = getattr(ctx, "uia", None)
        status_fn = getattr(uia, "background_status", None)
        if ctx is None or status_fn is None or not self.config.uia.devtools_apps:
            return []
        try:
            status = await ctx.run_blocking(status_fn)
            media = await ctx.run_blocking(_media_sessions) if sys.platform == "win32" else []
        except Exception as exc:
            logger.debug("Applis en arrière-plan illisibles : {}", exc)
            return []
        from mikata.tools.media_sessions import matching

        tasks: list[dict[str, object]] = []
        for app in status:
            session = matching(str(app.get("name", "")), media) if app.get("open") else None
            tasks.append(
                {
                    "id": f"{APP_TASK}{app.get('name')}",
                    "label": str(app.get("name", "")),
                    "running": bool(app.get("open")),
                    "driven": bool(app.get("driven")),
                    "visible": bool(app.get("visible")),
                    "media": ("playing" if session.playing else "paused") if session else "",
                    "now": session.label if session is not None else "",
                }
            )
        return tasks

    async def _app_action(self, command: dict[str, object]) -> None:
        """Afficher, masquer, Pause / Reprendre, fermer une appli tenue en arrière-plan,
        puis renvoyer l'état à l'interface."""
        registry = self._agent.tools if self._agent is not None else None
        ctx = registry.ctx if registry is not None else None
        uia = getattr(ctx, "uia", None)
        name, action = str(command.get("name") or ""), command.get("action")
        if ctx is None or uia is None or not name:
            return
        try:
            if action == "pause":
                from mikata.tools.media_sessions import toggle

                done = await ctx.run_blocking(toggle, name)
                logger.info("{} : {}.", name, done or "aucune lecture à commander")
            elif action in ("show", "hide", "close"):
                logger.info(await ctx.run_blocking(uia.background_action, name, action))
        except Exception as exc:
            logger.warning("{} ({}) : {}", name, action, exc)
        await self._publish_browser_state()

    def _background_tasks(
        self, tabs: list[dict[str, str]] | None = None
    ) -> list[dict[str, object]]:
        """Ce qui tourne en arrière-plan (onglet « En cours » des réglages) : chaque tâche
        est fermable par `close_task`. Le navigateur de travail est la première ; les
        minuteurs, rappels et surveillances s'y ajouteront sans changer le protocole."""
        registry = self._agent.tools if self._agent is not None else None
        browser = getattr(registry.ctx, "browser", None) if registry is not None else None
        running = bool(getattr(browser, "running", False))
        return [
            {
                "id": "browser",
                "label": "Navigateur de travail — lecture en arrière-plan",
                "running": running,
                "visible": bool(getattr(browser, "visible", False)),
                "paused": bool(getattr(browser, "media_paused", False)),
                "tabs": tabs or [],
            },
            *self._app_tasks,  # phase 17e : relu par _publish_browser_state
        ]

    def set_paused(self, paused: bool) -> None:
        """Micro coupé côté logiciel : plus de mot d'activation, d'écoute ni de collecte."""
        if paused == self.paused:
            return
        self.paused = paused
        logger.info("Écoute {}.", "en pause (micro coupé)" if paused else "reprise")
        if paused:
            for request in list(self._active.values()):
                if request is self._listening and request.recorder is not None:
                    request.recorder.cancel()
                    self._finish(request)
        self._set_mic_status("paused" if paused else "ok")

    def _set_mic_status(self, status: str) -> None:
        if status != self._mic_status:
            self._mic_status = status
            self._publish_ui("mic", status=status)

    async def _mic_watchdog(self) -> None:
        """Micro occupé ou débranché : plus aucune trame depuis `MIC_SILENT_S`."""
        while True:
            await asyncio.sleep(MIC_SILENT_S / 2)
            if self.paused:
                continue
            silent = time.monotonic() - self._last_frame_at > MIC_SILENT_S
            if silent and self._mic_status == "ok":
                logger.warning("Micro : aucune trame depuis {:.0f} s (occupé ?).", MIC_SILENT_S)
            self._set_mic_status("busy" if silent else "ok")

    async def reload_config(self) -> None:
        """config.yaml réécrit par l'interface : applique ce qui peut l'être sans
        redémarrer, et dit le reste à l'interface."""
        from mikata.config import ConfigError, load_config

        try:
            new = await asyncio.to_thread(load_config, self._config_path)
        except ConfigError as exc:
            logger.error("Réglages non appliqués : {}", exc)
            self._publish_ui("config", applied=[], restart=[], error=str(exc))
            return
        changed = changed_settings(self.config, new)
        old_secrets = self.config.secrets
        live = [k for k in changed if is_live_setting(k)]
        restart = [k for k in changed if not is_live_setting(k)]
        self.config = new
        wake = new.wake
        self._trigger.threshold = wake.threshold
        self._trigger.consecutive_frames = wake.consecutive_frames
        self._trigger.refractory_s = wake.refractory_s
        for name in ("threshold", "verifier_threshold"):
            if hasattr(self._detector, name):
                setattr(self._detector, name, getattr(wake, name))
        if self._agent is not None:
            self._agent.config = new
            if "agent.history_persist_h" in live:
                self._agent.memory.set_keep(new.agent.history_persist_h * 3600)
        if "audio.output_device" in live or "audio.earcon_volume" in live:
            await self._switch_output(new, "audio.output_device" in live)
        if "audio.input_device" in live:
            await self._switch_input(new)
        live += await self._apply_secrets(old_secrets, new.secrets)
        voice_keys = ("ui.skin", "ui.profiles", "tts.backend", "tts.elevenlabs")
        apply_voice = getattr(self._speaker.tts, "apply", None) if self._speaker else None
        if apply_voice is not None and any(k in live for k in voice_keys):
            try:
                await apply_voice(new)
            except Exception as exc:  # modèle de voix absent, etc. : la voix d'avant reste
                logger.exception("Voix du profil non appliquée.")
                self._publish_ui("config", applied=[], restart=[], error=str(exc))
                return
        logger.info(
            "Réglages relus : {} appliqué(s) tout de suite{}.",
            len(live),
            f", {len(restart)} au prochain démarrage ({', '.join(restart)})" if restart else "",
        )
        self._publish_ui("config", applied=live, restart=restart, error=None)

    async def _switch_output(self, config: Any, new_device: bool) -> None:
        """Phase 15b : nouvelle sortie audio (ou nouveau volume des sons) sans redémarrer.
        Les sons d'interface sont refaits au taux de la nouvelle sortie."""
        from mikata.__main__ import find_output_device

        player = self._player
        if player is None or not hasattr(player, "switch"):
            return
        from mikata.audio.devices import DeviceNotFoundError

        try:
            if new_device:
                try:
                    device = await asyncio.to_thread(find_output_device, config)
                except DeviceNotFoundError:  # branché après le démarrage : liste relue
                    await self._rescan_audio("sortie choisie dans les réglages")
                    return
                await asyncio.to_thread(player.switch, device, asyncio.get_running_loop())
            await self._rebuild_earcons()
        except Exception as exc:  # introuvable ou refusé : l'ancienne sortie reste
            logger.error("Sortie audio non appliquée : {}", exc)
            self._publish_ui("config", applied=[], restart=[], error=f"sortie audio : {exc}")

    async def _switch_input(self, config: Any) -> None:
        """Phase 15g : nouveau micro choisi dans les réglages, sans redémarrer."""
        import sounddevice as sd

        from mikata.audio.devices import (
            DeviceNotFoundError,
            collect_devices,
            default_device_index,
            find_device,
        )

        reopen = getattr(self._audio, "reopen", None)
        if reopen is None:
            return
        try:
            device = await asyncio.to_thread(
                find_device,
                collect_devices(sd),
                config.audio.input_device,
                "input",
                default_device_index(sd, "input"),
            )
        except DeviceNotFoundError:  # branché après le démarrage : liste relue
            await self._rescan_audio("micro choisi dans les réglages")
            return
        try:
            await asyncio.to_thread(reopen, device, asyncio.get_running_loop())
        except Exception as exc:
            logger.error("Micro non changé : {}", exc)
            self._publish_ui("config", applied=[], restart=[], error=f"micro : {exc}")

    async def _apply_secrets(self, old: Any, new: Any) -> list[str]:
        """Phase 15g : une clé remplacée dans les réglages sert tout de suite."""

        def value(secrets: Any, name: str) -> str | None:
            secret = getattr(secrets, name, None)
            return secret.get_secret_value() if secret is not None else None

        applied = []
        for name in ("llm_api_key", "search_api_key", "elevenlabs_api_key", "openrouter_api_key"):
            if value(old, name) == value(new, name):
                continue
            applied.append(f"secrets.{name}")
            if name == "llm_api_key" and self._agent is not None:
                set_key = getattr(self._agent.llm, "set_api_key", None)
                key = getattr(new, name)
                if set_key is not None and key is not None:
                    await set_key(key)
            elif name == "search_api_key" and self._agent is not None:
                registry = self._agent.tools
                if registry is not None:
                    registry.ctx.search_api_key = value(new, name)
            elif name == "openrouter_api_key" and self._agent is not None:
                from mikata.__main__ import build_jev

                if self._agent.jev is not None:
                    await self._agent.jev.aclose()
                self._agent.jev = build_jev(self.config)
            elif name == "elevenlabs_api_key":
                apply_voice = getattr(self._speaker.tts, "apply", None) if self._speaker else None
                if apply_voice is not None:
                    await apply_voice(self.config)
        if applied:
            logger.info("Clés appliquées sans redémarrer : {}.", ", ".join(applied))
        return applied

    async def _rebuild_earcons(self) -> None:
        """Sons d'interface au taux de la sortie en cours."""
        from mikata.audio.earcons import Earcons

        if self._player is None:
            return
        self._earcons = await asyncio.to_thread(
            Earcons,
            self.config.resolve("assets/sounds"),
            self._player.sample_rate,
            self.config.audio.earcon_volume,
        )

    async def _follow_output(self, source: Any) -> None:
        """Phase 15f : avec `audio.output_device: default`, MIKATA parle toujours dans la
        sortie de Windows, même changée après le démarrage (casque Bluetooth allumé, écran
        débranché). Jamais au milieu d'une commande : la phrase en cours n'est pas coupée."""
        from mikata.audio.follow import POLL_S, DefaultOutputWatch

        watch = DefaultOutputWatch()
        try:
            while True:
                await asyncio.sleep(POLL_S)
                player = self._player
                if player is None or self.config.audio.output_device.casefold() != "default":
                    continue
                if self._audio_broken:  # relecture précédente ratée : on réessaie
                    await self._rescan_audio("nouvel essai")
                    continue
                if self._active:
                    continue
                target = watch.update(await source.name(), player.device.name)
                if target is not None:
                    await self._move_output(target)
        finally:
            source.close()

    async def _move_output(self, name: str) -> None:
        """Bascule sur la sortie `name` ; si PortAudio ne la connaît pas encore (branchée
        après le démarrage), relit toute la liste des périphériques."""
        import sounddevice as sd

        from mikata.audio.devices import collect_devices, filter_preferred
        from mikata.audio.follow import same_device

        assert self._player is not None
        devices = filter_preferred(await asyncio.to_thread(collect_devices, sd))
        found = next(
            (d for d in devices if d.supports("output") and same_device(d.name, name)), None
        )
        if found is None:
            await self._rescan_audio(f"nouvelle sortie « {name} »")
            return
        logger.info("Sortie de Windows changée : MIKATA passe sur « {} ».", found.name)
        try:
            await asyncio.to_thread(self._player.switch, found, asyncio.get_running_loop())
            await self._rebuild_earcons()
        except Exception as exc:
            logger.error("Changement de sortie impossible : {}", exc)

    async def _rescan_audio(self, reason: str) -> None:
        """Relit la liste des périphériques de PortAudio (figée à son démarrage) et rouvre
        le micro et la sortie. Les flux doivent être fermés pendant la relecture : une
        seconde environ sans micro."""
        import sounddevice as sd

        from mikata.__main__ import _refresh_portaudio
        from mikata.audio.devices import collect_devices, default_device_index, find_device

        player, capture = self._player, self._audio
        if player is None:
            return
        cfg = self.config.audio
        loop = asyncio.get_running_loop()
        logger.info("Périphériques audio relus ({}).", reason)

        def reopen() -> None:
            if capture is not None:
                capture.stop()
            player.close()
            _refresh_portaudio(sd)
            devices = collect_devices(sd)
            out = find_device(
                devices, cfg.output_device, "output", default_device_index(sd, "output")
            )
            mic = find_device(devices, cfg.input_device, "input", default_device_index(sd, "input"))
            player.switch(out, loop)
            reopen_mic = getattr(capture, "reopen", None)
            if reopen_mic is not None:
                reopen_mic(mic, loop)

        try:
            await asyncio.to_thread(reopen)
        except Exception as exc:  # périphérique absent à cet instant : nouvel essai bientôt
            self._audio_broken = True
            logger.warning("Périphériques audio non rouverts ({}) : nouvel essai.", exc)
            return
        self._audio_broken = False
        await self._rebuild_earcons()

    def _refresh_state(self) -> None:
        """État affiché = le plus prioritaire des états des commandes en cours."""
        states = {request.state for request in self._active.values()}
        state = next((s for s in DISPLAY_PRIORITY if s in states), State.IDLE)
        if state is not self.state:
            self._set_state(state)

    def _set_listening(self, request: _Request | None) -> None:
        """Commande qui a le micro. Tant que quelqu'un parle, aucune nouvelle phrase de
        réponse ne commence (`Speaker.hold`)."""
        if (request is None) == (self._listening is None):
            self._listening = request
            return
        self._listening = request
        if self._speaker is not None:
            if request is None:
                self._speaker.release()
            else:
                self._speaker.hold()

    def _set_request_state(self, request: _Request, state: State) -> None:
        if request.state is not state:
            logger.debug("Commande #{} : {} -> {}", request.id, request.state, state)
            request.state = state
        self._refresh_state()

    async def run(self) -> None:
        """Boucle principale : rend la main après un `Shutdown` ou une annulation (Ctrl+C)."""
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        self._wake_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-wake")
        self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mikata-io")
        try:
            if self._tray is not None:
                # Mascotte masquée : l'icône de la barre de menus devient le seul moyen de
                # la faire revenir ; l'icône suit data/ui_state.json (écrit par l'interface).
                watch = getattr(self._tray, "watch_mascot_state", None)
                if callable(watch):
                    from mikata.ui.prefs import STATE_FILE

                    watch(self.config.resolve(STATE_FILE))
                hidden = list(self.config.uia.devtools_apps)  # phase 17e : « Fermer Spotify »
                self._tray.start(
                    on_quit=lambda: self.request_shutdown("menu Quitter"),
                    on_action=self._on_tray_action if self._ui is not None else None,
                    **({"apps": hidden} if hidden and self._ui is not None else {}),
                )
            if self._ui is not None:
                try:
                    await self._ui.start()
                except OSError as exc:  # l'interface est un plus : jamais bloquante
                    logger.warning("Canal de l'interface indisponible : {}", exc)
                    self._ui = None
                else:
                    if self.config.ui.mascot and self._ui_launcher is not None:
                        self._ui_launcher([])
            if self._player is not None:
                self._player.start(loop)
            if self._stt is not None:
                logger.info("Chargement de Whisper ({})...", self.config.stt.model)
                await asyncio.to_thread(self._stt.load)
            if self._speaker is not None:
                logger.info("Chargement de la voix...")
                await asyncio.to_thread(self._speaker.load)
            if self._agent is not None:
                self._spawn(self._agent.prewarm(), "préchauffage LLM")
            if self._audio is not None:
                self._audio.start(loop)
                self._tasks.append(asyncio.create_task(self._audio_loop(), name="audio"))
                self._last_frame_at = time.monotonic()
                if self._ui is not None:
                    self._tasks.append(
                        asyncio.create_task(self._mic_watchdog(), name="chien de garde micro")
                    )
            if self._hotkeys is not None:
                self._start_hotkeys()
            if self._player is not None and hasattr(self._player, "switch"):
                from mikata.audio.follow import default_output_source

                source = default_output_source()
                if source is not None:  # phase 15f : la sortie suit celle de Windows
                    self._tasks.append(
                        asyncio.create_task(self._follow_output(source), name="suivi sortie")
                    )
            if self._cpu is not None and self.config.logging.stats_interval_s > 0:
                self._tasks.append(asyncio.create_task(self._stats_loop(), name="stats"))
            self._tasks.append(asyncio.create_task(self._maintenance_loop(), name="maintenance"))
            self.prewarm_work_browser()
            if self.config.mcp.enabled and self._agent is not None:
                self._spawn(self._start_mcp(), "serveurs MCP")  # phase 16f
            if self._collector is not None:
                self._negatives_saved = sum(1 for _ in self._negatives_dir.rglob("*.wav"))
                logger.warning(
                    "Mode collect_negatives actif : extraits dans {} ({:.1f} h sur {:.0f} h).",
                    self._negatives_dir,
                    self._negatives_saved * NEGATIVE_CLIP_S / 3600,
                    self.config.wake.negatives_max_hours,
                )
                self._check_negatives_cap()

            self._set_state(State.IDLE)
            logger.info("MIKATA démarré, en attente.")
            while True:
                event = await self.bus.get()
                if isinstance(event, Shutdown):
                    logger.info("Arrêt demandé ({}).", event.reason or "sans motif")
                    break
                try:
                    await self._handle(event)
                except Exception:  # un événement raté ne doit jamais arrêter l'assistant
                    logger.exception("Erreur pendant le traitement de {}.", type(event).__name__)
        finally:
            await self._shutdown()

    def _start_hotkeys(self) -> None:
        assert self._hotkeys is not None

        def on_trigger() -> None:  # thread pynput
            self.bus.publish_threadsafe(WakeDetected(score=1.0, source="hotkey"))

        try:
            self._hotkeys.start(on_trigger)
        except Exception as exc:  # le raccourci est un secours : jamais bloquant
            logger.warning("Raccourci clavier indisponible : {}", exc)
            self._hotkeys = None

    async def _start_mcp(self) -> None:
        """Phase 16f : serveurs MCP démarrés en arrière-plan ; leurs outils rejoignent le
        registre, et le SDK des programmes est reconstruit."""
        from mikata.mcp.bridge import McpBridge

        agent = self._agent
        if agent is None or agent.tools is None:
            return
        bridge = McpBridge(self.config.mcp, self.config.resolve(".env"))
        self._mcp = bridge
        try:
            added = await bridge.start(agent.tools)
        except Exception:
            logger.exception("Démarrage des serveurs MCP impossible.")
            return
        if added:
            agent.refresh_tools()
            logger.info("Outils MCP disponibles : {}.", ", ".join(added))

    async def _shutdown(self) -> None:
        if self._ui is not None:
            try:
                await self._ui.close()  # « bye » : l'interface se ferme aussi
            except Exception:
                logger.exception("Erreur à la fermeture du canal de l'interface.")
        if self._voice_stop is not None:
            self._voice_stop.reset()
        for request in self._active.values():
            if request.recorder is not None:
                request.recorder.cancel()
            if request.stream is not None:
                request.stream.cancel()
        if self._mcp is not None:
            try:
                await self._mcp.close()
            except Exception:
                logger.exception("Erreur à l'arrêt des serveurs MCP.")
        tasks = [*self._tasks, *self._request_tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._request_tasks.clear()
        for name, aclose in (
            ("voix", self._speaker.aclose if self._speaker else None),
            ("LLM", self._agent.aclose if self._agent else None),
        ):
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.exception("Erreur à l'arrêt ({}).", name)
        for name, stop in (
            ("raccourci", self._hotkeys.stop if self._hotkeys else None),
            ("micro", self._audio.stop if self._audio else None),
            ("sortie", self._player.close if self._player else None),
            ("whisper", self._stt.close if self._stt else None),
            ("icône", self._tray.stop if self._tray else None),
        ):
            if stop is None:
                continue
            try:
                stop()
            except Exception:
                logger.exception("Erreur à l'arrêt ({}).", name)
        if self._wake_executor is not None:
            self._wake_executor.shutdown(wait=True, cancel_futures=True)
        if self._io_executor is not None:  # les écritures en attente vont jusqu'au bout
            self._io_executor.shutdown(wait=True)
        logger.info("MIKATA arrêté proprement.")

    # -- audio et mot d'activation ----------------------------------------------------------

    async def _audio_loop(self) -> None:
        assert self._audio is not None
        loop = asyncio.get_running_loop()
        half_threshold = self._trigger.threshold / 2
        while True:
            captured = await self._audio.read()
            self._last_frame_at = time.monotonic()
            if self.paused:  # micro coupé côté logiciel (phase 8) : la trame est jetée
                continue
            self.ring.write(captured.samples)
            self._written += len(captured.samples)
            # Pas d'await entre l'écriture dans le tampon et cette lecture : une trame est soit
            # dans le pré-roll d'une commande, soit ajoutée à son enregistrement, jamais les deux.
            request = self._listening
            if self._collector is not None:
                clip = self._collector.add(captured.samples)
                if clip is not None:
                    self._submit_io(self._save_negative, clip, datetime.now())
                    self._count_negative()
            recorder = request.recorder if request is not None else None
            if recorder is not None and not recorder.finished and self._vad is not None:
                speech = await loop.run_in_executor(
                    self._wake_executor, self._vad.is_speech, captured.samples
                )
                recorder.add(captured.samples, speech, captured.captured_at)
                if speech and request is not None and request.state is State.FOLLOW_UP:
                    self._set_request_state(request, State.LISTENING)  # tu parles
            if self._voice_stop is not None:
                await self._spot_voice_stop(request is None, captured)
            if self._detector is None:
                continue
            score = await loop.run_in_executor(
                self._wake_executor, self._detector.process, captured.samples
            )
            if score >= half_threshold:
                logger.debug("Score mot d'activation : {:.3f}", score)
            if self._trigger.update(score, captured.captured_at):
                self._wake_audio = self.ring.read_last(round(WAKE_LOG_S * SAMPLE_RATE))
                self._wake_written = self._written
                voice = getattr(self._detector, "last_verifier_score", None)
                self.bus.publish(WakeDetected(score=score, ts=captured.captured_at, verifier=voice))

    async def _spot_voice_stop(self, mic_free: bool, captured: CapturedFrame) -> None:
        """Arrêt vocal : VAD + découpe des phrases tant qu'une réponse est jouée et que le
        micro ne sert à aucune commande (sinon c'est l'écoute qui entend « stop »)."""
        assert self._voice_stop is not None and self._vad is not None
        if not mic_free or self._playing_stream() is None:
            if self._voice_stop_armed:
                self._voice_stop_armed = False
                self._voice_stop.reset()
            return
        loop = asyncio.get_running_loop()
        if not self._voice_stop_armed:
            self._voice_stop_armed = True
            if self._wake_executor is not None:  # même thread que is_speech : ordre garanti
                self._wake_executor.submit(self._vad.reset)
        speech = await loop.run_in_executor(
            self._wake_executor, self._vad.is_speech, captured.samples
        )
        self._voice_stop.add(captured.samples, speech, captured.captured_at)

    def _on_voice_stop(self, heard: StopHeard) -> None:
        """« stop » dit sans « MIKATA » pendant une réponse : coupe le son de la réponse en
        train d'être jouée, et arrête sa commande si elle travaille encore (comme « MIKATA,
        stop »). Les autres commandes continuent."""
        stream = self._playing_stream()
        if stream is None or self._listening is not None:
            logger.info("Arrêt vocal : « {} » entendu, plus rien à couper.", heard.text)
            return
        stream.cancel()  # AudioPlayer.stop() : silence dès le bloc audio suivant
        cut_at = time.monotonic()
        owner = next((r for r in self._active.values() if r.stream is stream), None)
        stopped = ""
        if (
            owner is not None
            and "answered" not in owner.marks
            and owner.task is not None
            and not owner.task.done()
        ):
            owner.task.cancel()  # plus d'outil ni d'appel LLM pour une réponse coupée
            stopped = " ; commande arrêtée"
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        total_ms = (cut_at - heard.last_speech_at) * 1000 + out_ms
        label = f"#{owner.id}" if owner is not None else stream.label or "annonce"
        log = logger.info if total_ms < VOICE_STOP_TARGET_MS else logger.warning
        log(
            "Arrêt vocal : « {} » -> réponse {} coupée ≈ {:.0f} ms après la fin de parole "
            "(silence {:.0f} + attente transcription {:.0f} + sortie {:.0f} ms ; Whisper "
            "{:.0f} ms ; objectif < {} ms){}.",
            heard.text,
            label,
            total_ms,
            (heard.ended_at - heard.last_speech_at) * 1000,
            (cut_at - heard.ended_at) * 1000,
            out_ms,
            heard.transcript.latency_s * 1000,
            VOICE_STOP_TARGET_MS,
            stopped,
        )

    def _playing_stream(self) -> SpeechStream | None:
        """Réponse en train d'être jouée (premier son parti, pas coupée), ou None."""
        stream = self._speaker.current if self._speaker is not None else None
        if stream is None or stream.first_sound_at is None or stream.cancelled:
            return None
        return stream

    async def _handle(self, event: BusEvent) -> None:
        if isinstance(event, WakeDetected):
            self._on_wake(event)
        else:
            logger.debug("Événement non géré : {}", type(event).__name__)

    def _on_wake(self, event: WakeDetected) -> None:
        if self.paused:
            logger.info("Écoute en pause : {} ignoré.", event.source)
            return
        now = time.monotonic()
        listening = self._listening
        if listening is not None:  # le micro sert déjà à une commande
            if listening.source == FOLLOW_UP_SOURCE and "wake" not in listening.marks:
                # « MIKATA » dit dans la fenêtre de suivi : elle écoute déjà, le mot sera
                # retiré du texte ; le bip confirme qu'on t'entend.
                listening.marks["wake"] = now
                self._play("wake")
                self._set_request_state(listening, State.LISTENING)
            logger.debug("Mot d'activation ignoré pendant une écoute ({}).", event.source)
            return
        barged = self._barge_in(event)
        miss = self._miss_audio(event)
        if miss is None:
            self._play("wake")
        else:  # bip d'écoute puis deux tics : le raté est noté (un seul effet à la fois)
            self._play_chain("wake", "miss")
        self._publish_ui("wake", score=round(event.score, 3), voice=event.verifier)
        self.detections += 1
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        pipeline_ms = (now - event.ts) * 1000
        if self._agent is not None:  # TLS prête quand la commande partira (§2, technique 2)
            self._spawn(self._agent.prewarm(), "préchauffage LLM")
        if self._active:
            logger.info("Nouvelle commande pendant {} autre(s) en cours.", len(self._active))

        if event.source == "hotkey":
            self._trigger.suppress(now)
            logger.info("Raccourci clavier : bip lancé (+ sortie {:.0f} ms).", out_ms)
            if miss is not None:
                self.misses += 1
                logger.info(
                    "Raté du mot d'activation noté (raccourci) : {:.1f} s d'audio gardées "
                    "pour le prochain entraînement.",
                    len(miss) / SAMPLE_RATE,
                )
                self._submit_io(self._save_miss, miss, datetime.now())
            self._start_listening(event, now, barged)
            return

        if self._detector is not None and self._wake_executor is not None:
            self._wake_executor.submit(self._detector.reset)
        audio, self._wake_audio = self._wake_audio, None
        tail_s = estimate_tail_silence_s(audio) if audio is not None else None
        scores = f"score {event.score:.2f}"
        if event.verifier is not None:
            scores += f", voix {event.verifier:.2f}"
        if tail_s is None:
            logger.info(
                "Mot d'activation détecté ({}) : bip {:.0f} ms après la trame "
                "(+ sortie {:.0f} ms).",
                scores,
                pipeline_ms,
                out_ms,
            )
        else:
            logger.info(
                "Mot d'activation détecté ({}) : bip ≈ {:.0f} ms après la fin du mot "
                "(détection {:.0f} + traitement {:.0f} + sortie {:.0f} ms).",
                scores,
                tail_s * 1000 + pipeline_ms + out_ms,
                tail_s * 1000,
                pipeline_ms,
                out_ms,
            )
        if audio is not None:
            self._submit_io(self._save_wake_log, audio, event.score, datetime.now())
        self._start_listening(event, now, barged)

    def _barge_in(self, event: WakeDetected) -> _Request | None:
        """Coupe la réponse **en train d'être jouée** (et seulement elle) ; renvoie sa
        commande. Une réponse pas encore commencée n'est pas touchée : elle attendra la fin
        de l'écoute."""
        stream = self._playing_stream()
        if stream is None:
            return None
        stream.cancel()  # AudioPlayer.stop() : silence dès le bloc audio suivant
        cut_at = time.monotonic()
        owner = next((r for r in self._active.values() if r.stream is stream), None)
        out_ms = (self._player.output_latency_s if self._player else 0.0) * 1000
        tail_s = None
        if event.source != "hotkey" and self._wake_audio is not None:
            tail_s = estimate_tail_silence_s(self._wake_audio)
        label = f"#{owner.id}" if owner is not None else stream.label or "annonce"
        if tail_s is None:
            logger.info(
                "Barge-in : réponse {} coupée {:.0f} ms après la trame (+ sortie {:.0f} ms).",
                label,
                (cut_at - event.ts) * 1000,
                out_ms,
            )
        else:
            logger.info(
                "Barge-in : réponse {} coupée ≈ {:.0f} ms après la fin du mot (détection {:.0f} "
                "+ traitement {:.0f} + sortie {:.0f} ms).",
                label,
                tail_s * 1000 + (cut_at - event.ts) * 1000 + out_ms,
                tail_s * 1000,
                (cut_at - event.ts) * 1000,
                out_ms,
            )
        return owner

    # -- écoute (une seule à la fois : il n'y a qu'un micro) -------------------------------

    @property
    def _can_listen(self) -> bool:
        return self._stt is not None and self._vad is not None and self._audio is not None

    async def listen_once(self, timeout_s: float) -> str:
        """Écoute une réponse courte sans mot d'activation (« oui », « stop »...).

        Attend que le micro soit libre, bipe, écoute jusqu'à la fin de la phrase (ou
        `timeout_s` sans parole), puis transcrit. Renvoie "" si rien n'a été dit.
        """
        if not self._can_listen:
            return ""
        assert self._stt is not None and self._vad is not None
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout_s
        while self._listening is not None:  # une autre commande a le micro
            if time.monotonic() >= deadline:
                return ""
            await asyncio.sleep(0.05)
        now = time.monotonic()
        self._play("wake")
        if self._wake_executor is not None:
            self._wake_executor.submit(self._vad.reset)
        ended: asyncio.Future[EndReason] = loop.create_future()

        def on_end(reason: EndReason) -> None:
            if not ended.done():
                ended.set_result(reason)

        request = _Request(id=next(self._ids), source="réponse", wake_at=now, wake_frame_at=now)
        vad = self.config.vad.model_copy(update={"no_speech_timeout_s": max(0.5, deadline - now)})
        pre_roll = self.ring.read_last(round(HOTKEY_PRE_ROLL_S * SAMPLE_RATE))
        request.recorder = CommandRecorder(
            self._stt, pre_roll, vad, self._wake_pattern, on_end=on_end, check_pre_roll=False
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._set_request_state(request, State.LISTENING)
        try:
            if await ended is EndReason.NO_SPEECH:
                logger.info("Réponse orale : rien entendu en {:.0f} s.", timeout_s)
                return ""
            if self._listening is request:
                self._set_listening(None)
            self._set_request_state(request, State.TRANSCRIBING)
            transcript = await request.recorder.transcribe()
            text, _ = strip_wake_word(clean_transcript(transcript.text), self._wake_pattern)
            logger.info("Réponse orale : « {} »", text)
            return text
        finally:
            self._finish(request)

    def _start_listening(
        self, event: WakeDetected, now: float, barged: _Request | None = None
    ) -> None:
        if self._stt is None or self._vad is None or self._audio is None:
            return
        hotkey = event.source == "hotkey"
        if hotkey:
            samples = round(HOTKEY_PRE_ROLL_S * SAMPLE_RATE)
        else:  # pré-roll complet avant la trame de détection + les trames arrivées depuis
            samples = round(self.config.audio.pre_roll_s * SAMPLE_RATE)
            samples += self._written - self._wake_written
        pre_roll = self.ring.read_last(min(samples, self.ring.capacity))
        if self._wake_executor is not None:  # même thread que is_speech : ordre garanti
            self._wake_executor.submit(self._vad.reset)
        request = _Request(
            id=next(self._ids),
            source=event.source,
            wake_at=now,
            wake_frame_at=event.ts,
            barged=barged,
        )
        request.recorder = CommandRecorder(
            self._stt,
            pre_roll,
            self.config.vad,
            self._wake_pattern,
            on_end=functools.partial(self._on_recorder_end, request),
            check_pre_roll=not hotkey,
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._set_request_state(request, State.LISTENING)
        logger.debug(
            "Commande #{} : écoute, {:.1f} s de pré-roll.", request.id, len(pre_roll) / SAMPLE_RATE
        )

    def _on_recorder_end(self, request: _Request, reason: EndReason) -> None:
        request.end_at = time.monotonic()
        request.reason = reason
        if self._listening is request:
            self._set_listening(None)  # micro libre : une autre commande peut commencer
        if reason is EndReason.NO_SPEECH:
            if request.source == FOLLOW_UP_SOURCE:
                logger.info("Fenêtre de suivi terminée : rien de dit.")
            else:
                logger.info("Écoute terminée : {}.", reason)
            self._play("end")
            self._finish(request)
            return
        self._set_request_state(request, State.TRANSCRIBING)
        request.task = self._spawn(self._process(request), f"commande #{request.id}")

    def _start_follow_up(self) -> None:
        """Fenêtre de suivi : écoute `agent.follow_up_s` sans mot d'activation, seulement si
        plus rien n'est en cours (sinon la dernière réponse à finir l'ouvrira)."""
        seconds = self.config.agent.follow_up_s
        if seconds <= 0 or not self._can_listen or self._listening is not None or self._active:
            return
        if self._speaker is not None and self._speaker.busy:
            return
        assert self._stt is not None and self._vad is not None
        now = time.monotonic()
        if self._wake_executor is not None:
            self._wake_executor.submit(self._vad.reset)
        request = _Request(
            id=next(self._ids),
            source=FOLLOW_UP_SOURCE,
            wake_at=now,
            wake_frame_at=now,
            state=State.FOLLOW_UP,
        )
        request.recorder = CommandRecorder(
            self._stt,
            self.ring.read_last(round(FOLLOW_UP_PRE_ROLL_S * SAMPLE_RATE)),
            self.config.vad.model_copy(update={"no_speech_timeout_s": seconds}),
            self._wake_pattern,
            on_end=functools.partial(self._on_recorder_end, request),
            check_pre_roll=False,
        )
        self._set_listening(request)
        self._active[request.id] = request
        self._refresh_state()
        logger.debug("Commande #{} : fenêtre de suivi ({:.0f} s).", request.id, seconds)

    # -- traitement d'une commande (plusieurs en parallèle) --------------------------------

    def _spawn(self, coro: object, name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        self._request_tasks.add(task)
        task.add_done_callback(self._request_tasks.discard)
        return task

    async def _process(self, request: _Request) -> None:
        try:
            text = await self._transcribe(request)
            if text and self._cancel_requested(request, text):
                return
            if text and self._agent is not None and self._speaker is not None:
                await self._respond(request, text)
        except asyncio.CancelledError:
            if request.stream is not None:
                request.stream.cancel()  # plus un mot de cette réponse
            self._publish_ui("result", id=request.id, outcome="cancelled")
            raise
        except Exception as exc:
            logger.exception("Commande #{} : erreur.", request.id)
            self._play("error")
            self._publish_ui("error", id=request.id, message=str(exc) or type(exc).__name__)
        finally:
            if request.stream is not None:
                request.stream.close()
            self._finish(request)
        if request.follow_up:
            self._start_follow_up()

    def _cancel_requested(self, request: _Request, text: str) -> bool:
        """« stop », « annule »... : True si la commande est une annulation, traitée ici
        (sans LLM). Cible : la commande dont la réponse vient d'être coupée si elle tourne
        encore, sinon la dernière commande en cours."""
        if not is_cancel_command(text, self.config.agent.cancel_phrases):
            return False
        running = [
            r
            for r in self._active.values()
            if r is not request and r.task is not None and not r.task.done()
        ]
        barged = request.barged
        target = barged if barged is not None and barged in running else None
        if target is None and running:
            target = max(running, key=lambda r: r.id)
        if target is not None:
            assert target.task is not None
            target.task.cancel()
            logger.info(
                "Commande #{} : « {} » -> commande #{} annulée.", request.id, text, target.id
            )
        elif barged is not None or request.source == FOLLOW_UP_SOURCE:
            logger.info("Commande #{} : « {} » -> rien d'autre à annuler.", request.id, text)
        else:
            return False  # rien à annuler : « stop » est une vraie commande (musique...)
        self._play("end")
        return True

    async def _transcribe(self, request: _Request) -> str:
        assert request.recorder is not None
        recorder = request.recorder
        transcript = await recorder.transcribe()
        request.marks["stt_ready"] = time.monotonic()
        raw = transcript.text
        text, found = strip_wake_word(clean_transcript(raw), self._wake_pattern)
        logger.info(
            "Mesures : fin_parole +{:.0f} ms après le bip, stt_prêt +{:.0f} ms après "
            "fin_parole ({}), Whisper {:.2f} s pour {:.1f} s d'audio, fin : {}.",
            (request.end_at - request.wake_at) * 1000,
            (request.marks["stt_ready"] - request.end_at) * 1000,
            "spéculatif" if recorder.used_speculative else "direct",
            transcript.latency_s,
            transcript.duration_s,
            request.reason,
        )
        if request.source != "hotkey" and not found:
            logger.debug("Mot d'activation absent du texte : rien de retiré.")
        if not text:
            logger.info("Rien compris (texte brut : « {} »).", raw)
            self._play("end")
            return ""
        if request.source == FOLLOW_UP_SOURCE:
            min_confidence = self.config.agent.follow_up_min_confidence
            rejection = follow_up_rejection(text, transcript, min_confidence)
            if rejection is not None:
                logger.info("Fenêtre de suivi : « {} » ignorée ({}).", text, rejection)
                self._play("end")
                return ""
            logger.info(
                "Fenêtre de suivi : « {} » retenue ({}).", text, describe_confidence(transcript)
            )
        logger.success("Commande #{} : « {} » (texte brut : « {} »)", request.id, text, raw)
        self.commands.append(text)
        self._publish_ui("heard", id=request.id, text=text)
        return text

    async def _respond(self, request: _Request, text: str) -> None:
        assert self._agent is not None and self._speaker is not None
        self._set_request_state(request, State.THINKING)
        stream = self._speaker.open(label=f"#{request.id}")
        request.stream = stream
        stream.on_first_sound = functools.partial(self._on_first_sound, request)
        if self._ui is not None:
            stream.on_say = functools.partial(self._on_say, request)
        gated = request.source == FOLLOW_UP_SOURCE and self.config.agent.follow_up_use_llm_gate
        if not gated:  # suivi jugé par le LLM : pas de son tant qu'on ne sait pas si on répond
            self._spawn(self._thinking_sound(request), f"réflexion #{request.id}")
        result = await self._agent.respond(
            text,
            stream,
            label=f"#{request.id}",
            on_phase=functools.partial(self._on_agent_phase, request),
            listen=functools.partial(self._listen_for, stream) if self._can_listen else None,
            follow_up=gated,
        )
        request.marks["answered"] = time.monotonic()  # le LLM a fini ; reste à tout dire
        stream.close()
        await stream.wait()
        request.marks["done"] = time.monotonic()
        self._publish_ui("result", id=request.id, outcome=ui_outcome(result, stream.cancelled))
        if result.outcome == "ignored":  # pas pour MIKATA : bip de fin, retour au repos
            self._play("end")
            return
        request.follow_up = not stream.cancelled
        if result.answer:
            self.answers.append(result.answer)
        self._log_response(request, result)

    def _on_say(self, request: _Request, text: str) -> None:
        self._publish_ui("say", id=request.id, text=text)

    async def _listen_for(self, stream: SpeechStream, timeout_s: float) -> str:
        """Réponse orale pour une commande. Si sa réponse a été coupée (barge-in), la
        question ou l'annonce n'a pas été entendue : réponse « stop », qui refuse une
        confirmation et annule une extinction (jamais l'inverse)."""
        if stream.cancelled:
            logger.info("Réponse coupée avant l'écoute : vaut « {} ».", INTERRUPTED_ANSWER)
            return INTERRUPTED_ANSWER
        return await self.listen_once(timeout_s)

    def _on_first_sound(self, request: _Request) -> None:
        request.marks["first_sound"] = time.monotonic()
        self._set_request_state(request, State.SPEAKING)

    def _on_agent_phase(self, request: _Request, phase: str) -> None:
        """Outils en cours (ACTING), puis retour au LLM (THINKING, ou SPEAKING s'il a déjà
        parlé)."""
        if phase == "acting":
            self._set_request_state(request, State.ACTING)
        elif request.stream is not None and request.stream.first_sound_at is not None:
            self._set_request_state(request, State.SPEAKING)
        else:
            self._set_request_state(request, State.THINKING)

    def voice_event(self, kind: str, text: str) -> None:
        """Routeur de voix (phase 11, thread de synthèse) : « fallback » (ElevenLabs
        indisponible, raison), « resume » (de retour), « profile » (voix changée). L'annonce
        orale du secours part avec le fragment lui-même (tts/router.py)."""
        loop = self.bus.loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):  # boucle fermée
            loop.call_soon_threadsafe(self._publish_voice, kind, text)

    def _publish_voice(self, kind: str, text: str) -> None:
        engine = "kokoro" if kind == "fallback" else "elevenlabs" if kind == "resume" else ""
        if kind == "fallback":
            message = f"Je passe sur ma voix de base : {text}."
        elif kind == "resume":
            message = f"Retour sur la voix {text}."
        else:
            message = f"Voix : {text}."
        self._publish_ui("voice", event=kind, engine=engine, message=message)

    async def announce(self, text: str) -> None:
        """Annonce hors commande (minuteur, rappel) : son « timer » puis phrase dite, à la
        suite des réponses en cours."""
        self._play("timer")
        if self._speaker is None:
            return
        stream = self._speaker.open(label="annonce")
        stream.say(text)
        stream.close()
        await stream.wait()

    async def _thinking_sound(self, request: _Request) -> None:
        """Petit son si rien n'est joué `thinking_sound_after_s` après la fin de parole (§2)."""
        delay = request.end_at + self.config.agent.thinking_sound_after_s - time.monotonic()
        await asyncio.sleep(max(0.0, delay))
        stream = request.stream
        if request.id in self._active and stream is not None and stream.first_sound_at is None:
            self._play("thinking")

    def _log_response(self, request: _Request, result: TurnResult) -> None:
        def since_end(t: float | None) -> str:
            return "—" if t is None else f"+{(t - request.end_at) * 1000:.0f} ms"

        stream = request.stream
        first_sound = stream.first_sound_at if stream is not None else None
        after_syllable = (
            f"{(first_sound - request.last_syllable_at) * 1000:.0f} ms"
            if first_sound is not None
            else "—"
        )
        logger.info(
            "Réponse #{} : premier son ≈ {} après la dernière syllabe, réflexion {} (depuis "
            "fin_parole : stt_prêt {}, llm_premier_token {}, premier_son {}, total {}) ; "
            "chemin : {} ; issue : {}.",
            request.id,
            after_syllable,
            result.effort_label,
            since_end(request.marks.get("stt_ready")),
            since_end(result.first_token_at),
            since_end(first_sound),
            since_end(request.marks.get("done")),
            result.path,
            result.outcome,
        )
        if result.answer:
            logger.info("Réponse #{} : « {} »", request.id, result.answer)

    def _finish(self, request: _Request) -> None:
        if self._listening is request:
            self._set_listening(None)
        if request.recorder is not None:
            request.recorder.cancel()
        self._active.pop(request.id, None)
        self._refresh_state()

    def _play(self, earcon: str) -> None:
        if self._player is not None and earcon in self._earcons:
            self._player.play_nowait(self._earcons[earcon])

    def _play_chain(self, *earcons: str) -> None:
        """Plusieurs sons à la suite, en un seul effet (le lecteur n'en joue qu'un à la fois)."""
        sounds = [self._earcons[name] for name in earcons if name in self._earcons]
        if self._player is None or not sounds:
            return
        rate = getattr(self._earcons, "sample_rate", 48_000)
        gap = np.zeros(round(0.06 * rate), dtype=np.float32)
        parts = [part for sound in sounds for part in (sound, gap)][:-1]
        self._player.play_nowait(np.concatenate(parts))

    def _miss_audio(self, event: WakeDetected) -> np.ndarray | None:
        """Raccourci clavier après un « MIKATA » raté : les dernières secondes du tampon, à
        garder pour le prochain entraînement (phase 9). None si rien à garder."""
        if event.source != "hotkey" or not self.config.wake.save_misses or self._audio is None:
            return None
        audio = self.ring.read_last(min(round(MISS_CLIP_S * SAMPLE_RATE), len(self.ring)))
        return audio if len(audio) else None

    # -- écritures disque (thread mikata-io) -----------------------------------------------

    def _submit_io(self, fn: Callable[..., None], *args: object) -> None:
        if self._io_executor is not None:
            self._io_executor.submit(self._guarded, fn, *args)

    @staticmethod
    def _guarded(fn: Callable[..., None], *args: object) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("Erreur d'écriture sur le disque.")

    def _save_wake_log(self, audio: np.ndarray, score: float, now: datetime) -> None:
        path = save_detection(self._wake_logs_dir, audio, score, now)
        logger.debug("Audio de détection : {}", path)
        self._purge_wake_logs()

    def _save_miss(self, audio: np.ndarray, now: datetime) -> None:
        path = save_miss(self._misses_dir, audio, now)
        logger.info("Raté enregistré : {}", path)
        count = purge_older_than(self._misses_dir, self.config.wake.miss_days)
        if count:
            logger.info("{} ancien(s) raté(s) du mot d'activation mis à la corbeille.", count)

    def _purge_wake_logs(self) -> None:
        count = purge_older_than(self._wake_logs_dir, self.config.wake.log_days)
        if count:
            logger.info("{} ancien(s) audio(s) de détection mis à la corbeille.", count)

    async def _maintenance_loop(self) -> None:
        """Purge au démarrage puis toutes les 24 h : MIKATA tourne sans redémarrer (phase 7)."""
        while True:
            self._submit_io(self._maintenance)
            await asyncio.sleep(MAINTENANCE_INTERVAL_S)

    def _maintenance(self) -> None:
        counts = run_maintenance(self.config)
        done = {name: n for name, n in counts.items() if n}
        if done:
            details = ", ".join(f"{n} dans {name}" for name, n in done.items())
            logger.info("Purge : {} mis à la corbeille.", details)

    def _count_negative(self) -> None:
        self._negatives_saved += 1
        self._check_negatives_cap()

    def _check_negatives_cap(self) -> None:
        """Collecte bornée à `wake.negatives_max_hours` : au-delà, elle s'arrête d'elle-même."""
        limit = self.config.wake.negatives_max_hours
        if self._collector is not None and self._negatives_saved * NEGATIVE_CLIP_S >= limit * 3600:
            self._collector = None
            logger.warning(
                "Collecte de la vie courante arrêtée : {:.0f} h dans {} "
                "(wake.negatives_max_hours). Remets wake.collect_negatives: false.",
                limit,
                self._negatives_dir,
            )

    def _save_negative(self, clip: np.ndarray, now: datetime) -> None:
        from mikata.audio.wavfile import write_wav

        write_wav(NegativeCollector.path_for(self._negatives_dir, now), clip, SAMPLE_RATE)

    # -- statistiques -----------------------------------------------------------------------

    async def _stats_loop(self) -> None:
        assert self._cpu is not None
        interval = self.config.logging.stats_interval_s
        while True:
            await asyncio.sleep(interval)
            one_core, total = self._cpu.sample()
            dropped = self._audio.dropped_frames if self._audio else 0
            overflows = self._audio.overflows if self._audio else 0
            logger.info(
                "CPU du processus : {:.1f} % d'un cœur ({:.2f} % du PC) ; "
                "détections : {} ; trames perdues : {} ; débordements micro : {}",
                one_core,
                total,
                self.detections,
                dropped,
                overflows,
            )


def ui_outcome(result: TurnResult, cancelled: bool) -> str:
    """Issue d'une commande pour l'interface : success (« C'est fait », outils réussis), ok
    (réponse), error, ignored (suivi pas pour MIKATA), cancelled (réponse coupée)."""
    if cancelled:
        return "cancelled"
    if result.outcome in ("server", "timeout", "limit"):
        return "error"
    if result.outcome == "ignored":
        return "ignored"
    if result.shortcut or (result.tool_results and all(r.ok for r in result.tool_results)):
        return "success"
    return "ok"
