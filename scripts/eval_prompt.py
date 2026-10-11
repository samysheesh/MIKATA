"""Évaluation du prompt système sur 26 phrases approximatives (vrai LLM, actions simulées).

Usage : uv run python scripts/eval_prompt.py [--only 1 3 ...] [--json rapport.json]

Chaque phrase passe par la vraie boucle agent et le vrai LLM (config.yaml + .env), mais
**rien n'est fait sur le PC** : bureau, volume, clavier, souris, PowerShell et corbeille sont
simulés et notent ce qui aurait été fait. Scénario fixe : Bloc-notes au premier plan, Chrome,
Spotify et Discord ouverts, une liste d'applications installées connue.

Une phrase réussit si le LLM appelle l'outil attendu avec les bons arguments, sans
run_powershell, sans question de précision, et répond court sans parler de PowerShell ni de
syntaxe. Ajout phase 6 : histoire, blague, explication -> réponse dite à voix haute, sans
aucun outil (ni Bloc-notes ni fichier), en 3 à 5 phrases. Coût : une vingtaine d'appels LLM
(≈ 0,01 $).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mikata.tools.apps import fold
from mikata.tools.base import AppEntry, UiControlInfo, UiWindowInfo, WindowInfo
from mikata.tools.input import parse_keys

FORBIDDEN_TOOLS = ("run_powershell", "run_script")
FORBIDDEN_WORDS = ("powershell", "stop-process", "syntaxe", "argument", "outil ")
MAX_WORDS = 25

SCENARIO_APPS = (
    "Bloc-notes", "Brave", "Calculatrice", "Counter-Strike 2", "Discord", "Excel",
    "Explorateur de fichiers", "Google Chrome", "Microsoft Edge", "Netflix", "OBS Studio",
    "Paint", "Paramètres", "Photos", "Spotify", "Steam", "Visual Studio Code",
    "VLC media player", "WhatsApp", "Word",
)  # fmt: skip
SCENARIO_WINDOWS = (
    WindowInfo(101, "Sans titre - Bloc-notes", "notepad"),
    WindowInfo(102, "YouTube - Google Chrome", "chrome"),
    WindowInfo(103, "Spotify Premium", "Spotify"),
    WindowInfo(104, "Discord", "Discord"),
)

Check = Callable[[dict[str, Any]], bool]


def named(*options: str) -> Check:
    """L'argument `name` désigne l'une des applis (sans casse ni accents)."""
    wanted = [fold(option) for option in options]
    return lambda args: any(w in fold(str(args.get("name") or "")) for w in wanted)


def anything(args: dict[str, Any]) -> bool:
    return True


def keys_are(*expected: str) -> Check:
    def check(args: dict[str, Any]) -> bool:
        try:
            return parse_keys(str(args.get("keys"))) == list(expected)
        except ValueError:
            return False

    return check


@dataclass
class Case:
    phrase: str
    tools: tuple[str, ...]  # un seul de ces outils suffit ; () : aucun outil, réponse dite
    expect: str  # attendu, lisible
    check: Check = anything
    sentences: tuple[int, int] | None = None  # nombre de phrases dites (min, max)
    max_words: int = MAX_WORDS


SPOKEN = ()  # aucun outil : MIKATA répond lui-même à voix haute


NOTEPAD = named("bloc-notes", "bloc notes", "notepad")
CLOSE = ("close_app", "close_window")
CASES: tuple[Case, ...] = (
    Case("stop process name notepad", CLOSE, "fermer le Bloc-notes", NOTEPAD),
    Case("arrête notepad", CLOSE, "fermer le Bloc-notes", NOTEPAD),
    Case("ferme le bloc-notes", CLOSE, "fermer le Bloc-notes", NOTEPAD),
    Case("ouvre spotifaille", ("open_app", "focus_window"), "Spotify", named("spotify")),
    Case("lance Team", ("open_app",), "ouvrir Steam", named("steam")),
    Case("ouvre brève", ("open_app",), "ouvrir Brave", named("brave")),
    Case("ouvre visuel studio code", ("open_app",), "ouvrir Visual Studio Code",
         named("visual studio code", "vs code", "vscode")),
    Case("ferme crôme", CLOSE, "fermer Chrome", named("chrome")),
    Case("ferme cette fenêtre", ("close_window", "close_app"),
         "fermer la fenêtre active (Bloc-notes)",
         lambda a: not a.get("name") or NOTEPAD(a)),
    Case("mets spotify devant", ("focus_window", "open_app"), "Spotify au premier plan",
         named("spotify")),
    Case("mets le son à trente", ("set_volume",), "volume à 30", lambda a: a.get("level") == 30),
    Case("monte un peu le son", ("set_volume",), "volume plus fort",
         lambda a: (a.get("change") or 0) > 0 or (a.get("level") or 0) > 50),
    Case("coupe le son", ("mute",), "sourdine", lambda a: a.get("mute", True) is True),
    Case("met pause", ("media_control",), "pause",
         lambda a: a.get("action") in ("pause", "play_pause")),
    Case("passe à la chanson d'après", ("media_control",), "piste suivante",
         lambda a: a.get("action") == "next"),
    Case("écris bonjour tout le monde", ("type_text",), "taper « Bonjour tout le monde »",
         lambda a: fold(str(a.get("text"))).startswith("bonjour tout le monde")),
    Case("fais contrôle S", ("press_keys",), "Ctrl+S", keys_are("ctrl", "s")),
    Case("mets un minuteur de cinq minutes", ("set_timer",), "minuteur de 300 s",
         lambda a: a.get("seconds") == 300),
    Case("quel temps il fait à Lyon", ("get_weather",), "météo à Lyon",
         lambda a: "lyon" in fold(str(a.get("city")))),
    Case("verrouille l'ordi", ("lock",), "verrouiller la session"),
    # Phase 12a : l'état ambiant Discord se lit sans rien ouvrir.
    Case("qui est dans mon salon vocal discord", ("uia_state", "uia_window", "uia_read"),
         "l'état ambiant Discord",
         lambda a: not a or "discord" in fold(str(a.get("process") or ""))),
    # Phase 12b : une tâche web enchaîne ouverture, lecture et clic.
    Case("lance le dernier opening de black clover",
         ("browser_open", "browser_click"),
         "ouvrir YouTube et cliquer la vidéo",
         lambda a: "youtube" in fold(str(a.get("url") or ""))
         or "black" in fold(str(a.get("text") or ""))),
    # Phase 12c : une procédure apprise se rejoue sans appel LLM.
    Case("passe le pc en silencieux", ("set_volume", "mute"),
         "rejouer la procédure apprise",
         lambda a: a.get("level") == 20),
    # Phase 12d : l'index local répond d'abord, puis l'Explorateur s'ouvre dessus.
    Case("où est cyberpunk 2077", ("find_path",),
         "chercher dans l'index local",
         lambda a: "cyberpunk" in fold(str(a.get("name")))),
    Case("montre-moi où est cyberpunk 2077 dans l'explorateur", ("open_in_explorer",),
         "ouvrir l'Explorateur sur le chemin",
         lambda a: "cyberpunk" in fold(str(a.get("path")))),
    # Phase 6 : « raconte-moi une longue histoire » avait ouvert le Bloc-notes pour l'y coller.
    Case("raconte-moi une longue histoire", SPOKEN, "histoire dite, 3 à 5 phrases",
         sentences=(3, 5), max_words=140),
    Case("raconte-moi une blague", SPOKEN, "blague dite", sentences=(1, 5), max_words=80),
    Case("explique-moi comment marche un frigo", SPOKEN, "explication dite, 3 à 5 phrases",
         sentences=(3, 5), max_words=140),
)  # fmt: skip


@dataclass
class Outcome:
    number: int
    phrase: str
    expect: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    answer: str = ""
    llm_calls: int = 0
    ms: float = 0.0
    cost_usd: float = 0.0


def evaluate(case: Case, calls: list[tuple[str, dict[str, Any]]], answer: str) -> list[str]:
    """Raisons de l'échec (liste vide : réussi)."""
    reasons = []
    names = [name for name, _ in calls]
    if not case.tools and calls:
        reasons.append(f"aucun outil attendu ({case.expect}), appelé : {', '.join(names)}")
    elif case.tools and not any(name in case.tools and case.check(args) for name, args in calls):
        reasons.append(f"attendu {'/'.join(case.tools)} ({case.expect})")
    if used := [name for name in names if name in FORBIDDEN_TOOLS]:
        reasons.append(f"outil à éviter : {', '.join(used)}")
    lowered = answer.casefold()
    if words := [w.strip() for w in FORBIDDEN_WORDS if w in lowered]:
        reasons.append(f"parle de {', '.join(words)}")
    if len(answer.split()) > case.max_words:
        reasons.append(f"réponse trop longue ({len(answer.split())} mots)")
    if case.sentences is not None:
        count = count_sentences(answer)
        low, high = case.sentences
        if not low <= count <= high:
            reasons.append(f"{count} phrase(s) dite(s), attendu {low} à {high}")
    elif not calls and "?" in answer:
        reasons.append("pose une question au lieu d'agir")
    return reasons


def count_sentences(text: str) -> int:
    """Phrases d'une réponse dite (« Pourquoi... ? Parce que... » : 2)."""
    parts = re.split(r"[.!?…]+(?=\s|$)", text)
    return sum(1 for part in parts if re.search(r"[^\W\d_]{2,}", part))


# -- environnement simulé : rien n'est fait sur le PC -------------------------------------


class DryRunDesktop:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.apps = [AppEntry(name, f"simulé:{name}") for name in SCENARIO_APPS]

    def list_apps(self, refresh: bool = False) -> list[AppEntry]:
        return list(self.apps)

    def cached_apps(self) -> list[AppEntry]:
        return list(self.apps)

    def windows(self) -> list[WindowInfo]:
        return list(SCENARIO_WINDOWS)

    def foreground_window(self) -> WindowInfo | None:
        return SCENARIO_WINDOWS[0]

    def launch(self, app: AppEntry) -> None:
        self.actions.append(f"ouvrir {app.name}")

    def open_target(self, target: str) -> None:
        self.actions.append(f"ouvrir {target}")

    def open_folder(self, path: str, select: bool = False) -> None:
        self.actions.append(f"explorateur {path}" + (" (sélection)" if select else ""))

    def close_window(self, window: WindowInfo) -> None:
        self.actions.append(f"fermer « {window.title} »")

    def focus_window(self, window: WindowInfo) -> None:
        self.actions.append(f"premier plan « {window.title} »")

    def media_key(self, action: str) -> None:
        self.actions.append(f"touche média {action}")

    def lock(self) -> None:
        self.actions.append("verrouiller")

    def power(self, action: str) -> None:
        self.actions.append(f"alimentation {action}")

    def clipboard_read(self) -> str:
        return ""

    def clipboard_write(self, text: str) -> None:
        self.actions.append(f"copier {text!r}")


class DryRunAudio:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    def get_volume(self) -> float:
        return 50.0

    def set_volume(self, percent: float) -> None:
        self.actions.append(f"volume {percent:.0f} %")

    def set_mute(self, mute: bool) -> None:
        self.actions.append("sourdine" if mute else "son remis")

    def output_devices(self) -> list[tuple[str, str]]:
        return [("1", "Casque (AirPods)"), ("2", "Haut-parleurs (Realtek(R) Audio)")]

    def set_default_output(self, device_id: str) -> None:
        self.actions.append(f"sortie {device_id}")


class DryRunInput:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    def type_text(self, text: str) -> None:
        self.actions.append(f"taper {text!r}")

    def press_keys(self, keys: list[str]) -> None:
        self.actions.append(f"touches {'+'.join(keys)}")

    def mouse(self, action: str, x: int | None, y: int | None, amount: int) -> None:
        self.actions.append(f"souris {action}")


class DryRunShell:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    async def run(
        self, argv: list[str], timeout_s: float, cwd: Path | None = None
    ) -> tuple[int, str]:
        self.actions.append(f"lancer {argv[-1][-80:]!r}")
        return 0, "(simulation : rien n'a été lancé)"


class DryRunUi:
    """Pilotage UIA simulé (phase 12a) : l'état ambiant répond, rien n'est cliqué."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    def window(self, title: str | None, process: str | None) -> UiWindowInfo:
        self.actions.append(f"fenêtre {process or title or 'active'}")
        return UiWindowInfo(title or "Discord", process or "Discord")

    def controls(self, window: UiWindowInfo, max_controls: int) -> list[UiControlInfo]:
        return [UiControlInfo("Rechercher", "bouton")]

    def read(self, window: UiWindowInfo, control: str | None, max_chars: int) -> str:
        self.actions.append(f"lire {control or window.title}")
        return "Texte simulé."

    def click(self, window: UiWindowInfo, control: str, kind: str | None, nth: int) -> str:
        self.actions.append(f"cliquer {control!r}")
        return f"« {control} » cliqué."

    def set_text(self, window: UiWindowInfo, control: str, text: str) -> str:
        self.actions.append(f"écrire dans {control!r}")
        return f"« {control} » rempli."

    def state_summary(self) -> str:
        self.actions.append("état ambiant")
        return "Discord : salon vocal « Général ». En vocal : Samy."

    def pause_state(self, paused: bool) -> None:
        pass

    def aclose(self) -> None:
        pass


class DryRunBrowser:
    """Navigateur simulé (phase 12b) : pages scriptées, rien n'est ouvert ni cliqué."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    async def tabs(self) -> str:
        self.actions.append("onglets du navigateur")
        return "1 onglet : YouTube — https://youtube.com"

    async def open_url(self, url: str) -> str:
        self.actions.append(f"naviguer {url}")
        return f"Page ouverte : {url}"

    async def read(self, max_chars: int) -> str:
        self.actions.append("lire la page")
        return "Titre : black clover - YouTube\nURL : https://youtube.com\nTexte :\nRésultats."

    async def find(self, text: str, nth: int) -> dict[str, Any]:
        return {"ok": True, "label": text, "tag": "A", "href": "https://youtube.com/watch?v=x"}

    async def click(self, text: str, nth: int) -> str:
        self.actions.append(f"cliquer {text!r}")
        return f"« {text} » cliqué."

    async def type_text(self, field: str, text: str) -> str:
        self.actions.append(f"écrire {text!r}")
        return f"Texte écrit dans le champ « {field} »."

    async def back(self) -> str:
        self.actions.append("page précédente")
        return "Page précédente."

    async def quit(self) -> str:
        self.actions.append("fermer le navigateur de travail")
        return "Navigateur de travail fermé."

    async def aclose(self) -> None:
        pass


class DryRunIndex:
    """Index de fichiers simulé (phase 12d) : un résultat fixe, jamais de vrai disque."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions

    def ready(self) -> bool:
        return True

    def stats(self) -> dict[str, Any]:
        return {"ready": True, "entries": 3}

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self.actions.append(f"chercher dans l'index {query!r}")
        return [
            {
                "path": "D:\\Steam\\steamapps\\common\\Cyberpunk 2077.exe",
                "name": "Cyberpunk 2077.exe",
                "is_dir": False,
                "mtime": 1_700_000_000.0,
            }
        ]

    def list_folder(self, folder: str, limit: int) -> list[dict[str, Any]]:
        self.actions.append(f"lister l'index {folder!r}")
        return [{"name": "Cyberpunk 2077.exe", "is_dir": False, "mtime": 1_700_000_000.0}]

    def aclose(self) -> None:
        pass


class RecordingSpeech:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)

    def say_audio(self, pcm: Any, sample_rate: int) -> None:
        self.said.append("C'est fait.")

    async def drain(self) -> None:
        pass


async def run_cases(
    llm: Any, config: Any, cases: Sequence[tuple[int, Case]], search_api_key: str | None = None
) -> list[Outcome]:
    from mikata.agent.guard import Guard
    from mikata.agent.loop import Agent
    from mikata.agent.memory import LongTermMemory, ShortTermMemory
    from mikata.llm.usage import cost_usd
    from mikata.skills.library import SkillLibrary
    from mikata.tools.apps import installed_app_names, open_apps
    from mikata.tools.base import ToolContext
    from mikata.tools.registry import ToolRegistry
    from mikata.tools.web import create_http_client

    actions: list[str] = []
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="mikata-eval-") as folder:
        sandbox = Path(folder)
        guard_cfg = config.guard.model_copy(update={"allowed_dirs": [sandbox]})
        skills_cfg = config.skills.model_copy(update={"auto_learn": False})  # banc figé
        sandboxed = config.model_copy(update={"guard": guard_cfg, "skills": skills_cfg})
        trashed: list[str] = []

        async def no_wait(seconds: float) -> None:
            return None

        async def silent(text: str) -> None:
            return None

        desktop = DryRunDesktop(actions)
        ctx = ToolContext(
            sandboxed,
            desktop=desktop,
            audio=DryRunAudio(actions),
            http=create_http_client(),
            search_api_key=search_api_key,
            memory=LongTermMemory(sandbox / "memory.json"),
            announce=silent,
            sleep=no_wait,
            guard=Guard(sandboxed, home=sandbox, trash=trashed.append),
            input=DryRunInput(actions),
            shell=DryRunShell(actions),
            uia=DryRunUi(actions),
            browser=DryRunBrowser(actions),
            index=DryRunIndex(actions),
            skills=(
                SkillLibrary(
                    sandbox / "skills.json",
                    sandboxed.skills.max_skills,
                    sandboxed.skills.max_failures,
                )
                if sandboxed.skills.enabled
                else None
            ),
        )

        async def running() -> list[str]:
            return open_apps(SCENARIO_WINDOWS)

        registry = ToolRegistry(ctx)
        if ctx.skills is not None:
            # Deux procédures déjà apprises (comme après quelques jours d'usage).
            ctx.skills.learn(
                "pc_silencieux",
                "mettre le PC en silencieux",
                "passe le pc en silencieux",
                [
                    {"tool": "set_volume", "args": {"level": 20}},
                    {"tool": "mute", "args": {"mute": True}},
                ],
            )
            ctx.skills.learn(
                "youtube_recent_video",
                "chercher et lancer une vidéo récente sur YouTube",
                "lance le dernier opening de black clover",
                [
                    {
                        "tool": "browser_open",
                        "args": {"url": "https://youtube.com/results?search_query={query}"},
                    }
                ],
                params=["query"],
            )
        agent = Agent(
            llm,
            config,
            ShortTermMemory(),  # en mémoire : ne touche pas à data/history.json
            window_title=lambda: SCENARIO_WINDOWS[0].title,
            tools=registry,
            long_memory=ctx.memory,
            skills=ctx.skills,
            open_apps=running,
            installed_apps=lambda: installed_app_names(desktop.cached_apps()),
        )
        try:
            await agent.prewarm()
            for number, case in cases:
                agent.memory.clear()  # chaque phrase comme une nouvelle conversation
                actions.clear()
                speech = RecordingSpeech()
                start = time.perf_counter()
                result = await agent.respond(case.phrase, speech, label=f"éval {number}")
                elapsed = (time.perf_counter() - start) * 1000
                calls = [(call.name, dict(call.arguments)) for call in result.tool_calls]
                answer = " ".join(speech.said).strip()
                reasons = evaluate(case, calls, answer)
                outcomes.append(
                    Outcome(
                        number=number,
                        phrase=case.phrase,
                        expect=case.expect,
                        ok=not reasons,
                        reasons=reasons,
                        calls=[f"{name}({_args(args)})" for name, args in calls],
                        answer=answer,
                        llm_calls=result.llm_calls,
                        ms=elapsed,
                        cost_usd=cost_usd(result.usage, config.llm) if result.usage else 0.0,
                    )
                )
        finally:
            await agent.aclose()
    return outcomes


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def report(outcomes: Sequence[Outcome]) -> str:
    lines = []
    for o in outcomes:
        mark = "OK " if o.ok else "ÉCHEC"
        lines.append(f"{o.number:>2}. {mark} « {o.phrase} »")
        lines.append(f"      appels : {'; '.join(o.calls) or 'aucun'}")
        lines.append(f"      dit : « {o.answer} » ({o.llm_calls} appel(s) LLM, {o.ms:.0f} ms)")
        if o.reasons:
            lines.append(f"      problème : {' ; '.join(o.reasons)}")
    passed = sum(o.ok for o in outcomes)
    single = sum(o.llm_calls == 1 for o in outcomes)
    cost = sum(o.cost_usd for o in outcomes)
    lines.append("")
    lines.append(
        f"Réussies : {passed}/{len(outcomes)} ; en un seul appel LLM : {single} ; "
        f"coût ≈ {cost:.4f} $."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", type=int, nargs="+", help="numéros des phrases à tester")
    parser.add_argument("--json", type=Path, help="écrit le détail dans ce fichier")
    args = parser.parse_args(argv)

    from loguru import logger

    from mikata.config import ConfigError, load_config
    from mikata.llm.client import OpenAICompatClient, check_llm_config

    logger.remove()  # seulement les avertissements : le rapport reste lisible
    logger.add(sys.stderr, level="WARNING")

    try:
        config = load_config()
        api_key = check_llm_config(config.llm, config.secrets)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    selected = [
        (number, case)
        for number, case in enumerate(CASES, 1)
        if not args.only or number in args.only
    ]
    key = config.secrets.search_api_key
    llm = OpenAICompatClient(config.llm, api_key)
    outcomes = asyncio.run(
        run_cases(llm, config, selected, key.get_secret_value() if key else None)
    )
    print(report(outcomes))
    if args.json:
        args.json.write_text(
            json.dumps([asdict(o) for o in outcomes], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
