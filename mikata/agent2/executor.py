"""Agent v2 (phase 13c) : exécuter un sous-but, essai après essai, sans jamais réessayer
ce que l'utilisateur a refusé.

Chaque essai reçoit un Sdk neuf (compteur d'actions remis à zéro) et un Recorder neuf ; le
code est écrit par le LLM (effort rapide) puis filtré par le bac à sable, exécuté, et le
prédicat du sous-but est vérifié. Une erreur est montrée au LLM au format `type: message`
pour qu'il corrige ; un refus du garde-fou ou une annulation arrête le sous-but aussitôt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, TypeAdapter

from mikata.agent2.llm_json import ask_json
from mikata.agent2.prompts import STEP_SYSTEM, STEP_USER
from mikata.agent2.sandbox import CodeRejected, run_code
from mikata.agent2.schema import Observation, Predicate, Resolved, Step, SubGoal
from mikata.agent2.sdk import Recorder, Sdk, SdkError, sdk_reference
from mikata.agent2.verify import check
from mikata.config import Agent2Config
from mikata.llm.base import LLMClient
from mikata.tools.base import ActionCancelled, GuardDenied

_PRED_ADAPTER = TypeAdapter(Predicate)


class _StepCode(BaseModel):
    code: str


_REFUSAL_MARKERS = ("refusé par le garde-fou", "l'utilisateur n'a pas confirmé")


def _is_refusal(error: Exception) -> bool:
    """Un refus du garde-fou ou une annulation de l'utilisateur : on n'essaie plus."""
    return isinstance(error, (GuardDenied, ActionCancelled)) or (
        isinstance(error, SdkError) and any(marker in str(error) for marker in _REFUSAL_MARKERS)
    )


def _observation_text(obs: Observation) -> str:
    lines = [
        f"App active : {obs.active_app}",
        f"Titre : {obs.title}",
        "Page du navigateur de travail :",
    ]
    lines.extend((obs.page or "").splitlines())
    lines.append("Résumé de l'écran :")
    lines.extend(obs.summary.splitlines())
    return "\n".join(lines) or "(écran illisible)"


async def run_subgoal(
    index: int,
    sub: SubGoal,
    *,
    sdk_factory: Callable[[Recorder], Sdk],
    llm: LLMClient,
    resolved: list[Resolved],
    cfg: Agent2Config,
    os_name: str,
    skills: Any = None,
) -> list[Step]:
    """Tous les essais du sous-but. Le dernier Step a `goal_reached` si c'est réussi."""
    steps: list[Step] = []
    previous: list[str] = []
    for attempt in range(1, cfg.max_attempts_per_goal + 1):
        recorder = Recorder()
        sdk = sdk_factory(recorder)
        obs = await sdk.observe()
        error: str | None = None
        code = ""
        reached = False
        if sub.done_when.type != "judge":
            # La condition est déjà vraie à l'écran ? On n'appelle pas le LLM et on ne
            # refait rien (un « refaire » pourrait casser ce qui vient d'être fait, ex.
            # rouvrir la recherche pendant que la vidéo tourne).
            already, _ = await check(sub.done_when, obs, [], llm, effort=cfg.fast_effort)
            if already:
                steps.append(_make_step(index, attempt, "", recorder, None, obs, True))
                return steps
        if sub.skill and skills is not None:
            code = f"skill:{sub.skill}"
            try:
                await skills.run(sub.skill, sdk, sub.skill_args)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            answer, _, _ = await ask_json(
                llm,
                STEP_SYSTEM.format(os=os_name, sdk_reference=sdk_reference()),
                STEP_USER.format(
                    goal=sub.goal,
                    done_when=_PRED_ADAPTER.dump_json(sub.done_when).decode(),
                    resolved="\n".join(f"- {item.name} = {item.value}" for item in resolved)
                    or "(aucune)",
                    observation=_observation_text(obs),
                    previous="\n".join(previous) or "(aucun)",
                ),
                _StepCode,
                effort=cfg.fast_effort,
            )
            if answer is None or not answer.code.strip():
                error = "pas de code"
            else:
                code = answer.code.strip()
                try:
                    await run_code(code, sdk, timeout_s=cfg.code_timeout_s)
                except CodeRejected as exc:
                    error = f"{type(exc).__name__}: {exc}"
                except SdkError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if _is_refusal(exc):
                        steps.append(
                            _make_step(index, attempt, code, recorder, error, obs, reached)
                        )
                        return steps
                except TimeoutError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
        obs2 = await sdk.observe()
        if error is None:
            reached, _ = await check(
                sub.done_when, obs2, recorder.actions, llm, effort=cfg.fast_effort
            )
        steps.append(_make_step(index, attempt, code, recorder, error, obs2, reached))
        if reached:
            return steps
        previous.append(
            f"essai {attempt} : code « {code[:200]} » ; {error or 'condition non atteinte'}"
        )
    return steps


def _make_step(
    index: int,
    attempt: int,
    code: str,
    recorder: Recorder,
    error: str | None,
    obs: Observation,
    reached: bool,
) -> Step:
    return Step(
        subgoal_index=index,
        attempt=attempt,
        code=code,
        actions=recorder.actions,
        error=error,
        observation=obs,
        goal_reached=reached,
    )
