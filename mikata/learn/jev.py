"""Premier étage de l'aiguillage (phase 15k) : Jev, le modèle de décision de TypeSafe.

Jev ne génère pas de texte : on lui donne la phrase et des questions à choix, il rend
pour chaque question une distribution de probabilités sur les options. Une seule requête
pose deux sortes de questions, traitées en parallèle par Jev :

- « competence » : laquelle des compétences actives la phrase demande, ou « aucune » ;
- une question par paramètre : lequel des morceaux de la phrase (1 à 5 mots, recopiés tels
  quels) en est la valeur. Jev ne sait pas extraire un mot ; il choisit donc parmi les
  morceaux de la phrase, et la valeur ne peut pas être inventée.

Décision : compétence si sa probabilité atteint `threshold` (0,9) ; « aucune » si celle-ci
l'atteint ; sinon « incertain », et le petit appel LLM de la phase 15j décide. Mesuré le
14/10 (16 phrases, 4 langues, 3 passages) : les cas nets à p >= 0,96, les cas douteux
entre 0,63 et 0,77, médiane 375 à 391 ms, environ 0,00004 $ par phrase.

Appelé par OpenRouter (`OPENROUTER_API_KEY` dans .env) ; sans clé, l'aiguillage reste
celui de la phase 15j.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from mikata.learn.skills import Skill

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
NONE_LABEL = "aucune"
MAX_SPAN_WORDS = 5
MAX_SPANS = 120
NONE_CRITERION = (
    "Toute autre demande : une autre action (fermer, couper, mettre en pause, chercher une "
    "information sans rien lancer), plusieurs actions à la fois, une négation, ou une demande "
    "sans rapport avec ces compétences."
)
_WORD = re.compile(r"[\w'’.-]+")
_TRIM = ".,;:!?'’\"«»-"


def spans(sentence: str) -> list[str]:
    """Les morceaux de 1 à MAX_SPAN_WORDS mots de la phrase, recopiés tels quels
    (« Dr. Stone » garde son point), sans doublon."""
    words = [m for m in _WORD.finditer(sentence) if m.group().strip(_TRIM)]
    found: list[str] = []
    for size in range(1, MAX_SPAN_WORDS + 1):
        for start in range(len(words) - size + 1):
            first, last = words[start], words[start + size - 1]
            span = sentence[first.start() : last.end()].strip(_TRIM)
            if span and span not in found:
                found.append(span)
                if len(found) >= MAX_SPANS:
                    return found
    return found


def _skill_criterion(skill: Skill) -> str:
    examples = " ; ".join(f"« {p} »" for p in skill.patterns[:3])
    return f"{skill.description} Exemples, dans n'importe quelle langue : {examples}."


def _param_key(skill: Skill, param: str) -> str:
    return f"{skill.call_name}__{param}"


def build_request(sentence: str, skills: list[Skill]) -> tuple[dict[str, Any], list[str]]:
    """Corps de la requête et morceaux candidats (dans l'ordre des options m0, m1...)."""
    candidates = spans(sentence)
    criteria = {skill.call_name: _skill_criterion(skill) for skill in skills}
    criteria[NONE_LABEL] = NONE_CRITERION
    questions: dict[str, Any] = {
        "competence": {
            "type": "choice",
            "instructions": (
                "Un assistant vocal a reçu cette demande. Laquelle de ses compétences "
                "demande-t-elle exactement ?"
            ),
            "criteria": criteria,
        }
    }
    options = {f"m{i}": f"« {span} »" for i, span in enumerate(candidates)}
    for skill in skills:
        for param in skill.params:
            questions[_param_key(skill, param)] = {
                "type": "choice",
                "instructions": (
                    f"Quel morceau de la demande est exactement la valeur « "
                    f"{param.replace('_', ' ')} » (pour : {skill.description}), sans mot en "
                    "plus ni en moins ?"
                ),
                "criteria": options,
            }
    return {"model": JEV_MODEL, "state": {"demande": sentence}, "questions": questions}, (
        candidates
    )


@dataclass
class JevDecision:
    kind: str  # « compétence », « aucune » ou « incertain »
    skill: Skill | None = None
    values: dict[str, str] = field(default_factory=dict)
    probability: float = 0.0
    cost_usd: float = 0.0


def read_answers(
    data: dict[str, Any], skills: list[Skill], candidates: list[str], threshold: float
) -> JevDecision:
    """Décision tirée de la réponse de Jev ; « incertain » au moindre doute."""
    answers = data.get("answers") or {}
    cost = float((data.get("usage") or {}).get("cost") or 0)
    probabilities = (answers.get("competence") or {}).get("probabilities") or {}
    if not probabilities:
        return JevDecision("incertain", cost_usd=cost)
    label, best = max(probabilities.items(), key=lambda item: float(item[1]))
    best = float(best)
    if best < threshold:
        return JevDecision("incertain", probability=best, cost_usd=cost)
    if label == NONE_LABEL:
        return JevDecision("aucune", probability=best, cost_usd=cost)
    skill = next((s for s in skills if s.call_name == label), None)
    if skill is None:
        return JevDecision("incertain", probability=best, cost_usd=cost)
    values: dict[str, str] = {}
    for param in skill.params:
        choice = str((answers.get(_param_key(skill, param)) or {}).get("choice") or "")
        index = int(choice[1:]) if re.fullmatch(r"m\d+", choice) else -1
        if not 0 <= index < len(candidates):
            return JevDecision("incertain", probability=best, cost_usd=cost)
        values[param] = candidates[index]
    return JevDecision("compétence", skill, values, best, cost)


class JevRouter:
    """Client de Jev (OpenRouter). Une connexion gardée ouverte, préchauffée au mot
    d'activation comme celle du LLM."""

    def __init__(
        self,
        api_key: str,
        *,
        threshold: float = 0.9,
        timeout_s: float = 1.5,
        transport: Any = None,
    ) -> None:
        import httpx2

        self.threshold = threshold
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._http = httpx2.AsyncClient(timeout=timeout_s, transport=transport)

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key

    async def prewarm(self) -> None:
        """Ouvre la connexion TLS d'avance (la réponse elle-même n'importe pas)."""
        try:
            await self._http.head(JEV_URL)
        except Exception as exc:
            logger.debug("Préchauffage de Jev : {}", exc)

    async def decide(self, sentence: str, skills: list[Skill]) -> JevDecision:
        body, candidates = build_request(sentence, skills)
        start = time.monotonic()
        try:
            response = await self._http.post(
                JEV_URL, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
            response.raise_for_status()
            decision = read_answers(response.json(), skills, candidates, self.threshold)
        except Exception as exc:  # hors ligne, clé refusée, réponse illisible
            logger.info("Jev sans réponse utilisable : {}", str(exc) or type(exc).__name__)
            return JevDecision("incertain")
        logger.info(
            "Jev : {}{} (p = {:.2f}, {:.0f} ms).",
            decision.kind,
            f" « {decision.skill.name} » {decision.values}" if decision.skill else "",
            decision.probability,
            (time.monotonic() - start) * 1000,
        )
        return decision

    async def aclose(self) -> None:
        await self._http.aclose()
