"""Mémoire de conversation (§6).

Court terme (phase 3) : historique en RAM, remis à zéro après `session_reset_min` sans
interaction ou sur « nouvelle conversation ». Plafond ~`history_max_tokens` : les échanges
les plus anciens sautent en premier. Phase 4 : un échange garde aussi ses appels d'outils
(« ouvre Spotify » puis « ferme-le » doit marcher).

Long terme (ajout phase 4, outils `remember` / `forget`) : `data/memory.json`, au plus
30 faits, injectés dans le prompt système.

Plusieurs commandes peuvent tourner en parallèle : chacune lit un instantané de l'historique
au départ et y ajoute son échange à la fin.

Ajouts phase 6 :
- `agent.history_persist` (activé par défaut) : l'historique est aussi écrit dans
  `data/history.json` à chaque échange et relu au démarrage ; il est alors gardé
  `agent.history_persist_h` heures sans interaction (7 jours) au lieu de
  `session_reset_min` minutes. Durée modifiable à la voix (outil `set_history_duration`).
- `agent.history_summarize` : au-delà du plafond de tokens, les échanges les plus anciens
  sont résumés par le LLM au lieu d'être supprimés.
- « nouvelle conversation » vide tout : échanges, résumé (fichier compris).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

CHARS_PER_TOKEN = 3.5  # estimation pour du français, sans tokeniseur
MAX_FACTS = 30
FACT_MAX_CHARS = 300
_RESET_RE = re.compile(r"^\W*nouvelle\s+conversation\W*$", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 4  # + surcoût du message


def is_reset_command(text: str) -> bool:
    return _RESET_RE.match(text) is not None


@dataclass
class _Exchange:
    user: str
    assistant: str
    steps: list[dict[str, Any]] = field(default_factory=list)  # appels d'outils + résultats

    @property
    def tokens(self) -> int:
        steps = sum(estimate_tokens(json.dumps(step, ensure_ascii=False)) for step in self.steps)
        return estimate_tokens(self.user) + estimate_tokens(self.assistant) + steps


SUMMARY_MAX_CHARS = 1500  # ~400 tokens


def _exchange_from(item: dict[str, Any]) -> _Exchange:
    return _Exchange(str(item["user"]), str(item["assistant"]), list(item.get("steps", [])))


def _exchange_to(exchange: _Exchange) -> dict[str, Any]:
    return {"user": exchange.user, "assistant": exchange.assistant, "steps": exchange.steps}


class ShortTermMemory:
    def __init__(
        self,
        max_tokens: int = 6000,
        reset_after_s: float = 600.0,
        clock: Callable[[], float] = time.time,
        path: Path | None = None,
        summarize: bool = False,
    ) -> None:
        """`path` (ajout phase 6) : fichier où l'historique est gardé entre deux démarrages ;
        None = en mémoire seulement. `clock` : heure murale (comparée à celle du fichier).

        `summarize` (ajout phase 6) : au-delà de `max_tokens`, les échanges les plus anciens
        ne sont pas supprimés mais mis de côté (`pending`) pour être résumés par le LLM
        (boucle agent) ; le résumé (`summary`) ouvre l'historique envoyé.
        """
        self.max_tokens = max_tokens
        self.reset_after_s = reset_after_s
        self.path = path
        self.summarize = summarize
        self.summary = ""
        self.keep_override_s: float | None = None  # durée choisie à la voix
        self._clock = clock
        self._exchanges: list[_Exchange] = []
        self._pending: list[_Exchange] = []
        self._generation = 0  # change à chaque remise à zéro (résumé en cours périmé)
        self._last_activity = clock()
        if path is not None:
            self._load(path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            last = float(data["last_activity"])
            exchanges = [_exchange_from(item) for item in data["exchanges"]]
            pending = [_exchange_from(item) for item in data.get("pending", [])]
            summary = str(data.get("summary", ""))[:SUMMARY_MAX_CHARS]
            keep = data.get("keep_s")
            keep_s = float(keep) if keep is not None else None
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Historique illisible ({}) : {}", path, exc)
            return
        if keep_s is not None and keep_s > 0:
            self.keep_override_s = self.reset_after_s = keep_s
        if self._clock() - last > self.reset_after_s:
            logger.info("Historique de conversation trop ancien : nouvelle conversation.")
            return
        self._exchanges, self._pending, self.summary = exchanges, pending, summary
        self._last_activity = last
        logger.info(
            "Historique de conversation repris : {} échange(s){}.",
            len(exchanges),
            " et un résumé des plus anciens" if summary else "",
        )

    def _save(self) -> None:
        if self.path is None:
            return
        payload = {
            "last_activity": self._last_activity,
            "keep_s": self.keep_override_s,
            "summary": self.summary,
            "pending": [_exchange_to(e) for e in self._pending],
            "exchanges": [_exchange_to(e) for e in self._exchanges],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".part")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)  # écriture atomique ; aucun fichier supprimé
        except OSError as exc:
            logger.warning("Historique non enregistré ({}) : {}", self.path, exc)

    def _reset(self) -> None:
        self._exchanges.clear()
        self._pending.clear()
        self.summary = ""
        self._generation += 1

    def _expire(self) -> None:
        if self._clock() - self._last_activity > self.reset_after_s and (
            self._exchanges or self._pending or self.summary
        ):
            self._reset()

    def messages(self) -> list[dict[str, Any]]:
        """Historique à envoyer au LLM (instantané) : résumé des anciens échanges, puis les
        échanges récents en entier."""
        self._expire()
        out: list[dict[str, Any]] = []
        if self.summary:
            out.append(
                {"role": "system", "content": f"Résumé de la conversation plus ancienne : "
                 f"{self.summary}"}
            )  # fmt: skip
        for exchange in self._exchanges:
            out.append({"role": "user", "content": exchange.user})
            out.extend(exchange.steps)
            out.append({"role": "assistant", "content": exchange.assistant})
        return out

    def last_tools(self) -> list[str]:
        """Phase 16g : les outils appelés par le dernier échange (dans l'ordre)."""
        self._expire()
        if not self._exchanges:
            return []
        return [
            str(call.get("function", {}).get("name", ""))
            for step in self._exchanges[-1].steps
            if step.get("role") == "assistant"
            for call in step.get("tool_calls") or []
        ]

    def add(self, user: str, assistant: str, steps: Sequence[dict[str, Any]] = ()) -> None:
        """Ajoute un échange ; `steps` : messages d'appels d'outils et de résultats."""
        self._expire()
        self._exchanges.append(_Exchange(user, assistant, list(steps)))
        self._last_activity = self._clock()
        while len(self._exchanges) > 1 and self.tokens > self.max_tokens:
            oldest = self._exchanges.pop(0)
            if self.summarize:
                self._pending.append(oldest)
        # Résumés en échec (hors ligne) : on ne garde pas indéfiniment l'arriéré.
        while self._pending and sum(e.tokens for e in self._pending) > self.max_tokens:
            dropped = self._pending.pop(0)
            logger.warning("Échange trop ancien oublié sans résumé : « {} »", dropped.user)
        self._save()

    @property
    def pending(self) -> int:
        """Échanges anciens en attente de résumé."""
        return len(self._pending)

    def summary_request(self) -> tuple[int, str, list[tuple[str, str]]]:
        """(génération, résumé actuel, échanges à y ajouter (utilisateur, MIKATA))."""
        exchanges = [(e.user, e.assistant) for e in self._pending]
        return self._generation, self.summary, exchanges

    def apply_summary(self, generation: int, count: int, summary: str) -> bool:
        """Remplace le résumé et retire les `count` premiers échanges résumés. Ignoré (False)
        si la conversation a été remise à zéro entre-temps."""
        summary = " ".join(summary.split())[:SUMMARY_MAX_CHARS]
        if generation != self._generation or not summary:
            return False
        self.summary = summary
        del self._pending[:count]
        self._save()
        return True

    def set_keep(self, seconds: float) -> None:
        """Durée de conservation sans interaction, choisie à la voix (gardée dans le fichier,
        prioritaire sur la config)."""
        self.keep_override_s = self.reset_after_s = seconds
        self._save()

    def touch(self) -> None:
        self._expire()
        self._last_activity = self._clock()

    def clear(self) -> None:
        """« Nouvelle conversation » : échanges, résumé et arriéré vidés (pas la durée)."""
        self._reset()
        self._last_activity = self._clock()
        self._save()

    @property
    def tokens(self) -> int:
        summary = estimate_tokens(self.summary) if self.summary else 0
        return summary + sum(exchange.tokens for exchange in self._exchanges)

    def __len__(self) -> int:
        return len(self._exchanges)


def _fold(text: str) -> str:
    """Forme de comparaison : minuscules, sans accents ni ponctuation."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", text).strip()


class MemoryFullError(ValueError):
    pass


class LongTermMemory:
    """Faits durables sur l'utilisateur (`remember` / `forget`), dans un fichier JSON.

    Le fichier est réécrit en entier à chaque changement (écriture atomique) ; oublier un
    fait le retire de la liste, aucun fichier n'est supprimé.
    """

    def __init__(
        self,
        path: Path,
        max_facts: int = MAX_FACTS,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.path = path
        self.max_facts = max_facts
        self._clock = clock
        self._facts: list[dict[str, str]] = self._load()

    def _load(self) -> list[dict[str, str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            facts = [
                {"text": str(item["text"]), "added": str(item.get("added", ""))}
                for item in data.get("facts", [])
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning("Mémoire illisible ({}) : {}", self.path, exc)
            return []
        return facts[: self.max_facts]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        payload = {"facts": self._facts}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def facts(self) -> list[str]:
        return [fact["text"] for fact in self._facts]

    def remember(self, fact: str) -> bool:
        """Mémorise `fact` ; False s'il était déjà connu. `MemoryFullError` si plein."""
        fact = " ".join(fact.split())[:FACT_MAX_CHARS]
        if not fact:
            raise ValueError("fait vide")
        if any(_fold(known["text"]) == _fold(fact) for known in self._facts):
            return False
        if len(self._facts) >= self.max_facts:
            raise MemoryFullError(f"mémoire pleine ({self.max_facts} faits)")
        self._facts.append({"text": fact, "added": self._clock().isoformat(timespec="seconds")})
        self._save()
        return True

    def find(self, query: str, cutoff: float = 70) -> str | None:
        """Fait le plus proche de `query` (correspondance floue), sinon None."""
        from rapidfuzz import fuzz, process

        if not self._facts:
            return None
        choices = [_fold(fact["text"]) for fact in self._facts]
        best = process.extractOne(_fold(query), choices, scorer=fuzz.WRatio, score_cutoff=cutoff)
        return self._facts[best[2]]["text"] if best is not None else None

    def forget(self, query: str) -> str | None:
        """Retire le fait le plus proche de `query` ; renvoie son texte, ou None."""
        text = self.find(query)
        if text is None:
            return None
        self._facts = [fact for fact in self._facts if fact["text"] != text]
        self._save()
        return text
