"""Appel au modèle, en flux : le texte part à la voix fragment par fragment, les appels
d'outils sont rassemblés. Un retry immédiat sur une erreur réseau avant le premier mot
(docs/ARCHITECTURE.md §6) ; rôle « fort » si configuré (`llm.strong_model`)."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any

from loguru import logger

from mikata.agent.turn import FollowUpGate, Reply, SpeechSink, TurnResult
from mikata.config import Config
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq
from mikata.tts.normalize import FragmentSplitter

MAX_ATTEMPTS = 2  # un retry immédiat (§6)


def with_reasoning(messages: list[dict[str, Any]], effort: str | None) -> list[dict[str, Any]]:
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


# Phase 17b : le modèle annonce parfois ce qu'il va faire en anglais (« I'll open the shop
# and add the headset to the cart. »), malgré la consigne ; MIKATA l'aurait dit tel quel.
# Seule cette tournure d'annonce est écartée : une traduction demandée reste dite.
ENGLISH_NARRATION = re.compile(
    r"\s*(?:I['’]ll|I will|Let me|I['’]m going to|I am going to|(?:Now|First),? I['’]ll)\b"
    r"[^.!?]*[.!?]?\s*",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")

# Phase 17f : une réponse qui commence par dire qu'une action est faite. Le 15/10, « Quitte
# le salon vocal ? » a reçu « C'est fait. » sans aucun outil appelé : le modèle imitait les
# réponses d'avant. Sans action dans le tour, une telle réponse n'est pas dite (texte replié :
# minuscules, sans accents).
CLAIM = re.compile(
    r"\s*(?:(?:voila|ok|d'?accord|parfait)\W+)?(?:"
    r"c'?est (?:fait|bon|lance|lancee|coupe|coupee|ferme|fermee|ouvert|ouverte|regle|reglee"
    r"|parti|eteint|allume|active|activee|desactive|desactivee|envoye|envoyee|note|termine)"
    r"|j'?ai (?:bien )?(?:coupe|lance|ferme|ouvert|quitte|rejoint|envoye|active|desactive|mis"
    r"|regle|baisse|monte|deconnecte|connecte|supprime|deplace|copie|cree|ecrit|tape|clique"
    r"|retire|ajoute|coupe)"
    r"|tu es (?:bien |maintenant |desormais )?(?:de)?connecte"
    r"|tu n'?es plus (?:dans|connecte))\b"
)


def claims_done(text: str) -> bool:
    """Le texte commence-t-il par annoncer une action faite (« C'est fait », « J'ai
    quitté… ») ? « Oui, c'est fait » (une réponse sur le passé) ne compte pas."""
    folded = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    return CLAIM.match("".join(c for c in folded if not unicodedata.combining(c))) is not None


def drop_english_narration(text: str) -> str:
    """Le texte sans ses phrases d'annonce en anglais (« I'll find... »)."""
    sentences = _SENTENCES.split(text)
    kept = [s for s in sentences if not ENGLISH_NARRATION.fullmatch(s)]
    return " ".join(kept).strip() if len(kept) != len(sentences) else text


class ModelCaller:
    def __init__(self, llm: LLMClient, config: Callable[[], Config]) -> None:
        self.llm = llm
        self._config = config

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        speech: SpeechSink,
        result: TurnResult,
        gate: FollowUpGate | None = None,
        strong: bool = False,
    ) -> Reply | None:
        """Un appel (avec son retry) ; None si le serveur reste injoignable. Phase 17f : une
        réponse qui commence par « c'est fait » alors que rien n'a été fait dans le tour est
        retenue (`Reply.unbacked`), jamais dite."""
        config = self._config()
        gate = gate or FollowUpGate(active=False)
        acted = bool(result.tool_calls or result.scripts or result.skill_uses)
        held: list[str] = []
        spoke = False

        def say(fragments: list[str]) -> None:
            nonlocal spoke
            for fragment in fragments:
                if ENGLISH_NARRATION.match(fragment):  # tic du modèle, jamais dit
                    logger.info("Annonce en anglais non dite : « {} »", fragment.strip())
                    continue
                if not gate.allow(fragment):
                    continue
                if not acted and not spoke and (held or claims_done(fragment)):
                    held.append(fragment)  # la suite dira si une action l'appuie
                    continue
                spoke = True
                speech.say(fragment)

        splitter = FragmentSplitter(config.tts.first_chunk_min_words)
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
                request = with_reasoning(messages, effort)
                # Rôle « fort » : un autre modèle seulement s'il est configuré (sinon l'appel
                # reste identique, la réflexion relevée fait la différence).
                extra = (
                    {"model": config.llm.strong_model} if strong and config.llm.strong_model else {}
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
        _log_call(result.llm_calls, effort, start, first_at, done, calls)
        text = drop_english_narration("".join(parts))
        if held:
            # Annoncée avant d'agir (des outils suivent) : l'action conclura elle-même. Sans
            # outil : une action jamais faite, rien n'est dit.
            logger.info(
                "Réponse « {} » non dite : {}.",
                "".join(held).strip()[:80],
                "l'action suit" if calls else "rien n'a été fait dans ce tour",
            )
        return Reply(
            "" if held else text,
            calls,
            done.reasoning if done is not None else "",
            gate.ignored,
            unbacked=bool(held) and not calls,
            held=text if held else "",
        )


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
