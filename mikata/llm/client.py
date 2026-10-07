"""Client LLM compatible OpenAI, en streaming (§6).

- Un seul `openai.AsyncOpenAI` sur un `httpx2.AsyncClient` persistant (HTTP/2, keep-alive).
- `prewarm()` : requête légère envoyée dès la détection du mot d'activation, pendant que
  l'utilisateur parle encore, pour que la poignée de main TLS soit faite quand la commande
  part (§2, technique 2).
- `stream()` : traduit le flux SSE en `TextDelta` / `ToolCallReq` / `Done` (llm/base.py).
  Les appels d'outils arrivent en morceaux : ils sont réassemblés par index.
Les nouvelles tentatives sont gérées par la boucle agent (un retry immédiat, §6), pas ici.

Ajout phase 6 (avertissement « pas de réponse en 20 s » vu en test) :
- `llm.first_chunk_timeout_s` : si le serveur n'envoie **rien** (pas même un morceau de
  réflexion) dans ce délai, l'appel échoue tout de suite au lieu d'attendre
  `request_timeout_s`, et la boucle agent relance (réflexion none / low seulement : en
  medium / high, le modèle peut réfléchir longtemps en silence) ;
- après un délai dépassé ou une erreur de connexion, la connexion est **recréée** avant la
  relance : une connexion HTTP/2 restée ouverte mais morte (coupée côté serveur ou box
  pendant l'inactivité) ferait échouer la relance de la même façon.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import SecretStr

from mikata.config import ConfigError, LlmConfig, Secrets
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq, Usage

PREWARM_TIMEOUT_S = 5.0
PREWARM_MIN_INTERVAL_S = 10.0  # inutile de réveiller une connexion encore chaude
KEEPALIVE_S = 120.0
MAX_RETIRED = 3  # anciennes connexions gardées le temps que leurs requêtes finissent
FAST_EFFORTS = ("none", "low")  # niveaux de réflexion où le premier morceau arrive vite
USE_CONFIG = "config"  # stream() : niveau de réflexion de llm.reasoning_effort


class LlmNotConfiguredError(ConfigError):
    """Clé API absente ou endpoint non renseigné : MIKATA refuse de démarrer."""


class LlmError(Exception):
    """Échec d'un appel LLM (réseau, délai, refus du serveur), message en français."""


def check_llm_config(cfg: LlmConfig, secrets: Secrets) -> SecretStr:
    """Clé API utilisable, sinon `LlmNotConfiguredError` avec la marche à suivre."""
    problems = []
    if secrets.llm_api_key is None:
        problems.append("LLM_API_KEY absente : ajoute-la dans le fichier .env (voir .env.example).")
    if cfg.is_placeholder:
        problems.append(
            "llm.base_url / llm.model non renseignés dans config.yaml (endpoint et nom du "
            "modèle « flash » de CommandCode)."
        )
    if problems:
        raise LlmNotConfiguredError(
            "LLM non configuré, démarrage impossible :\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nPour tester sans LLM (phases 1 et 2) : python -m mikata --no-llm"
        )
    assert secrets.llm_api_key is not None
    return secrets.llm_api_key


@dataclass
class _PendingCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)

    def build(self) -> ToolCallReq:
        raw = "".join(self.arguments).strip()
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.warning("Arguments d'outil illisibles pour {} : {!r}", self.name, raw)
            args = {}
        return ToolCallReq(
            id=self.id, name=self.name, arguments=args if isinstance(args, dict) else {}
        )


def _reasoning_text(delta: Any) -> str:
    """Texte de réflexion d'un morceau : `reasoning_content` (DeepSeek) ou `reasoning`."""
    extra = getattr(delta, "model_extra", None) or {}
    value = extra.get("reasoning_content") or extra.get("reasoning")
    return value if isinstance(value, str) else ""


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def _usage(raw: Any) -> Usage:
    """Décompte OpenAI (`prompt_tokens_details.cached_tokens`) ou DeepSeek natif
    (`prompt_cache_hit_tokens`)."""
    cached = _field(_field(raw, "prompt_tokens_details"), "cached_tokens")
    if cached is None:
        cached = _field(raw, "prompt_cache_hit_tokens")
    reasoning = _field(_field(raw, "completion_tokens_details"), "reasoning_tokens")
    return Usage(
        prompt=int(_field(raw, "prompt_tokens") or 0),
        completion=int(_field(raw, "completion_tokens") or 0),
        cached=int(cached or 0),
        reasoning=int(reasoning or 0),
    )


class OpenAICompatClient:
    """Implémente l'interface `LLMClient`."""

    def __init__(self, cfg: LlmConfig, api_key: SecretStr, transport: Any = None) -> None:
        self.cfg = cfg
        self._base_url = cfg.base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        self._retired: list[Any] = []  # (client openai, client httpx) remplacés
        self._http, self._client = self._connect()
        self._last_prewarm = float("-inf")
        self.renewals = 0

    def _connect(self) -> tuple[Any, Any]:
        import httpx2
        from openai import AsyncOpenAI

        http = httpx2.AsyncClient(
            http2=True,
            timeout=httpx2.Timeout(self.cfg.request_timeout_s, connect=10.0),
            limits=httpx2.Limits(keepalive_expiry=KEEPALIVE_S),
            transport=self._transport,
        )
        client = AsyncOpenAI(
            api_key=self._api_key.get_secret_value(),
            base_url=self._base_url,
            http_client=http,
            max_retries=0,  # le retry immédiat est décidé par la boucle agent
            timeout=self.cfg.request_timeout_s,
        )
        return http, client

    async def _renew(self, reason: str) -> None:
        """Nouvelle connexion pour les appels suivants. L'ancienne n'est pas fermée tout de
        suite : une autre commande peut encore s'en servir."""
        self._retired.append((self._client, self._http))
        self._http, self._client = self._connect()
        self._last_prewarm = float("-inf")
        self.renewals += 1
        logger.info("Connexion LLM recréée ({}).", reason)
        while len(self._retired) > MAX_RETIRED:
            await self._close_pair(self._retired.pop(0))

    @staticmethod
    async def _close_pair(pair: tuple[Any, Any]) -> None:
        client, http = pair
        with contextlib.suppress(Exception):
            await client.close()
            await http.aclose()

    async def prewarm(self) -> None:
        """Ouvre (ou garde ouverte) la connexion TLS. N'échoue jamais."""
        now = time.monotonic()
        if now - self._last_prewarm < PREWARM_MIN_INTERVAL_S:
            return
        self._last_prewarm = now
        try:
            response = await self._http.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                timeout=PREWARM_TIMEOUT_S,
            )
            logger.debug(
                "Connexion LLM préchauffée ({} ms, HTTP {}).",
                round((time.monotonic() - now) * 1000),
                response.status_code,
            )
        except Exception as exc:  # préchauffage seulement : l'appel réel dira la vraie erreur
            logger.debug("Préchauffage LLM impossible : {}", exc)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = USE_CONFIG,
    ) -> AsyncIterator[LLMEvent]:
        """`reasoning_effort` (ajout phase 4) : niveau de réflexion de cet appel ; par défaut
        celui de la config ; None = paramètre non envoyé."""
        import openai

        kwargs: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        if self.cfg.include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        effort = self.cfg.reasoning_effort if reasoning_effort == USE_CONFIG else reasoning_effort
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        calls: dict[int, _PendingCall] = {}
        finish_reason = "stop"
        usage: Usage | None = None
        reasoning: list[str] = []
        start = time.monotonic()
        reasoning_end: float | None = None
        # Réflexion medium / high : le modèle peut penser longtemps avant d'envoyer quoi que
        # ce soit, seul request_timeout_s s'applique.
        quick = effort in FAST_EFFORTS
        first_s = self.cfg.first_chunk_timeout_s if quick else self.cfg.request_timeout_s
        waiting = True  # aucun morceau reçu : délai `first_s`
        try:
            async with asyncio.timeout(first_s) as first_chunk:
                response = await self._client.chat.completions.create(**kwargs)
                async for chunk in response:
                    if waiting:  # le serveur répond : plus que le délai entre morceaux
                        waiting = False
                        first_chunk.reschedule(None)
                    text = _absorb(chunk, calls, reasoning)
                    if reasoning and reasoning_end is None and _answer_started(chunk):
                        reasoning_end = time.monotonic()  # fin de la réflexion : premier mot
                    if text:
                        yield TextDelta(text)
                    if chunk.usage is not None:
                        usage = _usage(chunk.usage)
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
        except TimeoutError as exc:
            await self._renew("aucune réponse")
            raise LlmError(f"pas de réponse en {first_s:.0f} s") from exc
        except openai.AuthenticationError as exc:
            raise LlmError("clé API refusée par le serveur (LLM_API_KEY)") from exc
        except openai.APITimeoutError as exc:
            await self._renew("délai dépassé")
            raise LlmError(f"pas de réponse en {self.cfg.request_timeout_s:.0f} s") from exc
        except openai.APIConnectionError as exc:
            await self._renew("connexion en échec")
            raise LlmError(f"serveur injoignable ({exc})") from exc
        except openai.APIStatusError as exc:
            detail = str(getattr(exc, "message", "") or "")[:200]
            raise LlmError(f"erreur HTTP {exc.status_code} du serveur LLM : {detail}") from exc
        for index in sorted(calls):
            yield calls[index].build()
        if reasoning and reasoning_end is None:
            reasoning_end = time.monotonic()
        yield Done(
            finish_reason,
            usage=usage,
            reasoning_s=reasoning_end - start if reasoning_end is not None else None,
            reasoning="".join(reasoning),
        )

    async def aclose(self) -> None:
        for pair in [*self._retired, (self._client, self._http)]:
            await self._close_pair(pair)
        self._retired.clear()


def _absorb(chunk: Any, calls: dict[int, _PendingCall], reasoning: list[str]) -> str:
    """Texte d'un morceau ; la réflexion et les morceaux d'appels d'outils sont mis de côté."""
    if not chunk.choices or chunk.choices[0].delta is None:
        return ""
    delta = chunk.choices[0].delta
    if thought := _reasoning_text(delta):
        reasoning.append(thought)
    for call in delta.tool_calls or []:
        pending = calls.setdefault(call.index, _PendingCall())
        pending.id = call.id or pending.id
        if call.function is not None:
            pending.name = call.function.name or pending.name
            if call.function.arguments:
                pending.arguments.append(call.function.arguments)
    return str(delta.content or "")


def _answer_started(chunk: Any) -> bool:
    """Le morceau contient du texte ou un appel d'outil (et pas que de la réflexion)."""
    if not chunk.choices or chunk.choices[0].delta is None:
        return False
    delta = chunk.choices[0].delta
    return bool(delta.content or delta.tool_calls) and not _reasoning_text(delta)
