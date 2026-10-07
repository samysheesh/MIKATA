"""Client LLM compatible OpenAI, en streaming (§6).

- Un seul `openai.AsyncOpenAI` sur un `httpx2.AsyncClient` persistant (HTTP/2, keep-alive).
- `prewarm()` : requête légère envoyée dès la détection du mot d'activation, pendant que
  l'utilisateur parle encore, pour que la poignée de main TLS soit faite quand la commande
  part (§2, technique 2).
- `stream()` : traduit le flux SSE en `TextDelta` / `ToolCallReq` / `Done` (llm/base.py).
  Les appels d'outils arrivent en morceaux : ils sont réassemblés par index.
Les nouvelles tentatives sont gérées par la boucle agent (un retry immédiat, §6), pas ici.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import SecretStr

from mikata.config import ConfigError, LlmConfig, Secrets
from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq

PREWARM_TIMEOUT_S = 5.0
PREWARM_MIN_INTERVAL_S = 10.0  # inutile de réveiller une connexion encore chaude
KEEPALIVE_S = 120.0


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


class OpenAICompatClient:
    """Implémente l'interface `LLMClient`."""

    def __init__(self, cfg: LlmConfig, api_key: SecretStr, transport: Any = None) -> None:
        import httpx2
        from openai import AsyncOpenAI

        self.cfg = cfg
        self._base_url = cfg.base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx2.AsyncClient(
            http2=True,
            timeout=httpx2.Timeout(cfg.request_timeout_s, connect=10.0),
            limits=httpx2.Limits(keepalive_expiry=KEEPALIVE_S),
            transport=transport,
        )
        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=self._base_url,
            http_client=self._http,
            max_retries=0,  # le retry immédiat est décidé par la boucle agent
            timeout=cfg.request_timeout_s,
        )
        self._last_prewarm = float("-inf")

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
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        import openai

        kwargs: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        calls: dict[int, _PendingCall] = {}
        finish_reason = "stop"
        try:
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None and delta.content:
                    yield TextDelta(delta.content)
                for call in (delta.tool_calls if delta is not None else None) or []:
                    pending = calls.setdefault(call.index, _PendingCall())
                    pending.id = call.id or pending.id
                    if call.function is not None:
                        pending.name = call.function.name or pending.name
                        if call.function.arguments:
                            pending.arguments.append(call.function.arguments)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except openai.AuthenticationError as exc:
            raise LlmError("clé API refusée par le serveur (LLM_API_KEY)") from exc
        except openai.APITimeoutError as exc:
            raise LlmError(f"pas de réponse en {self.cfg.request_timeout_s:.0f} s") from exc
        except openai.APIConnectionError as exc:
            raise LlmError(f"serveur injoignable ({exc})") from exc
        except openai.APIStatusError as exc:
            raise LlmError(f"erreur HTTP {exc.status_code} du serveur LLM") from exc
        for index in sorted(calls):
            yield calls[index].build()
        yield Done(finish_reason)

    async def aclose(self) -> None:
        await self._client.close()
        await self._http.aclose()
