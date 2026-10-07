"""Client LLM testé avec le vrai SDK openai, branché sur un faux serveur (httpx2.MockTransport)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from mikata.config import ConfigError, LlmConfig, Secrets
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq, Usage
from mikata.llm.client import (
    LlmError,
    LlmNotConfiguredError,
    OpenAICompatClient,
    check_llm_config,
)

CFG = LlmConfig(base_url="https://api.example.test/v1/", model="deepseek-flash")


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(chunks: list[dict[str, Any]]) -> httpx2.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx2.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class Server:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": []})
        return self.reply(request) if callable(self.reply) else self.reply


def _collect(server: Server, tools: list[dict[str, Any]] | None = None) -> list[Any]:
    async def run() -> list[Any]:
        client = OpenAICompatClient(CFG, SecretStr("sk-test"), httpx2.MockTransport(server))
        try:
            return [e async for e in client.stream([{"role": "user", "content": "x"}], tools or [])]
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_streams_text_then_done() -> None:
    server = Server(
        _sse(
            [
                _chunk({"role": "assistant", "content": "Il est "}),
                _chunk({"content": "midi."}),
                _chunk({}, "stop"),
            ]
        )
    )
    events = _collect(server)
    assert events == [TextDelta("Il est "), TextDelta("midi."), Done("stop")]
    request = server.requests[0]
    assert str(request.url) == "https://api.example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    payload = json.loads(request.content)
    assert payload["model"] == "deepseek-flash" and payload["stream"] is True
    assert "tools" not in payload  # aucun outil demandé


def test_tool_calls_are_reassembled() -> None:
    server = Server(
        _sse(
            [
                _chunk({"tool_calls": [{"index": 0, "id": "a", "type": "function",
                                        "function": {"name": "open_app", "arguments": '{"na'}}]}),
                _chunk({"tool_calls": [{"index": 1, "id": "b", "type": "function",
                                        "function": {"name": "set_volume", "arguments": "{}"}}]}),
                _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'me": "Steam"}'}}]}),
                _chunk({}, "tool_calls"),
            ]
        )
    )  # fmt: skip
    tools = [{"type": "function", "function": {"name": "open_app", "parameters": {}}}]
    events = _collect(server, tools)
    assert events == [
        ToolCallReq("a", "open_app", {"name": "Steam"}),
        ToolCallReq("b", "set_volume", {}),
        Done("tool_calls"),
    ]
    assert json.loads(server.requests[0].content)["tools"] == tools


def test_bad_tool_arguments_become_empty() -> None:
    server = Server(
        _sse([_chunk({"tool_calls": [{"index": 0, "id": "a", "type": "function",
                                      "function": {"name": "x", "arguments": "{oups"}}]})])
    )  # fmt: skip
    assert _collect(server)[0] == ToolCallReq("a", "x", {})


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "clé API refusée"), (500, "erreur HTTP 500"), (429, "erreur HTTP 429")],
)
def test_http_errors(status: int, message: str) -> None:
    server = Server(httpx2.Response(status, json={"error": {"message": "non"}}))
    with pytest.raises(LlmError, match=message):
        _collect(server)


def test_connection_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refusée", request=request)

    with pytest.raises(LlmError, match="injoignable"):
        _collect(Server(refuse))


def test_prewarm_once_and_never_fails() -> None:
    server = Server(httpx2.Response(200))

    async def run() -> None:
        client = OpenAICompatClient(CFG, SecretStr("sk-test"), httpx2.MockTransport(server))
        assert isinstance(client, LLMClient)
        await client.prewarm()
        await client.prewarm()  # connexion encore chaude : pas de 2e requête
        await client.aclose()

        def boom(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("hors ligne", request=request)

        offline = OpenAICompatClient(CFG, SecretStr("k"), httpx2.MockTransport(boom))
        await offline.prewarm()  # ne lève pas
        await offline.aclose()

    asyncio.run(run())
    assert [str(r.url) for r in server.requests] == ["https://api.example.test/v1/models"]
    assert server.requests[0].headers["authorization"] == "Bearer sk-test"


def test_check_llm_config() -> None:
    key = SecretStr("sk-1")
    assert check_llm_config(CFG, Secrets(llm_api_key=key)) is key
    with pytest.raises(LlmNotConfiguredError) as info:
        check_llm_config(LlmConfig(), Secrets())
    message = str(info.value)
    assert "LLM_API_KEY absente" in message and ".env" in message
    assert "llm.base_url" in message and "--no-llm" in message
    assert isinstance(info.value, ConfigError)
    with pytest.raises(LlmNotConfiguredError, match="LLM_API_KEY"):
        check_llm_config(CFG, Secrets())


# -- Ajouts phase 4 : niveau de réflexion, décompte des tokens, réflexion du modèle ---------


def _payload(server: Server) -> dict[str, Any]:
    return json.loads(server.requests[-1].content)


def _stream_with(server: Server, cfg: LlmConfig = CFG, **kwargs: Any) -> list[Any]:
    async def run() -> list[Any]:
        client = OpenAICompatClient(cfg, SecretStr("sk-test"), httpx2.MockTransport(server))
        try:
            messages = [{"role": "user", "content": "x"}]
            return [e async for e in client.stream(messages, [], **kwargs)]
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_reasoning_effort_and_usage_are_requested() -> None:
    server = Server(_sse([_chunk({"content": "ok"}), _chunk({}, "stop")]))
    _stream_with(server)
    payload = _payload(server)
    assert payload["reasoning_effort"] == "none"  # défaut de llm.reasoning_effort
    assert payload["stream_options"] == {"include_usage": True}
    assert "thinking" not in payload
    _stream_with(server, reasoning_effort="high")  # niveau choisi pour cette commande
    assert _payload(server)["reasoning_effort"] == "high"
    _stream_with(server, reasoning_effort=None)  # null : choix du fournisseur
    assert "reasoning_effort" not in _payload(server)
    quiet = CFG.model_copy(update={"include_usage": False, "reasoning_effort": None})
    _stream_with(server, quiet)
    assert "stream_options" not in _payload(server) and "reasoning_effort" not in _payload(server)


def _usage_chunk(usage: dict[str, Any]) -> dict[str, Any]:
    return {**_chunk({}), "choices": [], "usage": usage}


@pytest.mark.parametrize(
    "usage",
    [
        {  # format OpenAI (OpenCode Zen)
            "prompt_tokens": 1200,
            "completion_tokens": 40,
            "total_tokens": 1240,
            "prompt_tokens_details": {"cached_tokens": 1024},
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
        {  # format DeepSeek natif
            "prompt_tokens": 1200,
            "completion_tokens": 40,
            "total_tokens": 1240,
            "prompt_cache_hit_tokens": 1024,
            "prompt_cache_miss_tokens": 176,
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
    ],
)
def test_usage_is_parsed(usage: dict[str, Any]) -> None:
    server = Server(_sse([_chunk({"content": "ok"}), _chunk({}, "stop"), _usage_chunk(usage)]))
    done = _stream_with(server)[-1]
    assert done.usage == Usage(prompt=1200, completion=40, cached=1024, reasoning=25)


def test_reasoning_content_is_kept_and_timed() -> None:
    server = Server(
        _sse(
            [
                _chunk({"role": "assistant", "reasoning_content": "17 h 05 + 76 min "}),
                _chunk({"reasoning_content": "= 18 h 21."}),
                _chunk({"content": "18 h 21."}),
                _chunk({}, "stop"),
            ]
        )
    )
    events = _stream_with(server, reasoning_effort="low")
    assert events[0] == TextDelta("18 h 21.")
    done = events[-1]
    assert done.reasoning == "17 h 05 + 76 min = 18 h 21."
    assert done.reasoning_s is not None and done.reasoning_s >= 0
    plain = _stream_with(Server(_sse([_chunk({"content": "ok"}), _chunk({}, "stop")])))[-1]
    assert plain.reasoning == "" and plain.reasoning_s is None and plain.usage is None


def test_silent_server_fails_fast_on_a_new_connection() -> None:
    """Phase 6 : rien reçu en `first_chunk_timeout_s` -> échec tout de suite (la boucle agent
    relance) et nouvelle connexion pour la relance."""
    cfg = CFG.model_copy(update={"first_chunk_timeout_s": 0.05})
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(5)  # serveur muet (connexion morte)
        return _sse([_chunk({"content": "Ok."}), _chunk({}, "stop")])

    async def run() -> tuple[float, list[Any], int]:
        client = OpenAICompatClient(cfg, SecretStr("k"), httpx2.MockTransport(handler))
        first_http = client._http
        start = asyncio.get_running_loop().time()
        with pytest.raises(LlmError, match="pas de réponse en 0 s"):
            [e async for e in client.stream([{"role": "user", "content": "x"}], [])]
        elapsed = asyncio.get_running_loop().time() - start
        assert client._http is not first_http and client.renewals == 1
        events = [e async for e in client.stream([{"role": "user", "content": "x"}], [])]
        await client.aclose()
        return elapsed, events, calls

    elapsed, events, count = asyncio.run(run())
    assert elapsed < 1.0 and count == 2
    assert events == [TextDelta("Ok."), Done("stop")]


def test_slow_but_alive_stream_is_not_cut() -> None:
    """Le délai ne vaut que pour le premier morceau : une longue réponse continue."""
    cfg = CFG.model_copy(update={"first_chunk_timeout_s": 0.05})

    async def body() -> Any:
        yield f"data: {json.dumps(_chunk({'content': 'Un.'}))}\n\n".encode()
        await asyncio.sleep(0.15)  # plus long que first_chunk_timeout_s
        yield f"data: {json.dumps(_chunk({'content': ' Deux.'}, 'stop'))}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    client_events: list[Any] = []

    async def run() -> int:
        client = OpenAICompatClient(cfg, SecretStr("k"), httpx2.MockTransport(handler))
        client_events.extend(
            [e async for e in client.stream([{"role": "user", "content": "x"}], [])]
        )
        await client.aclose()
        return client.renewals

    assert asyncio.run(run()) == 0
    assert client_events == [TextDelta("Un."), TextDelta(" Deux."), Done("stop")]


def test_connection_error_renews_the_connection() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refusée", request=request)

    async def run() -> int:
        client = OpenAICompatClient(CFG, SecretStr("k"), httpx2.MockTransport(refuse))
        for _ in range(5):  # les anciennes connexions sont fermées au-delà de MAX_RETIRED
            with pytest.raises(LlmError):
                [e async for e in client.stream([{"role": "user", "content": "x"}], [])]
        assert len(client._retired) == 3
        await client.aclose()
        assert client._retired == []
        return client.renewals

    assert asyncio.run(run()) == 5


def test_long_reasoning_keeps_the_full_timeout() -> None:
    """Réflexion medium / high : pas de délai court sur le premier morceau."""
    cfg = CFG.model_copy(update={"first_chunk_timeout_s": 0.05})

    async def handler(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.15)  # le modèle réfléchit sans rien envoyer
        return _sse([_chunk({"content": "Réfléchi."}), _chunk({}, "stop")])

    async def run(effort: str) -> list[Any]:
        client = OpenAICompatClient(cfg, SecretStr("k"), httpx2.MockTransport(handler))
        try:
            return [
                e
                async for e in client.stream(
                    [{"role": "user", "content": "x"}], [], reasoning_effort=effort
                )
            ]
        finally:
            await client.aclose()

    assert asyncio.run(run("high"))[0] == TextDelta("Réfléchi.")
    with pytest.raises(LlmError, match="pas de réponse"):
        asyncio.run(run("low"))
