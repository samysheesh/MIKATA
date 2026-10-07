"""Client LLM testé avec le vrai SDK openai, branché sur un faux serveur (httpx2.MockTransport)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from mikata.config import ConfigError, LlmConfig, Secrets
from mikata.llm.base import Done, LLMClient, TextDelta, ToolCallReq
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
    assert "tools" not in payload  # pas d'outils en phase 3


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
