"""Agent v2 (phase 13a) : le bac à sable refuse les constructions interdites et borne
l'exécution du code écrit par le LLM."""

from __future__ import annotations

import asyncio

import pytest

from mikata.agent2.sandbox import CodeRejected, check_code, run_code


class EchoSdk:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def open_app(self, name: str) -> str:
        await asyncio.sleep(0)  # cède le contrôle : le timeout du bac à sable peut expirer
        self.calls.append(name)
        return f"{name} ouvert"

    async def volume(self, percent: int) -> str:
        await asyncio.sleep(0)
        self.calls.append(str(percent))
        return f"volume {percent}"


def _run(code: str, sdk: EchoSdk, timeout_s: float = 5.0) -> str:
    async def scenario() -> str:
        return await run_code(code, sdk, timeout_s=timeout_s)

    return asyncio.run(scenario())


def test_valid_code_runs_and_returns() -> None:
    sdk = EchoSdk()
    result = _run(
        "await sdk.open_app('Spotify')\nawait sdk.volume(40)\nreturn 'fait'",
        sdk,
    )
    assert result == "fait"
    assert sdk.calls == ["Spotify", "40"]


@pytest.mark.parametrize(
    ("code", "raison"),
    [
        ("import os\nreturn 'x'", "imports"),
        ("open('x')", "open"),
        ("x = sdk._registry", "privés"),
        ("__import__('os')", "nom interdit"),
        ("eval('1+1')", "nom interdit"),
        ("f = lambda x: x\nreturn 'x'", "lambda"),
        ("class X:\n    pass\nreturn 'x'", "classes"),
        ("x = 1\ndel x", "del"),
        ("x = 1\ndef", "syntaxe"),
    ],
)
def test_forbidden_constructions_are_rejected(code: str, raison: str) -> None:
    with pytest.raises(CodeRejected):
        check_code(code)


def test_code_too_long_is_rejected() -> None:
    with pytest.raises(CodeRejected):
        check_code("x = " + "1" * 4001)


def test_infinite_loop_is_timed_out() -> None:
    sdk = EchoSdk()

    def scenario() -> None:
        async def run() -> None:
            with pytest.raises(TimeoutError):
                await run_code("while True:\n    await sdk.open_app('x')", sdk, timeout_s=0.2)

        asyncio.run(run())

    scenario()


def test_print_is_captured_not_printed(capsys: pytest.CaptureFixture[str]) -> None:
    sdk = EchoSdk()
    result = _run("print('bonjour', 'monde')\nreturn 'ok'", sdk)
    assert "bonjour monde" in result and "ok" in result
    assert capsys.readouterr().out == ""  # rien sur la console
