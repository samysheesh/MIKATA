"""scripts/eval_prompt.py : 20 phrases approximatives, vraie boucle agent, actions simulées.

Ici le LLM est simulé (pas de réseau en CI) : on vérifie le banc d'essai lui-même, avec un
« bon » LLM (tout réussit) et un « mauvais » (PowerShell, question, réponse bavarde). Le vrai
test du prompt se lance sur le PC, avec la clé LLM : uv run python scripts/eval_prompt.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from scripts_loader import load_script
from tool_fakes import make_config

from mikata.llm.base import Done, LLMEvent, TextDelta, ToolCallReq, Usage

ev = load_script("eval_prompt")

GOOD: dict[str, tuple[str, dict[str, Any]]] = {
    "stop process name notepad": ("close_app", {"name": "Bloc-notes"}),
    "arrête notepad": ("close_app", {"name": "notepad"}),
    "ferme le bloc-notes": ("close_app", {"name": "Bloc-notes"}),
    "ouvre spotifaille": ("open_app", {"name": "Spotify"}),
    "lance Team": ("open_app", {"name": "Steam"}),
    "ouvre brève": ("open_app", {"name": "Brave"}),
    "ouvre visuel studio code": ("open_app", {"name": "Visual Studio Code"}),
    "ferme crôme": ("close_app", {"name": "Google Chrome"}),
    "ferme cette fenêtre": ("close_window", {}),
    "mets spotify devant": ("focus_window", {"name": "Spotify"}),
    "mets le son à trente": ("set_volume", {"level": 30}),
    "monte un peu le son": ("set_volume", {"change": 10}),
    "coupe le son": ("mute", {"mute": True}),
    "met pause": ("media_control", {"action": "pause"}),
    "passe à la chanson d'après": ("media_control", {"action": "next"}),
    "écris bonjour tout le monde": ("type_text", {"text": "Bonjour tout le monde"}),
    "fais contrôle S": ("press_keys", {"keys": "ctrl+s"}),
    "mets un minuteur de cinq minutes": ("set_timer", {"seconds": 300}),
    "quel temps il fait à Lyon": ("get_weather", {"city": "Lyon"}),
    "verrouille l'ordi": ("lock", {}),
}


class PhraseLLM:
    """Répond à chaque phrase par l'appel d'outil de `answers`, puis par `final`."""

    def __init__(self, answers: dict[str, tuple[str, dict[str, Any]] | str], final: str) -> None:
        self.answers = answers
        self.final = final

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        assert any("Applications installées" in m["content"] for m in messages[:3])
        if messages[-1]["role"] == "tool":  # après un outil non terminal (météo)
            yield TextDelta(self.final)
            yield Done("stop", usage=Usage(3000, 15, 2900))
            return
        phrase = messages[-1]["content"].split("« ")[-1].rstrip(" »")
        answer = self.answers[phrase]
        if isinstance(answer, str):
            yield TextDelta(answer)
            yield Done("stop", usage=Usage(3000, 15, 2900))
            return
        name, arguments = answer
        yield ToolCallReq("c1", name, arguments)
        yield Done("tool_calls", usage=Usage(3000, 12, 2900))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_weather sans réseau : réponse Open-Meteo simulée."""
    import httpx2

    def server(request: httpx2.Request) -> httpx2.Response:
        if "geocoding" in request.url.host:
            place = {"name": "Lyon", "country": "France", "latitude": 45.7, "longitude": 4.8}
            return httpx2.Response(200, json={"results": [place]})
        return httpx2.Response(200, json={"current": {}, "daily": {}})

    monkeypatch.setattr(
        "mikata.tools.web.create_http_client",
        lambda: httpx2.AsyncClient(transport=httpx2.MockTransport(server)),
    )


def _run(llm: PhraseLLM, tmp_path: Path) -> list[Any]:
    config = make_config(tmp_path)
    return asyncio.run(ev.run_cases(llm, config, list(enumerate(ev.CASES, 1))))


def test_twenty_phrases_including_the_notepad_ones() -> None:
    phrases = [case.phrase for case in ev.CASES]
    assert len(phrases) == 20 and len(set(phrases)) == 20
    assert phrases[:3] == ["stop process name notepad", "arrête notepad", "ferme le bloc-notes"]
    assert set(phrases) == set(GOOD)


def test_a_good_llm_passes_everything_without_touching_the_pc(
    tmp_path: Path, no_network: None
) -> None:
    outcomes = _run(PhraseLLM(GOOD, "Il fait doux à Lyon."), tmp_path)
    assert [o.reasons for o in outcomes if not o.ok] == []
    by_phrase = {o.phrase: o for o in outcomes}
    assert by_phrase["stop process name notepad"].calls == ["close_app(name='Bloc-notes')"]
    assert by_phrase["mets le son à trente"].answer == "C'est fait."
    assert by_phrase["mets le son à trente"].llm_calls == 1
    assert by_phrase["quel temps il fait à Lyon"].llm_calls == 2
    text = ev.report(outcomes)
    assert "Réussies : 20/20" in text and "en un seul appel LLM : 19" in text


def test_a_literal_llm_fails_with_reasons(tmp_path: Path, no_network: None) -> None:
    bad: dict[str, Any] = dict(GOOD)
    bad["stop process name notepad"] = ("run_powershell", {"command": "Stop-Process -Name notepad"})
    bad["arrête notepad"] = "Tu veux que je ferme le Bloc-notes ?"
    bad["ouvre spotifaille"] = ("open_app", {"name": "Spotifaille"})
    bad["met pause"] = (
        "J'ai utilisé PowerShell pour envoyer la touche pause au lecteur multimédia, qui "
        "devrait maintenant être en pause si tout s'est bien passé comme prévu ce soir."
    )
    outcomes = _run(PhraseLLM(bad, "Fait."), tmp_path)
    failed = {o.phrase: o.reasons for o in outcomes if not o.ok}
    assert set(failed) == {
        "stop process name notepad",
        "arrête notepad",
        "ouvre spotifaille",
        "met pause",
    }
    assert "outil à éviter : run_powershell" in failed["stop process name notepad"]
    assert "pose une question au lieu d'agir" in failed["arrête notepad"]
    assert failed["ouvre spotifaille"] == ["attendu open_app/focus_window (Spotify)"]
    assert any("parle de powershell" in r for r in failed["met pause"])
    assert any("trop longue" in r for r in failed["met pause"])


def test_main_refuses_without_llm_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from conftest import write_test_config

    path = write_test_config(tmp_path / "config.yaml")
    monkeypatch.setenv("MIKATA_CONFIG", str(path))
    assert ev.main([]) == 2  # LLM non configuré : message clair, rien de lancé
