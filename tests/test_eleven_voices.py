"""Phase 11 : scripts/eleven_voices.py avec l'API ElevenLabs simulée (aucun appel réel)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import httpx2
import pytest

from mikata.audio.wavfile import read_wav
from mikata.config import load_config

ROOT = Path(__file__).resolve().parent.parent
KEY = "el-test-0123456789"  # gitleaks:allow
THEO = "NyxenPOqNyllHIzSoPbJ"


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "eleven_voices", ROOT / "scripts" / "eleven_voices.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Api:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.headers["xi-api-key"] == KEY
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, request.url.path, {**dict(request.url.params), **body}))
        path = request.url.path
        if path == "/v1/voices":
            return httpx2.Response(200, json={"voices": [
                {"voice_id": THEO, "name": "Théo", "category": "professional",
                 "labels": {"gender": "male", "accent": "parisian"}},
                {"voice_id": "zzz", "name": "Alice", "category": "premade", "labels": {}},
            ]})  # fmt: skip
        if path == "/v1/shared-voices":
            return httpx2.Response(200, json={"voices": [
                {"public_owner_id": "own1", "voice_id": "v1", "name": "Camille",
                 "gender": "female", "accent": "standard", "language": "fr",
                 "description": "Voix douce", "preview_url": "https://example.test/p.mp3"},
            ], "has_more": False})  # fmt: skip
        if path.startswith("/v1/voices/add/"):
            return httpx2.Response(200, json={"voice_id": "nouvelle"})
        if path.startswith("/v1/text-to-speech/"):
            return httpx2.Response(200, content=b"\x00\x10" * 2400)
        return httpx2.Response(404, json={"detail": {"status": "not_found", "message": "?"}})


@pytest.fixture
def project(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "config.yaml", tmp_path / "config.yaml")
    (tmp_path / ".env").write_text(f"ELEVENLABS_API_KEY={KEY}\n", encoding="utf-8")
    return tmp_path


def _run(project: Path, *argv: str, api: Api | None = None) -> int:
    transport = httpx2.MockTransport(api or Api())
    return _script().main(["--config", str(project / "config.yaml"), *argv], transport)  # type: ignore[no-any-return]


def test_list_shows_account_voices_and_profiles(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert _run(project, "list") == 0
    out = capsys.readouterr().out
    assert "Alice" in out and "Théo" in out and "profil(s) gars" in out
    assert KEY not in out


def test_search_and_add(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = Api()
    assert _run(project, "search", "--gender", "female", "--query", "douce", api=api) == 0
    out = capsys.readouterr().out
    assert "Camille" in out and "owner=own1" in out and "voice=v1" in out
    assert api.calls[-1][2] == {"page_size": "20", "language": "fr", "gender": "female",
                                "search": "douce"}  # fmt: skip
    assert _run(project, "add", "own1", "v1", "--name", "Camille", api=api) == 0
    assert api.calls[-1][:2] == ("POST", "/v1/voices/add/own1/v1")
    assert api.calls[-1][2]["new_name"] == "Camille"
    assert "use <skin> nouvelle" in capsys.readouterr().out


def test_try_writes_a_wav(project: Path) -> None:
    api = Api()
    assert _run(project, "try", THEO, "--text", "Salut.", api=api) == 0
    _, path, sent = api.calls[-1]
    assert path == f"/v1/text-to-speech/{THEO}/stream" and sent["model_id"] == "eleven_v4_turbo"
    pcm, rate = read_wav(project / "data" / f"eleven_essai_{THEO}.wav")
    assert rate == 24_000 and len(pcm) == 2400


def test_use_writes_the_profile(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(project, "use", "femme", "v1", "--name", "Camille") == 0
    config = load_config(project / "config.yaml")
    assert config.ui.profiles["femme"].voice_id == "v1"
    assert config.ui.profiles["femme"].voice_name == "Camille"
    assert config.ui.profiles["gars"].voice_id == THEO  # les autres profils restent
    assert list((project / "data" / "backups").rglob("config.yaml"))  # ancienne version gardée
    assert _run(project, "use", "sakura", "--kokoro") == 0
    assert load_config(project / "config.yaml").ui.profiles["sakura"].voice == "kokoro"


def test_missing_key_is_reported(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    (project / ".env").write_text("", encoding="utf-8")
    assert _run(project, "list") == 1
    assert "ELEVENLABS_API_KEY absente" in capsys.readouterr().err
