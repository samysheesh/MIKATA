from __future__ import annotations

import re
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

import mikata.__main__ as entry
from mikata.app import State
from mikata.config import PROJECT_ROOT


def _log(config_file: Path) -> str:
    return (config_file.parent / "data" / "logs" / "mikata.log").read_text(encoding="utf-8")


def test_check_config_ok(config_file: Path) -> None:
    assert entry.main(["--config", str(config_file), "--check-config"]) == entry.EXIT_OK
    log = _log(config_file)
    assert "Configuration valide" in log
    assert "LLM_API_KEY : absente" in log


def test_check_config_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "config_invalide.yaml"
    bad.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    assert entry.main(["--config", str(bad), "--check-config"]) == entry.EXIT_CONFIG
    log = _log(bad)
    assert "wake.threshold" in log
    assert "wake.treshold" in log


def test_missing_config(tmp_path: Path) -> None:
    assert entry.main(["--config", str(tmp_path / "absent.yaml")]) == entry.EXIT_CONFIG


def test_secret_is_never_logged(config_file: Path) -> None:
    (config_file.parent / ".env").write_text("LLM_API_KEY=sk-ne-pas-loguer\n", encoding="utf-8")
    assert entry.main(["--config", str(config_file), "--check-config"]) == entry.EXIT_OK
    log = _log(config_file)
    assert "LLM_API_KEY : définie" in log
    assert "sk-ne-pas-loguer" not in log


class AutoQuitTray:
    instances: ClassVar[list[AutoQuitTray]] = []

    def __init__(self) -> None:
        self.stopped = False
        AutoQuitTray.instances.append(self)

    def start(self, on_quit: Callable[[], None]) -> None:
        threading.Timer(0.05, on_quit).start()

    def set_state(self, state: State) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def test_run_and_quit_from_tray(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entry, "TrayIcon", AutoQuitTray)
    assert entry.main(["--config", str(config_file), "--no-audio"]) == entry.EXIT_OK
    assert AutoQuitTray.instances[-1].stopped
    log = _log(config_file)
    assert "Arrêt demandé (menu Quitter)" in log
    assert "MIKATA arrêté proprement." in log


def test_tray_failure_is_fatal(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenTray(AutoQuitTray):
        def start(self, on_quit: Callable[[], None]) -> None:
            raise ImportError("No module named 'pystray'")

    monkeypatch.setattr(entry, "TrayIcon", BrokenTray)
    assert entry.main(["--config", str(config_file), "--no-audio"]) == entry.EXIT_ERROR
    assert "Erreur fatale" in _log(config_file)


def test_keyboard_interrupt_is_clean(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(coro: object) -> None:
        coro.close()  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(entry.asyncio, "run", interrupted)
    assert entry.main(["--config", str(config_file), "--no-tray", "--no-audio"]) == entry.EXIT_OK
    assert "Ctrl+C" in _log(config_file)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        entry.main(["--version"])
    assert info.value.code == 0
    assert "MIKATA" in capsys.readouterr().out


# -- phase 1 : assemblage des composants audio ---------------------------------------------


@pytest.fixture
def no_priority_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mikata.governor.lower_priority", lambda: None)


def _with_audio(config_file: Path, **audio: object) -> Path:
    text = config_file.read_text(encoding="utf-8")
    for key, value in audio.items():
        text = re.sub(rf"(?m)^  {key}: .*$", f"  {key}: {value}", text)
    config_file.write_text(text, encoding="utf-8")
    return config_file


def test_missing_models_exit_with_hint(
    config_file: Path, fake_sd: object, no_priority_change: None
) -> None:
    assert entry.main(["--config", str(config_file), "--no-tray", "--no-llm"]) == entry.EXIT_CONFIG
    log = _log(config_file)
    assert "hey_jarvis.onnx" in log
    assert "download_models.py" in log


def test_unknown_device_exit(config_file: Path, fake_sd: object, no_priority_change: None) -> None:
    _with_audio(config_file, input_device='"Rode NT-USB"')
    assert entry.main(["--config", str(config_file), "--no-tray", "--no-llm"]) == entry.EXIT_CONFIG
    assert "aucun périphérique" in _log(config_file)


def test_channel_out_of_range(config_file: Path, fake_sd: object, no_priority_change: None) -> None:
    _with_audio(config_file, input_channel="3")
    assert entry.main(["--config", str(config_file), "--no-tray", "--no-llm"]) == entry.EXIT_CONFIG
    assert "audio.input_channel = 3" in _log(config_file)


def test_build_app_wires_real_components(
    config_file: Path,
    fake_sd: object,
    no_priority_change: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Detector:
        name = "hey_jarvis"

        def __init__(self, model: Path, verifier: Path | None) -> None:
            self.model = model

    class Vad:
        def __init__(self, model: Path, threshold: float) -> None:
            self.model, self.threshold = model, threshold

    monkeypatch.setattr("mikata.wake.oww.OpenWakeWordDetector", Detector)
    monkeypatch.setattr("mikata.vad.silero.SileroVAD", Vad)
    config = entry.load_config(config_file)
    app = entry.build_app(config, tray=False, audio=True, llm=False)
    assert app._agent is None and app._speaker is None
    assert app._audio.device.name == "Microphone (RØDE NT-USB+)"  # type: ignore[union-attr]
    assert app._player.device.name == "Casque (HyperX Cloud)"  # type: ignore[union-attr]
    assert app._detector.model == config_file.parent / "models" / "hey_jarvis.onnx"  # type: ignore[union-attr]
    assert app._earcons["wake"].dtype.name == "float32"
    assert app._hotkeys is not None and app._cpu is not None
    assert app._vad.model == config_file.parent / "models" / "silero_vad.onnx"  # type: ignore[union-attr]
    assert app._stt.download_root == config_file.parent / "models" / "whisper"  # type: ignore[union-attr]
    assert not app._stt.loaded  # type: ignore[union-attr]  # chargé au démarrage de run()


def test_missing_vad_model_exit(
    config_file: Path, fake_sd: object, no_priority_change: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mikata.wake.oww.OpenWakeWordDetector", lambda *a: types.SimpleNamespace(name="x")
    )
    assert entry.main(["--config", str(config_file), "--no-tray", "--no-llm"]) == entry.EXIT_CONFIG
    assert "silero_vad.onnx" in _log(config_file)


def test_gpu_unavailable_exit_code(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mikata.app import App
    from mikata.stt.whisper_local import SttUnavailableError

    class NoGpu:
        def load(self) -> None:
            raise SttUnavailableError("Aucun GPU NVIDIA utilisable par Whisper.")

        def close(self) -> None:
            pass

    def build(config: object, *, tray: bool, audio: bool, llm: bool = True) -> App:
        return App(entry.load_config(config_file), stt=NoGpu())  # type: ignore[arg-type]

    monkeypatch.setattr(entry, "build_app", build)
    assert entry.main(["--config", str(config_file), "--no-tray"]) == entry.EXIT_STARTUP
    log = _log(config_file)
    assert "Aucun GPU NVIDIA" in log
    assert "Traceback" not in log


# -- phase 3 : LLM obligatoire, voix --------------------------------------------------------


def _with_llm(config_file: Path, key: str | None = "sk-test") -> Path:
    text = config_file.read_text(encoding="utf-8")
    text = text.replace('base_url: "https://..."', 'base_url: "https://api.example.test/v1"')
    text = text.replace('model: "..."', 'model: "deepseek-flash"')
    config_file.write_text(text, encoding="utf-8")
    if key is not None:
        (config_file.parent / ".env").write_text(f"LLM_API_KEY={key}\n", encoding="utf-8")
    return config_file


def test_refuses_to_start_without_llm_key(
    config_file: Path, fake_sd: object, no_priority_change: None
) -> None:
    from conftest import FakeInputStream

    _with_llm(config_file, key=None)
    assert entry.main(["--config", str(config_file), "--no-tray"]) == entry.EXIT_CONFIG
    log = _log(config_file)
    assert "LLM_API_KEY absente" in log
    assert ".env" in log and "--no-llm" in log
    assert "Traceback" not in log
    assert FakeInputStream.instances == []  # refus avant d'ouvrir le micro


def test_refuses_to_start_with_placeholder_endpoint(
    config_file: Path, fake_sd: object, no_priority_change: None
) -> None:
    (config_file.parent / ".env").write_text("LLM_API_KEY=sk-test\n", encoding="utf-8")
    assert entry.main(["--config", str(config_file), "--no-tray"]) == entry.EXIT_CONFIG
    log = _log(config_file)
    assert "llm.base_url / llm.model non renseignés" in log
    assert "LLM_API_KEY absente" not in log


def test_build_app_with_llm_and_voice(
    config_file: Path, fake_sd: object, no_priority_change: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mikata.wake.oww.OpenWakeWordDetector", lambda *a: types.SimpleNamespace(name="x")
    )
    monkeypatch.setattr("mikata.vad.silero.SileroVAD", lambda *a: object())
    config = entry.load_config(_with_llm(config_file))
    app = entry.build_app(config, tray=False, audio=True)
    assert app._agent is not None and app._speaker is not None
    assert app._agent.llm.cfg.model == "deepseek-flash"  # type: ignore[attr-defined]
    assert app._speaker.tts.name == "piper"
    assert app._speaker.tts.model == config_file.parent / "models" / "fr_FR-siwis-medium.onnx"  # type: ignore[attr-defined]
    assert app._speaker.player is app._player  # même flux de sortie que les bips
    assert app._agent.phrases is app._speaker.cache


def test_ask_without_audio_prints_answer(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mikata.agent.loop import Agent
    from mikata.llm.base import Done, TextDelta

    class FakeLLM:
        async def stream(self, messages: list, tools: list):  # type: ignore[no-untyped-def]
            assert "quelle heure" in messages[-1]["content"]
            yield TextDelta("Il est midi.")
            yield Done("stop")

    monkeypatch.setattr(entry, "build_agent", lambda config, phrases=None: Agent(FakeLLM(), config))
    code = entry.main(
        ["--config", str(_with_llm(config_file)), "--ask", "quelle heure", "--no-audio"]
    )
    assert code == entry.EXIT_OK
    assert capsys.readouterr().out.strip() == "Il est midi."
    assert "premier token" in _log(config_file)


def test_ask_refuses_without_key(config_file: Path) -> None:
    code = entry.main(
        ["--config", str(_with_llm(config_file, key=None)), "--ask", "x", "--no-audio"]
    )
    assert code == entry.EXIT_CONFIG
