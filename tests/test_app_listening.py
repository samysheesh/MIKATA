"""Phase 2 : écoute depuis le pré-roll, fin de parole, transcription, retrait du mot."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_app import RecordingPlayer, RecordingTray, ScriptedAudio, ScriptedDetector

from mikata.app import App, State
from mikata.audio.wavfile import float_to_int16, read_wav
from mikata.config import PROJECT_ROOT, load_config
from mikata.events import WakeDetected
from mikata.stt.base import Transcript

EARCONS = {
    name: np.full(10, i + 1, dtype=np.float32) for i, name in enumerate(["wake", "end", "error"])
}


class ScriptedVAD:
    """Parole sur les trames dont la valeur (= index, cf. ScriptedAudio) est dans `speech`."""

    def __init__(self, speech: set[int]) -> None:
        self.speech = speech
        self.resets = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        return int(frame[0]) in self.speech

    def reset(self) -> None:
        self.resets += 1


class FakeSTT:
    def __init__(
        self, text: str | Callable[[np.ndarray], str] = "Hey Jarvis, ouvre Spotify."
    ) -> None:
        self.text = text
        self.calls: list[np.ndarray] = []
        self.loaded = False
        self.closed = False
        self.fail = False

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        self.calls.append(audio.copy())
        await asyncio.sleep(0.005)
        if self.fail:
            raise RuntimeError("CUDA failed")
        text = self.text if isinstance(self.text, str) else self.text(audio)
        return Transcript(text=text, duration_s=len(audio) / 16_000, latency_s=0.005)


def _make(
    config_file: Path,
    *,
    detections: dict[int, float],
    speech: set[int],
    frames: int = 120,
    stt: FakeSTT | None = None,
) -> tuple[App, RecordingTray, FakeSTT, ScriptedVAD, RecordingPlayer]:
    tray, player = RecordingTray(quit_from_thread=False), RecordingPlayer()
    stt, vad = stt or FakeSTT(), ScriptedVAD(speech)
    app = App(
        load_config(config_file),
        tray=tray,
        audio=ScriptedAudio(frames),
        detector=ScriptedDetector(detections),
        player=player,
        earcons=EARCONS,
        vad=vad,
        stt=stt,
    )
    return app, tray, stt, vad, player


def _run_until(app: App, done: Callable[[], bool], timeout: float = 5.0) -> None:
    async def scenario() -> None:
        task = asyncio.create_task(app.run())
        deadline = asyncio.get_running_loop().time() + timeout
        while not done():
            assert asyncio.get_running_loop().time() < deadline, "délai dépassé"
            assert not task.done(), task
            await asyncio.sleep(0.005)
        app.request_shutdown("test")
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


def _frame_ids(audio: np.ndarray) -> list[int]:
    return [int(v) for v in audio[::1280]]


def _played(player: RecordingPlayer) -> list[str]:
    names = {float(pcm[0]): name for name, pcm in EARCONS.items()}
    return [names[float(pcm[0])] for pcm in player.played]


def test_command_said_in_one_breath_starts_from_pre_roll(config_file: Path) -> None:
    # Mot d'activation détecté sur la trame 31, la commande continue jusqu'à la trame 37.
    app, tray, stt, vad, player = _make(
        config_file, detections={31: 0.9}, speech=set(range(20, 38))
    )
    _run_until(app, lambda: bool(app.commands) and app.state is State.IDLE)

    assert list(app.commands) == ["ouvre Spotify."]
    assert stt.loaded and stt.closed
    assert vad.resets == 1
    assert _played(player) == ["wake"]
    ids = _frame_ids(stt.calls[-1])
    assert len(ids) >= 25 + 6  # 2 s de pré-roll + la fin de la commande et le silence
    assert ids == list(range(ids[0], ids[0] + len(ids)))  # aucune trame perdue ni doublée
    assert ids[0] == 31 - 24  # 2 s pleines avant la trame de détection : « Hey Jarvis » inclus
    assert tray.states == [State.IDLE, State.LISTENING, State.TRANSCRIBING, State.IDLE]
    # La détection a continué pendant l'écoute : toutes les trames transcrites y sont passées.
    assert app._detector.calls >= ids[-1]  # type: ignore[union-attr]


def test_nothing_said_returns_to_idle_with_end_beep(config_file: Path) -> None:
    app, _, stt, _, player = _make(
        config_file, detections={31: 0.9}, speech=set(), stt=FakeSTT("Hey Jarvis.")
    )
    _run_until(app, lambda: "end" in _played(player) and app.state is State.IDLE)
    assert list(app.commands) == []
    assert len(stt.calls) == 1  # seulement la vérification du pré-roll


def test_short_command_finished_before_beep(config_file: Path) -> None:
    app, _, stt, _, _ = _make(
        config_file, detections={31: 0.9}, speech=set(), stt=FakeSTT("Hey Jarvis, pause.")
    )
    _run_until(app, lambda: bool(app.commands))
    assert list(app.commands) == ["pause."]
    assert len(stt.calls) == 1


def test_wake_word_alone_then_command(config_file: Path) -> None:
    def text(audio: np.ndarray) -> str:
        return "Hey Jarvis." if len(audio) <= 32_000 else "Hey Jarvis. Mets le volume à 30."

    app, _, _, _, _ = _make(
        config_file, detections={31: 0.9}, speech=set(range(45, 55)), stt=FakeSTT(text)
    )
    _run_until(app, lambda: bool(app.commands))
    assert list(app.commands) == ["Mets le volume à 30."]


def test_empty_text_plays_end_beep(config_file: Path) -> None:
    app, _, _, _, player = _make(
        config_file,
        detections={31: 0.9},
        speech=set(range(32, 36)),
        stt=FakeSTT("Sous-titres réalisés par la communauté d'Amara.org"),
    )
    _run_until(app, lambda: "end" in _played(player))
    assert list(app.commands) == []
    assert app.state is State.IDLE


def test_stt_failure_plays_error_beep(config_file: Path) -> None:
    stt = FakeSTT()
    stt.fail = True
    app, _, _, _, player = _make(
        config_file, detections={31: 0.9}, speech=set(range(32, 36)), stt=stt
    )
    _run_until(app, lambda: "error" in _played(player) and app.state is State.IDLE)
    assert list(app.commands) == []


def test_wake_word_ignored_while_listening(config_file: Path) -> None:
    app, _, _, _, player = _make(
        config_file, detections={31: 0.9}, speech=set(range(32, 60)), frames=200
    )
    sent = []

    def check() -> bool:
        if not sent and app.state is State.LISTENING:
            app.bus.publish(WakeDetected(score=1.0, source="hotkey"))
            sent.append(1)
        return bool(app.commands)

    _run_until(app, check)
    assert app.detections == 1  # le 2e déclenchement n'a ni bip ni nouvelle écoute
    assert _played(player) == ["wake"]


def test_hotkey_uses_short_pre_roll(config_file: Path) -> None:
    app, _, stt, _, _ = _make(
        config_file, detections={}, speech=set(range(42, 46)), stt=FakeSTT("ouvre Discord")
    )

    class HotkeyAt40(ScriptedAudio):
        async def read(self):  # type: ignore[no-untyped-def]
            if self._i == 40:  # appui sur Ctrl+Alt+M pendant la trame 40
                app.bus.publish(WakeDetected(score=1.0, source="hotkey"))
            return await super().read()

    app._audio = HotkeyAt40(120)
    _run_until(app, lambda: bool(app.commands))
    assert list(app.commands) == ["ouvre Discord"]
    assert len(stt.calls) == 1  # pas de vérification du pré-roll
    ids = _frame_ids(stt.calls[0])
    assert ids == list(range(ids[0], ids[0] + len(ids)))
    assert 36 <= ids[0] <= 41  # ~0,3 s de pré-roll, pas 2 s


MODELS = PROJECT_ROOT / "models"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "hey_jarvis_ouvre_spotify.wav"


class WavAudio(ScriptedAudio):
    """Trames réelles : bruit de fond, « hey jarvis ouvre Spotify » d'une traite, silence."""

    def __init__(self, pcm: np.ndarray) -> None:
        super().__init__(len(pcm) // 1280)
        self.pcm = pcm

    async def read(self):  # type: ignore[no-untyped-def]
        frame = await super().read()
        i = self._i - 1
        return type(frame)(self.pcm[i * 1280 : (i + 1) * 1280], frame.captured_at)


@pytest.mark.skipif(
    not (MODELS / "hey_jarvis.onnx").is_file() or not (MODELS / "silero_vad.onnx").is_file(),
    reason="modèles non téléchargés",
)
def test_real_wake_word_and_vad_keep_the_whole_sentence(config_file: Path) -> None:
    from mikata.vad.silero import SileroVAD
    from mikata.wake.oww import OpenWakeWordDetector

    speech, _ = read_wav(FIXTURE)
    rng = np.random.default_rng(0)
    noise = lambda s: rng.normal(0, 0.003, round(s * 16_000))  # noqa: E731
    pcm = float_to_int16(np.concatenate([noise(1.5), speech, noise(3.0)]).astype(np.float32))
    stt = FakeSTT()
    app = App(
        load_config(config_file),
        audio=WavAudio(pcm),
        detector=OpenWakeWordDetector(MODELS / "hey_jarvis.onnx"),
        vad=SileroVAD(MODELS / "silero_vad.onnx"),
        stt=stt,
    )
    _run_until(app, lambda: bool(app.commands), timeout=20)

    sent = stt.calls[-1]
    # L'audio transcrit est une tranche contiguë de trames : on retrouve sa position exacte.
    first = next(
        k for k in range(0, len(pcm), 1280) if np.array_equal(sent[:1280], pcm[k : k + 1280])
    )
    assert np.array_equal(sent, pcm[first : first + len(sent)])
    speech_start = round(1.5 * 16_000)
    assert first <= speech_start  # « hey jarvis » compris, dès le début
    assert first + len(sent) >= speech_start + len(speech)  # jusqu'au bout de « ouvre Spotify »
