from __future__ import annotations

import asyncio
import time
import types

import numpy as np
import pytest
from conftest import FakeInputStream

from mikata.audio.base import FRAME_SAMPLES
from mikata.audio.capture import MAX_QUEUED_FRAMES, AudioCapture, CapturedFrame, FrameAssembler
from mikata.audio.devices import DeviceInfo

MIC = DeviceInfo(1, "Analogue 1 + 2 (Focusrite USB Audio)", "Windows WASAPI", 2, 0, 48_000)


def _sine(freq: float, rate: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_passthrough_at_16k() -> None:
    asm = FrameAssembler(16_000, 1)
    data = (np.arange(3000) / 32768).astype(np.float32)[:, None]
    frames = asm.push(data[:1000]) + asm.push(data[1000:])
    assert [len(f) for f in frames] == [FRAME_SAMPLES, FRAME_SAMPLES]
    assert frames[0].dtype == np.int16
    assert frames[1][0] == FRAME_SAMPLES
    assert asm.push(np.zeros((40, 1), np.float32)) == []  # 440 + 40 < 1280


def test_resample_48k_stereo_keeps_frequency() -> None:
    asm = FrameAssembler(48_000, 2)
    tone = _sine(1000, 48_000, 1.0)
    stereo = np.stack([tone, tone], axis=1)
    frames = [f for i in range(0, len(stereo), 960) for f in asm.push(stereo[i : i + 960])]
    assert 11 <= len(frames) <= 12.5  # ~1 s de 16 kHz en trames de 80 ms
    audio = np.concatenate(frames[2:]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * 16_000 / len(audio)
    assert abs(peak_hz - 1000) < 20
    assert 0.4 < np.abs(audio).max() / 32768 < 0.6


def test_channel_selection() -> None:
    silent = np.zeros(48_000, dtype=np.float32)
    tone = _sine(440, 48_000, 1.0)
    block = np.stack([silent, tone], axis=1)
    left = FrameAssembler(48_000, 2, channel=1).push(block)
    right = FrameAssembler(48_000, 2, channel=2).push(block)
    mixed = FrameAssembler(48_000, 2).push(block)
    assert np.abs(np.concatenate(left)).max() == 0
    peak_right = np.abs(np.concatenate(right)).max()
    peak_mixed = np.abs(np.concatenate(mixed)).max()
    assert peak_right > 15_000
    assert peak_mixed == pytest.approx(peak_right / 2, rel=0.05)


def test_invalid_channel() -> None:
    with pytest.raises(ValueError, match="canal 3"):
        FrameAssembler(48_000, 2, channel=3)


def test_capture_end_to_end(fake_sd: types.ModuleType) -> None:
    FakeInputStream.signal = np.stack([_sine(500, 48_000, 0.5)] * 2, axis=1)

    async def scenario() -> list[CapturedFrame]:
        capture = AudioCapture(MIC, channel=1)
        capture.start(asyncio.get_running_loop())
        frames = [await asyncio.wait_for(capture.read(), timeout=5) for _ in range(5)]
        capture.stop()
        capture.stop()  # idempotent
        return frames

    before = time.monotonic()
    frames = asyncio.run(scenario())
    stream = FakeInputStream.instances[-1]
    assert stream.kwargs["device"] == 1
    assert stream.kwargs["samplerate"] == 48_000
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["blocksize"] == 960
    assert stream.kwargs["dtype"] == "float32"
    assert stream.closed
    assert all(len(f.samples) == FRAME_SAMPLES for f in frames)
    assert all(before <= f.captured_at <= time.monotonic() for f in frames)
    assert [f.captured_at for f in frames] == sorted(f.captured_at for f in frames)
    assert np.abs(frames[2].samples).max() > 10_000


def test_callback_only_copies() -> None:
    capture = AudioCapture(MIC)
    block = np.ones((960, 2), dtype=np.float32)
    capture._callback(block, 960, None, False)
    block[:] = 0  # PortAudio réutilise son tampon
    copied, captured_at, flagged = capture._raw.get_nowait()
    assert copied.shape == (960, 2)
    assert copied.max() == 1.0
    assert not flagged
    assert captured_at <= time.monotonic()


def test_queue_drops_oldest_when_consumer_stalls() -> None:
    async def scenario() -> tuple[int, CapturedFrame]:
        capture = AudioCapture(MIC)
        for i in range(MAX_QUEUED_FRAMES + 3):
            capture._enqueue(CapturedFrame(np.full(FRAME_SAMPLES, i, np.int16), float(i)))
        return capture.dropped_frames, await capture.read()

    dropped, first = asyncio.run(scenario())
    assert dropped == 3
    assert first.captured_at == 3.0
