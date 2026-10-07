from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

from mikata.audio.wavfile import read_wav
from mikata.wake.recordings import (
    NegativeCollector,
    estimate_tail_silence_s,
    purge_older_than,
    save_detection,
)

NOW = datetime(2026, 10, 7, 14, 30, 5, 123456)


def test_save_detection(tmp_path: Path) -> None:
    audio = np.arange(32_000, dtype=np.int16)
    path = save_detection(tmp_path / "wake_logs", audio, 0.734, NOW)
    assert path.name == "20261007-143005-123456_score0.73.wav"
    pcm, rate = read_wav(path)
    assert rate == 16_000
    assert len(pcm) == 32_000


def test_purge_sends_old_files_to_trash(tmp_path: Path) -> None:
    old = tmp_path / "a" / "old.wav"
    recent = tmp_path / "recent.wav"
    other = tmp_path / "notes.txt"
    for p in (old, recent, other):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    now = 1_000_000_000.0
    os.utime(old, (now - 8 * 86_400, now - 8 * 86_400))
    os.utime(recent, (now - 1 * 86_400, now - 1 * 86_400))
    os.utime(other, (now - 30 * 86_400, now - 30 * 86_400))
    trashed: list[str] = []
    assert purge_older_than(tmp_path, 7, now=now, trash=trashed.append) == 1
    assert trashed == [str(old)]
    assert purge_older_than(tmp_path / "absent", 7, trash=trashed.append) == 0


def test_negative_collector_makes_10s_clips() -> None:
    collector = NegativeCollector()
    clips = []
    for i in range(130):  # 130 trames de 80 ms = 10,4 s
        clip = collector.add(np.full(1280, i, dtype=np.int16))
        if clip is not None:
            clips.append(clip)
    assert len(clips) == 1
    assert len(clips[0]) == 160_000
    assert clips[0][0] == 0 and clips[0][-1] == 124
    assert NegativeCollector.path_for(Path("d"), NOW) == Path("d/2026-10-07/143005.wav")


def test_estimate_tail_silence() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 50, 32_000)
    audio = noise.copy()
    audio[8_000:27_200] += 8_000 * np.sin(np.arange(19_200) * 0.3)  # voix jusqu'à 1,7 s
    tail = estimate_tail_silence_s(audio.astype(np.int16))
    assert tail is not None
    assert abs(tail - 0.3) < 0.02
    assert estimate_tail_silence_s(noise.astype(np.int16)) is None
    assert estimate_tail_silence_s(np.zeros(50, dtype=np.int16)) is None
