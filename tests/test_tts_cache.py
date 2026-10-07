from __future__ import annotations

from pathlib import Path

import numpy as np

from mikata.tts.cache import PHRASES, PhraseCache


class CountingTTS:
    name = "fake"
    sample_rate = 16_000

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize_sync(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.full(len(text) * 10, 0.25, dtype=np.float32)


def test_warm_synthesizes_once_then_reads_disk(tmp_path: Path) -> None:
    tts = CountingTTS()
    cache = PhraseCache(tts, tmp_path, "voix-a")
    cache.warm()
    assert sorted(tts.calls) == sorted(PHRASES.values())
    assert len(list(tmp_path.glob("fake-*.wav"))) == len(PHRASES)
    done = cache.get("done")
    assert done is not None and len(done) == len("C'est fait.") * 10

    again = CountingTTS()
    PhraseCache(again, tmp_path, "voix-a").warm()
    assert again.calls == []  # relu depuis data/tts_cache

    other_voice = CountingTTS()
    PhraseCache(other_voice, tmp_path, "voix-b").warm()
    assert len(other_voice.calls) == len(PHRASES)  # autre voix : autre clé


def test_lookup_matches_spoken_text(tmp_path: Path) -> None:
    cache = PhraseCache(CountingTTS(), tmp_path, "v")
    cache.warm()
    assert cache.lookup("C'est fait.") is cache.get("done")
    assert cache.lookup("c'est fait !") is cache.get("done")
    assert cache.lookup("C'est fait, j'ai ouvert Spotify.") is None
    assert cache.get("inconnu") is None
    assert cache.sample_rate == 16_000


def test_corrupt_cache_file_is_regenerated(tmp_path: Path) -> None:
    tts = CountingTTS()
    cache = PhraseCache(tts, tmp_path, "v")
    cache.warm()
    for path in tmp_path.glob("*.wav"):
        path.write_bytes(b"pas un wav")
    tts2 = CountingTTS()
    PhraseCache(tts2, tmp_path, "v").warm()
    assert len(tts2.calls) == len(PHRASES)
