from __future__ import annotations

import numpy as np
import pytest

from mikata.audio.ringbuffer import RingBuffer


def test_read_before_full() -> None:
    rb = RingBuffer(10)
    rb.write(np.arange(4, dtype=np.int16))
    assert len(rb) == 4
    assert rb.read_last(10).tolist() == [0, 1, 2, 3]
    assert rb.read_last(2).tolist() == [2, 3]
    assert rb.read_last(0).tolist() == []


def test_wraparound_keeps_latest() -> None:
    rb = RingBuffer(10)
    for start in range(0, 25, 5):
        rb.write(np.arange(start, start + 5, dtype=np.int16))
    assert len(rb) == 10
    assert rb.read_last(10).tolist() == list(range(15, 25))
    assert rb.read_last(3).tolist() == [22, 23, 24]


def test_write_larger_than_capacity() -> None:
    rb = RingBuffer(4)
    rb.write(np.arange(3, dtype=np.int16))
    rb.write(np.arange(100, 110, dtype=np.int16))
    assert rb.read_last(4).tolist() == [106, 107, 108, 109]
    rb.write(np.array([1], dtype=np.int16))
    assert rb.read_last(4).tolist() == [107, 108, 109, 1]


def test_read_returns_copy_and_clear() -> None:
    rb = RingBuffer(4)
    rb.write(np.array([1, 2], dtype=np.int16))
    out = rb.read_last(2)
    out[:] = 0
    assert rb.read_last(2).tolist() == [1, 2]
    rb.clear()
    assert len(rb) == 0
    assert rb.read_last(4).tolist() == []


def test_for_seconds() -> None:
    assert RingBuffer.for_seconds(2.0).capacity == 32_000
    with pytest.raises(ValueError):
        RingBuffer(0)
