# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A failed stream must wake the other thread; a short first reply may finish before the second starts."""
from pathlib import Path
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench" / "replay"))
import two_at_once


def fail(url, body, on_piece):
    assert body == "first"
    raise RuntimeError("first stream failed")


def short(url, body, on_piece=None):
    if on_piece:
        on_piece(1)
    return body


if __name__ == "__main__":
    with patch.object(two_at_once, "stream", fail):
        try:
            two_at_once.concurrent_pair("", "first", "second")
        except RuntimeError as exc:
            assert str(exc) == "first stream failed"
        else:
            raise AssertionError("the failed first stream was lost")
    with patch.object(two_at_once, "stream", short):
        assert two_at_once.concurrent_pair("", "first", "second") == ("first", "second")

    second_started = threading.Event()

    def overlap(url, body, on_piece=None):
        if on_piece:
            on_piece(32)
            assert second_started.wait(5)
        else:
            second_started.set()
        return body

    with patch.object(two_at_once, "stream", overlap):
        assert two_at_once.concurrent_pair("", "first", "second") == ("first", "second")
