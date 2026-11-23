# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the Python binding against the built C library and the tiny CPU model."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Barrier, Event
from unittest import TestCase
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
import flyx


def check_tokenization(engine):
    for text in ("", "a coding agent", "🙂"):
        data = text.encode()
        n = engine._lib.flyx_tokenize(engine._h, data, None, 0)
        expected = (ctypes.c_int32 * n)()
        engine._lib.flyx_tokenize(engine._h, data, expected, n)
        with patch.object(engine._lib, "flyx_tokenize", wraps=engine._lib.flyx_tokenize) as tokenize:
            assert engine.tokenize(text) == list(expected)
            assert tokenize.call_count == (2 if n > len(data) else 1)

    # Also exercise the retry branch when flyx_tokenize needs more slots than the initial UTF-8 byte length.
    calls = 0
    def grow(_h, _data, buf, cap):
        nonlocal calls
        calls += 1
        if cap < 3:
            return 3
        buf[0], buf[1], buf[2] = 7, 8, 9
        return 3
    with patch.object(engine._lib, "flyx_tokenize", side_effect=grow):
        assert engine.tokenize("x") == [7, 8, 9]
        assert calls == 2


def check_paths(model, library):
    # A forward slash is also a path separator on Windows.
    opts = dict(lib=library.as_posix(), backend="cpu_ref", max_ctx=128, speculative=False)
    with patch.dict(os.environ, FLYX_BACKEND_DIR=""):
        engine = flyx.Engine(model, **opts)
        assert engine.backend == "cpu_ref"
        check_tokenization(engine)
        engine.close()

    with tempfile.TemporaryDirectory() as empty, patch.dict(os.environ, FLYX_BACKEND_DIR=empty):
        try:
            flyx.Engine(model, **opts)
        except flyx.FlyxError as error:
            assert empty in str(error), error
        else:
            raise AssertionError("FLYX_BACKEND_DIR was ignored")
        engine = flyx.Engine(model, backend_dir=library.parent, **opts)
        assert engine.backend == "cpu_ref"
        engine.close()

    with tempfile.TemporaryDirectory(prefix="flyx-backend-") as tmp:
        directory = Path(tmp) / "测试 café 🗂"
        directory.mkdir()
        plugin = "libflyx-cpu_ref" + library.suffix
        shutil.copyfile(library.parent / plugin, directory / plugin)
        with patch.dict(os.environ, FLYX_BACKEND_DIR=str(directory)):
            engine = flyx.Engine(model, **opts)
            assert engine.backend == "cpu_ref"
            engine.close()


def check_callback(model, library):
    engine = flyx.Engine(model, lib=library, backend="cpu_ref", backend_dir=library.parent,
                         max_ctx=128, speculative=False)
    session = engine.session()
    failure = ValueError("writing the streamed text failed")

    def fail(text):
        raise failure

    try:
        session.generate(text="hello", on_text=fail, max_new_tokens=12, ignore_eos=True)
    except ValueError as error:
        assert error is failure
    else:
        raise AssertionError("callback failure returned a partial response")
    assert session.usage.finish_reason == 3  # FLYX_FINISH_CANCEL
    assert len(session.generate(text="hello", max_new_tokens=2, ignore_eos=True)) == 2
    # The C API delivers pointer + length, including text with an embedded U+0000.
    raw = "A\0文".encode()
    buffer = ctypes.create_string_buffer(raw)

    def emit(_session, _request, callback, _user, _usage, _error):
        event = flyx._Event(1, ctypes.addressof(buffer), len(raw), None, 0, None)
        return callback(ctypes.pointer(event), None)

    with patch.object(engine._lib, "flyx_generate", emit):
        assert session.chat([{"role": "user", "content": "example"}]) == raw.decode()
    session.close()
    engine.close()


def check_close(model, library):
    engine = flyx.Engine(model, lib=library, backend_dir=library.parent, backend="cpu_ref",
                         max_ctx=128, speculative=False)
    session = engine.session()
    engine.close()
    engine.close()
    raises = TestCase().assertRaisesRegex
    with patch.object(engine, "_lib") as closed_lib:
        closed_lib.flyx_session_new.return_value = None
        for use_engine in (lambda: engine.backend, lambda: engine.tokenize("test"),
                           engine.session, lambda: flyx.Session(engine)):
            with raises(flyx.FlyxError, "engine is closed"):
                use_engine()
        assert not closed_lib.mock_calls

    # An existing session owns its model even after Engine.close().
    assert len(session.generate(text="hello", max_new_tokens=2, ignore_eos=True)) == 2
    free = engine._lib.flyx_session_free
    handle = session._h

    def close_on_text(_text):
        session.close()
        session.close()
        release.assert_not_called()  # still on the native generator's stack

    # Record disposal and free outside the callback even when an assertion fails.
    with patch.object(engine._lib, "flyx_session_free") as release:
        try:
            session.generate(text="hello", max_new_tokens=12, ignore_eos=True, on_text=close_on_text)
            assert session.usage.finish_reason == 3  # FLYX_FINISH_CANCEL
        finally:
            session.close()
            free(handle)
        release.assert_called_once_with(handle)

    with patch.object(engine, "_lib") as closed_lib:
        for use_session in (lambda: session.generate(text="test"), lambda: session.chat([]),
                            lambda: session.chat_full([]), session.cancel):
            with raises(flyx.FlyxError, "session is closed"):
                use_session()
        assert not closed_lib.mock_calls


def check_threaded_close():
    # Park calls at the C boundary so their ordering does not depend on model speed.
    lib = Mock()
    lib.flyx_load.return_value, lib.flyx_session_new.return_value = 1, 2
    with patch.object(flyx, "_load_lib", return_value=lib):
        engine = flyx.Engine("fixture")
    session = engine.session()
    tokenizers = Barrier(3)
    resume = Event()

    def tokenize(_handle, data, out, _cap):
        tokenizers.wait(5)
        if data == b"b":
            assert resume.wait(5)
        out[0] = 7
        return 1

    lib.flyx_tokenize.side_effect = tokenize
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(engine.tokenize, "a")
        second = pool.submit(engine.tokenize, "b")
        tokenizers.wait(5)  # both tokenizers have borrowed the handle
        try:
            engine.close()
            assert first.result(timeout=5) == [7]
            lib.flyx_unload.assert_not_called()
        finally:
            resume.set()
        assert second.result(timeout=5) == [7]
        lib.flyx_unload.assert_called_once_with(1)

        entered = Event()
        resume.clear()

        def generate(_handle, _req, callback, _user, _usage, _err):
            entered.set()
            assert resume.wait(5)
            assert callback(ctypes.pointer(flyx._Event()), None) == 1
            return 0

        lib.flyx_generate.side_effect = generate
        result = pool.submit(session.generate, text="hello")
        assert entered.wait(5)
        try:
            with TestCase().assertRaisesRegex(flyx.FlyxError, "generation in progress"):
                session.generate(text="hello")
            pool.submit(session.close).result(timeout=5)
            lib.flyx_cancel.assert_called_once_with(2)
            lib.flyx_session_free.assert_not_called()
        finally:
            resume.set()
        assert result.result(timeout=5) == []
        session.close()
        lib.flyx_session_free.assert_called_once_with(2)


if __name__ == "__main__":
    model, library = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
    check_paths(model, library)
    check_callback(model, library)
    check_close(model, library)
    check_threaded_close()
