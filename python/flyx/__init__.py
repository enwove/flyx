# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Python binding over the C API in include/flyx/flyx.h (ctypes, no build step).

    import flyx
    eng = flyx.Engine("qwen35-4b.flyx", lib="build/bin/libflyx.so")
    s = eng.session()
    out = s.generate(tokens=[...], max_new_tokens=64)          # -> list of token ids
    text = s.chat([{"role": "user", "content": "hi"}], max_new_tokens=64, on_text=print)
    reply = s.chat_full(messages, tools=[...], temperature=0.6)  # -> {"content", "reasoning", "tool_calls"}

Messages use the OpenAI chat format (tool_calls, tool results, reasoning_content).
A session keeps its state on the GPU: the next call that extends the same
conversation only pays for the new tokens.
"""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import json
import os
import threading

__all__ = [
    "Engine", "Session", "Usage", "FlyxError",
    "FLYX_EVENT_TEXT", "FLYX_EVENT_DONE", "FLYX_EVENT_REASONING", "FLYX_EVENT_TOOL_CALL",
    "FLYX_FINISH_STOP", "FLYX_FINISH_LENGTH", "FLYX_FINISH_CANCEL", "FLYX_FINISH_ERROR",
]

# The version of the C API the structures below mirror (FLYX_API_VERSION in include/flyx/flyx.h): they
# change together.  A library of another version lays its structs out differently, so it is refused when loaded.
_API_VERSION = 3

FLYX_EVENT_TEXT = 1
FLYX_EVENT_DONE = 2
FLYX_EVENT_REASONING = 3
FLYX_EVENT_TOOL_CALL = 4

FLYX_FINISH_STOP = 1
FLYX_FINISH_LENGTH = 2
FLYX_FINISH_CANCEL = 3
FLYX_FINISH_ERROR = 4

_EVENT_NAMES = {
    FLYX_EVENT_TEXT: "text",
    FLYX_EVENT_REASONING: "reasoning",
    FLYX_EVENT_TOOL_CALL: "tool_call",
}


class FlyxError(RuntimeError):
    pass


class _Err(C.Structure):
    _fields_ = [("message", C.c_char * 512)]


class _LoadOpts(C.Structure):
    _fields_ = [("max_ctx", C.c_int32), ("backend", C.c_char_p), ("backend_dir", C.c_char_p),
                ("speculative", C.c_int32), ("verbose", C.c_int32), ("drafter", C.c_char_p), ("width", C.c_int32),
                ("prefill_chunk", C.c_int32), ("kv", C.c_int32),
                ("backend_options", C.c_char_p), ("activations", C.c_char_p), ("chat_template", C.c_char_p),
                ("drafter_file", C.c_char_p), ("mmproj_file", C.c_char_p), ("no_vision", C.c_int32),
                ("vision_device", C.c_char_p), ("vision_quant", C.c_char_p), ("vision_max_tokens", C.c_int32)]


class _Msg(C.Structure):
    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p), ("reasoning", C.c_char_p)]


class _Req(C.Structure):
    _fields_ = [("tokens", C.POINTER(C.c_int32)), ("n_tokens", C.c_int32), ("messages", C.POINTER(_Msg)),
                ("n_messages", C.c_int32), ("text", C.c_char_p), ("enable_thinking", C.c_int32),
                ("max_new_tokens", C.c_int32), ("temperature", C.c_float), ("ignore_eos", C.c_int32),
                ("top_p", C.c_float), ("top_k", C.c_int32), ("seed", C.c_uint64),
                ("stop", C.POINTER(C.c_char_p)), ("n_stop", C.c_int32), ("messages_json", C.c_char_p),
                ("tools_json", C.c_char_p), ("split_output", C.c_int32), ("tool_choice_json", C.c_char_p),
                ("response_format_json", C.c_char_p), ("single_tool_call", C.c_int32),
                ("presence_penalty", C.c_float), ("frequency_penalty", C.c_float)]


class Usage(C.Structure):
    _fields_ = [("prompt_tokens", C.c_int32), ("reused_tokens", C.c_int32), ("new_tokens", C.c_int32),
                ("rounds", C.c_int32), ("accepted_drafts", C.c_int32), ("drafted", C.c_int32),
                ("prefill_ms", C.c_double),
                ("decode_ms", C.c_double), ("finish_reason", C.c_int32)]


class _Event(C.Structure):
    # Keep the pointer until text_len bytes are read. c_char_p converts it to NUL-terminated bytes on access.
    _fields_ = [("kind", C.c_int32), ("text", C.c_void_p), ("text_len", C.c_size_t),
                ("tokens", C.POINTER(C.c_int32)), ("n_tokens", C.c_int32), ("usage", C.POINTER(Usage))]


_EventFn = C.CFUNCTYPE(C.c_int, C.POINTER(_Event), C.c_void_p)


_LIB_NAME = "flyx.dll" if os.name == "nt" else "libflyx.so"


def _default_lib() -> str:
    """The library shipped with this wheel or unpacked release, else the system's."""
    package = os.path.dirname(os.path.realpath(__file__))
    wheel = os.path.join(package, "_lib", _LIB_NAME)
    if os.path.isfile(wheel):
        return wheel
    python_dir = os.path.dirname(package)
    if os.path.normcase(os.path.basename(python_dir)) == "python":
        archive = os.path.join(os.path.dirname(python_dir), "lib", _LIB_NAME)
        if os.path.isfile(archive):
            return archive
    return _LIB_NAME


_DLL_DIRS: dict[str, object] = {}


def _load_lib(path: str):
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        lib_dir = os.path.dirname(os.path.abspath(path))
        key = os.path.normcase(lib_dir)
        if os.path.isdir(lib_dir) and key not in _DLL_DIRS:
            _DLL_DIRS[key] = os.add_dll_directory(lib_dir)
    lib = C.CDLL(path)
    try:
        lib.flyx_api_version.restype = C.c_uint32
        lib.flyx_api_version.argtypes = []
        version = lib.flyx_api_version()
    except AttributeError:  # a library from before the API had a version
        version = None
    if version != _API_VERSION:
        raise FlyxError(f"{path} has C API version {'none' if version is None else version}; this binding is "
                        f"written for version {_API_VERSION}: use the binding that came with the library")
    lib.flyx_load.restype = C.c_void_p
    lib.flyx_load.argtypes = [C.c_char_p, C.POINTER(_LoadOpts), C.POINTER(_Err)]
    lib.flyx_unload.argtypes = [C.c_void_p]
    lib.flyx_backend_name.restype = C.c_char_p
    lib.flyx_backend_name.argtypes = [C.c_void_p]
    lib.flyx_session_new.restype = C.c_void_p
    lib.flyx_session_new.argtypes = [C.c_void_p]
    lib.flyx_session_free.argtypes = [C.c_void_p]
    lib.flyx_generate.restype = C.c_int
    lib.flyx_generate.argtypes = [C.c_void_p, C.POINTER(_Req), _EventFn, C.c_void_p, C.POINTER(Usage),
                                  C.POINTER(_Err)]
    lib.flyx_cancel.argtypes = [C.c_void_p]
    lib.flyx_tokenize.restype = C.c_int
    lib.flyx_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int32]
    return lib


class Engine:
    def __init__(self, model: str, lib: str | None = None, backend: str | None = None, max_ctx: int | str = 8192,
                 speculative: bool = True, verbose: bool = False, backend_dir: str | None = None,
                 drafter: str | None = None, width: int = 0, prefill_chunk: int = 0, kv: str | None = None,
                 backend_options: str | None = None, activations: str | None = None,
                 chat_template: str | None = None, drafter_file: str | list[str] | None = None,
                 mmproj_file: str | None = None, vision: bool = True,
                 vision_device: str | None = None, vision_quant: str | None = None,
                 vision_max_tokens: int = 0):
        """max_ctx: tokens, or "max": the model's own context, or the longest the GPU and host memory hold.
        width: drafts per round (0: chosen each round by the speculation controller).
        activations: None (as the file has it), "exact" (every format's own rounding everywhere) or "nvfp4"
        (NVFP4 weights take NVFP4 inputs in prefill passes whatever the file says).
        kv: None (the model's: FP8, binary16 on Llama models), "fp8" or "16" (binary16, llama.cpp's).
        chat_template: a Jinja template's source (not a path), used instead of the model file's.
        drafter_file: drafters in files of their own, one path or a list: the MTP layer from a GGUF (unsloth's
        MTP/mtp-*.gguf), for a GGUF without it, or a DFlash2 drafter file (flyx_convert --draft alone).
        mmproj_file: optional companion mmproj-*.gguf vision projector path.
        vision: False skips loading any vision encoder even if present.
        vision_device: "disk" (default, 0 VRAM / 0 locked RAM), "host", "gpu", or "auto".
        vision_quant: "q8" (default), "exact", or "nvfp4" for float vision linears.
        vision_max_tokens: per-image merged visual token cap (0 = auto)."""
        if isinstance(drafter_file, (list, tuple)):
            drafter_file = "\n".join(os.path.expanduser(os.fspath(p)) for p in drafter_file)
        elif drafter_file is not None:
            drafter_file = os.path.expanduser(os.fspath(drafter_file))
        self._lock = threading.Lock()
        self._active = 0
        self._h = None
        lib = os.path.expanduser(lib or os.environ.get("FLYX_LIB") or _default_lib())
        self._lib = _load_lib(lib)
        if backend_dir is None and not os.environ.get("FLYX_BACKEND_DIR") and os.path.dirname(lib):
            backend_dir = os.path.dirname(os.path.abspath(lib))
        kv_modes = {None: 0, "fp8": 1, "16": 2}
        if kv not in kv_modes:
            raise ValueError(f"kv must be None, 'fp8', or '16' (got {kv!r})")
        opts = _LoadOpts(-1 if max_ctx == "max" else max_ctx, backend.encode() if backend else None,
                         os.path.expanduser(backend_dir).encode() if backend_dir else None, int(speculative), int(verbose),
                         drafter.encode() if drafter else None, int(width), prefill_chunk, kv_modes[kv],
                         backend_options.encode() if backend_options else None,
                         activations.encode() if activations else None,
                         chat_template.encode() if chat_template else None,
                         drafter_file.encode() if drafter_file else None,
                         os.path.expanduser(os.fspath(mmproj_file)).encode() if mmproj_file else None,
                         0 if vision else 1,
                         vision_device.encode() if vision_device else None,
                         vision_quant.encode() if vision_quant else None,
                         int(vision_max_tokens))
        err = _Err()
        self._h = self._lib.flyx_load(os.path.expanduser(model).encode(), C.byref(opts), C.byref(err))
        if not self._h:
            raise FlyxError(err.message.decode(errors="replace"))

    @property
    def backend(self) -> str:
        with self._borrow_handle() as handle:
            return self._lib.flyx_backend_name(handle).decode()

    @contextmanager
    def _borrow_handle(self):
        # Tokenization can run in parallel. Keep the handle alive across both native calls.
        with self._lock:
            if not self._h:
                raise FlyxError("engine is closed")
            handle = self._h
            self._active += 1
        try:
            yield handle
        finally:
            with self._lock:
                self._active -= 1
                if self._h is None and not self._active:
                    self._lib.flyx_unload(handle)

    def tokenize(self, text: str) -> list[int]:
        """The tokens generate(text=...) feeds: the tokenizer's prefix (MiniCPM5's <s>), then the text's."""
        data = text.encode()
        buf = (C.c_int32 * max(len(data), 1))()
        with self._borrow_handle() as handle:
            n = self._lib.flyx_tokenize(handle, data, buf, len(buf))
            if n < 0:
                raise FlyxError("tokenization failed")
            if n > len(buf):  # a character can take several byte-level tokens
                buf = (C.c_int32 * n)()
                n = self._lib.flyx_tokenize(handle, data, buf, n)
                if n < 0:
                    raise FlyxError("tokenization failed")
        return buf[:n]

    def session(self) -> "Session":
        return Session(self)

    def close(self):
        """Close the engine after outstanding calls return. Sessions already created keep their model."""
        with self._lock:
            if self._h:
                handle, self._h = self._h, None
                if not self._active:
                    self._lib.flyx_unload(handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Session:
    """A session runs one generation at a time. It can be closed from a stream callback or another thread."""

    def __init__(self, engine: Engine):
        self._e = engine
        self._lock = threading.Lock()
        self._running = False
        self._h = None
        with engine._borrow_handle() as handle:
            self._h = engine._lib.flyx_session_new(handle)
        if not self._h:
            raise FlyxError("could not create a session")
        self.usage = Usage()

    def _open_handle(self):
        if not self._h:
            raise FlyxError("session is closed")
        return self._h

    def _run(self, req: _Req, on_text, on_event=None):
        toks: list[int] = []
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[dict] = []

        def handle_event(e):
            s = C.string_at(e.text, e.text_len).decode(errors="replace") if e.text_len else ""
            if e.kind == FLYX_EVENT_TEXT:
                if e.n_tokens:
                    toks.extend(e.tokens[: e.n_tokens])
                text.append(s)
            elif e.kind == FLYX_EVENT_REASONING:
                reasoning.append(s)
            elif e.kind == FLYX_EVENT_TOOL_CALL:
                calls.append(json.loads(s))
            if e.kind in (FLYX_EVENT_TEXT, FLYX_EVENT_REASONING) and s and on_text and on_text(s) is False:
                return 1
            if on_event and e.kind in _EVENT_NAMES and on_event(_EVENT_NAMES[e.kind], s) is False:
                return 1
            return 0

        callback_error = None

        @_EventFn
        def callback(ev, _user):
            # ctypes cannot propagate a Python exception across the C callback.
            nonlocal callback_error
            if self._h is None:
                return 1  # close may have arrived before the native generation started
            try:
                return handle_event(ev.contents)
            except BaseException as error:
                callback_error = error
                return 1

        err = _Err()
        with self._lock:
            handle = self._open_handle()
            if self._running:
                raise FlyxError("session already has a generation in progress")
            self._running = True
        try:
            rc = self._e._lib.flyx_generate(handle, C.byref(req), callback, None, C.byref(self.usage), C.byref(err))
        finally:
            with self._lock:
                self._running = False
                if self._h is None:
                    self._e._lib.flyx_session_free(handle)
        if callback_error is not None:
            raise callback_error
        if rc != 0:
            raise FlyxError(err.message.decode(errors="replace"))
        return toks, "".join(text), "".join(reasoning), calls

    @staticmethod
    def _sampling(req, temperature, top_p, top_k, seed, stop, keep, presence_penalty=0.0, frequency_penalty=0.0):
        req.temperature, req.top_p, req.top_k, req.seed = float(temperature), float(top_p), int(top_k), int(seed)
        req.presence_penalty, req.frequency_penalty = float(presence_penalty), float(frequency_penalty)
        if stop:
            stops = [x.encode() for x in ([stop] if isinstance(stop, str) else stop)]
            keep.append(stops)
            arr = (C.c_char_p * len(stops))(*stops)
            keep.append(arr)
            req.stop, req.n_stop = arr, len(stops)

    def generate(self, tokens=None, text=None, max_new_tokens=256, ignore_eos=False, on_text=None,
                 temperature=0.0, top_p=1.0, top_k=0, seed=0, stop=None, presence_penalty=0.0,
                 frequency_penalty=0.0) -> list[int]:
        """Continue a token sequence (or raw text, after its tokenizer's prefix: Engine.tokenize's tokens).  Returns
        the new token ids.

        presence_penalty / frequency_penalty (the OpenAI definition): what a
        token already chosen in this reply loses from its logit, once and per
        time it was chosen."""
        req, keep = _Req(), []
        if tokens is not None and text is not None:
            raise ValueError("pass either tokens or text, not both")
        if tokens is not None:
            if isinstance(tokens, (str, bytes)) or hasattr(tokens, "keys"):
                raise TypeError("tokens must be a sequence of token ids (got %s)" % type(tokens).__name__)
            tokens = [int(t) for t in tokens]
            arr = (C.c_int32 * len(tokens))(*tokens)
            keep.append(arr)
            req.tokens, req.n_tokens = arr, len(tokens)
        elif text is not None:
            raw = text.encode()
            keep.append(raw)
            req.text = raw
        else:
            raise ValueError("generate() requires either tokens or text")
        req.max_new_tokens, req.ignore_eos = max_new_tokens, int(ignore_eos)
        self._sampling(req, temperature, top_p, top_k, seed, stop, keep, presence_penalty, frequency_penalty)
        return self._run(req, on_text)[0]

    @classmethod
    def _chat_req(cls, messages, max_new_tokens, enable_thinking, temperature, top_p, top_k, seed, stop,
                  presence_penalty, frequency_penalty, *, split_output=False, tools=None, tool_choice=None,
                  parallel_tool_calls=True, response_format=None):
        req, keep = _Req(), []
        mj = json.dumps(messages, ensure_ascii=False).encode()
        keep.append(mj)
        req.messages_json = mj
        if split_output:
            tj = json.dumps(tools, ensure_ascii=False).encode() if tools else None
            cj = json.dumps(tool_choice).encode() if tool_choice is not None else None
            fj = json.dumps(response_format).encode() if response_format is not None else None
            keep += [tj, cj, fj]
            req.tools_json, req.split_output = tj, 1
            req.tool_choice_json, req.response_format_json = cj, fj
            req.single_tool_call = int(not parallel_tool_calls)
        req.enable_thinking, req.max_new_tokens = int(enable_thinking), max_new_tokens
        cls._sampling(req, temperature, top_p, top_k, seed, stop, keep, presence_penalty, frequency_penalty)
        return req, keep

    def chat_full(self, messages, tools=None, max_new_tokens=1024, enable_thinking=True, temperature=0.0,
                  top_p=1.0, top_k=0, seed=0, stop=None, on_text=None, on_event=None, tool_choice=None,
                  parallel_tool_calls=True, response_format=None, presence_penalty=0.0,
                  frequency_penalty=0.0) -> dict:
        """One assistant turn.  Returns {"content", "reasoning", "tool_calls": [{"name", "arguments"}]}.

        tool_choice and response_format take the OpenAI values ("required",
        {"type": "function", "function": {"name": ...}}, {"type": "json_object"}, ...)."""
        req, _keep = self._chat_req(
            messages, max_new_tokens, enable_thinking, temperature, top_p, top_k, seed, stop,
            presence_penalty, frequency_penalty, split_output=True, tools=tools, tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls, response_format=response_format)
        _, content, reasoning, calls = self._run(req, on_text, on_event)
        return {"content": content, "reasoning": reasoning, "tool_calls": calls}

    def chat(self, messages, max_new_tokens=256, enable_thinking=True, on_text=None, *,
             temperature=0.0, top_p=1.0, top_k=0, seed=0, stop=None, presence_penalty=0.0,
             frequency_penalty=0.0) -> str:
        """One assistant turn as raw text (thinking included, if any)."""
        req, _keep = self._chat_req(
            messages, max_new_tokens, enable_thinking, temperature, top_p, top_k, seed, stop,
            presence_penalty, frequency_penalty)
        return self._run(req, on_text)[1]

    def cancel(self):
        with self._lock:
            self._e._lib.flyx_cancel(self._open_handle())

    def close(self):
        """Close the session. An active generation is cancelled and releases its handle when it returns."""
        with self._lock:
            if self._h:
                handle, self._h = self._h, None
                if self._running:
                    self._e._lib.flyx_cancel(handle)
                else:
                    self._e._lib.flyx_session_free(handle)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
