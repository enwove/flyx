# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The .flyx model file: one self-describing file per model.

Layout (all integers little-endian):

    0   8 bytes  magic  b"FLYX" and 4 zero bytes (files from before the name flyx: OLD_MAGIC, read the same)
    8   u32      format version (FORMAT_VERSION; REQUIRES_VERSION for a header with ``requires``)
    12  u32      reserved (0)
    16  u64      header length in bytes
    24  header   UTF-8 JSON
    ...          zero padding up to a multiple of ALIGN_DATA -> data start
    data         tensors and blobs; every entry starts at a multiple of ALIGN_ENTRY
                 relative to data start; offsets in the header are relative to it

Quantized linears (NVFP4, from ModelOpt checkpoints) are three tensors:
``NAME`` dtype ``nvfp4`` with the logical shape [N, K] (two E2M1 values per
byte, low nibble first), ``NAME.scale`` dtype ``f8e4m3`` [N, K/16] (one scale
per 16 values) and ``NAME.scale2`` f32 [1]:  w = e2m1 * scale * scale2.

Header keys: ``arch``, ``config`` (canonical, what the engine reads),
``draft`` (canonical drafter config or null), ``tokenizer``, ``generation``,
``tensors`` {name: {dtype, shape, offset, nbytes}}, ``blobs`` {name: {offset,
nbytes}}, ``source`` (provenance only); ``requires`` (format 2 only): the
parts of the file a reader must not ignore, by name (``lora``: the target's
own adapters, which a reader that dropped would decode another model with).
A file with none of them is format 1, which every reader reads; a reader of
format 1 only refuses format 2.

A file packed from a GGUF (``flyx pack``, core/modelfile/flyx_pack.cpp) requires
``gguf``: its header is the C++ GGUF reader's (``gguf``, ``source.format``
``gguf``), its tensors GGML blocks (dtype ``ggml_<type>``) under their GGUF
names, its drafters ``bundled`` files ({role, header, tensors}) whose tensors
are in the same data section, and ``recommended`` the settings measured best
for it.  This module neither writes nor reads such files.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import mmap
import os
from pathlib import Path
import stat
import struct
import tempfile

import numpy as np

MAGIC = b"FLYX\0\0\0\0"
OLD_MAGIC = bytes([0x46, 0x41, 0x53, 0x54, 0x45, 0x53, 0x54, 0])  # the project's name before flyx
FORMAT_VERSION = 1
REQUIRES_VERSION = 2
ALIGN_DATA = 4096
ALIGN_ENTRY = 256

DTYPES = {"bf16": 2, "f16": 2, "f32": 4, "f64": 8, "i32": 4, "u32": 4, "u8": 1, "i64": 8, "f8e4m3": 1,
          "nvfp4": 0}


def expected_nbytes(dtype: str, shape) -> int:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    if dtype == "nvfp4":
        if numel % 2 != 0:
            raise ValueError(f"nvfp4 shape {list(shape)} has odd element count {numel}")
        return numel // 2
    return numel * DTYPES[dtype]
NUMPY = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "i32": np.int32,
         "u32": np.uint32, "u8": np.uint8, "i64": np.int64, "bf16": np.uint16, "f8e4m3": np.uint8,
         "nvfp4": np.uint8}

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def e4m3_to_float(b: np.ndarray) -> np.ndarray:
    """FP8 E4M3 (fn: no infinities) bytes -> float64."""
    b = b.astype(np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    return s * v


def dequant_nvfp4(packed: np.ndarray, scale: np.ndarray, scale2: float) -> np.ndarray:
    """[N, K/2] bytes, [N, K/16] e4m3, scalar -> [N, K] float64."""
    lo, hi = E2M1[packed & 0xF], E2M1[packed >> 4]
    w = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)
    return w * np.repeat(e4m3_to_float(scale), 16, axis=1) * float(scale2)


def _check_requires(parts, path: str) -> None:
    if not isinstance(parts, list) or not parts or not all(isinstance(p, str) and p for p in parts):
        raise ValueError(f"{path}: 'requires' {parts!r} is not a list of the parts a reader must not ignore")


def _pad(n: int, a: int) -> int:
    return (n + a - 1) // a * a


@contextmanager
def _output_file(path: str):
    target = Path(path)
    try:
        mode = target.stat().st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None:
        if not stat.S_ISREG(mode):  # pipes and devices receive the stream directly
            with open(target, "wb") as f:
                yield f
            return
        os.close(os.open(target, os.O_WRONLY))  # respect an existing file's write permissions
    target = target.resolve()  # a symlink to the model keeps pointing to it
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".flyx-") as directory:
        staged = Path(directory) / "model"
        with open(staged, "wb") as f:
            yield f
        if mode is not None:
            staged.chmod(stat.S_IMODE(mode))
        os.replace(staged, target)


class Writer:
    """Collect tensors (raw little-endian bytes) and blobs, then write once."""

    def __init__(self):
        self.entries: list[tuple[str, str, bytes | memoryview, dict]] = []
        self.meta: dict = {}

    def add_tensor(self, name: str, dtype: str, shape, data) -> None:
        data = memoryview(data).cast("B")
        need = expected_nbytes(dtype, shape)
        if data.nbytes != need:
            raise ValueError(f"{name}: {data.nbytes} bytes, expected {need} for {dtype}{list(shape)}")
        self.entries.append(("tensor", name, data, {"dtype": dtype, "shape": [int(s) for s in shape]}))

    def add_blob(self, name: str, data: bytes) -> None:
        self.entries.append(("blob", name, memoryview(data).cast("B"), {}))

    def write(self, path: str) -> None:
        tensors, blobs, off = {}, {}, 0
        for kind, name, data, info in self.entries:
            off = _pad(off, ALIGN_ENTRY)
            rec = dict(info, offset=off, nbytes=data.nbytes)
            (tensors if kind == "tensor" else blobs)[name] = rec
            off += data.nbytes
        header = dict(self.meta, tensors=tensors, blobs=blobs)
        if "requires" in header:
            _check_requires(header["requires"], path)
        version = REQUIRES_VERSION if "requires" in header else FORMAT_VERSION
        hj = json.dumps(header, ensure_ascii=False, sort_keys=True).encode()
        data_start = _pad(24 + len(hj), ALIGN_DATA)
        with _output_file(path) as f:
            f.write(MAGIC + struct.pack("<IIQ", version, 0, len(hj)) + hj)
            f.write(b"\0" * (data_start - 24 - len(hj)))
            pos = 0
            for _kind, _name, data, _info in self.entries:
                aligned = _pad(pos, ALIGN_ENTRY)
                f.write(b"\0" * (aligned - pos))
                f.write(data)
                pos = aligned + data.nbytes


class Reader:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        try:
            self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
            if self.mm[:8] not in (MAGIC, OLD_MAGIC):
                raise ValueError(f"{path}: not a .flyx file")
            ver, _, hlen = struct.unpack("<IIQ", self.mm[8:24])
            if ver not in (FORMAT_VERSION, REQUIRES_VERSION):
                raise ValueError(f"{path}: format version {ver}, expected {FORMAT_VERSION} or {REQUIRES_VERSION}")
            self.version = ver
            self.header = json.loads(self.mm[24:24 + hlen])
            if ("requires" in self.header) != (ver == REQUIRES_VERSION):
                raise ValueError(f"{path}: format version {ver} {'with' if 'requires' in self.header else 'without'} "
                                 f"'requires' (format {REQUIRES_VERSION}'s only, and always)")
            if ver == REQUIRES_VERSION:
                _check_requires(self.header["requires"], path)
            self.data_start = _pad(24 + hlen, ALIGN_DATA)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        mm = getattr(self, "mm", None)
        if mm is not None:
            mm.close()
            self.mm = None
        f = getattr(self, "f", None)
        if f is not None:
            f.close()
            self.f = None

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def requires(self) -> list:
        """The parts of the file a reader must not ignore (format 2), by name."""
        return list(self.header.get("requires", []))

    def raw(self, rec) -> memoryview:
        a = self.data_start + rec["offset"]
        return memoryview(self.mm)[a:a + rec["nbytes"]]

    def numpy(self, name: str) -> np.ndarray:
        rec = self.header["tensors"][name]
        shape = list(rec["shape"])
        if rec["dtype"] == "nvfp4":
            shape[-1] //= 2
        return np.frombuffer(self.raw(rec), dtype=NUMPY[rec["dtype"]]).reshape(shape)

    def dequant(self, name: str) -> np.ndarray:
        """An nvfp4 tensor as float64 [N, K]."""
        return dequant_nvfp4(self.numpy(name), self.numpy(name + ".scale"), float(self.numpy(name + ".scale2")[0]))

    def torch(self, name: str):
        import torch
        rec = self.header["tensors"][name]
        a = self.numpy(name)
        if rec["dtype"] == "bf16":
            return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(a.copy())

    def blob(self, name: str) -> bytes:
        return bytes(self.raw(self.header["blobs"][name]))

    def all_torch(self) -> dict:
        """Every tensor; nvfp4 linears dequantized to float64 (their scales folded in)."""
        import torch
        out = {}
        for n, rec in self.header["tensors"].items():
            if rec["dtype"] == "nvfp4":
                out[n] = torch.from_numpy(self.dequant(n))
            elif n.endswith(".scale") or n.endswith(".scale2"):
                base = n.rsplit(".", 1)[0]
                if self.header["tensors"].get(base, {}).get("dtype") == "nvfp4":
                    continue
                out[n] = self.torch(n)
            else:
                out[n] = self.torch(n)
        return out
