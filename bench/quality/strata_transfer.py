# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Measure a lossless transport format on a real flyx CUDA snapshot.

Input is a version-1 snapshot of the 27B's 16 target FP8 attention layers,
with a 2- or 3-bit Strata copy. Extra state following those layers is ignored.
Each 1,024-position page has an offset and one length per bitplane per K/V.
The page's grids travel verbatim. Each plane travels raw or in a zlib frame,
whichever fits in fewer bytes. All sizes count metadata, including a sentinel
offset. This is an offline byte count, with no timing of a GPU decoder.
"""
import argparse
from collections import Counter
import os
import struct
import zlib

import numpy as np


HEADS = 4
DIM = 256
PAGE = 1024


def addresses(bits, is_key):
    """Map a logical [position in tile, dimension] to StrataLayout word and bit."""
    pos, dim = np.ogrid[:16, :DIM]
    per_half = 16 // bits
    groups = 4 if is_key else 8
    pairs = DIM // (2 * groups)
    own, tail = divmod(pairs, per_half)
    sharing = per_half // tail if tail else 1
    shared = (groups + sharing - 1) // sharing if tail else 0
    words = groups * own + shared
    if is_key:
        group, pair, half = (dim // 2) % 4, dim // 8, dim % 2
        regular = ((pair // per_half) * 2 + pos // 8) * 32 + (pos % 8) * 4 + group
        trailing_group = pos
    else:
        group, pair, half = dim % 8, dim // 16, (dim // 8) % 2
        regular = (((pair // per_half) * 2 + pos // 8) * 2 + pos % 2) * 32 + group * 4 + (pos % 8) // 2
        trailing_group = (pos // 8 * 2 + pos % 2) * 4 + (pos % 8) // 2
    trailing = groups * own * 16 + trailing_group * shared + group // sharing
    word = np.where(pair < own * per_half, regular, trailing)
    slot = np.where(pair < own * per_half, pair % per_half, (group % sharing) * tail + pair - own * per_half)
    shift = 16 * half + bits * slot
    # Each field is disjoint, and accounts for every dimension of the tile.
    used = word * 32 + shift + np.arange(bits)[:, None, None]
    assert used.min() == 0 and used.max() < 16 * words * 32 and np.unique(used).size == used.size
    return words, word, shift


def measure_page(tile_words, word, shift, bits):
    codes = ((tile_words[:, :, word] >> shift) & ((1 << bits) - 1)).astype(np.uint8)
    codes = codes.transpose(0, 2, 1, 3).reshape(-1, HEADS, DIM)
    gray = codes ^ (codes >> 1)
    restored = np.zeros_like(codes)
    total = 0
    for bit in reversed(range(bits)):
        plane = np.packbits((gray >> bit) & 1, axis=0).transpose(1, 2, 0).tobytes()
        compressed = zlib.compress(plane, level=1)
        if len(compressed) < len(plane):
            total += len(compressed)
            plane = zlib.decompress(compressed)
        else:
            total += len(plane)
        decoded = np.frombuffer(plane, np.uint8).reshape(HEADS, DIM, -1).transpose(2, 0, 1)
        restored |= (np.unpackbits(decoded, axis=0) ^ ((restored >> (bit + 1)) & 1)) << bit
    assert np.array_equal(codes, restored)
    return total + 4 * bits  # each plane's byte length; raw length identifies a raw plane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("bits", type=int, choices=(2, 3))
    args = parser.parse_args()
    layouts = [addresses(args.bits, is_key) for is_key in (True, False)]
    sums = Counter()
    with open(args.snapshot, "rb") as stream:
        magic, version, _, pos, _, size = struct.unpack("<IIQiiQ", stream.read(32))
        if magic != 0x504E5346 or version != 1 or pos <= 0 or size != os.fstat(stream.fileno()).st_size:
            raise ValueError("expected a complete version-1 CUDA snapshot")
        print(f"positions {pos}, bits {args.bits}, page {PAGE}, zlib {zlib.ZLIB_RUNTIME_VERSION}")
        tiles = (pos + 15) // 16
        for layer in range(16):
            stream.seek(2 * pos * HEADS * (DIM + 4), os.SEEK_CUR)  # exact FP8 K and V and f32 scales
            raw = 0
            wire = 4 * ((tiles * 16 + PAGE - 1) // PAGE + 1)  # page offsets and end of last page
            for words, index, shift in layouts:
                data = np.fromfile(stream, dtype="<u4", count=tiles * HEADS * 16 * words)
                data = data.reshape(tiles, HEADS, 16 * words)
                raw += data.nbytes
                # Verify no nonzero padding would be lost by tight repacking.
                mask = np.zeros(16 * words, dtype=np.uint32)
                np.bitwise_or.at(mask, index.ravel(), np.broadcast_to(
                    ((1 << args.bits) - 1) << shift, index.shape).astype(np.uint32).ravel())
                if np.any(data & ~mask):
                    raise ValueError(f"unexpected padding in layer {layer}, check snapshot format and bit width")
                for start in range(0, tiles, PAGE // 16):
                    wire += measure_page(data[start:start + PAGE // 16], index, shift, args.bits)
            grids = np.fromfile(stream, dtype="<f2", count=pos * HEADS * 4)
            if grids.size != pos * HEADS * 4 or not np.isfinite(grids).all() or (grids[1::2] < 0).any():
                raise ValueError(f"invalid grids in layer {layer}, check cache type and bit width")
            raw += grids.nbytes
            wire += grids.nbytes
            sums.update(raw=raw, wire=wire)
            print(f"layer {layer:2}  device {raw}  wire {wire}  fraction {wire / raw:.4f}")
    print(f"total device {sums['raw']}  wire {sums['wire']}  fraction {sums['wire'] / sums['raw']:.4f}")


if __name__ == "__main__":
    main()
