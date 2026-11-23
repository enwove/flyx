# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DV reference's loaders (python/reference/flyx_ref/dv.py) against the earlier CUDA port's export of the same
checkpoints.

    python -I tests/python/reference/check_dv_export.py CANVAS.pt SELECTOR.pt EXPORT_DIR

Run by hand: EXPORT_DIR is what the earlier CUDA port's python/export_dv.py wrote from CANVAS.pt and SELECTOR.pt
(canvas.safetensors, selector.safetensors and dv.json, whose source names the two checkpoints).  canvas_from_pt must
give the tensors of canvas.safetensors as dv.*, and selector_from_pt those of selector.safetensors as dv.sel.*: the
same names, dtypes, shapes and bytes.  Checkpoints are read with torch.load(weights_only=True).
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
from flyx_ref.dv import canvas_from_pt, selector_from_pt  # noqa: E402


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def compare(what, mine, export):
    """Prints how many of the export's tensors the loader gives byte for byte; returns whether all, and no others."""
    names = sorted(export)
    same = sum(k in mine and same_bytes(mine[k], export[k]) for k in names)
    extra = sorted(set(mine) - set(export))
    dtypes = sorted({str(v.dtype).removeprefix("torch.") for v in export.values()})
    ok = same == len(names) and not extra
    print(f"{'ok' if ok else 'FAIL'} {what}: {same}/{len(names)} of the export's tensors byte for byte "
          f"({', '.join(dtypes)})" + (f"; not in the export: {extra[:5]}" if extra else ""), flush=True)
    return ok


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    canvas_pt, selector_pt, export = sys.argv[1:]
    meta = json.load(open(os.path.join(export, "dv.json"), encoding="utf-8"))
    source = meta.get("source", {})
    print(f"{export}: made from {source.get('canvas')} and {source.get('selector')}; given "
          f"{os.path.basename(canvas_pt)} and {os.path.basename(selector_pt)}", flush=True)
    canvas, _ = canvas_from_pt(canvas_pt, meta["layers"])
    ok = compare("canvas_from_pt as dv.*",
                 canvas, {"dv." + k: v for k, v in load_file(os.path.join(export, "canvas.safetensors")).items()})
    ok &= compare("selector_from_pt as dv.sel.*", selector_from_pt(selector_pt),
                  {"dv.sel." + k: v for k, v in load_file(os.path.join(export, "selector.safetensors")).items()})
    print("dv loaders vs the export: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
