# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Load a .flyx file into the reference model."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "convert"))
from flyx_convert.fileformat import Reader  # noqa: E402

from .draft import Draft, draft_config_from_canonical  # noqa: E402
from .mtp import Mtp  # noqa: E402
from .target import Target, TargetConfig  # noqa: E402


READS = ("lora",)  # the parts a file may require (fileformat.py: format 2) that the reference reads


def load(path: str, dtype=torch.float64, device="cpu", with_mtp=False):
    with Reader(path) as r:
        unread = [p for p in r.requires if p not in READS]
        if unread:
            raise ValueError(f"{path}: requires {unread}, which the reference does not read")
        W = {k: v.to(device) for k, v in r.all_torch().items()}
        h = r.header
    cfg = TargetConfig.from_canonical(h["config"])
    a16 = {n for n, rec in h["tensors"].items() if rec["dtype"] == "nvfp4"}
    target = Target(cfg, W, dtype, a16, lora_scale=float((h.get("lora") or {}).get("scale", 1.0)))
    draft = Draft(draft_config_from_canonical(h["draft"]), W, dtype) if h.get("draft") else None
    if with_mtp:
        return target, draft, (Mtp(cfg, W, dtype) if h.get("mtp") else None), h, None
    return target, draft, h, None
