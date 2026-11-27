# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the converter and the candidate cutoff on the tiny DSpine companion."""
import copy
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root / "python" / d) for d in ("convert", "reference")]
from flyx_convert.dspine import canonical_dspine_config
from flyx_ref.load import load


def main(assets):
    config = json.loads((assets / "hf_dspine_window" / "config.json").read_text(encoding="utf-8"))
    canonical_dspine_config(config, None)
    for section, keys in (
            ("selected_prefix_code_bridge", ("code_size", "rank", "eps", "kappa")),
            ("selected_prefix_last_write_refine", ("version", "head_topk", "weight")),
            ("code_bridge_second_pass", ("head_topk", "first_exact"))):
        for key in keys:
            bad = copy.deepcopy(config)
            parent = bad if section == "code_bridge_second_pass" else bad["dflash_config"]
            parent[section][key] = True
            try:
                canonical_dspine_config(bad, None)
            except ValueError:
                continue
            raise AssertionError(f"{section}.{key} accepted a JSON boolean")

    _, draft, _, _ = load(assets / "tiny_dspine.flyx")
    draft.head = torch.zeros_like(draft.head)  # every score ties, including the first excluded candidate
    ids, debug = draft.block(draft.new_state(), 5)
    expected = torch.arange(draft.cfg.topk).expand(draft.block_drafts, -1)
    assert torch.equal(debug["cand"], expected) and ids == [0] * draft.block_drafts


if __name__ == "__main__":
    main(Path(sys.argv[1]))
