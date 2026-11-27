# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The reference block drafter's candidates on tied logits: the lowest token ids first, as ops/sampling/ref.h takes
them (tiny.flyx's DFlash2 and tiny_dflash.flyx's DFlash, whose walk over one candidate is the head's best).

    python tests/python/reference/draft_test.py ASSET_DIR
"""
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root / "python" / "reference")]
from flyx_ref.load import load  # noqa: E402


def main(assets):
    for name in ("tiny.flyx", "tiny_dflash.flyx"):
        _, draft, _, _ = load(assets / name)
        draft.head = torch.zeros_like(draft.head)  # every logit ties
        ids, debug = draft.block(draft.new_state(), 5)
        expected = torch.arange(draft.cfg.sel_topk).expand(draft.block_drafts, -1)
        assert torch.equal(debug["cand"], expected), (name, debug["cand"])
        if not draft.cfg.sel_rank:
            assert ids == [0] * draft.block_drafts, (name, ids)
    print("draft_test: PASSED")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
