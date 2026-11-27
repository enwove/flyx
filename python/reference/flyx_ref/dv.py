# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""DV drafting on a Llama target (reference): MiniCPM5-1B, and tiny models shaped like it.

DV is bifill-dv's drafter (docs/prior-art.md).  Each round is one forward of two groups of rows:

* the target's rows: the anchor (the token the last round chose, at position p, the first not yet in the cache) and
  a tree of T drafts, each at p + its depth, seeing the cache and its ancestors.  Target weights, and the target's
  own LoRA (V-LoRA, ``target.L{l}.{attn|mlp}.{name}.lora_{a,b}``: ``y = W x + lora_scale B (A x)``, unmerged, as
  bifill-dv's evalgen.py VLORA runs raft.py's WithLoRA, whose scale is 1) when it has one.  The deepest draft whose path
  matches the target's choices is accepted (the lower index among the deepest); the target's choice after it is the
  next anchor.  The round commits the anchor and the accepted path: 1 + a tokens.
* the canvas: N slots at p + 1 .. p + N, seeing the cache, the anchor and every slot (only the slots before them if
  causal), never the tree.  The target's base weights (never its V-LoRA) plus the canvas adapter (``dv.*``): a
  rank-r LoRA ``y += B (A x)`` on every linear, its own norms (``ln1c``, ``ln2c``, ``fnormc``), and the input

      x_w = (cv_w == MASK ? mask_emb : E[cv_w]) + res_out (silu(g) * u),   [u | g] = res_in z_w

  where z_w is the slot's expected embedding from the round before (0 for a new slot).  The canvas's logits give
  every slot its top 16 (ties to the lower id), their probabilities p, and z' = sum_k p_k E[i_k].

After the check the canvas slides by the n tokens committed: slot w takes slot w + n (``slide``; a slot past the
canvas holds no candidates).  The next tree grows from the slid canvas with the selector (``selector_table``):

    S_s(prev, b) = alpha * log max(p_s[b], 1e-9) + <A E[prev] * H h_s, B E[cand_s[b]]> / sqrt(rank)

(h_s: the slot's final-normed canvas hidden state; prev: the anchor for slot 0, else one of the slot before's
candidates), as log-softmax over the 16 b, then either tree:

* ``beam_tree`` (bifill's DYN=T:W[:cov], selector.py dyn_tree_fast): a beam of W nodes per depth by the summed
  log-probabilities (plus with cov each slot's log top-16 mass, minus 1e-4 a depth), then the T best beam nodes;
* ``pool_tree`` (ST T/R/m=1, the earlier CUDA port's dv.cu): a fixed pool of rank tuples that leave the selector's
  best path at most once, at rank <= R; the T best by summed log-probability, minus 1e-6 a depth.

The first round after the prefill drafts T MASKs under the anchor (a star; the canvas is all MASK, z = 0), so it
commits the anchor alone.  ``Dv.generate`` follows bifill-dv's evalgen.py mode DV (FAST_STEP's _tree_post) round for
round, computed in the dtype asked for (float64 for the tiny models' golden drafts, float32 for MiniCPM5 on a CPU,
bfloat16 like bifill's engine); ties everywhere go to the lower index (torch.topk on a GPU leaves them open).  With
``kv_round`` and ``canvas_round`` it rounds what flyx's CUDA backend rounds, the rest in its dtype: the K and V the
attention reads (binary16 caches, or e4m3 ones with a scale per position and head), and the inputs of the canvas's
linears and head (bf16: its prefill GEMMs, a canvas of more than 16 slots).

Weights use the canonical names of python/convert (``target.*``) plus ``dv.L{l}.{q,k,v,o,gate,up,down}.lora_{a,b}``,
``dv.L{l}.ln1c``, ``dv.L{l}.ln2c``, ``dv.fnormc``, ``dv.mask_emb``, ``dv.res_in`` [2D, D], ``dv.res_out`` [D, D]
and the selector ``dv.sel.{A,H,B}`` [rank, D], ``dv.sel.alpha`` [1] (float32).  ``hf_llama``, ``canvas_from_pt``,
``selector_from_pt`` and ``vlora_from_pt`` read them from a Hugging Face directory and bifill-dv's checkpoints;
``from_flyx`` from a .flyx file python/convert wrote with --dv (the weights as stored, the DvConfig its header names),
``llama_from_flyx`` from any Llama .flyx file (plain decoding, a V-LoRA alone: python/convert --lora).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .target import rope

LOG_FLOOR = 1e-9      # canvas probabilities are clamped here before their log (evalgen.py: clamp(min=1e-9).log())
MASS_FLOOR = 1e-6     # cov: log max(top-16 mass, this)
NO_SLOT = -1e9        # beam: an edge into a slot without candidates
BEAM_PAD = -2e9       # beam: the places a depth's children cannot fill (fewer than W of them)
UNDRAFTED = -1e8      # beam: a node worth no more than this drafts MASK
BEAM_STEP = 1e-4      # beam: what a depth costs, so a parent is worth strictly more than its children
POOL_DEAD = -1e4      # pool: a node at a slot without candidates
POOL_STEP = 1e-6      # pool: what a depth costs in the selection


@dataclass
class LlamaConfig:
    hidden: int
    layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    inter: int
    vocab: int
    eps: float
    rope_theta: float
    lora_scale: float = 1.0     # the target's own adapters' (a V-LoRA's): y = W x + lora_scale B (A x)

    @staticmethod
    def from_hf(c: dict) -> "LlamaConfig":
        rp = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if rp.get("rope_type", rp.get("type", "default")) != "default":
            raise ValueError(f"rope {rp} is not supported (only the default rotary)")
        return LlamaConfig(hidden=c["hidden_size"], layers=c["num_hidden_layers"], n_heads=c["num_attention_heads"],
                           n_kv=c["num_key_value_heads"],
                           head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                           inter=c["intermediate_size"], vocab=c["vocab_size"], eps=c["rms_norm_eps"],
                           rope_theta=float(rp.get("rope_theta", c.get("rope_theta", 10000.0))))

    @staticmethod
    def from_canonical(c: dict) -> "LlamaConfig":
        if c["rot_dim"] != c["head_dim"]:
            raise ValueError("a partial rotary is not supported")
        return LlamaConfig(hidden=c["hidden"], layers=c["layers"], n_heads=c["n_heads"], n_kv=c["n_kv"],
                           head_dim=c["head_dim"], inter=c["inter"], vocab=c["vocab"], eps=c["eps"],
                           rope_theta=float(c["rope_theta"]))


@dataclass
class DvConfig:
    slots: int                  # N canvas slots
    mask_token: int
    width: int = 31             # T drafts a round
    tree: str = "beam"          # "beam": DYN=width:beam[:cov]; "pool": ST width/pool_r/1
    beam: int = 8
    cov: bool = True
    pool_r: int = 7
    topk: int = 16              # candidates a slot keeps (bifill's SEL_K and res_k: one number here)
    fill: float = 0.9           # a slot keeps its top token as the next canvas input from this probability on
    causal: bool = False        # canvas slots see only the slots before them
    temp: float = 1.0           # the selector's scores are divided by this before the softmax

    @property
    def depths(self) -> int:
        """Slots a tree reaches.  Slot N - 1 never holds candidates after a slide; and a node deeper than T has T
        ancestors worth more, so trees of T never reach past depth T (bifill's BEAM_CAP: the same trees)."""
        return min(self.slots - 1, self.width)

    def set_tree(self, spec: str) -> "DvConfig":
        """beam:T:W[:cov] or pool:T:R[:1]."""
        parts = spec.split(":")
        if parts[0] == "beam" and len(parts) in (3, 4):
            self.tree, self.width, self.beam = "beam", int(parts[1]), int(parts[2])
            self.cov = len(parts) == 4 and parts[3] == "cov"
            if len(parts) == 4 and parts[3] != "cov":
                raise ValueError(f"tree {spec!r}: the fourth field can only be cov")
        elif parts[0] == "pool" and len(parts) in (3, 4):
            if len(parts) == 4 and parts[3] != "1":
                raise ValueError(f"tree {spec!r}: pools leave the best path at most once (m = 1)")
            self.tree, self.width, self.pool_r = "pool", int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"tree {spec!r}: beam:T:W[:cov] or pool:T:R[:1]")
        return self

    @classmethod
    def from_header(cls, d: dict) -> "DvConfig":
        """A .flyx header's dv section (python/convert/flyx_convert/dv.py) -> the DvConfig a run with the same settings
        uses (set_tree): the fields of the other tree kind keep their defaults."""
        t = d["tree"]
        dv = cls(slots=d["slots"], mask_token=d["mask_token"], width=t["width"], topk=d["topk"], fill=d["fill"],
                 causal=bool(d["causal"]), temp=d.get("temp", 1.0))
        if t["kind"] == "beam":
            dv.tree, dv.beam, dv.cov = "beam", t["beam"], bool(t["cov"])
        elif t["kind"] == "pool":
            dv.tree, dv.pool_r = "pool", t["ranks"]
        else:
            raise ValueError(f"DV tree kind {t['kind']!r}: beam or pool")
        return dv


# ---- the pieces, each on its own (unit tests and the C++ references' cases use them) ------------------------------

def rms(x, w, eps):
    """LlamaRMSNorm with a given weight: bfloat16 rows are normed in float32 and rounded before the weight."""
    xf = x.float() if x.dtype == torch.bfloat16 else x
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def attend(q, k, v, mask):
    """q [R, H, d], k/v [C, Hkv, d], mask [R, C] (True: seen).  bfloat16 in float32, as SDPA does inside."""
    dt = q.dtype
    if dt == torch.bfloat16:
        q, k, v = q.float(), k.float(), v.float()
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), v).to(dt)


def canvas_input(tokens, z, embed, mask_emb, res_in, res_out, mask_token):
    """x_w = (tokens_w == MASK ? mask_emb : E[tokens_w]) + res_out (silu(g) * u), [u | g] = res_in z_w (bifill-dv
    dv.py CanvasAdapter.embed).  tokens [N], z [N, D] -> [N, D]."""
    x = torch.where((tokens == mask_token)[:, None], mask_emb.to(embed.dtype)[None], embed[tokens])
    u, g = F.linear(z, res_in).chunk(2, -1)
    return x + F.linear(F.silu(g) * u, res_out).to(x.dtype)


def topk_low(x, k):
    """Each row's k largest, highest first, ties to the lower index: [R, V] -> indices [R, k]."""
    kth = torch.topk(x, k, dim=-1).values[:, -1:]
    out = []
    for r in range(x.shape[0]):
        idx = (x[r] >= kth[r]).nonzero()[:, 0]            # every index as high as the k-th, in ascending order
        out.append(idx[torch.sort(x[r, idx], descending=True, stable=True).indices[:k]])
    return torch.stack(out)


def canvas_stats(logits, embed, k):
    """A canvas's logits [N, V] -> its top k ids (ties to the lower id), their probabilities under the whole
    vocabulary (not renormalised), the log-sum-exp, and the expected embedding z' = sum_k p_k E[i_k] (evalgen.py
    _dv_tree_heads; in bfloat16 its products are bfloat16 as there)."""
    lf = logits.float() if logits.dtype == torch.bfloat16 else logits
    lse = torch.logsumexp(lf, -1)
    ids = topk_low(lf, k)
    p = torch.exp(lf.gather(-1, ids) - lse[:, None])
    zn = torch.einsum("wk,wkd->wd", p.to(embed.dtype), embed[ids])
    return {"ids": ids, "p": p, "lse": lse, "zn": zn}


def slide(stats, hid, n, fill, mask_token):
    """The canvas after a round that committed n tokens: slot w takes this round's slot j = w + n.  A slot past the
    canvas (j >= N) holds no candidates (ok = False), an input of MASK and z = 0; its candidates, log-probabilities
    and hidden state are slot N - 1's (clamped, as evalgen.py gathers them; no tree reads them).  The slot's input is
    its top token when that has probability >= fill, else MASK (evalgen.py _tree_post)."""
    ids, p, zn = stats["ids"], stats["p"], stats["zn"]
    N = ids.shape[0]
    j = torch.arange(N) + n
    ok, jc = j < N, j.clamp(max=N - 1)
    top, conf = ids[jc, 0], p[jc, 0]
    return {"cv": torch.where(ok & (conf >= fill), top, torch.full_like(top, mask_token)),
            "z": zn[jc] * ok[:, None].to(zn.dtype),
            "cand": ids[jc], "logp": p.clamp(min=LOG_FLOOR).log()[jc], "ok": ok, "hid": hid[jc]}


def fresh_canvas(N, k, hidden, mask_token, dtype):
    """The canvas after a prefill (or anything that leaves it unknown): MASK inputs, z = 0, no candidates."""
    return {"cv": torch.full((N,), mask_token, dtype=torch.long), "z": torch.zeros(N, hidden, dtype=dtype),
            "cand": torch.full((N, k), mask_token, dtype=torch.long), "logp": torch.full((N, k), math.log(LOG_FLOOR)),
            "ok": torch.zeros(N, dtype=torch.bool), "hid": torch.zeros(N, hidden, dtype=dtype)}


@dataclass
class Selector:
    """bifill-dv selector.py's PathSelector on the target's embedding table: A, H, B [rank, D], alpha."""
    embed: torch.Tensor
    A: torch.Tensor
    H: torch.Tensor
    B: torch.Tensor
    alpha: torch.Tensor
    temp: float = 1.0

    @property
    def dtype(self):
        return self.A.dtype


def selector_table(sel: Selector, anchor, cand, logp, hid):
    """Every slot's log-probabilities over its candidates given each candidate of the slot before as the token
    before (slot 0: the anchor, its rows all the same): [S, K (before), K].  PathSelector.table's arithmetic, in the
    selector's dtype (float32 there; float64 for the float64 reference).  cand/logp [S, K], hid [S, D]."""
    e = lambda t: sel.embed[t].to(sel.dtype)
    K = cand.shape[-1]
    qh = F.linear(hid.to(sel.dtype)[None], sel.H)                                     # [1, S, r]
    ec = e(cand)[None]
    qa = F.linear(ec, sel.A)                                                          # [1, S, K, r]
    prev = torch.cat([F.linear(e(torch.as_tensor([anchor])), sel.A)[:, None, None].expand(-1, 1, K, -1),
                      qa[:, :-1]], 1)
    q = prev * qh[:, :, None]
    s = torch.einsum("bsir,bsjr->bsij", q, F.linear(ec, sel.B)) / math.sqrt(sel.A.shape[0])
    return ((s + sel.alpha * logp.to(sel.dtype)[None, :, None, :]) / sel.temp).log_softmax(-1)[0]


def beam_edges(table, logp, ok, cov):
    """The beam's edge values: the table, plus with cov each slot's log top-k mass log max(sum_k p_k, 1e-6), and
    NO_SLOT into a slot without candidates (PathSelector.table(cov=...))."""
    if cov:
        table = table + logp.to(table.dtype).exp().sum(-1).clamp(min=MASS_FLOOR).log()[:, None, None]
    return torch.where(ok[:, None, None], table, torch.full_like(table, NO_SLOT))


def beam_tree(edges, cand, width, beam, mask_token):
    """bifill's dynamic tree (selector.py dyn_tree_fast) on edge values [D, K, K] (``beam_edges``).  Depth d + 1
    keeps the W best children of depth d's beam: a child's value is (its parent's + edge) - 1e-4, the anchor's 0;
    fewer than W children are padded with BEAM_PAD (a padded place continues the last real child: its parent and
    candidate).  The drafts are the T best of all D * W beam nodes, in beam order (depth-major, then beam rank); a
    node worth <= -1e8 drafts MASK.  Ties go to the lower index everywhere.  Needs D * W >= T.
    Returns drafts [T], parents [T] (the draft each continues, -1: the anchor), depths [T]."""
    D, K = edges.shape[0], edges.shape[-1]
    if D * beam < width:
        raise ValueError(f"a beam of {beam} over {D} depths has fewer than {width} nodes")
    bval = torch.zeros(1, dtype=edges.dtype)
    bidx = torch.zeros(1, dtype=torch.long)           # each beam node's candidate at its slot
    bid = torch.zeros(1, dtype=torch.long)            # its id: 0 the anchor, 1 + d * W + rank
    toks, vals, pars = [], [], []
    for d in range(D):
        m = bval.shape[0]
        flat = (bval[:, None] + edges[d][bidx] - BEAM_STEP).reshape(-1)
        if flat.shape[0] < beam:
            flat = torch.cat([flat, torch.full((beam - flat.shape[0],), BEAM_PAD, dtype=flat.dtype)])
        i = torch.sort(flat, descending=True, stable=True).indices[:beam]
        bval = flat[i]
        i = i.clamp(max=m * K - 1)
        pars.append(bid[i // K])
        bidx = i % K
        toks.append(cand[d][bidx])
        vals.append(bval)
        bid = 1 + d * beam + torch.arange(beam)
    tok, val, par = torch.cat(toks), torch.cat(vals), torch.cat(pars)
    pick = torch.sort(torch.sort(val, descending=True, stable=True).indices[:width]).values
    at = torch.zeros(1 + D * beam, dtype=torch.long)
    at[1 + pick] = torch.arange(1, width + 1)
    drafts = torch.where(val[pick] > UNDRAFTED, tok[pick], torch.full_like(pick, mask_token))
    return drafts, at[par[pick]] - 1, pick // beam + 1


def pool_layout(D, R):
    """bifill-dv's TreePool(D, R, m=1) tuples, depth-major (the earlier CUDA port's tree_pool): at depth d the
    all-zero tuple (the selector's best path), then for each slot s < d each rank 1..R at s.  A tuple's parent is
    the same deviation one depth up, or the all-zero tuple when the deviation is at its last slot; lastr is the rank
    of its own candidate under its parent's (R at its deviation, else 0).  Returns depth, parent, lastr, offsets."""
    depth, parent, lastr, off = [], [], [], []
    for d in range(1, D + 1):
        off.append(len(depth))
        prev = off[d - 2] if d > 1 else -1               # depth d - 1's all-zero tuple
        depth.append(d); parent.append(prev); lastr.append(0)
        for s in range(d):
            for r in range(1, R + 1):
                depth.append(d)
                parent.append(-1 if d == 1 else (prev if s == d - 1 else prev + 1 + s * R + (r - 1)))
                lastr.append(r if s == d - 1 else 0)
    off.append(len(depth))
    return torch.tensor(depth), torch.tensor(parent), torch.tensor(lastr), off


def pool_tree(table, cand, ok, width, R, mask_token):
    """The selector-scored pool tree (ST, the earlier CUDA port's k_dv_tree / bifill-dv's SelTreePool.select) on the
    selector's log-softmax table [D, K, K] (no cov).  A tuple's candidate is the lastr-th best (stable, high to low)
    under its parent's candidate, and its value is its parent's + that log-probability; at a slot without candidates
    every rank is worth 0 at candidate index = rank, plus POOL_DEAD, and drafts MASK.  The drafts are the T best by
    value - 1e-6 * depth (ties: the lower pool index, so a parent before its children), in pool order.
    Returns drafts [T], parents [T] (-1: the anchor), depths [T]."""
    D, K = table.shape[0], table.shape[-1]
    depth, parent, lastr, off = pool_layout(D, R)
    if off[-1] < width:
        raise ValueError(f"a pool of {off[-1]} tuples cannot give {width} drafts")
    sv, si = torch.sort(table, dim=-1, descending=True, stable=True)
    idx = torch.zeros(off[-1], dtype=torch.long)
    cs = torch.zeros(off[-1], dtype=table.dtype)
    for d in range(1, D + 1):
        a, b = off[d - 1], off[d]
        lr = lastr[a:b]
        row = torch.zeros(b - a, dtype=torch.long) if d == 1 else idx[parent[a:b]]
        if bool(ok[d - 1]):
            idx[a:b], v = si[d - 1][row, lr], sv[d - 1][row, lr]
        else:
            idx[a:b], v = lr, torch.zeros(b - a, dtype=table.dtype)
        base = torch.zeros(b - a, dtype=table.dtype) if d == 1 else cs[parent[a:b]]
        cs[a:b] = base + v + (0.0 if bool(ok[d - 1]) else POOL_DEAD)
    score = cs - POOL_STEP * depth.to(table.dtype)
    sel = torch.sort(torch.sort(score, descending=True, stable=True).indices[:width]).values
    node = torch.full((off[-1],), -1, dtype=torch.long)
    node[sel] = torch.arange(width)
    dep = depth[sel]
    tok = torch.where(ok[dep - 1], cand[dep - 1, idx[sel]], torch.full_like(sel, mask_token))
    par = torch.where(dep > 1, node[parent[sel].clamp(min=0)], torch.full_like(sel, -1))
    return tok, par, dep


def accept(drafts, parents, depths, choice):
    """The exact check (evalgen.py _tree_post): a draft is right when it is the target's choice after its parent
    (choice[0] after the anchor, choice[i + 1] after draft i), taken when its whole path is right; the deepest taken
    draft wins, the lower index among the deepest.  Returns the accepted path (draft indices by depth)."""
    taken, best = [], -1
    for t in range(len(drafts)):
        p = int(parents[t])
        taken.append(int(drafts[t]) == int(choice[p + 1]) and (p < 0 or taken[p]))
        if taken[t] and (best < 0 or int(depths[t]) > int(depths[best])):
            best = t
    path = []
    while best >= 0:
        path.append(best)
        best = int(parents[best])
    return path[::-1]


def rounds_needed(accepted, length):
    """bifill's count of forwards for an answer of `length` tokens: forward k commits 1 + a_k (the anchor and the
    accepted drafts), so need = min{K : sum_{k<=K} (1 + a_k) >= length}; one more than were run if never."""
    c = 0
    for k, a in enumerate(accepted, 1):
        c += 1 + a
        if c >= length:
            return k
    return len(accepted) + 1


def flyx_rounds(accepted, length):
    """The same answer in flyx's count: the prefill emits the first token and round k emits a_k + 1, so the rounds
    are min{R : 1 + sum_{k<=R} (1 + a_k) >= length}."""
    return 0 if length <= 1 else rounds_needed(accepted, length - 1)


# ---- the model ----------------------------------------------------------------------------------------------------

@dataclass
class Cache:
    k: list = field(default_factory=list)        # layer -> [cap, n_kv, hd]
    v: list = field(default_factory=list)
    n: int = 0                                   # positions held: the anchor sits at n


def round_kv(x, kv_round):
    """K or V [C, Hkv, d] as a cache of `kv_round` holds them: torch.float16 (binary16), "e4m3" (FP8: each position
    and head scaled by its largest magnitude / 448 in float32, as attn_prep writes them), None as they are."""
    if kv_round is None:
        return x
    if kv_round != "e4m3":
        return x.to(kv_round).to(x.dtype)
    x32 = x.float()
    scale = x32.abs().amax(-1, keepdim=True) / 448
    q = torch.where(scale > 0, x32 / torch.where(scale > 0, scale, 1), 0).to(torch.float8_e4m3fn)
    return (q.to(x.dtype) * scale.to(x.dtype)).to(x.dtype)


class Dv:
    """A Llama target with DV drafting.  W: canonical weights (target.*, dv.*); dtype of the computation.
    kv_round: what the attention reads of K and V (round_kv); canvas_round: the dtype the canvas's linears and head
    take their inputs in (None: as they are)."""

    def __init__(self, cfg: LlamaConfig, W: dict, dv: DvConfig, dtype=torch.float64, kv_round=None,
                 canvas_round=None):
        self.cfg, self.dv, self.dtype = cfg, dv, dtype
        self.kv_round, self.canvas_round = kv_round, canvas_round
        self.W = {k: v.to(dtype) for k, v in W.items() if not k.startswith("dv.sel.")}
        self.embed = self.W["target.embed"]
        self.head = self.W.get("target.lm_head", self.embed)
        sdt = torch.float64 if dtype == torch.float64 else torch.float32   # bifill keeps the selector in float32
        self.sel = Selector(self.embed, *(W["dv.sel." + n].to(sdt) for n in ("A", "H", "B")),
                            W["dv.sel.alpha"].to(sdt).reshape(()), dv.temp) if "dv.sel.A" in W else None
        self.has_canvas = "dv.mask_emb" in W

    def w(self, name):
        return self.W[name]

    def _canvas_in(self, x):
        """A canvas linear's input, as canvas_round rounds it."""
        return x if self.canvas_round is None else x.to(self.canvas_round).to(x.dtype)

    # -- one pass of rows: [0, n_v) the target's, [n_v, R) the canvas's -----------------------------------------------
    def _lin(self, h, n_v, l, blk, name):
        w = self.W[f"target.L{l}.{blk}.{name}"]
        y = F.linear(h, w)
        parts = [y[:n_v], y[n_v:]]
        a = self.W.get(f"target.L{l}.{blk}.{name}.lora_a")
        if a is not None and n_v:  # u = scale (A x) first, as the engine forms it (bifill-dv's scale 1: A x itself)
            u = F.linear(h[:n_v], a)
            u = u if self.cfg.lora_scale == 1.0 else u * self.cfg.lora_scale
            parts[0] = parts[0] + F.linear(u, self.W[f"target.L{l}.{blk}.{name}.lora_b"])
        if n_v < h.shape[0]:
            hc = self._canvas_in(h[n_v:])
            if self.canvas_round is not None:
                parts[1] = F.linear(hc, w)
            parts[1] = parts[1] + F.linear(self._canvas_in(F.linear(hc, self.W[f"dv.L{l}.{name}.lora_a"])),
                                           self.W[f"dv.L{l}.{name}.lora_b"])
        return torch.cat(parts)

    def _norm(self, x, n_v, target_w, canvas_w):
        eps = self.cfg.eps
        if n_v == x.shape[0]:
            return rms(x, self.W[target_w], eps)
        return torch.cat([rms(x[:n_v], self.W[target_w], eps), rms(x[n_v:], self.W[canvas_w], eps)])

    def _pass(self, cache: Cache, x, pos, mask, n_v):
        """mask [R, C + R]: which of the C cached positions and of the pass's own rows each row sees.  Returns the
        final-normed hidden [R, D] (the canvas rows normed with fnormc) and every layer's (k, v) of the rows."""
        c = self.cfg
        R, C = x.shape[0], cache.n
        kv = []
        for l in range(c.layers):
            h = self._norm(x, n_v, f"target.L{l}.in_norm", f"dv.L{l}.ln1c")
            q = self._lin(h, n_v, l, "attn", "q").view(R, c.n_heads, c.head_dim)
            k = self._lin(h, n_v, l, "attn", "k").view(R, c.n_kv, c.head_dim)
            v = self._lin(h, n_v, l, "attn", "v").view(R, c.n_kv, c.head_dim)
            q, k = rope(q, pos, c.rope_theta, c.head_dim), rope(k, pos, c.rope_theta, c.head_dim)
            o = attend(q, round_kv(torch.cat([cache.k[l][:C], k]), self.kv_round),
                       round_kv(torch.cat([cache.v[l][:C], v]), self.kv_round), mask).reshape(R, -1)
            x = x + self._lin(o, n_v, l, "attn", "o")
            h = self._norm(x, n_v, f"target.L{l}.post_norm", f"dv.L{l}.ln2c")
            g = F.silu(self._lin(h, n_v, l, "mlp", "gate")) * self._lin(h, n_v, l, "mlp", "up")
            x = x + self._lin(g, n_v, l, "mlp", "down")
            kv.append((k, v))
        return self._norm(x, n_v, "target.norm", "dv.fnormc"), kv

    def _keep(self, cache: Cache, kv, rows):
        """Append the pass's rows `rows` (in this order) to the cache."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        for l, (k, v) in enumerate(kv):
            need = cache.n + rows.shape[0]
            if cache.k[l].shape[0] < need:
                grow = max(need, 2 * cache.k[l].shape[0])
                cache.k[l] = torch.cat([cache.k[l], cache.k[l].new_zeros(grow - cache.k[l].shape[0], *k.shape[1:])])
                cache.v[l] = torch.cat([cache.v[l], cache.v[l].new_zeros(grow - cache.v[l].shape[0], *v.shape[1:])])
            cache.k[l][cache.n:need] = k[rows]
            cache.v[l][cache.n:need] = v[rows]
        cache.n += rows.shape[0]

    def new_cache(self, cap=1024) -> Cache:
        c = self.cfg
        z = lambda: torch.zeros(cap, c.n_kv, c.head_dim, dtype=self.dtype)
        return Cache([z() for _ in range(c.layers)], [z() for _ in range(c.layers)], 0)

    def prefill(self, cache: Cache, prompt):
        """The prompt's rows, causal (target weights and V-LoRA).  Returns the last row's logits."""
        T = len(prompt)
        pos = torch.arange(cache.n, cache.n + T)
        mask = torch.cat([torch.ones(T, cache.n, dtype=torch.bool), torch.ones(T, T, dtype=torch.bool).tril()], 1)
        h, kv = self._pass(cache, self.embed[torch.as_tensor(prompt)], pos, mask, T)
        self._keep(cache, kv, range(T))
        return F.linear(h[-1:], self.head)[0]

    def plain(self, prompt, max_new, eos=()):
        """Greedy decoding one token a round (the target, with its V-LoRA)."""
        cache = self.new_cache(len(prompt) + max_new + 8)
        out = [int(torch.argmax(self.prefill(cache, prompt)))]
        while len(out) < max_new and out[-1] not in eos:
            out.append(int(torch.argmax(self.prefill(cache, [out[-1]]))))
        return out

    def round(self, cache: Cache, anchor, drafts, parents, depths, canvas, choose):
        """One DV forward: the anchor and the tree, and the canvas.  choose(logits_fn, rows_pos) gives the target's
        choice after each of the 1 + T rows.  Commits the anchor and the accepted path to the cache.
        Returns (path, choice, canvas stats, canvas final hidden)."""
        dv, c = self.dv, self.cfg
        T, N, p = len(drafts), dv.slots, cache.n
        par = [int(x) for x in parents]
        # ancestor-or-self among the target's rows (row 0 the anchor, row t + 1 draft t)
        anc = torch.zeros(1 + T, 1 + T, dtype=torch.bool)
        anc[0, 0] = True
        for t in range(T):
            anc[t + 1] = anc[par[t] + 1]
            anc[t + 1, t + 1] = True
        n_v = 1 + T
        rows = n_v + (N if self.has_canvas else 0)
        mask = torch.zeros(rows, p + rows, dtype=torch.bool)
        mask[:, :p] = True
        mask[:n_v, p:p + n_v] = anc
        vpos = p + torch.cat([torch.zeros(1, dtype=torch.long), torch.as_tensor(depths, dtype=torch.long)])
        x = self.embed[torch.as_tensor([anchor] + [int(d) for d in drafts])]
        pos = vpos
        if self.has_canvas:
            mask[n_v:, p] = True                                   # the canvas sees the anchor, never the tree
            mask[n_v:, p + n_v:] = torch.ones(N, N, dtype=torch.bool).tril() if dv.causal else True
            xc = canvas_input(canvas["cv"], canvas["z"], self.embed, self.w("dv.mask_emb"), self.w("dv.res_in"),
                              self.w("dv.res_out"), dv.mask_token)
            x = torch.cat([x, xc])
            pos = torch.cat([vpos, p + 1 + torch.arange(N)])
        h, kv = self._pass(cache, x, pos, mask, n_v)
        choice = choose(lambda: F.linear(h[:n_v], self.head), vpos)
        path = accept(drafts, par, depths, choice)
        self._keep(cache, kv, [0] + [t + 1 for t in path])
        stats = (canvas_stats(F.linear(self._canvas_in(h[n_v:]), self.head), self.embed, dv.topk)
                 if self.has_canvas else None)
        return path, choice, stats, h[n_v:]

    def tree(self, anchor, canvas):
        """The next round's drafts from the slid canvas: (drafts, parents, depths)."""
        dv = self.dv
        D = dv.depths
        # every slot's table, as bifill computes it (on fewer slots float32's batched products round differently)
        table = selector_table(self.sel, anchor, canvas["cand"], canvas["logp"], canvas["hid"])[:D]
        if dv.tree == "beam":
            edges = beam_edges(table, canvas["logp"][:D], canvas["ok"][:D], dv.cov)
            return beam_tree(edges, canvas["cand"][:D], dv.width, dv.beam, dv.mask_token)
        return pool_tree(table, canvas["cand"][:D], canvas["ok"][:D], dv.width, dv.pool_r, dv.mask_token)

    def star(self):
        T = self.dv.width
        return (torch.full((T,), self.dv.mask_token, dtype=torch.long), torch.full((T,), -1, dtype=torch.long),
                torch.ones(T, dtype=torch.long))

    def generate(self, prompt, max_new=512, eos=(), forced=None, oracle=None, commits=None, rounds=None):
        """DV decoding as bifill-dv evalgen.py mode DV runs it.  forced: the tokens to check the drafts against
        (teacher forcing; the first is the prefill's), instead of the target's own choices.  commits (with forced):
        the drafts forward k commits, commits[k], instead of the ones it accepts, while the list lasts: every forward
        then starts where another run's did (its anchor, its canvas slid by that run's counts), so that its own a_k
        compares with that run's round for round.  oracle(pos, drafts, parents, depths) -> drafts, or (drafts,
        parents, depths): tests replace drafts, or the whole tree (the CUDA backend's test chain, golden_test's DV
        rounds), before a round.  rounds: stop after that many forwards (a run's first forwards are the same whatever
        comes after them).
        Returns {ids (stopped at the first eos or at max_new), accepted (a_k a forward: the drafts it accepted),
        committed (the drafts it committed: accepted's but where commits says otherwise), need and flyx_rounds (of
        the committed counts), rounds: per forward {pos, anchor, drafts, parents, accepted}}."""
        if commits is not None and forced is None:
            raise ValueError("commits: a run that commits another run's counts is forced to that run's tokens")
        dv = self.dv
        P = len(prompt)
        cache = self.new_cache(P + max_new + dv.width + 2)
        logits = self.prefill(cache, prompt)
        anchor = int(forced[0]) if forced is not None else int(torch.argmax(logits))

        def choose(logits_fn, vpos):
            if forced is None:
                return [int(t) for t in torch.argmax(logits_fn(), -1)]
            at = (vpos + 1 - P).tolist()
            return [int(forced[i]) if i < len(forced) else -1 for i in at]

        canvas = fresh_canvas(dv.slots, dv.topk, self.cfg.hidden, dv.mask_token, self.dtype)
        drafts, parents, depths = self.star()
        out, accepted, committed, run = [], [], [], []
        limit = min(max_new, len(forced)) if forced is not None else max_new
        while rounds is None or len(run) < rounds:
            if oracle is not None:
                got = oracle(cache.n, drafts.clone(), parents, depths)
                if isinstance(got, tuple):
                    drafts, parents, depths = (torch.as_tensor(x, dtype=torch.long) for x in got)
                else:
                    drafts = torch.as_tensor(got, dtype=torch.long)
            pos = cache.n
            path, choice, stats, hid = self.round(cache, anchor, drafts, parents, depths, canvas, choose)
            run.append({"pos": pos, "anchor": anchor, "drafts": [int(t) for t in drafts],
                        "parents": [int(t) for t in parents], "accepted": len(path)})
            n = commits[len(accepted)] if commits is not None and len(accepted) < len(commits) else len(path)
            accepted.append(len(path))
            committed.append(n)
            tokens = [anchor] + [int(drafts[i]) for i in path[:n]]
            if n > len(path):
                tokens += [int(t) for t in forced[len(out) + 1 + len(path):len(out) + 1 + n]]
            done = False
            for t in tokens:
                out.append(t)
                if t in eos or len(out) >= limit:
                    done = True
                    break
            if done:
                break
            if n == len(path):
                anchor = choice[path[-1] + 1 if path else 0]
            else:
                # The round kept the anchor's and its path's rows: the ones past the n committed go, and the tokens
                # committed past the path get theirs as the text's (a pass of the target over them).
                cache.n -= max(0, len(path) - n)
                if n > len(path):
                    self.prefill(cache, tokens[1 + len(path):])
                anchor = int(forced[len(out)])
            canvas = slide(stats, hid, 1 + n, dv.fill, dv.mask_token)
            drafts, parents, depths = self.tree(anchor, canvas)
        return {"ids": out, "accepted": accepted, "committed": committed,
                "need": rounds_needed(committed, len(out)), "flyx_rounds": flyx_rounds(committed, len(out)),
                "rounds": run}


# ---- weights ------------------------------------------------------------------------------------------------------

_LINEARS = (("q", "attn"), ("k", "attn"), ("v", "attn"), ("o", "attn"), ("gate", "mlp"), ("up", "mlp"),
            ("down", "mlp"))


def hf_llama(model_dir):
    """A Hugging Face Llama directory -> (LlamaConfig, canonical target.* weights as stored)."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    names = {"model.embed_tokens.weight": "target.embed", "model.norm.weight": "target.norm",
             "lm_head.weight": "target.lm_head"}
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        names[p + "input_layernorm.weight"] = f"target.L{l}.in_norm"
        names[p + "post_attention_layernorm.weight"] = f"target.L{l}.post_norm"
        for n, blk in _LINEARS:
            names[f"{p}{'self_attn' if blk == 'attn' else 'mlp'}.{n}_proj.weight"] = f"target.L{l}.{blk}.{n}"
    W = {}
    for f in sorted(x for x in os.listdir(model_dir) if x.endswith(".safetensors")):
        for k, v in load_file(os.path.join(model_dir, f)).items():
            if k not in names:
                raise ValueError(f"{k}: not a Llama weight")
            W[names[k]] = v
    if cfg.get("tie_word_embeddings"):
        W.pop("target.lm_head", None)
    return LlamaConfig.from_hf(cfg), W


def _load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=True)   # never unpickle code from a checkpoint


def canvas_from_pt(path, layers):
    """bifill-dv's CanvasAdapter checkpoint {'adapter', 'cfg'} -> (dv.* weights rounded to bfloat16, as its engine and
    the earlier CUDA port's export_dv.py hold them; its cfg).  Only the plain residual canvas: no mixers, spine,
    see-gate, second residual or candidate memory, and none of the cfg flags that change its pass without a tensor
    (soft_in: a MASK slot's input its z; see_guess: it attends to the guesses); z over the top 16."""
    ck = _load_pt(path)
    cfg, ad = ck["cfg"], ck["adapter"]
    if cfg.get("mixers") or not cfg.get("residual"):
        raise ValueError(f"{path}: only residual canvases without mixers are supported")
    flags = [f for f in ("soft_in", "see_guess", "see_gate", "residual_h", "spine", "mem_layers") if cfg.get(f)]
    if flags or int(cfg.get("res_k", 16)) != 16:
        raise ValueError(f"{path}: a canvas trained with {flags or 'res_k ' + str(cfg['res_k'])} is not supported")
    if any(k.startswith(("spine", "mixers")) or "see_gate" in k or k.startswith("res_h") for k in ad):
        raise ValueError(f"{path}: a spine, see-gate or second residual is not supported")
    W = {"dv.fnormc": ad["norm"], "dv.mask_emb": ad["mask_emb"], "dv.res_in": ad["res_in.weight"],
         "dv.res_out": ad["res_out.weight"]}
    for l in range(layers):
        W[f"dv.L{l}.ln1c"], W[f"dv.L{l}.ln2c"] = ad[f"layers.{l}.ln1"], ad[f"layers.{l}.ln2"]
        for n, _ in _LINEARS:
            W[f"dv.L{l}.{n}.lora_a"] = ad[f"layers.{l}.{n}.A"]
            W[f"dv.L{l}.{n}.lora_b"] = ad[f"layers.{l}.{n}.B"]
    if len(W) != 4 + layers * 16 or len(ad) != len(W):
        raise ValueError(f"{path}: {len(ad)} adapter tensors, expected {4 + layers * 16}")
    return {k: v.to(torch.bfloat16) for k, v in W.items()}, cfg


def selector_from_pt(path):
    """bifill-dv's PathSelector checkpoint {'sel', 'r'} -> dv.sel.* (float32).  Only the one-token bilinear one."""
    ck = _load_pt(path)
    if ck.get("two") or ck.get("mlp"):
        raise ValueError(f"{path}: two-token and MLP selectors are not supported")
    s = ck["sel"]
    return {"dv.sel.A": s["A.weight"].float(), "dv.sel.H": s["H.weight"].float(), "dv.sel.B": s["B.weight"].float(),
            "dv.sel.alpha": s["alpha"].float().reshape(1)}


def _flyx(path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
    from flyx_convert.fileformat import Reader
    r = Reader(path)
    h = r.header
    if h.get("arch") != "llama":
        raise ValueError(f"{path}: not a Llama file")
    if any(p != "lora" for p in r.requires):  # the parts a reader must not ignore (fileformat.py: format 2)
        raise ValueError(f"{path}: requires {r.requires}, of which the reference reads lora only")
    cfg = LlamaConfig.from_canonical(h["config"])
    if h.get("lora"):  # the target's own adapters (python/convert --lora)
        cfg.lora_scale = float(h["lora"]["scale"])
    return r, h, cfg


def llama_from_flyx(path):
    """A Llama .flyx file -> (LlamaConfig, its lora_scale the header's lora section's; every tensor as stored)."""
    r, h, cfg = _flyx(path)
    return cfg, {n: r.torch(n) for n in h["tensors"]}


def from_flyx(path):
    """A .flyx file with a DV drafter (python/convert --dv) -> (LlamaConfig, every tensor as stored: target.* (with
    its V-LoRA when it has one), dv.*, dv.sel.*; the DvConfig of its header's dv section)."""
    r, h, cfg = _flyx(path)
    if not h.get("dv"):
        raise ValueError(f"{path}: not a Llama file with a DV drafter")
    return cfg, {n: r.torch(n) for n in h["tensors"]}, DvConfig.from_header(h["dv"])


def vlora_from_pt(path, layers):
    """bifill-dv's V LoRA {'lora': [7 a layer: q k v o gate up down] of {'A', 'B'}} -> the target's own adapters,
    rounded to bfloat16 as its engine holds them (y = W x + B (A x), no scale)."""
    ck = _load_pt(path)
    if len(ck["lora"]) != 7 * layers:
        raise ValueError(f"{path}: {len(ck['lora'])} adapters, expected {7 * layers}")
    W = {}
    for i, ab in enumerate(ck["lora"]):
        n, blk = _LINEARS[i % 7]
        W[f"target.L{i // 7}.{blk}.{n}.lora_a"] = ab["A"].to(torch.bfloat16)
        W[f"target.L{i // 7}.{blk}.{n}.lora_b"] = ab["B"].to(torch.bfloat16)
    return W
