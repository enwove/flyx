# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DV reference (python/reference/flyx_ref/dv.py) against bifill-dv's own functions, on random inputs.

    python -I tests/python/reference/check_dv_bifill.py BIFILL_CODE [--n 1000]

Run by hand: BIFILL_CODE is a copy of bifill-dv's code/ at the commit its records ran (839ac35: evalgen.py,
selector.py, ...) with code/bifill/{__init__,tree,combo,lossy_tree,dv_fused}.py beside it (branch dv-batch1-tree:
combo.py's selector pool tree, the source of the earlier CUDA port's dv.cu).  Triton's attention is replaced by a
stub; nothing here runs a model.  For each case, float32 (bifill's selector is float32 only) and, where the input can
be given, float64:

1. selector_table against PathSelector.table, without and with cov and slots past the canvas;
2. beam_edges + beam_tree against PathSelector.dyn_tree_fast (DYN=T:W:cov), from the selector's inputs (float32) and
   from one table given to both (float64);
3. pool_tree against combo.py's SelTreePool.select on Combo.sel_slide's tables (ST T/R/m=1);
4. accept, slide and the next tree against evalgen.py Engine._tree_post (what a round does after its forward);
5. the canvas's input and statistics: canvas_input against dv.py CanvasAdapter.embed (MASK slots, z, new slots with
   z = 0) and canvas_stats against the canvas part of evalgen.py Engine._dv_tree_heads (top 16, p, z').

The canvas pass itself is bifill-dv's flex_attention in bfloat16 there; tests/python/reference/dv_test.py (ctest
reference_dv) holds it to transformers' Llama in float64 instead.

Ties: bifill breaks them with torch.topk, which leaves their order open; the reference takes the lower index.  Inputs
here are continuous, so only slots past the canvas (every edge NO_SLOT, every draft MASK) tie; the counts below say
how often their MASK drafts hang under other parents, and that nothing else differs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import types
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
from flyx_ref import dv as ref  # noqa: E402

MASK = 1023


def bifill_modules(code):
    sys.path.insert(0, code)
    attn = types.ModuleType("bifill.attn")
    attn.attention = None                       # Triton's kernel: dv_fused imports it, nothing here calls it
    sys.modules["bifill.attn"] = attn
    import dv as bifill_dv
    import evalgen
    import selector
    from bifill import combo
    return evalgen, selector, combo, bifill_dv


def random_slots(g, N, V, K, D, dtype):
    """A slid canvas: distinct candidates a slot, descending probabilities with a tail, hidden states, ok."""
    cand = torch.stack([torch.randperm(V - 1, generator=g)[:K] for _ in range(N)])
    u = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    raw = u(N, K) ** (8 * u(N, 1))                           # from flat to peaked
    p = raw / raw.sum(-1, keepdim=True) * u(N, 1)            # the top 16 hold part of the mass
    p[torch.rand(N, K, generator=g) < 0.05] = 1e-12          # some below the 1e-9 floor
    p = p.sort(-1, descending=True).values.to(dtype)
    h = torch.randn(N, D, generator=g, dtype=dtype)
    n = int(torch.randint(1, N, (1,), generator=g))
    ok = torch.arange(N) + n < N if torch.rand(1, generator=g) < 0.7 else torch.ones(N, dtype=torch.bool)
    return cand, p, h, ok, n


def make_selector(selector, g, V, D, r, dtype):
    emb = torch.randn(V, D, generator=g, dtype=dtype)
    sel = selector.PathSelector(emb, D, r).eval()
    with torch.no_grad():
        for m in (sel.A, sel.H, sel.B):
            m.weight.copy_(torch.randn(r, D, generator=g) / math.sqrt(D))
        sel.alpha.fill_(float(0.5 + torch.rand(1, generator=g)))
    mine = ref.Selector(emb, sel.A.weight.detach().clone(), sel.H.weight.detach().clone(),
                        sel.B.weight.detach().clone(), sel.alpha.detach().clone())
    return sel, mine


def mask_nodes(t):
    return torch.where(t >= 0, t, torch.full_like(t, MASK))


class Tally:
    def __init__(self):
        self.rows = {}

    def add(self, name, same, mask_only=False):
        r = self.rows.setdefault(name, [0, 0, 0])
        r[0] += 1
        r[1] += bool(same)
        r[2] += bool(mask_only) and not same

    def report(self):
        bad = 0
        for name, (n, same, mask_only) in self.rows.items():
            other = n - same - mask_only
            bad += other
            print(f"{'ok ' if other == 0 else 'FAIL'} {name}: identical {same}/{n}"
                  + (f", MASK drafts under other parents only {mask_only}" if mask_only else "")
                  + (f", OTHER DIFFERENCES {other}" if other else ""), flush=True)
        return bad


def same_live(a, b):
    """Two trees (drafts, parents, depths) that differ only in where MASK drafts hang."""
    (ta, pa, da), (tb, pb, db) = a, b
    live = ta != MASK
    return bool(torch.equal(ta, tb) and torch.equal(da, db) and torch.equal(pa[live], pb[live]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    evalgen, selector, combo, bifill_dv = bifill_modules(a.code)
    g = torch.Generator().manual_seed(2026)
    tally = Tally()
    worst_fewer = 0.0
    V, Dm, r, K = MASK + 1, 64, 32, 16
    for case in range(a.n):
        N = [8, 16, 32, 32, 64][case % 5]
        T = [7, 15, 31, 31, 31][case % 5]
        W = [8, 4, 8, 8, 8][case % 5]
        cov = case % 3 != 0
        sel, mine = make_selector(selector, g, V, Dm, r, torch.float32)
        cand, p, h, ok, n = random_slots(g, N, V, K, Dm, torch.float32)
        logp = p.clamp(min=1e-9).log()
        start = int(torch.randint(0, V - 1, (1,), generator=g))
        D = min(N - 1, T)

        # 1. the table
        with torch.no_grad():
            theirs = sel.table(torch.tensor([start]), cand[None], logp[None], h[None], ok[None], cov)[0]
        table = ref.selector_table(mine, start, cand, logp, h)
        tally.add("selector_table vs PathSelector.table (float32)",
                  torch.equal(ref.beam_edges(table, logp, ok, cov), theirs))
        # on fewer slots the batched products round differently (why Dv.tree computes every slot's)
        fewer = (ref.selector_table(mine, start, cand[:D], logp[:D], h[:D]) - table[:D]).abs().max().item()
        worst_fewer = max(worst_fewer, fewer)

        # 2. the beam tree, from the selector's inputs (float32)
        args = (torch.tensor([start]), cand[None], logp[None], h[None], ok[None], T, W, cov)
        with torch.no_grad():
            tk, par, dep = sel.dyn_tree_fast(*args)
        theirs = (mask_nodes(tk[0]), par[0] - 1, dep[0])
        edges = ref.beam_edges(table[:D], logp[:D], ok[:D], cov)
        ours = ref.beam_tree(edges, cand[:D], T, W, MASK)
        same = all(torch.equal(x, y) for x, y in zip(ours, theirs))
        tally.add(f"beam_tree vs dyn_tree_fast (float32, DYN={T}:{W}{':cov' if cov else ''}, {N} slots)", same,
                  same_live(ours, theirs))
        # ... and from one float64 table given to both (dyn_tree_fast's own table step replaced)
        e64 = ref.beam_edges(table.double() + 1e-3 * torch.randn(table.shape, generator=g, dtype=torch.float64),
                             logp.double(), ok, cov)
        sel.table = lambda *x, **k: e64[None]
        with torch.no_grad():
            tk, par, dep = sel.dyn_tree_fast(*args)
        del sel.table
        theirs = (mask_nodes(tk[0]), par[0] - 1, dep[0])
        ours = ref.beam_tree(e64[:D], cand[:D], T, W, MASK)
        same = all(torch.equal(x, y) for x, y in zip(ours, theirs))
        tally.add("beam_tree vs dyn_tree_fast (float64 table given)", same, same_live(ours, theirs))

        # 3. the pool tree: combo.py's tables (Combo.sel_slide) and SelTreePool.select on the same slid canvas
        R = [7, 3, 7, 5, 7][case % 5]
        pool = combo.SelTreePool(N - 1, R, 1, device="cpu")
        hc = torch.randn(1, N, Dm, generator=g)                  # this round's canvas hidden, before the slide
        jc = (torch.arange(N) + n).clamp(max=N - 1)
        dead = ~ok
        gis = torch.where(dead[:, None], torch.full_like(cand, MASK), cand)
        lqs = torch.where(dead[:, None], torch.full_like(logp, combo.DEAD), logp)
        fake = SimpleNamespace(N=N, sr=r, hcs=torch.zeros(N, Dm), gis=gis, lqs=lqs,
                               embed=SimpleNamespace(weight=mine.embed), sA=mine.A, sH=mine.H, sB=mine.B,
                               salpha=mine.alpha, so0=torch.zeros(K, dtype=torch.long), sl0=torch.zeros(K),
                               so=torch.zeros(N - 1, K, K, dtype=torch.long), sl=torch.zeros(N - 1, K, K),
                               lt0=torch.zeros(K), lt=torch.zeros(N - 1, K, K), g=torch.zeros(1, N, dtype=torch.long))
        with torch.no_grad():
            combo.Combo.sel_slide(fake, torch.tensor([start]), hc, torch.tensor([n]))
            pen = torch.where(lqs[:, 0] <= combo.DEAD / 2, combo.DEAD, 0.0)
            tk, dp, pr, _, _ = pool.select(fake.so0, fake.sl0, fake.so, fake.sl, gis, pen, T)
        theirs = (tk, pr - 1, dp)
        ptable = ref.selector_table(mine, start, gis[:D], lqs[:D].clamp(min=math.log(1e-9)), hc[0][jc][:D])
        ours = ref.pool_tree(ptable, gis[:D], ok[:D], T, R, MASK)
        same = all(torch.equal(x, y) for x, y in zip(ours, theirs))
        tally.add(f"pool_tree vs SelTreePool.select (float32, ST {T}/{R}/1, {N} slots)", same, same_live(ours, theirs))

        # 4. a round's step after its forward: evalgen.py Engine._tree_post (DV, exact check)
        tw = T + 1
        par_pos = torch.zeros(T, dtype=torch.long)
        for t in range(1, T):
            par_pos[t] = int(torch.randint(0, t + 1, (1,), generator=g))   # a parent before each node
        depth = torch.zeros(tw, dtype=torch.long)
        for t in range(T):
            depth[t + 1] = depth[par_pos[t]] + 1
        order = torch.sort(depth[1:], stable=True).indices                   # bifill's trees are breadth-first
        remap = torch.zeros(tw, dtype=torch.long)
        remap[1 + order] = torch.arange(1, tw)
        par_pos = remap[par_pos[order]]
        dv_t = torch.randint(0, V - 1, (T,), generator=g)
        dv_t[torch.rand(T, generator=g) < 0.1] = MASK
        t_dep, t_anc = evalgen.Engine.tree_shape(par_pos[None], depth[1:][order][None])
        choice = torch.randint(0, V - 1, (tw,), generator=g)
        leaf = int(torch.randint(0, T, (1,), generator=g))                   # its path is the target's choices
        t = leaf
        while t >= 0 and torch.rand(1, generator=g) < 0.97:
            choice[par_pos[t]] = dv_t[t]
            t = int(par_pos[t]) - 1
        ids, pp = cand.clone(), p.clone()
        fill = 0.9
        pp[:, 0] = torch.where(torch.rand(N, generator=g) < 0.5, torch.full((N,), 0.95), pp[:, 0])
        zn = torch.randn(N, Dm, generator=g)
        heads = (choice[None], ids[None, :, 0], pp[None, :, 0], zn[None], ids[None], pp[None], hc, None)
        eng = SimpleNamespace(ND=N, TW=tw, sel=sel, dyn=(T, W, cov), mask=MASK, sel_k=K,
                              tree_shape=evalgen.Engine.tree_shape, _dv_tree_heads=lambda *x: heads)
        start_pos = torch.tensor([100])
        v_in = torch.cat([torch.tensor([7]), dv_t])[None]
        with torch.no_grad():
            nb, nxt, commit, _, _, dv_t2, par2, dep2, _, dv_cv, dv_z, _, _, _ = evalgen.Engine._tree_post(
                eng, torch.zeros(1, 1), v_in, dv_t[None], par_pos[None], t_dep, t_anc, start_pos, False, 0.9, fill)
        drafts, parents = dv_t, par_pos - 1
        path = ref.accept(drafts, parents, t_dep[0, 1:], choice)
        n_mine = 1 + len(path)
        nxt_mine = int(choice[path[-1] + 1 if path else 0])
        canvas = ref.slide({"ids": ids, "p": pp, "zn": zn}, hc[0], n_mine, fill, MASK)
        tree = ref.beam_tree(ref.beam_edges(ref.selector_table(mine, nxt_mine, canvas["cand"][:D],
                                                               canvas["logp"][:D], canvas["hid"][:D]),
                                            canvas["logp"][:D], canvas["ok"][:D], cov),
                             canvas["cand"][:D], T, W, MASK)
        their_tree = (dv_t2[0], par2[0] - 1, dep2[0, 1:])
        step_same = (int(nb) == n_mine and int(nxt) == nxt_mine
                     and commit[0, :n_mine].tolist() == [7] + [int(dv_t[i]) for i in path]
                     and torch.equal(dv_cv[0], canvas["cv"]) and torch.equal(dv_z[0], canvas["z"]))
        tally.add("accept + slide vs _tree_post (committed count, next anchor, commit, canvas input, z)", step_same)
        same = all(torch.equal(x, y) for x, y in zip(tree, their_tree))
        tally.add("the next tree vs _tree_post's", same, same_live(tree, their_tree))

        # 5. the canvas's input (dv.py CanvasAdapter.embed) and statistics (evalgen.py Engine._dv_tree_heads)
        lin = lambda i, o: torch.randn(o, i, generator=g) / math.sqrt(i)  # noqa: E731
        ad = SimpleNamespace(mask_id=MASK, mask_emb=torch.randn(Dm, generator=g), residual=True,
                             res_in=torch.nn.Linear(Dm, 2 * Dm, bias=False),
                             res_out=torch.nn.Linear(Dm, Dm, bias=False))
        with torch.no_grad():
            ad.res_in.weight.copy_(lin(Dm, 2 * Dm))
            ad.res_out.weight.copy_(lin(Dm, Dm))
            tokens = torch.where(torch.rand(N, generator=g) < 0.4, torch.full((N,), MASK), cand[:, 0])
            z = torch.randn(N, Dm, generator=g) * (torch.rand(N, 1, generator=g) < 0.7)
            theirs = bifill_dv.CanvasAdapter.embed(ad, tokens[None], mine.embed[tokens][None], z[None])[0]
            ours = ref.canvas_input(tokens, z, mine.embed, ad.mask_emb, ad.res_in.weight, ad.res_out.weight, MASK)
            tally.add("canvas_input vs CanvasAdapter.embed (float32)", torch.equal(ours, theirs))
            head = torch.nn.Linear(Dm, V, bias=False)
            head.weight.copy_(lin(Dm, V) * 8)
            hv = torch.randn(1, tw + N, Dm, generator=g)
            eng = SimpleNamespace(TW=tw, head=head, sel_k=K, embed=SimpleNamespace(weight=mine.embed))
            _, _, _, zn_t, ids_t, p_t, _, _ = evalgen.Engine._dv_tree_heads(eng, hv, par_pos[None])
            st = ref.canvas_stats(head(hv[0, tw:]), mine.embed, K)
            tally.add("canvas_stats vs _dv_tree_heads (top 16, p, z'; float32)", torch.equal(st["ids"], ids_t[0])
                      and torch.equal(st["p"], p_t[0]) and torch.equal(st["zn"], zn_t[0]))
        if case == 0:
            print(f"case 0: {N} slots, n = {n}, accepted {len(path)}", flush=True)
    bad = tally.report()
    print(f"selector_table on the first D slots only, against all N: max |difference| {worst_fewer:.3g}", flush=True)
    print("dv vs bifill-dv: " + ("ok" if not bad else f"{bad} differences beyond MASK placement"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
