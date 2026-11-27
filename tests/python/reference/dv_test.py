# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DV reference (python/reference/flyx_ref/dv.py) on the tiny Llama, in float64.

    python tests/python/reference/dv_test.py ASSETS

* Its plain decoding is transformers' (golden_llama.json, float64).
* DV decoding emits exactly the plain tokens: the beam tree and the pool tree, the canvas both ways and causal, the
  target with and without a V-LoRA, and with the plain continuation planted on a path of the tree (a random canvas
  almost never guesses right, so this is what makes rounds accept long paths).
* Checking the drafts against the plain tokens given as a list (teacher forcing) runs the same rounds.  Told to
  commit another run's counts (dv_forced.py --replay), every forward starts where that run's did, the cache keeps
  the text's own rows past the path and short of it, and every tree grows again from transformers' canvases slid by
  the counts committed.
* canvas_from_pt loads the tiny DV's checkpoint and refuses one whose cfg changes the canvas pass with no tensor of
  its own (bifill-dv's soft_in, see_guess) or sums z over other than 16 candidates.

Exact verification makes any drafts give the plain tokens, so those checks cannot see a wrong drafter.  What can:

* The canvas rows of a round are transformers' Llama run as bifill-dv describes the canvas: the base weights with the
  canvas LoRA merged in, its own norms, its input from the slot's guess or MASK and z, the slots after the anchor,
  seeing the text and the anchor (with the target's own V-LoRA in their keys) and the slots, never the tree.
* Each round's tree grows again from those transformers canvases: every slot's input from the slide before, the slide
  by what the round committed (the next round's position minus its own), then the selector and the tree.  The pieces
  that step uses (canvas_stats, slide, selector_table, the trees) are held to ops/'s float64 references by the ctests
  dv_canvas and dv_trees, and to bifill-dv's own functions by check_dv_bifill.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from flyx_ref import dv as ref  # noqa: E402
from flyx_ref.dv import Dv, DvConfig, hf_llama, _LINEARS  # noqa: E402
from check import check, finish  # noqa: E402


def random_dv(cfg, rank, sel_rank, gen, vlora_rank=0):
    """Random canvas adapter and selector weights for cfg (bf16 values, as converted files hold them)."""
    D, I, q, kv = cfg.hidden, cfg.inter, cfg.n_heads * cfg.head_dim, cfg.n_kv * cfg.head_dim
    dims = {"q": (D, q), "k": (D, kv), "v": (D, kv), "o": (q, D), "gate": (D, I), "up": (D, I), "down": (I, D)}
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    W = {"dv.fnormc": 1 + 0.1 * r(D), "dv.mask_emb": r(D), "dv.res_in": r(2 * D, D) / math.sqrt(D),
         "dv.res_out": r(D, D) / math.sqrt(D)}
    for l in range(cfg.layers):
        W[f"dv.L{l}.ln1c"], W[f"dv.L{l}.ln2c"] = 1 + 0.1 * r(D), 1 + 0.1 * r(D)
        for n, blk in _LINEARS:
            i, o = dims[n]
            W[f"dv.L{l}.{n}.lora_a"] = r(rank, i) / math.sqrt(i)
            W[f"dv.L{l}.{n}.lora_b"] = r(o, rank) / math.sqrt(rank)
            if vlora_rank:
                W[f"target.L{l}.{blk}.{n}.lora_a"] = r(vlora_rank, i) / math.sqrt(i)
                W[f"target.L{l}.{blk}.{n}.lora_b"] = 0.3 * r(o, vlora_rank) / math.sqrt(vlora_rank)
    W = {k: v.to(torch.bfloat16) for k, v in W.items()}
    W.update({"dv.sel.A": r(sel_rank, D).float(), "dv.sel.H": r(sel_rank, D).float(),
              "dv.sel.B": r(sel_rank, D).float(), "dv.sel.alpha": torch.tensor([0.85])})
    return W


def planted(prompt_len, plain):
    """Puts the plain continuation on the tree's first deepest path, every round but every third."""
    calls = [0]

    def oracle(pos, drafts, parents, depths):
        calls[0] += 1
        if calls[0] % 3 == 0:
            return drafts
        t = int(torch.argmax(depths))
        while t >= 0:
            at = pos + int(depths[t]) - prompt_len          # the plain token at the draft's position
            if at < len(plain):
                drafts[t] = plain[at]
            t = int(parents[t])
        return drafts
    return oracle


def recording(oracle, trees):
    """The oracle, keeping each tree as the reference grew it (before the oracle plants anything): (drafts, parents)."""
    def wrapped(pos, drafts, parents, depths):
        trees.append((drafts.tolist(), [int(p) for p in parents]))
        return oracle(pos, drafts, parents, depths)
    return wrapped


def float64_inside(model, cfg):
    """transformers computes RMSNorm and the rotary angles in float32 even in a float64 model; here both in float64,
    so that it and the reference agree to float64 rounding and a tree never turns on a float32 near-tie."""
    def rms_norm(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    for m in model.modules():
        if type(m).__name__ == "LlamaRMSNorm":
            m.forward = types.MethodType(rms_norm, m)
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim))

    def rotary(self, x, position_ids):
        angles = position_ids[..., None].to(torch.float64) * inv
        angles = torch.cat([angles, angles], -1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    model.model.rotary_emb.forward = types.MethodType(rotary, model.model.rotary_emb)
    return model


def transformers_pair(hf_dir, cfg, W):
    """transformers' Llama in float64, twice: the target (its V-LoRA merged into its weights, W + B A, when W has
    one) and the canvas (the base weights with the canvas LoRA merged in; ln1c, ln2c and fnormc as its norms)."""
    from transformers import AutoModelForCausalLM

    def load():
        return float64_inside(AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float64).eval(), cfg)

    def merged(name):
        return W[name + ".lora_b"].double() @ W[name + ".lora_a"].double()
    target, canvas = load(), load()
    with torch.no_grad():
        for l, (lt, lc) in enumerate(zip(target.model.layers, canvas.model.layers)):
            for n, blk in _LINEARS:
                if f"target.L{l}.{blk}.{n}.lora_a" in W:
                    getattr(lt.self_attn if blk == "attn" else lt.mlp, n + "_proj").weight += merged(
                        f"target.L{l}.{blk}.{n}")
                getattr(lc.self_attn if blk == "attn" else lc.mlp, n + "_proj").weight += merged(f"dv.L{l}.{n}")
            lc.input_layernorm.weight.copy_(W[f"dv.L{l}.ln1c"])
            lc.post_attention_layernorm.weight.copy_(W[f"dv.L{l}.ln2c"])
        canvas.model.norm.weight.copy_(W["dv.fnormc"])
    return target, canvas


def transformers_canvas(pair, W, text, cv, z, mask_token, causal):
    """The canvas after `text` (the committed tokens, the anchor last): its logits and final hidden states [N, .].
    The text's keys come from the target; slot w sits at position len(text) + w and sees the whole text and every
    slot (causal: the slots up to itself); its input is bifill-dv's CanvasAdapter.embed of its guess or MASK and z."""
    target, canvas = pair
    N, T = cv.shape[0], len(text)
    with torch.no_grad():
        cache = target(torch.tensor([text]), use_cache=True).past_key_values
        x = torch.where((cv == mask_token)[:, None], W["dv.mask_emb"].double(), W["target.embed"].double()[cv])
        u, g = (z.double() @ W["dv.res_in"].double().T).chunk(2, -1)
        x = x + (F.silu(g) * u) @ W["dv.res_out"].double().T
        seen = torch.ones(N, T + N, dtype=torch.bool)
        if causal:
            seen[:, T:] = torch.ones(N, N, dtype=torch.bool).tril()
        h = canvas.model(inputs_embeds=x[None], position_ids=torch.arange(T, T + N)[None],
                         attention_mask=seen[None, None], past_key_values=cache).last_hidden_state[0]
        return canvas.lm_head(h), h


def canvas_rows(cfg, W, pair, gen, mask):
    """One round's canvas rows from Dv.round against transformers_canvas, on random rounds: text, a tree of random
    drafts beside the canvas, slots holding guesses and MASKs with random z.  Returns the largest difference of the
    logits and the final hidden states, relative to their largest magnitude, over both canvas masks."""
    worst = 0.0
    for causal in (False, True):
        model = Dv(cfg, W, DvConfig(slots=8, mask_token=mask, causal=causal), torch.float64)
        N, T = model.dv.slots, model.dv.width
        for _ in range(3):
            text = torch.randint(0, mask, (int(torch.randint(1, 20, (1,), generator=gen)),), generator=gen).tolist()
            cache = model.new_cache()
            if len(text) > 1:
                model.prefill(cache, text[:-1])
            parents = [-1] + [int(torch.randint(-1, t, (1,), generator=gen)) for t in range(1, T)]
            depths = []
            for p in parents:
                depths.append(1 if p < 0 else depths[p] + 1)
            drafts = torch.randint(0, mask, (T,), generator=gen)
            cv = torch.randint(0, mask, (N,), generator=gen)
            cv[torch.rand(N, generator=gen) < 0.4] = mask
            z = 0.05 * torch.randn(N, cfg.hidden, generator=gen, dtype=torch.float64)
            _, _, _, hid = model.round(cache, text[-1], drafts, torch.tensor(parents), torch.tensor(depths),
                                       {"cv": cv, "z": z}, lambda logits, _: torch.argmax(logits(), -1).tolist())
            logits, h = transformers_canvas(pair, W, text, cv, z, mask, causal)
            worst = max(worst, ((F.linear(hid, model.head) - logits).abs().max() / logits.abs().max()).item(),
                        ((hid - h).abs().max() / h.abs().max()).item())
    return worst


def replay(model, pair, W, prompt, run, trees):
    """Grows every tree of a run but the first again from transformers' canvases: round k's canvas from the slots
    the slide after round k - 1 left (all MASK and z = 0 in the first round), slid by round k + 1's position minus
    round k's, then the reference's selector and tree from round k + 1's anchor.  Returns (trees the same, trees)."""
    dv, P, ids = model.dv, len(prompt), run["ids"]
    cv = torch.full((dv.slots,), dv.mask_token, dtype=torch.long)
    z = torch.zeros(dv.slots, model.cfg.hidden, dtype=torch.float64)
    same = 0
    for r, after, tree in zip(run["rounds"], run["rounds"][1:], trees[1:]):
        text = prompt + ids[:r["pos"] - P + 1]
        logits, h = transformers_canvas(pair, W, text, cv, z, dv.mask_token, dv.causal)
        slid = ref.slide(ref.canvas_stats(logits, model.embed, dv.topk), h, after["pos"] - r["pos"], dv.fill,
                         dv.mask_token)
        cv, z = slid["cv"], slid["z"]
        drafts, parents, _ = model.tree(after["anchor"], slid)
        same += text[-1] == r["anchor"] and (drafts.tolist(), parents.tolist()) == tree
    return same, len(run["rounds"]) - 1


def kept_rows(model, prompt, run):
    """run() (a generate of `model` on `prompt`), and the largest relative difference between the K and V rows its
    cache kept and the target's own pass over the prompt and the answer it emitted: a forward that commits other
    counts than its tree accepted must keep the text's rows, at the text's positions."""
    caches = []
    model.new_cache = lambda cap, make=Dv.new_cache.__get__(model): caches.append(make(cap)) or caches[-1]
    try:
        g = run()
    finally:
        del model.new_cache
    cache = caches[-1]
    text = (prompt + g["ids"])[:cache.n]
    whole = model.new_cache(len(text) + 8)
    model.prefill(whole, text)
    worst = max(((a[:cache.n] - b[:cache.n]).abs().max() / b[:cache.n].abs().max()).item()
                for x, y in ((cache.k, whole.k), (cache.v, whole.v)) for a, b in zip(x, y))
    return g, worst, cache.n == len(text)


def commits(model, pair, W, prompt, want, n):
    """A forced run told to commit other counts than its own (Dv.generate's commits): with its own counts it runs
    the same rounds; with a planted run's counts (longer paths than a random canvas guesses) and with counts shorter
    than the planted paths, every forward starts where that run's did, the cache keeps the text's rows, and every
    tree grows again from transformers' canvases slid by the counts committed (not the ones accepted)."""
    P = len(prompt)
    f = model.generate(prompt, n, forced=want)
    g = model.generate(prompt, n, forced=want, commits=f["accepted"])
    check(g["accepted"] == g["committed"] == f["accepted"] and g["ids"] == f["ids"] and g["rounds"] == f["rounds"],
          "commits: a forced run given its own counts runs the same rounds")
    planted_run = model.generate(prompt, n, oracle=planted(P, want))
    trees_l, trees_s = [], []
    longer, worst_l, whole_l = kept_rows(model, prompt, lambda: model.generate(
        prompt, n, forced=want, oracle=recording(lambda pos, d, p, dep: d, trees_l), commits=planted_run["accepted"]))
    short = [min(a, 1) for a in planted_run["accepted"]]
    shorter, worst_s, whole_s = kept_rows(model, prompt, lambda: model.generate(
        prompt, n, forced=want, oracle=recording(planted(P, want), trees_s), commits=short))
    schedule = lambda counts: [P + sum(1 + c for c in counts[:k]) for k in range(len(counts))]  # noqa: E731
    check(longer["ids"] == want and longer["committed"] == planted_run["accepted"]
          and [r["pos"] for r in longer["rounds"]] == [r["pos"] for r in planted_run["rounds"]]
          and [r["anchor"] for r in longer["rounds"]] == [r["anchor"] for r in planted_run["rounds"]]
          and longer["need"] == planted_run["need"] and sum(longer["accepted"]) < sum(longer["committed"]),
          f"commits: a planted run's counts ({sum(planted_run['accepted'])} drafts; this canvas accepts "
          f"{sum(longer['accepted'])} of them itself): its forwards' positions, anchors and need")
    check(shorter["ids"] == want and shorter["committed"][:len(short)] == short
          and [r["pos"] for r in shorter["rounds"]] == schedule(shorter["committed"])
          and sum(a > c for a, c in zip(shorter["accepted"], shorter["committed"])) >= 3,
          f"commits: counts shorter than the planted paths ({sum(shorter['accepted'])} drafts accepted, "
          f"{sum(shorter['committed'])} committed): every forward starts after the counts committed")
    check(whole_l and whole_s and max(worst_l, worst_s) < 1e-12,
          f"commits: the cache keeps the text's own rows, past the path and short of it (largest relative difference "
          f"{max(worst_l, worst_s):.1e})")
    regrown = [a + b for a, b in zip(replay(model, pair, W, prompt, longer, trees_l),
                                     replay(model, pair, W, prompt, shorter, trees_s))]
    check(regrown[0] == regrown[1] > 0, f"commits: the trees grow again from transformers' canvases slid by the counts "
          f"committed ({regrown[0]}/{regrown[1]})")
    try:
        model.generate(prompt, n, commits=[0])
        check(False, "commits without the tokens they commit: refused")
    except ValueError:
        check(True, "commits without the tokens they commit: refused")


def canvas_checkpoints(path, layers):
    """canvas_from_pt on the tiny DV's checkpoint (make_tiny.py): loaded; with a cfg flag that changes the canvas pass
    without a tensor of its own (bifill-dv's soft_in, see_guess) or z over other than 16 candidates, refused."""
    import tempfile
    ck = torch.load(path, map_location="cpu", weights_only=True)
    W, _ = ref.canvas_from_pt(path, layers)
    check(len(W) == 4 + 16 * layers, f"canvas_from_pt: the tiny canvas, {len(W)} tensors")
    with tempfile.TemporaryDirectory() as tmp:
        for flag, value in (("soft_in", True), ("see_guess", True), ("residual_h", True), ("mem_layers", [1]),
                            ("res_k", 8)):
            changed = os.path.join(tmp, f"{flag}.pt")
            torch.save({**ck, "cfg": {**ck["cfg"], flag: value}}, changed)
            try:
                ref.canvas_from_pt(changed, layers)
                ok = False
            except ValueError as e:
                ok = flag in str(e)
            check(ok, f"canvas_from_pt: a canvas with cfg {flag}={value}: refused")


def main(assets):
    cfg, W = hf_llama(os.path.join(assets, "hf_llama"))
    golden = json.load(open(os.path.join(assets, "golden_llama.json"), encoding="utf-8"))
    gen = torch.Generator().manual_seed(5)
    W.update(random_dv(cfg, rank=8, sel_rank=8, gen=gen))
    Wv = dict(W, **{k: v for k, v in random_dv(cfg, 8, 8, gen, vlora_rank=4).items() if k.startswith("target.")})
    mask = cfg.vocab - 1
    cases = golden["cases"][:4]
    n = len(cases[0]["tokens"])

    plain_model = Dv(cfg, W, DvConfig(slots=8, mask_token=mask, width=7, beam=4), torch.float64)
    plains = [plain_model.plain(c["prompt"], n) for c in cases]
    check(all(p == c["tokens"] for p, c in zip(plains, cases)),
          f"plain decoding is transformers' on {len(cases)} prompts of {n} tokens")
    with_v = Dv(cfg, Wv, DvConfig(slots=8, mask_token=mask), torch.float64)
    vplains = [with_v.plain(c["prompt"], n) for c in cases]
    check(any(a != b for a, b in zip(vplains, plains)), "the V-LoRA changes the plain tokens")

    weights = {"W": W, "Wv": Wv}
    pairs = {k: transformers_pair(os.path.join(assets, "hf_llama"), cfg, w) for k, w in weights.items()}
    worst = max(canvas_rows(cfg, w, pairs[k], gen, mask) for k, w in weights.items())
    check(worst < 1e-12, f"a round's canvas rows are transformers' (12 rounds: guesses, MASKs and z, a tree beside "
          f"them, causal or not, with and without a V-LoRA; largest relative difference {worst:.1e})")

    variants = [("beam 7:4:cov", "beam:7:4:cov", False, "W", plains), ("beam 7:2", "beam:7:2", False, "W", plains),
                ("beam 7:20 (padded)", "beam:7:20", False, "W", plains), ("pool 7/3", "pool:7:3:1", False, "W", plains),
                ("pool 9/2", "pool:9:2", False, "W", plains), ("causal canvas", "beam:7:4:cov", True, "W", plains),
                ("V-LoRA", "beam:7:4:cov", False, "Wv", vplains), ("V-LoRA pool", "pool:7:3", False, "Wv", vplains)]
    # The first variant on every prompt, the others (each a tree, canvas or target of its own; the same code around
    # them) on the first two.
    for v, (name, tree, causal, wk, want_all) in enumerate(variants):
        dv = DvConfig(slots=8, mask_token=mask, causal=causal).set_tree(tree)
        model = Dv(cfg, weights[wk], dv, torch.float64)
        same = acc = deep = masks = 0
        tf_same = shape = True
        regrown = [0, 0]
        held = cases if v == 0 else cases[:2]
        for case, (c, want) in enumerate(zip(held, want_all)):
            r = model.generate(c["prompt"], n)
            same += r["ids"] == want
            trees = []
            o = model.generate(c["prompt"], n, oracle=recording(planted(len(c["prompt"]), want), trees))
            same += o["ids"] == want
            if case < 2:
                regrown = [a + b for a, b in zip(regrown, replay(model, pairs[wk], weights[wk], c["prompt"], o, trees))]
            acc += sum(o["accepted"])
            deep = max(deep, max(o["accepted"]))
            masks += sum(d == mask for rd in o["rounds"][1:] for d in rd["drafts"])
            f = model.generate(c["prompt"], n, forced=want)
            tf_same &= f["ids"] == r["ids"] and f["accepted"] == r["accepted"]
            shape &= r["accepted"][0] == 0 and r["rounds"][0]["drafts"] == [mask] * dv.width
            shape &= all(len(rd["drafts"]) == dv.width and all(p < i for i, p in enumerate(rd["parents"]))
                         for rd in r["rounds"] + o["rounds"])
        check(shape, f"{name}: the first round drafts a star of MASKs; every round {dv.width} drafts, parents first")
        check(same == 2 * len(held), f"{name}: DV emits the plain tokens ({same}/{2 * len(held)} runs)")
        check(acc > n * len(held) / 2 and deep >= 5,
              f"{name}: planted paths are accepted ({acc} drafts, up to {deep} a round)")
        check(tf_same, f"{name}: teacher forcing on the plain tokens runs the same rounds")
        check(masks > 0, f"{name}: slots past the canvas draft MASK ({masks})")
        check(regrown[0] == regrown[1] > 0,
              f"{name}: the trees grow again from transformers' canvases ({regrown[0]}/{regrown[1]}, 2 prompts)")

    # Teacher forcing emits the list it is given, whatever the target would choose; stops at its end.
    model = Dv(cfg, W, DvConfig(slots=8, mask_token=mask).set_tree("beam:7:4:cov"), torch.float64)
    other = [t if i % 5 != 2 else (t + 1) % cfg.vocab for i, t in enumerate(plains[0])][:40]
    f = model.generate(cases[0]["prompt"], n, forced=other, oracle=planted(len(cases[0]["prompt"]), other))
    check(f["ids"] == other and max(f["accepted"]) >= 1,
          f"teacher forcing on a list the target disagrees with emits the list ({len(f['ids'])} tokens)")
    commits(model, pairs["W"], W, cases[0]["prompt"], plains[0], n)
    commits(Dv(cfg, Wv, DvConfig(slots=8, mask_token=mask).set_tree("pool:7:3"), torch.float64), pairs["Wv"], Wv,
            cases[1]["prompt"], vplains[1], n)
    # The same arithmetic in float32 (MiniCPM5 on a CPU) and bfloat16 (bifill-dv's engine).
    m32 = Dv(cfg, W, DvConfig(slots=8, mask_token=mask).set_tree("beam:7:4:cov"), torch.float32)
    check(all(m32.generate(c["prompt"], n)["ids"] == p for c, p in zip(cases, plains)),
          "float32: DV emits the plain tokens (the cases' margins are 0.02 or more)")
    m16 = Dv(cfg, W, DvConfig(slots=8, mask_token=mask).set_tree("pool:7:3"), torch.bfloat16)
    r16 = m16.generate(cases[0]["prompt"], n)
    same16 = r16["ids"] == m16.plain(cases[0]["prompt"], n)   # rows rounded together or alone: not promised
    check(len(r16["ids"]) == n, f"bfloat16: DV runs ({len(r16['ids'])} tokens in {len(r16['accepted'])} forwards, "
          f"{'the' if same16 else 'not the'} plain bfloat16 tokens)")
    canvas_checkpoints(os.path.join(assets, "dv_tiny", "canvas.pt"), cfg.layers)
    return finish("dv_test")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
