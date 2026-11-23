# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Which drafter and width a round should take, replayed from recorded rounds.

    python bench/replay/drafter_choice.py DIR

DIR holds, per model file M and prompt W, each drafter D's rounds alone at 7 drafts with lookup off
(`flyx run ... --drafter D --width 7 --rounds DIR/M.D.W.jsonl`, FLYX_LOOKUP=0), and the cost of every round
kind as the engine measured it at load (`flyx run ... -v 2> DIR/M.costs.err`, both drafters given):
bench/replay/drafter_traces_27b.sh records them for Qwen3.8-27B.

Greedy decoding writes the same text whichever drafter drafts, and every round keeps every drafter's context
current, so how many drafts drafter D gets accepted from a round starting at a point of the text depends on the
point, not on which drafters ran before.  A recorded round that started at L0 and had a drafts accepted covers the
points L0 .. L0 + a; a round starting at L0 + j is taken to have a - j accepted (the recorded drafts after it
matched, the next did not).  The replay runs a policy over the text and sums the measured costs: tokens per second
per policy with both drafters (the worse of the two orders of trying them first) and with each alone, against the
better drafter alone under the engine's initial chooser (the first policy).  Policy() with
two_stage, bayes 2, bayes_keep 0.95, forget 0.03, ucb 2, first_each and probe_plain_only is the engine's
(core/speculation/controller.h).
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

MAX_ROWS = 16


def load_costs(path: str) -> dict:
    """{(drafter or None, width): ms} from the engine's 'round cost:' line."""
    text = open(path, errors="replace").read()
    line = next(l for l in text.splitlines() if "round cost:" in l)
    costs = {}
    for name, width, ms in re.findall(r"(?:([a-z0-9]+) )?(\d+)=([\d.]+) ms", line.split("round cost:")[1]):
        costs[(name or None, int(width))] = float(ms)
    return costs


def load_tiles(path: str):
    """[(L0, accepted)] of a drafter's recorded rounds, in order, and where the text ends."""
    tiles, end = [], 0
    for line in open(path):
        r = json.loads(line)
        if r["tokens"] <= 0:
            continue
        end = r["position"]
        if r["drafts"] > 0:
            tiles.append((r["position"] - r["tokens"], r["accepted"]))
    return tiles, end


class Accept:
    """accepted(t, w): drafts of w a round starting at text point t gets accepted, from the recorded rounds."""

    def __init__(self, tiles):
        self.tiles = tiles

    def __call__(self, t: int, w: int) -> int:
        lo, hi = 0, len(self.tiles) - 1
        while lo < hi:  # the last tile starting at or before t
            mid = (lo + hi + 1) // 2
            if self.tiles[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        l0, a = self.tiles[lo]
        return max(0, min(w, a - (t - l0))) if l0 <= t else 0


@dataclass
class Policy:
    """SpecController (core/speculation/controller.cpp) with its knobs."""
    name: str
    keep: float = 0.9          # the moving average: kept a round the drafter drafts
    probe: int = 32            # an idle drafter drafts the probe-th round
    prior: float = 0.8         # P(draft 0 accepted) before any round; 0.02 less a position
    first_each: bool = False   # every drafter drafts once before choosing
    # > 0: the means: P(accepted) = (bayes * prior + hits) / (bayes + rounds), hits and rounds kept `keep` (two
    # stages: bayes_keep) a round the drafter drafts and forgotten `forget` a round it does not
    bayes: float = 0.0
    forget: float = 0.0
    # two stages: plain decoding or a drafted round by the moving averages; drafted, the drafter by the means (its
    # best round's rate), at its best width by its moving average (one drafter: the moving average's choice)
    two_stage: bool = False
    bayes_keep: float = 0.95
    probe_plain_only: bool = False  # the probe only after rounds of plain decoding
    ucb: float = 0.0           # two stages: the means plus this many of their standard errors

    def reset(self, kinds, order=()):
        self.kinds = kinds  # [(drafter or None, width, ms)]
        present = {d for d, _, _ in kinds if d is not None}
        names = [d for d in order if d in present] + sorted(present - set(order))
        self.acc = {d: [self.prior - 0.02 * i for i in range(MAX_ROWS)] for d in names}
        self.hits = {d: [0.0] * MAX_ROWS for d in names}
        self.n = {d: [0.0] * MAX_ROWS for d in names}
        self.idle = {d: 0 for d in names}
        self.untried = list(names) if self.first_each else []

    def p(self, d, i, means=None):
        if means is None:
            means = self.bayes > 0 and not self.two_stage
        if means:
            n = self.bayes + self.n[d][i]
            mean = (self.bayes * (self.prior - 0.02 * i) + self.hits[d][i]) / n
            return min(1.0, mean + self.ucb * (mean * (1 - mean) / n) ** 0.5) if self.ucb > 0 else mean
        return self.acc[d][i]

    def expected(self, d, w, means=None):
        e, p = 1.0, 1.0
        if d is None:
            return e
        for i in range(min(w, MAX_ROWS)):
            p *= self.p(d, i, means)
            e += p
        return e

    def best(self, drafter=None, means=None):
        best, rate = None, 0.0
        for k in self.kinds:
            d, w, ms = k
            if drafter is not None and d != drafter:
                continue
            r = self.expected(d, w, means) / ms
            if best is None or r > rate:
                best, rate = k, r
        return best

    def choose(self):
        if self.untried:
            k = self.best(self.untried.pop(0))
        else:
            k = self.best()
            if self.two_stage and k[0] is not None:
                pick = max(self.idle, key=lambda d: self.expected(d, self.best(d, True)[1], True) / self.best(d, True)[2])
                k = self.best(pick, False)
            probe = None
            for d, idle in self.idle.items():
                if self.probe_plain_only and k[0] is not None:
                    break
                if d != k[0] and idle >= self.probe - 1 and (probe is None or idle > self.idle[probe]):
                    probe = d
            if probe is not None:
                k = self.best(probe)
        for d in self.idle:
            if d == k[0]:
                self.idle[d] = 0
                continue
            self.idle[d] += 1
            for i in range(MAX_ROWS):
                self.hits[d][i] *= 1 - self.forget
                self.n[d][i] *= 1 - self.forget
        return k

    def observe(self, d, w, accepted):
        if d is None:
            return
        for i in range(min(w, MAX_ROWS)):
            if i > accepted:
                break
            hit = 1.0 if i < accepted else 0.0
            if self.bayes > 0:
                keep = self.bayes_keep if self.two_stage else self.keep
                self.hits[d][i] = keep * self.hits[d][i] + hit
                self.n[d][i] = keep * self.n[d][i] + 1.0
            self.acc[d][i] += (1 - self.keep) * (hit - self.acc[d][i])


def replay(policy: Policy, kinds, accept: dict, start: int, end: int, order=()):
    """Tokens per second of the policy over the text from start to end (drafters numbered in `order`)."""
    policy.reset(kinds, order)
    t, tokens, ms = start, 0, 0.0
    while t < end:
        d, w, cost = policy.choose()
        a = accept[d](t, w) if d is not None else 0
        a = min(a, end - t - 1) if t + a + 1 > end else a
        policy.observe(d, w, a)
        t += a + 1
        tokens += a + 1
        ms += cost
    return tokens / ms * 1000


def main(d: str):
    policies = [
        Policy("before 0038: one estimate (moving average), an idle drafter drafts the 32nd round"),
        Policy("  the same, the 16th", probe=16),
        Policy("  the same, the 8th", probe=8),
        Policy("one estimate (means with a prior of 2 rounds, kept 0.95, forgotten 0.03 a round idle), each first",
               bayes=2.0, keep=0.95, forget=0.03, first_each=True, probe=10**9),
        Policy("two estimates (moving average for widths; means for drafters), each first", two_stage=True, bayes=2.0,
               forget=0.03, first_each=True, probe_plain_only=True),
        Policy("the engine's: two estimates, the drafters' optimistic by 2 standard errors", two_stage=True, bayes=2.0,
               forget=0.03, first_each=True, probe_plain_only=True, ucb=2.0),
    ] + [Policy(f"  bonus {c} standard errors, kept {k}, forgotten {f}", two_stage=True, bayes=2.0, bayes_keep=k,
                forget=f, first_each=True, probe_plain_only=True, ucb=c)
         for c in (1.0, 3.0) for k in (0.9, 0.95) for f in (0.02, 0.05)]
    models = sorted({f.split(".")[0] for f in os.listdir(d) if f.endswith(".costs.err")})
    results, now = {}, policies[0]
    for m in models:
        costs = load_costs(os.path.join(d, f"{m}.costs.err"))
        drafters = sorted({k[0] for k in costs if k[0]})
        for wl in ("code", "cache", "list", "reason"):
            paths = {dr: os.path.join(d, f"{m}.{dr}.{wl}.jsonl") for dr in drafters}
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            accept, start, end = {}, None, None
            for dr, p in paths.items():
                tiles, e = load_tiles(p)
                accept[dr] = Accept(tiles)
                start = tiles[0][0] if tiles else 0
                end = e if end is None else min(end, e)
            kinds = [(k[0], k[1], ms) for k, ms in costs.items()]  # the engine's order: plain, then each drafter's
            engine_order = tuple(dict.fromkeys(k[0] for k in costs if k[0]))
            row = {}
            for p in policies:  # the worse of the two orders of trying the drafters first
                both = min(replay(p, kinds, accept, start, end, o) for o in (engine_order, engine_order[::-1]))
                alone = {dr: replay(p, [k for k in kinds if k[0] in (None, dr)], accept, start, end) for dr in drafters}
                row[p.name] = (both, alone)
            results[(m, wl)] = row
    # Against today's engine: each drafter alone under the current policy, the better of the two.
    for p in policies:
        print(f"== {p.name}")
        ratios, alone_ratios = [], []
        for (m, wl), row in results.items():
            both, alone = row[p.name]
            ref = row[now.name][1]
            best = max(ref.values())
            ratios.append(both / best)
            alone_ratios += [alone[dr] / ref[dr] for dr in alone]
            print(f"  {m:4s} {wl:7s} both {both:6.1f} ({both / best:5.3f} of the better alone today)  alone " +
                  "  ".join(f"{dr} {v:6.1f}" for dr, v in alone.items()))
        print(f"  both against the better drafter alone today: geometric mean {gmean(ratios):.4f}, worst {min(ratios):.3f}"
              f"; each drafter alone against today: geometric mean {gmean(alone_ratios):.4f}, worst {min(alone_ratios):.3f}")


def gmean(xs):
    return prod(xs) ** (1 / len(xs))


def prod(xs):
    r = 1.0
    for x in xs:
        r *= x
    return r


if __name__ == "__main__":
    main(sys.argv[1])
