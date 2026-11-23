# DV: a Canvas Drafter Verified as a Tree

DV (bifill-dv's drafting method, see [prior art](../prior-art.md)) drafts with a **canvas**: a pass of the target's
own weights, with a small adapter, over N slots after the round's anchor, each slot a guess at a later token. A
**selector** grows a tree of drafts from the canvas, and the round verifies that tree exactly, so the output is plain
decoding's token for token (greedy and, for one seed, sampled). This page is the design: what DV computes, how it plugs
into flyx's rounds, when its rounds run, the row limit, its numerics, how its tokens per forward are counted and what
it is held to. DV runs on the CUDA backend; `cpu_ref` decodes a DV file as without it and says why. It drafts only
when named (`--drafter dv`, or `all`).

## What DV Computes

Next to the target's rows a round runs the canvas: N slots (32 in the checkpoints on Hugging Face, `panmz/bifill-dv`,
private) after the anchor. The canvas runs the target's base weights with an adapter of its own: a rank-256 LoRA on each
of a layer's seven linears (`y = W x + B (A x)`, no scale), its own RMSNorms before the attention and the MLP (`ln1c`,
`ln2c`) and before the head (`fnormc`), a learned MASK embedding, and a residual context:

    x_w = (cv_w == MASK ? mask_emb : E[cv_w]) + res_out (silu(g) * u),   [u | g] = res_in z_w

where `cv_w` is the slot's guess (its top token from the round before where that had probability at least `fill`, 0.9,
else MASK) and `z_w` its expected embedding from the round before (`sum_k p_k E[i_k]` over its top 16; 0 for a new
slot). Slot w sits at the anchor's position + 1 + w and sees the text up to and with the anchor and every slot (only
the slots up to itself if the canvas is causal), never the tree. Its logits give every slot its top 16 candidates (ties
to the lower id) with their probabilities under the whole vocabulary.

After the round the canvas **slides** by the n tokens committed: slot w takes slot w + n's state. The next round's
tree grows from the slid canvas with the selector, a bilinear score of each candidate b of a slot given each candidate
of the slot before as the token before it (the anchor for slot 0):

    S(prev, b) = (alpha log max(p_b, 1e-9) + <A E[prev] * H h, B E[b]> / sqrt(rank)) / temp

as a log-softmax over the slot's 16 candidates (h: the slot's normed hidden state; A, H, B: [rank, D] float32; temp 1).
Two trees grow from that table:

* **beam** (bifill-dv's DYN=T:W[:cov]): a beam of the W best nodes a depth by summed log-probability (with cov, plus
  each slot's `log max(sum_k exp(lp_k), 1e-6)` over its 16 candidates; minus 1e-4 a depth), then the T best nodes;
* **pool** (ST T/R/1): a fixed pool of rank tuples that leave the selector's best path once, at rank at most R; the T
  best by summed log-probability, minus 1e-6 a depth.

The round verifies the tree exactly: the deepest draft whose whole path is the target's choices is accepted (the lower
index among the deepest), then the target's choice after it. The first round after a prefill drafts T MASKs under the
anchor (an all-MASK canvas, never accepted), so it commits the anchor alone. `python/reference/flyx_ref/dv.py` follows
bifill-dv's `evalgen.py` round for round; `ops/draft/ref.h` (`selector_table`, `slot_coverage`, `beam_tree`,
`pool_tree`) and `ops/canvas/ref.h` (`canvas_input`, `canvas_stats`, `slide`) are its pieces in float64.

## How It Plugs In: the Canvas as Its Own Pass

A DV round is the existing tree round ([tree verification](tree-verification.md)) with DV as its drafter; no new node
kind:

```
DRAFT(dv, T, FLYX_DRAFT_TREE)   the tree from the slid canvas: the selector's table, then the beam or the pool
RUN embed; RUN attn/mlp x L; RUN head      the verify pass: the anchor and T drafts (target weights)
ACCEPT(-1, T)                   accept_tree and compact_path
COMMIT                          nothing for a Llama
CONTEXT(dv)                     the canvas pass (N rows), its statistics, the slide by the tokens committed
```

The prefill program ends with `CONTEXT(dv)`, which there means: the canvas starts over (every slot MASK, z = 0). DV's
forward k verifies the tree grown from canvas k - 1, and its canvas k sees the text up to the token the same forward
commits first: CONTEXT of round k runs after the verify pass has written the anchor's K/V at the round's first position
s (compaction never moves slot s), and the round's position has moved on by the tokens committed, `n = position - s`,
the slide. Everything lives on the device, so two rounds stay in flight and a round is replayed as a CUDA graph.

**A guard instead of bookkeeping.** The canvas is not positional and is not kept in checkpoints. DRAFT checks that the
canvas follows the round's position (`next_start == position`); after a prefill, a rewind, a restore, a snapshot load
or a round of another kind it does not, and the round drafts DV's first tree and the canvas starts over. Correctness
never depends on it: any drafts give the target's tokens.

**DV's warm-up** (`round_program(0, dv)`): plain decoding's round, then `CONTEXT(dv)`. Its canvas pass starts a canvas
that does not follow over at the round's anchor exactly as the pass of DV's first tree does, for one verify row instead
of the tree's: the same canvas (the golden test holds the trees after the one and the other the same, draft for draft).
The engine runs it before DV's first round after any round of another kind, a prefill or a rewind.

**DV's rounds are rounds among others.** The plans hold plain decoding, DV's trees of 15 and of its file's width, its
warm-up and lookup rounds, and the controller chooses among them every round (below). Its rounds are always trees
(`Model::drafts_trees`), its canvas follows its own rounds only (`Model::restarts`). Pinned (`--width N`), the widest
width it takes, at most N, runs every round, as bifill-dv's references draft. The plans say which widths the backend
refused and why (a refused warm-up refuses DV); where the backend takes none (`cpu_ref`; the CUDA backend for a tree
its kernels cannot draft) the engine decodes as without it, says why, and `flyx bench --json` records
`"drafter": "none"` and `"drafter_refused"`; a DV the caller named (`--drafter dv`) then fails the load. Out of GPU
memory is no refusal: the load gives back what the backend keeps for speed and lowers again. Trees read every
position of the cache (`FLYX_KV_READ_ALL`), and so does a file that holds DV whether DV is named or not
(`Model::file_drafts_trees`): under `--kv-method auto`, which reads StrataFlux past its switch, the file decodes the same
tokens with `--drafter dv` as with `--drafter none` (AGENTS.md gate 4). `--kv-method strata` with DV drafting is
refused at load.

**The fused canvas (not built).** bifill-dv's own engine runs the canvas rows in the verify pass: one weight read a
round instead of two. In flyx that needs rows in one pass that are not verified (their own mask, starting state and
weights): row groups, per-group norms, adapters and key maps, and a GEMM whose verify tiles keep plain decoding's bits
beside canvas tiles.

## The Description and the File (ABI 42)

* `FLYX_BLOCK_DV_LAYER` (`flyx_dv_layer_block`): one per target layer, naming the target's attention and MLP blocks
  whose base weights the canvas runs (never an adapter of the target's own), the layer's canvas norms, and
  `flyx_lora lora[FLYX_LINS]` (q, k, v, o, gate, up, down; `a [rank, in]`, `b [out, rank]`).
* `FLYX_BLOCK_DV` (`flyx_dv_block`): slots, the tree (kind, T, W, cov, R), top-k and res_k (16 each), the MASK token,
  causal, fill; the target's embedding and head blocks; its layers; `fnormc`, `mask_emb`, `res_in [2D, D]`,
  `res_out [D, D]` (bf16); the selector (A, H, B [rank, D] and alpha [1], float32; temp); its canvas state.
* `FLYX_STATE_CANVAS`: per slot the next input, the expected embedding, the top-k and their log-probabilities, the
  selector's projection; the position it follows. Only drafting reads it.

`models/dv.h` builds them from a Llama `.flyx` file's `dv` header section and its `dv.*` tensors; `models/llama` adds DV
as its one drafter and writes the programs above.

**Making a DV file.** `python/convert/export_dv.py` turns bifill-dv's checkpoints (read with
`torch.load(weights_only=True)`) into `canvas.safetensors` (bf16), `selector.safetensors` (float32) and `dv.json`
(slots, tree, fill, ranks, the MASK token's text; `--slots`, `--tree beam:T:W[:cov] | pool:T:R[:1]`, `--fill`: the
run's, default 0.9); `flyx_convert --model DIR --dv EXPORT` writes them into the model file. Refused: other canvases
(mixers, a spine, a see-gate, a second residual, a candidate memory), and a checkpoint whose configuration changes the
canvas pass with no tensor of its own (`soft_in`, `see_guess`); `flyx_ref/dv.py`'s `canvas_from_pt` refuses the same.
The two files below are made from bifill-dv's MiniCPM5-1B checkpoints on Hugging Face, `panmz/bifill-dv`, a private
repository: without access to it the files cannot be made, and DV's numbers here cannot be reproduced.
[models](../models.md) has the download and both conversions:

```sh
python -I python/convert/export_dv.py --canvas canvas/dva_res2.pt --selector selector/sel_orig_w8.pt \
    --tree beam:31:8:cov -o export
PYTHONPATH=python/convert python -m flyx_convert --model MiniCPM5-1B --dv export -o minicpm5-1b-dv-res2.flyx
python -I tests/python/convert/check_dv_convert.py minicpm5-1b-dv-res2.flyx export \
    --canvas canvas/dva_res2.pt --selector selector/sel_orig_w8.pt --hf MiniCPM5-1B
```

| File | Canvas + selector (`panmz/bifill-dv`) | Tree | Target |
|---|---|---|---|
| `minicpm5-1b-dv-res2.flyx` | `canvas/dva_res2.pt` + `selector/sel_orig_w8.pt` | beam 31:8:cov | MiniCPM5-1B |
| `minicpm5-1b-dv-b2h.flyx` | `canvas/dva_cd_b2h.pt` + `selector/sel_b2h_w8.pt` | beam 31:8:cov | MiniCPM5-1B with its V-LoRA `v_lora/lora_g_b2h.pt` |

`check_dv_convert.py` holds the file's DV tensors to the export byte for byte, the reference's reading of the `dv`
section to the `DvConfig` a run of the export's settings uses, and the reference's loading of the file to its loading
of the checkpoints. The 32-slot DV weights take 0.38 GB on the GPU beside MiniCPM5-1B's 2.26 GB (`flyx run -v`'s
`weights ... on the GPU` line, `minicpm5-1b-dv-res2.flyx` against `minicpm5-1b.flyx`).

## When DV's Rounds Run

After long prompts DV's rounds accept little: after the three `bench/prompts/summarize_*.txt` prompts (3110 to 5949
tokens) `minicpm5-1b-dv-res2.flyx`'s rounds of 31 drafts give 1.75 to 1.95 tokens a round against 5.02 on GSM8K's
first 100 questions ([benchmarks](../benchmarks.md#5-minicpm5-1b-with-dv-rtx-5060-ti-8-gb)), and each row, verify or
canvas, reads every key. So the controller (`SpecController`, `core/speculation/controller.h`) chooses DV's rounds
where they pay:

* **Costs at the context.** Where the plans hold DV's warm-up, every round is timed at load at 64 and at 1024 tokens of
  context (each twice, the faster kept), linear past the first; costs another program disturbed
  (`spec::costs_plausible`) are timed again, twice at most. Other models' rounds are timed once.
* **A tree round's tokens.** A tree without a trunk is rated by the tokens its rounds give: what the engine's sessions'
  rounds of its width gave (`SharedTokens`, kept 0.998 a round; the guess 4.0 counting as 8 rounds), times the reply's
  level, how DV's rounds in this reply fared against those means ((8 + sum of tokens / mean) / (8 + n), kept 0.95 a DV
  round, kept while DV is away, new with every reply). The width that runs is the one the engine's means give more
  tokens a ms at this context; the other drafts one DV round in 64.
* **The rule.** DV stays while its tokens per ms are at least plain decoding's at this context. It is entered (its
  warm-up, then its rounds) where an entry pays over 8 rounds, (1 + 8 T) / (warm-up + 8 rounds) against plain
  decoding's rate. Within a reply it comes back after leaving only by a probe: after 32 rounds away, when the engine's
  credit covers a probe at its worst (the warm-up and a round accepting nothing, against two plain tokens); the credit
  is 0.5% of every round's time plus what DV's rounds gave over plain decoding's rate, between 0 and 1.25 probes at
  their worst, one for all the engine's sessions (0 at load). A probe holds DV for 2 rounds, up to 8 while its rounds
  pay. Where DV does not pay its probes cost about 0.5% of the decoding time.
* No lookup round while DV runs or would be entered (a lookup round leaves the canvas behind), and no DV tree while the
  session's rounds go out with other sessions' (the scheduler's rounds of several sessions: DV's trees run alone, its
  warm-ups only beside warm-ups).

`bench/replay/dv_choice` replays the controller as built on the pinned runs' rounds kept in `bench/replay/dv_traces/`
(GSM8K, short chat requests, GSM8K as 4- and 10-shot chats, the summarize prompts, and sets that mix them), with each
constant set otherwise one at a time: `python -I bench/replay/dv_choice.py table BUILD/bin/dv_choice`. The table it
prints is what set the constants (the level's prior counting 8 rounds, the other width one round in 64, entries paid
over 8 rounds, the level kept while away, the probe share 0.5%).

DV still drafts only when named: after long prompts a new engine's first reply may enter DV on the guess.

## Rows: the File's Width, and 15

DV drafts trees of its file's width T, min(T, `FLYX_MAX_WIDE_ROWS` - 1): 31 in MiniCPM5-1B's files. Rounds of 1 + 31
rows run on the rows wide lookup rounds have (ABI 36, `FLYX_MAX_WIDE_ROWS` = 64 rows and round results; chains,
batching and `flyx_room` keep 16; see [tree verification](tree-verification.md)): on CUDA a model that holds DV has
the rows of its widest round under `wide=auto`, 64 with `wide=1`, and 16 with `wide=0`, where its rounds of more
drafts are refused and said. The kernels draft up to 63 drafts over up to 63 depths (`dv_tree_takes`). DV offers two
widths where its file's is wider than 15: `draft_widths` = {15, min(T, 63)}, each rated by the tokens its own rounds
give. `--width N` pins the widest at most N.

Such rounds keep every verify row's one-row bits (AGENTS.md gate 4): a round of more than 16 rows runs as a wide lookup
round does, its GEMMs on the 1..16-row kernels' arithmetic for every 16-row tile (`k_gemm_skinny_mt` for bf16 weights,
the weights read once for up to four tiles; `gemm_fp4_rows` for NVFP4), its head and choice 16 rows at a time, each
tree row at its depth's position ([numerics](numerics.md) classes R and C). The canvas pass has no such rule: only
drafting reads it, so it must be deterministic, not row-count invariant.

Wide lookup rounds and DV: lookup rounds, of any width, run only where the controller would not run DV (no lookup
round while DV drafts), never batched; a lookup round leaves DV's canvas behind, so DV's next round after one is its
warm-up. Penalties, relaxed acceptance and a grammar keep lookup rounds at 15 drafts; DV's trees take them at any
width.

A session's round buffers hold the wide rows, and the K/V caches are N positions longer (`cap = max_ctx +
max(32, R) + N` for round buffers of R rows; the canvas's slots from `s0 = max_ctx + max(32, R)`). The wide rows and
the plans past 16 rows widen every session, in the session size the session cap divides by (`wide` in
[the server's options](../server.md) has a 27B's): on MiniCPM5-1B's DV files at 2048 tokens 0.13 GB a session with
`wide=0`, 0.15 with DV's 32 rows (its graph room 15.1 and 19.8 MiB of it), 0.17 with `wide=1`'s 64 (22.8 MiB).

## Numerics

* Verify rows: the target's own kernels in a tree round: plain decoding's bits.
* The canvas pass: past 16 slots its GEMMs take bf16 inputs (each input rounded once, as prefill's GEMMs round theirs)
  on `k_gemm_skinny_mt` in one part (`Launch::draft`, `gemm_draft`: float32 sums over K's chunks, 8, or 16 from
  K = 4096, by K alone on every GPU, so each weight of a grouped launch gets the sums of a launch of its own), the
  adapters as two more GEMMs whose products add (the `a` product rounded to bf16 as the `b` GEMM's input), the residual
  context's two products as a round's (three parts), attention over its own key map. Deterministic, the same partition
  on every GPU; its rounding moves which drafts are accepted, never the output.
* The selector's tables `A E` and `B E` ([vocab, rank], float32) are built when a plan first drafts with them; the
  table and the trees in float32 with ties to the lower index (bifill-dv's `torch.topk` leaves them open, so its trees
  and flyx's may differ at exact ties; only aggregates are compared).

## Counting Tokens per Forward

bifill-dv's convention, from flyx's per-round data (`flyx bench --prompts --json` writes each round's tokens and
accepted drafts): with a_k the drafts round k accepted (k = 1..R; DV's first round accepts none) and
L = min(512, the position of the first end token + 1) on the answer,

    C(K) = K + sum_{k <= K} a_k,   need = min{K : C(K) >= L} (R + 1 if never),   TPF = sum_q L_q / sum_q need_q

flyx bench's own `(new tokens - 1) / rounds` is reported beside it. `tests/e2e/check_dv.py` computes both, checks that
the drafted answers are plain decoding's token for token, scores GSM8K as bifill-dv does, and compares with a reference
run of bifill-dv's engine: within 1.5% on all 1319 questions, 2% on 500, 3% on 100.

## What DV Is Held To

* **Tokens: flyx's own plain decoding**, same build, same `--kv 16`, same backend options, byte for byte on every
  prompt (bifill-dv's outputs cannot serve: its lossless runs equal its own plain decoding on 526 of 1319 answers).
* **Tokens per forward: bifill-dv's references** (bifill-dv's unpublished result files, runs of its engine on all 1319
  GSM8K questions: `dyn1319_orig.pt`, `DVDT@0.9`, DYN 31:8:cov on `dva_res2`: 5.2067; `dyn1319_b2hcd.pt`, the `g_b2h`
  line with its V-LoRA: 7.1791; `tests/e2e/dv_ref_json.py` reads them). On all 1319 questions, `--kv 16 --ctx 2048`, DV
  drafting its file's 31 every round: `minicpm5-1b-dv-res2.flyx` 5.2534 (+0.90%), `minicpm5-1b-dv-b2h.flyx` 7.1444
  (-0.48%), every answer plain decoding's:

  ```sh
  flyx bench minicpm5-1b-dv-res2.flyx --prompts gsm8k_test.jsonl \
      --suffix '\nPlease reason step by step, and put your final answer within \boxed{}.' \
      --kv 16 --ctx 2048 --drafter dv --width 31 --json dv.json
  flyx bench minicpm5-1b.flyx --prompts gsm8k_test.jsonl \
      --suffix '\nPlease reason step by step, and put your final answer within \boxed{}.' \
      --kv 16 --ctx 2048 --no-lookup --json plain.json
  python -I tests/e2e/dv_ref_json.py dyn1319_orig.pt -o ref_dyn.json   # bifill-dv's result file, unpublished
  python -I tests/e2e/check_dv.py --drafted dv.json --plain plain.json --ref ref_dyn.json
  ```

* **Drafts: the Python reference.** `flyx_ref/dv.py` matches bifill-dv's own functions case for case
  (`tests/python/reference/check_dv_bifill.py`, run by hand with bifill-dv's code at `839ac35`, which is private too)
  and emits plain decoding's tokens on the tiny Llama with every tree. Its float64 drafts are what the CUDA rounds'
  drafts are held to on the tiny Llama (`golden_cuda_llama_dv*`). On MiniCPM5-1B `python/reference/dv_forced.py` forces
  it to a flyx run's answers and, with `--replay`, to each of flyx's rounds' commits, so that `check_dv.py --oracle`
  holds flyx's rounds to its a_k round for round (the reference's a_k in 97.8% of the rounds of the first 50 questions
  on `minicpm5-1b-dv-res2.flyx`).

## DV's Rounds on the CUDA Backend

`regions/plan.cpp`, `regions/layers.cpp` (`dv_draft`, `dv_context`), `runtime/state.cpp`, `runtime/runner.cpp`,
`loader/bind.cpp`:

* **DRAFT** (`dv_draft`): the round's selector table over the slots the tree reaches (`min(N - 1, T)`), then
  `dv_tree_beam` or `dv_tree_pool` into the round's drafts and its `TreeRows`. A canvas that does not follow the round
  drafts DV's first tree; the test's oracle (`set_oracle`) a chain of its tokens.
* **CONTEXT** in a round (`dv_context`), after the accept: `canvas_input`, then per layer the target's attention and
  MLP blocks with their base weights and the layer's canvas norms, each linear's adapter as two GEMMs more;
  `attn_prep` and `attention` with `CanvasKeys` (slot w at `P->start + 1 + w`, its K/V in the scratch slot `s0 + w`;
  keys the text up to and with the anchor and the canvas's slots), never the low-bit copy; with the K/V in host memory
  the text's host part staged once a layer; `fnormc`, the head over the N rows, `canvas_stats` and `dv_slide`. A
  layer's GEMMs on the same rows go in one launch each (`gemm_draft`): 8 launches a layer instead of 15.
* **CONTEXT** in a prefill, and every rewind, restore and snapshot load: `forget_canvas`, `next_start = -1`.
* **A no-op round** (launched before the round that ended the reply was read) leaves the canvas as it was: in a round's
  graph the canvas pass is the body of a conditional node (`dv_context_unless_noop`), which `dv_live` sets after the
  accept.
* **State**, all in `build_seq`'s allocations, so `seq_bytes` counts it: the canvas, what a pass leaves (`DvStats`),
  the round's table, the pass's rows and a GEMM scratch for N rows, and the longer K/V caches.
* **Graphs**, in the room `seq_bytes` counts for a session's round graphs ([architecture](architecture.md)): a graph of
  each of DV's plans (its trees are drafted trees, its warm-up a round like plain decoding's), each counted by its
  nodes, the conditional node's body (the canvas pass) among them (`graph_size`). On `minicpm5-1b-dv-res2.flyx` the
  graph of a round of 31 drafts holds 666 kernels with its body, counted at 2.5 MiB; the eight plans a session's room
  holds (plain decoding, DV's two trees and its warm-up, four lookup widths: 9, 11, 15 and 31 on its wide rows) come
  to 19.8 MiB with its share of a batch graph and the driver's part (34.9 with the V-LoRA, on
  `minicpm5-1b-dv-b2h.flyx`), and a session of 2048 tokens to 0.15 GB (`flyx run -v`). `cuda_device_memory_dv*` hold
  DV's graphs on the tiny Llama within what they are counted at, and the conditional nodes with their bodies on their
  own: 150 graphs (50 sessions) took 14 MiB more with the pass conditional than inline (backend option
  `dv_conditional=0`), 18 MiB with the V-LoRA, and are counted at 21.6 MiB more. A driver that refuses the conditional
  node, or does not give its body to be counted, gets the round with the pass inline.
* **Weights:** the adapters of every DV layer in one allocation; the selector's token tables (267 MB on MiniCPM5-1B
  at rank 256), made when a plan first drafts with them, are counted with the weights.
* **Refused in `lower`** (the plans then decode without DV and say why): a tree the kernels cannot draft
  (`dv_tree_takes`) and a canvas on attention blocks with an output gate.

The golden tests (`golden_cuda_llama_dv*`, [testing](../testing.md)) hold plain decoding's tokens in every mode (two
rounds in flight, graphs on and off, rewind, restore, stop strings, sampled and with penalties, FP8 and host K/V,
rounds mixed with lookup rounds, the scheduler's requests at once) and DV's drafts against `flyx_ref/dv.py` in float64
rounding what the backend rounds, on seven tiny DV files (15 and 31 drafts, beam and pool, 8, 16 and 32 slots, causal
and not).

## The Target's Own Adapters (V-LoRA, ABI 43)

bifill-dv's best runs use a LoRA of the target's own (the V-side LoRA: rank 64 on each of a layer's seven linears, no
scale; `lora_g_b2h`), unmerged on every row the target runs (the prompt, plain decoding, the verify rows), while the
canvas rows run the base linears. flyx does the same:

* **Description (ABI 43).** `flyx_lora lora[4]` on an attention block (q, k, v, o), `lora[3]` on an MLP block (gate,
  up, down): all of a block's linears or none, one rank, and a scale (`y = W x + scale b (a x)`). `models/llama` reads
  them from the header's `lora` section and `target.L{l}.{attn,mlp}.<linear>.lora_{a,b}` (bf16).
* **`.flyx` format 2.** A file with adapters is written as format 2, its header requiring them
  (`"requires": ["lora"]`): a build that reads format 1 only refuses it instead of decoding the plain model, and this
  one refuses adapters a file does not require and a part it does not read. A file without adapters stays format 1.
* **Converter.** `export_dv.py --vlora lora_g_b2h.pt` writes `vlora.safetensors` and `vlora.json`, with or without a
  DV; `flyx_convert --dv` takes the export's V-LoRA with its DV, `--lora` a V-LoRA alone.
* **Backends.** `cpu_ref` adds `scale b (a x)` in float64 (`ops::linear_lora`). The CUDA backend runs each block's
  adapters beside its linears on the rows they read (`lora_down`, then `gemm` on its product, after the linear's own),
  so a round's rows past 16 keep their one-row bits on `k_gemm_skinny_mt`, and a fused region gives the generic one's
  bits (`test_lora`). Adapters on GGML block weights, on an attention with an output gate, or on an MTP layer are
  refused.
* **Numerics.** float32 sums; unmerged, because a bf16 merge `bf16(W + B A)` loses about 70% of the delta.

On GSM8K's 1319 questions with `lora_g_b2h`, plain decoding gives 846 correct answers (bifill-dv's own plain decoding
with this LoRA also gets 846), and DV (`minicpm5-1b-dv-b2h.flyx`) gives plain decoding's answers on all 1319.