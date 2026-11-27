# Speculative Tree Verification

Design and ABI contract for bitwise-exact verification of speculative token trees on hybrid attention + gated delta net (GDN) models.

Status: **Phase 1 implemented** (trees up to `FLYX_MAX_ROWS = 16` rows, ABI 23/24, CUDA backend). **Phase 2** implemented for attention-only models (tree rounds up to `FLYX_MAX_WIDE_ROWS = 64` rows on the wide rows of ABI 36: DV's trees of 31, [DV](dv.md)); trees past 16 rows on gated delta nets remain a design note.

---

## 1. Definition and Core Requirement

A tree round provides `token[i]` and `parent[i]` (`parent[0] = -1` for the anchor row 0; parents strictly precede children). Row `i` sits at logical position `base + depth[i]`, attending to all committed keys plus its ancestors and itself (never siblings or cousins). Acceptance walks from row 0 down the child matching the parent's chosen token and commits that path into contiguous KV slots and recurrent state. A chain is the special case `parent[i] = i - 1`.

**Gate 4 requirement ([AGENTS.md](../../AGENTS.md)):** Every node's forward pass and token choice must produce the **exact same bits** as plain decoding along its ancestor path. Approximate/closed-form hybrid tree methods (e.g., TreeWY, Bole, LumoTree) alter floating-point rounding and violate Gate 4, so `flyx` executes every tree node with exact sequential ancestor arithmetic.

---

CUDA refuses given-tree and DFlash2 tree plans for a program containing relative attention or ShortConv. Inkling-2 uses chain verification because branching would require per-ancestor relative-attention and convolution state. Ring-backed state has a rewind floor: once a write overwrites a required position, the session must recompute from still-valid history.

## 2. Bitwise-Exact Tree Design (Phase 1, ≤ 16 Rows)

1. **Tree payload (ABI 23, `FLYX_GIVEN_TREE`):** `flyx_gen_params` carries `drafts`, `n_drafts`, and `parents`. Host launch computes `TreeRows` (per-row `depth` and `path` ancestor table) and streams it via a pinned ring buffer so a single captured CUDA graph handles any tree topology of width `3`, `7`, or `15` (`Engine::tree_widths()`, `tree_plan()`). Chain rounds stay unchanged.
2. **RoPE & KV slots (`attn_prep`):** Rotates row `i` to `base + depth[i]` and writes K/V to slot `base + i`.
3. **Attention (`attention`, `attention_split`, `attention_rows_mma`):** Logical key `j >= base` maps via `tree_slot` to `base + path[i][j - base]`. Tiles and key splits follow logical position order so float32 sums match plain decoding bit-for-bit. Float32 kernels run one row per block; tensor-core decode attention (`≥ 8192` keys) shares pre-round key tiles across all rows and processes round-key tiles per row.
4. **Hybrid GDN layers (`gdn_conv`, `gdn_recur_tree`):** Short causal convolution gathers inputs from the row's 3 preceding ancestors (falling back to the committed window above the anchor). `gdn_recur_tree` walks rows sequentially; whenever `parent[i] != i - 1`, it reloads the round start state and replays the ancestor path with identical operations. Tree plans use the 4-kernel GDN sequence (never the chain-only fused 1-kernel round).
5. **Sampling & penalties (`penalize_rows`, `choose_rows`):** Row `i` draws Gumbel-max noise keyed on logical position `base + depth[i]` and counts only its ancestors' drafts for presence/frequency penalties.
6. **Accept & compact (`accept_tree`, `compact_kv`, `gdn_compact`, `compact_rows`):** Walks the accepted path from row 0, compacts accepted KV slots/e4m3 scales, GDN replay rows, tokens, final hidden states, and drafter taps into consecutive slots `0..n`, then commits and updates drafter contexts identically to chain rounds.
7. **Opt-in relaxed acceptance (lossy, off by default):** `accept_min_p` (target prob ≥ `tau * p_max`) and `accept_typical` (Medusa rule: prob ≥ `min(eps, sqrt(eps) * exp(-H))`), using `row_stats` + `accept_tree` in dedicated CUDA graphs. Picks the highest-probability qualifying child at each step and ends with the target's exact choice.

---

## 3. DFlash2 Dynamic Trees (ABI 24, `--draft-trees`)

DFlash2 scores 16 candidates per slot across 7 draft positions. With `FLYX_DRAFT_TREE` (enabled via `--draft-trees` or `FLYX_DRAFT_TREES=1`):
- `selector_tree` (verified against `ops/draft/ref.h::best_first_tree`) writes a dynamic tree directly on device into `TreeRows` (zero host upload; one CUDA graph per width `9`, `11`, `15`):
  - **Trunk first:** Preserves the greedy chain walk so a tree never accepts fewer tokens than its chain.
  - **Best-first branches:** Expands highest log-probability path children (EAGLE-2 style) over the remaining draft budget.
- `SpecController` times each tree width at load alongside chains and runs trees only when learned branch yield per round cost beats every chain. Off by default because on current 27B hardware, extra rows beyond width 8 cost more than the +0.15–0.23 extra tokens yielded by branches.
- **DSpark** grows the same trees from its Markov walk: a position's candidates are its row's (the block is not causal, so every branch shares them), each child scored by the head's logit plus the Markov bias from its parent's token, over up to 1024 candidates a position (`selector_tree`'s dynamic shared memory). Sampling, the trunk is the walk's coupled chain. On by default with a DSpark drafter on a Qwen3 or Llama target (`--no-draft-trees` turns them off), where a tree of 15 drafts costs about a chain of 7: Qwen3-8B bf16 on an RTX PRO 6000 (DeepSpec's 9 datasets, 20 prompts each, greedy) verified 5.51 tokens a round against 5.01 for chains of 7 (360 against 334 tok/s), above DeepSpec's own evaluator (chains) on every dataset.

---

## 4. Verification

- `cuda_ops` (`test_select`): `selector_tree` vs `best_first_tree`.
- `golden_test` (CUDA, across BF16, NVFP4, unfused, GDN 128, tensor-core attention, Qwen3 BF16/GGUF): hides plain decoding's continuation on random paths inside 3/7/15-draft trees over 20 rounds (greedy, seeded sampling, penalties, FP8 KV, `min_p = 1.0`), asserting bit-exact tokens, exact path acceptance, and bit-exact subsequent MTP drafter rings after compaction.
- `tree_bench MODEL [--ctx-at P] [--drafted T]`: measures verification cost vs plain decoding break-even (`t(R) / t(1)`).

---

## 5. Phase 2: Trees Past 16 Rows (on the Wide Rows)

- **Row budget.** A tree round runs on the rows wide lookup rounds have (ABI 36: `FLYX_MAX_WIDE_ROWS = 64` rows, a round result of 64 tokens and drafts), where the backend has them (CUDA: backend option `wide`, and a model that holds DV has its widest round's rows under `auto`); chains, batched rounds and `flyx_room` keep 16. An engine lowers no tree past 16 rows of its own but DV's; tests and `tree_bench` lower wider given trees with `wide=1`. `accept_tree` stops a round at the context's end, so the round in flight behind it writes inside the K/V cap (`max_ctx` plus the wide rows).
- **Each row keeps its one-row bits.** Such a round runs as a wide lookup round does (`Launch::round_rows`): NVFP4 GEMMs through `gemm_fp4_rows`; bf16 GEMMs on `k_gemm_skinny_mt`, which reads each span of weights once for up to four 16-row tiles and gives each tile the MMAs, chunk order and sums the 1..16-row kernel gives it (three input parts); `gemm=rows` and GGML block GEMMs 16 rows at a time; the generic regions; the head and the choice 16 rows at a time, a tree's row choosing at its depth's position and counting its ancestors' drafts for penalties (`tree_from`), the relaxed rules' statistics on every 16 rows and each draft's logit in its parent's row gathered for `accept_tree` (`child_logits`). Tree rows never take `attention_flash`: a row takes `attention_rows_mma` by its own key count, as decoding it does.
- **Refused:** such rounds on gated delta nets (at lowering; wide chains run there) and over a strata copy (at launch: the wide strata strips are a chain's). The rounds in flight count their own rows (`ahead()`), so a round's strata phase and strata's fixed stagings take the reach a round stages (position + ahead + max(rows, 16)).
- **Cost:** `tree_bench MODEL --ctx-at 1024 --widths 1,15,31,63 --kv 16` times every width from the same context. On `minicpm5-1b-dv-res2.flyx` (BF16; RTX 5060 Ti, driver 610.88), `tree_bench minicpm5-1b-dv-res2.flyx --ctx-at 1024 --widths 1,15,31,63 --kv 16`: a round of 32 rows costs 1.90 to 1.94 plain rounds (a plain round 4.43 to 4.46 ms at 1024 tokens), one of 64 rows 3.36 to 3.41.
- **Remaining:** branch-grouped round-key tiles in `attention_rows_mma` and branch-state caching in `gdn_recur_tree` for trees past 16 rows on gated delta nets.
