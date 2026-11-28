# Models, Drafters, and Long-Context Sizing

Reference for supported model architectures (Qwen3.5 / Qwen3.8, Qwen3, Inkling-2, Llama), file formats, Hugging Face Hub loading, speculative drafters, and GPU/RAM/disk KV sizing.

---

## 1. Supported Models and Formats

| Format | Supported Architectures | Supported Weight Types | Refused / Unsupported |
|---|---|---|---|
| **GGUF** (native, zero conversion) | `qwen35` (Qwen3.5 / Qwen3.8 hybrid GDN + gated attention, plus multimodal vision via `--mmproj` `qwen3vl_merger` `mmproj-*.gguf`), `qwen3` (dense), `inkling2_loop` (Inkling-2 looped ShortConv + RelAttn + sparse MoE), `gpt-oss` (gpt-oss-20b / 120b: attention with biases, a learned sink a head, sliding windows of 128 on every other layer and YaRN rotary; sparse MoE with MXFP4 experts, their biases and gpt-oss's clamped SwiGLU; harmony replies: the analysis channel as thinking, messages to `functions.NAME` as tool calls) | `F32`, `F16`, `BF16`, `Q2_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, mixed quants, `MXFP4`, GGML `NVFP4`; a split file (`-00001-of-0000N.gguf`) with all its parts in one folder | Inkling-2 vision towers (`mmproj`), scaled RoPE (but gpt-oss's YaRN), Llama GGUFs; a gpt-oss .flyx |
| **`.flyx`** (via `python/convert`) | Qwen3.5 / Qwen3.8, Qwen3, Llama (e.g. MiniCPM5-1B) | `BF16`, NVIDIA ModelOpt `NVFP4` | Vision / MoE; a Llama GGUF |

Inkling-2-Mini stores 24 layers. Running its 12-layer core three times gives 48 effective layer applications. It combines sliding-window and global relative attention, ShortConv and AttnRes with 128 experts, routing each token to eight. Its trained window is 1,048,576 tokens. The usable window depends on available VRAM and host RAM.

Key execution properties:
- **Packed GGUF blocks:** GGML block weights remain packed on GPU and are decoded in registers (their inputs: [Precision](#precision)).
- **Host-resident untied embeddings:** When output head and token embedding are untied (e.g., Qwen3.8-27B's 0.42 GB embedding), the embedding stays in host RAM where under 32 GB of GPU memory are left after the weights, its rows fetched over PCIe at negligible cost; above that it goes to the GPU (`--backend-opt embed=host` or `embed=gpu` chooses).
- **Vision encoder offload and quantization:** Vision weights (`mmproj-*.gguf`, BF16/F16/F32 or pre-quantized GGML blocks) default to `--vision-device disk` + `--vision-quant q8`: floating-point vision linears are quantized on the fly to `Q8_0` (~46% smaller; 0.50 GB on Qwen3.8-27B, KL 0.00125 / 98.75% top-1 vs BF16 over 800 multimodal positions; pre-quantized `mmproj` files stay packed) and written to an auto-deleted temp file flushed clean to OS Standby/Page Cache (**0 bytes resident VRAM, 0 bytes locked host RAM at KV-budget time** — zero impact on `--kv-ram auto` context capacity). During `encode_vision`, weights stream through a 3-stage pipeline (lazy read-only OS page-cache map → 16 MiB lazy pinned host ring → double-buffered GPU staging) with **bit-identical outputs** to GPU-resident execution (~54–64 ms warm ViT on RTX 5070 Ti). Override with `--vision-device host|gpu|auto` (`host`: pinned host RAM; `auto`: GPU when >= 8 GB free after weights, else `disk`) and `--vision-quant exact|nvfp4` (`exact`: file precision; `nvfp4`: ~70% smaller, opt-in).
- **GGUF format details:** See [docs/design/gguf.md](design/gguf.md).
- **Llama (MiniCPM5-1B):** Qwen3's layers without the per-head q/k RMSNorms (`LlamaForCausalLM`), from `.flyx` only (llama.cpp's converter reorders a Llama's q and k rows, so a Llama GGUF is refused). MiniCPM5-1B's pre-tokenizer cuts numbers into groups of three before the Qwen2 split, and its `<s>` goes before every raw prompt as its tokenizer puts it; its chat template renders as transformers renders it (tool calls are not read back yet, so a request with tools is refused). Its K/V cache is binary16 by default: FP8 moved its next-token distributions visibly (KL 0.0064, top-1 97.2%, perplexity +1.4% against binary16), so a Llama keeps binary16 unless `--kv fp8`; Qwen3.5 and Qwen3 keep FP8.
- **`.flyx` format 2:** a file whose header requires a part a reader must not ignore (`"requires": ["lora"]`, the V-LoRA below) is written as format 2, so builds that read only format 1 refuse it instead of decoding the plain model; every other file stays format 1.
- **LoRA adapters (PEFT, `--lora DIR|org/name[:FOLDER]`):** an adapter as Hugging Face PEFT saves it (`adapter_config.json`, `adapter_model.safetensors`) runs unmerged beside the model's linears on every pass, its products summed in float32: attention q/k/v/o, MLP gate/up/down and the gated delta net's five in-projections and output (Qwen3, Qwen3.5 / Qwen3.8, Llama; `.flyx` and GGUF, GGML block weights included), any rank, scale alpha / r or alpha / sqrt(r) (rsLoRA). F32 and F16 tensors are rounded to bf16 at load (KL against their float32 products in [numerics](design/numerics.md#against-llamacpp)). Refused with the reason, never run in part: DoRA, ranks or alphas per module, `modules_to_save`, a bias, an activated LoRA (`alora_invocation_tokens`), retrained token embeddings, PiSSA / OLoRA / CorDA / LoftQ initializations not converted on saving (they assume a base those initializations changed), the head's, an embedding's or a vision tower's adapters, layers past the model's, tensors in F8, integers or a pickle (`adapter_model.bin`). The dashboard's LoRA tab lists the ones Hugging Face files for the model's base (`GET /v1/hub/loras`, `GET /v1/hub/lora`: [server](server.md)).

### Precision

`--activations PREFILL[/DECODE]` (or `FLYX_ACTIVATIONS`) names the inputs a weight multiplies in prefill passes (more than 16 rows), then in decoding rounds (1 to 16 rows: plain decoding and verifying drafts); `/DECODE` left out means the same as prefill. Unset, each weight takes what its file says. [Numerics](design/numerics.md#against-llamacpp) holds each choice against llama.cpp's.

| Weights | Unset | `--activations` |
|---|---|---|
| `BF16`, `F16` | prefill bf16 (tensor cores); decode three bf16 parts (24 bits: exact products, float32 sums) | — |
| `F32` matrices (Inkling-2's routers and loop controllers, for one) | float32; bf16 on tensor cores in prefill passes of more than 384 rows or more than 64 outputs | — |
| GGML blocks (GGUF `Q*`, `IQ*`, `MXFP4`) | `q8`: 8 bits with a scale per 32 (llama.cpp's `q8_1`) | `exact`: binary16 over a row's power-of-two scale |
| `NVFP4` matrices (ModelOpt `.flyx`, GGML `NVFP4`) | `nvfp4/bf16` on GPUs with FP4 tensor cores unless the file declares bf16; else bf16 | `nvfp4/bf16` whatever the file declares; `exact`: bf16 |
| `NVFP4` MoE experts (3D) | bf16 | — |
| `MXFP4` MoE experts (gpt-oss) | `q8` (llama.cpp's) | `bf16` (vLLM's, Marlin); `mxfp8`: e4m3 with a power-of-two scale per 32 (SGLang's, FlashInfer CUTLASS; sm_120); `mxfp8/q8` |

A row gets the same bits in a round of any width (so speculation never changes the output) and in a prefill pass of any length (so a reused prompt prefix continues as a fresh prefill does); prefill and rounds agree within float32 sums, or within their inputs' rounding where they differ (`nvfp4/bf16`, `mxfp8/q8`).

Beside the linears:

- **K/V cache** (`--kv`): FP8 (e4m3, a float32 scale per position and head) where under 32 GB of GPU memory remain after the weights, binary16 above that and on Llama models.
- **Residual stream:** float32; bf16 in prefill passes of stacks whose every residual add is a W4A4 GEMM.
- **GDN recurrence:** chunked prefill with bf16 products on sm_86, 89, 120 and 121; float32 elsewhere and in rounds.
- **LoRA adapters:** bf16 weights, float32 inputs and sums.
- **Vision encoder** (`--vision-quant`): `q8` (`Q8_0` weights, 8-bit inputs) by default on floating-point `mmproj` files; `exact` keeps the file's type; `nvfp4`.

By model:

| Model | Files | Unset | `--activations` |
|---|---|---|---|
| Qwen3.5 / Qwen3.8, Qwen3 | GGUF; `.flyx` `BF16` or `NVFP4` | GGUF `q8`; `NVFP4` files `nvfp4/bf16` (Blackwell) | `exact`, `nvfp4/bf16` |
| Llama (MiniCPM5-1B) | `.flyx` `BF16` or `NVFP4` | as Qwen3's `.flyx`; binary16 K/V | `exact`, `nvfp4/bf16` |
| Inkling-2 | GGUF | `q8`; its `F32` matrices as above | `exact` |
| gpt-oss | GGUF (`MXFP4` experts) | `q8` | `bf16`, `mxfp8`, `mxfp8/q8`, `exact` |

---

## 2. Hugging Face Hub Resolution

The CLI and server model loader accept a local path or a Hugging Face Hub specifier for targets and companion files. Sidecar discovery happens in the app loader. When opening files directly through the C or Python API, pass the companion paths explicitly.

```sh
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
flyx run unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS --chat -p "Hello"
flyx serve unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S --drafter-file z-lab/Qwen3.8-27B-DFlash2
```

| Rule | Behavior |
|---|---|
| `org/repo:QUANT` | Selects the GGUF carrying `QUANT` (falls back to `UD-<QUANT>` when unique). Passing `org/repo` without `:QUANT` lists available GGUFs and sizes. |
| `org/repo:QUANT@REV` | Pins a branch or commit hash (also used to address older cached snapshots). |
| Auto MTP sidecar | If a GGUF lacks an embedded MTP layer and the repo contains `MTP/*.gguf` (e.g., `unsloth/Qwen3.8-27B-GGUF`), `flyx` fetches and attaches it automatically. |
| Auto DSpark sidecar | With no drafter file named (or `--drafter dspark`/`all`), `flyx` fetches and attaches the repo's first `dspark-*.gguf` (e.g., `unsloth/Inkling-2-Mini-GGUF`), or, for a model GGUF on disk, a `dspark-<model name>*.gguf` beside it (`Inkling-2-Mini-UD-IQ3_XXS.gguf` pairs with `dspark-Inkling-2-Mini-Q8_0.gguf`); either only when its header says it is a DSpark drafter (`arch=dflash`, `has_markov_context=true`). Use `--drafter none` for no neural drafter. The modes `mtp` and `dflash2` also suppress DSpark, as does `--no-spec`. In auto mode, naming a drafter suppresses implicit DSpark. With `dspark` or `all`, a named drafter of another kind still allows discovery. The first sorted adjacent basename match with a valid Markov header is used. |
| Auto `mmproj` vision projector | If a Hub GGUF repo contains a vision projector (`mmproj-BF16.gguf` preferred over F16/F32), `flyx` fetches and attaches it automatically unless `--no-vision` or an explicit `--mmproj` path is given, or the model's architecture is one this build reads no vision for (anything but `qwen35`: a projector published beside such a model is left alone, and `--mmproj` with one is refused). |
| Direct block-drafter loading | `z-lab/*-DFlash2`, `vllm-project/*-DFlash2` and `incoai/*-DFlash2` (DFlash2), `z-lab/*-DFlash-b16` (DFlash), `deepseek-ai/dspark_*` (DSpark), `tencent/*-DSpine-b16` (DSpine) and `EAGLE-Corp/*-EAGLE4` (EAGLE-4) safetensors repos as well as llama.cpp `general.architecture=dflash` GGUF drafters (e.g. `z-lab/Qwen3.8-27B-DFlash2-GGUF`, `ufrik/Inkling-2-Mini-Dflash2-GGUF`) load directly with no conversion step across supported target architectures (`qwen35`, `qwen3`, `llama`, `inkling2_loop`). |
| Downloader & cache | 16 parallel range connections, SHA-256 verified against Hub metadata, resumable across restarts, multi-process coordinated via `.flyx-lock` files in standard HF cache (`HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE=1`). Requires `libcurl` on Linux; native WinHTTP on Windows. |

---

## 3. Speculative Drafters

`flyx` supports eight drafter sources and can run multiple simultaneously; a per-round controller (`RoundChooser`) chooses the drafter and width dynamically from each round's cost at the context (timed at load, then learned from the rounds a session reads alone: a long context's K/V read over the bus or through a 2-bit copy makes wide rounds dearer; `FLYX_LEARN_COSTS=0` keeps the load's) and live acceptance rates.

| Drafter | Source | Notes |
|---|---|---|
| **MTP** (Multi-Token Prediction) | Embedded in GGUF/HF checkpoint, or sidecar `--drafter-file MTP/mtp-*.gguf` | Default when present. Shares target embedding/head. At long contexts (> 32K), uses 4096-token exact ring (with one prefill pass of headroom, copied with every checkpoint) + 2-bit StrataFlux copy (`mtp_window`, `mtp_copy`). |
| **DFlash2** (block diffusion) | HF safetensors repo (`--drafter-file z-lab/Qwen3.8-27B-DFlash2` or `incoai/Inkling-2-Mini-DFlash2`), `arch=dflash` GGUF (BF16 or quantized GGML blocks), or converted `.flyx` | For Qwen3.5 / Qwen3.8, Qwen3, Llama, and Inkling-2 (`inkling2_loop`) targets (a drafter made for another model is refused; supports target embedding norm, unpadded vocab clamping, post-`output_norm` taps, and `before_correction` / `after_correction` selector softcapping). Under default `draft_weights=auto`, floating-point DFlash2 weights stream-quantize directly on GPU (32 MiB staging, no full BF16 upload spike) into target-matched precision when VRAM allows (BF16 beside BF16, Q8_0 beside GGUF, NVFP4 beside NVFP4; pre-quantized GGUF drafters stay packed) and drop automatically to NVFP4 (4-bit) when VRAM is tight; override via `--backend-opt draft_weights=auto\|nvfp4\|q8\|exact`. Beside `inkling2_loop`, adaptive rounds verify `{1, 2, 3, 5, 7}` by confidence and widen sustained high-acceptance runs to stretched width `15`. A GGUF drafter that is causal, samples from the anchor, or has a confidence head is refused. Strongest on code/structured text; `--drafter dflash2` selects it exclusively. |
| **DFlash** (DFlash2's predecessor) | HF repo (`--drafter-file z-lab/Qwen3-8B-DFlash-b16`) or converted `.flyx`; for Qwen3.5 / Qwen3.8, Qwen3, and Llama targets (a drafter made for another model is refused) | z-lab's `DFlashDraftModel`: DFlash2's block (the same context, attention both ways within the block) without the grouped convolutions or the candidate selector: each row's draft is the head's best (measured against walking the head's top 16 with the target's noise under sampling: Qwen3-8B, DeepSpec's 9 datasets at T 1, 4.14 against 3.93 tokens a round). Its projections are quantized as DFlash2's, but Q8_0 beside a BF16 target under `draft_weights=auto` (Qwen3-8B, DeepSpec's 9 datasets: 356.0 against 340.3 tok/s in BF16, the same tokens a round). `--drafter dflash` selects it exclusively. |
| **DSpark** (block drafter + Markov head) | DeepSpec's: HF repo (`--drafter-file deepseek-ai/dspark_qwen3_8b_block7`) or converted `.flyx`, beside a Qwen3 (or Qwen3.5) target of its width and vocabulary. llama.cpp's: sidecar GGUF (`--drafter-file dspark-Inkling-2-Mini-*.gguf`, `arch=dflash` with `has_markov_context=true`), attached by default from the model's repository or directory (Auto DSpark sidecar above) | One kind (`--drafter dspark`), its block by its file. DeepSpec's: DFlash2's block drafter without the convolutions: every row of its block drafts (block 7: 7 drafts), a Markov head (a low-rank bias from the previous token) re-ranks the head's top 1024 while walking the block (its acceptance within about 1% of DeepSpec's walk over the whole vocabulary), its projections NVFP4 beside an NVFP4 target and Q8_0 beside a bf16 one (`draft_weights=exact`: bf16). On a Qwen3 or Llama target it drafts trees by default beside its chains (`--no-draft-trees`), and sampling, its walk takes the target's own noise (the output unchanged): Qwen3-8B bf16 on an RTX PRO 6000 over DeepSpec's 9 datasets verified more tokens a round than DeepSpec's evaluator greedy on all 9, sampled at T 1 on 7 (below on math500 and aime25). Its confidence head is left out (the rounds do not use it). llama.cpp's: a companion 3-layer block-uniform SWA/Full drafter with autoregressive Markov refinement for `inkling2_loop` (its file beside any other model is refused); it can run alongside DFlash2 under `--drafter all` (one DSpark a model, of either source). Its layers' linears follow `draft_weights` as DFlash2's do (a Q8_0 file stays packed under `auto`). Uses bounded per-layer binary16 KV rings (`window = 1025` on SWA layers, `mtp_window` ring on the Full layer above 32K) to cap drafter KV at ~28 MiB at any context length. Its drafts are picked from 1-bit copies of the target's head and of its Markov head together and only the candidates scored exactly on both (`draft_head=full` reads both heads whole). On CUDA, adaptive rounds run the parallel backbone once and advance the sequential Markov head step-wise across checkpoints `{1, 2, 3, 5, 7}`, stopping early to skip remaining Markov head steps and choosing the optimal already-drafted verification prefix (`stop_decide_step`). |
| **DSpine** (adjacent causal injection) | HF repo (`--drafter-file tencent/Qwen3.8-27B-DSpine-b16`, `tencent/Qwen3-8B-DSpine-b16`) or converted `.flyx`; for Qwen3.5 / Qwen3.8, Qwen3, and Llama targets (a drafter made for another model is refused) | A DFlash-style block drafter (the same context from target taps) whose layers attend causally within the block and pass a low-rank message from each row to the next (the code bridge); its drafts walk the head's top 16 with the last layer's message redone from the token actually drafted. A round of fewer drafts runs fewer rows. Chains only (no draft trees); sampling, its walk takes none of the target's noise (the output stays the sampler's; fewer drafts accepted than DFlash2's coupled walk would); its layer projections are quantized to NVFP4 beside BF16 or NVFP4 weights and to Q8_0 beside GGML block weights (its walk re-scores the head's candidates, so NVFP4 layers cost it at most 1% of its tokens a round), its bridge stays BF16. Under `draft_weights=auto` beside a BF16 target its layers are NVFP4 (DFlash2's stay BF16). `--drafter dspine` selects it exclusively; a model holds one of DFlash2's block (DFlash2, DFlash or DeepSpec's DSpark) or DSpine, and llama.cpp's DSpark beside it. |
| **EAGLE-4** (one layer over target taps) | HF repo (`--drafter-file EAGLE-Corp/Qwen3.8-27B-EAGLE4`, `LlamaForCausalLMEagle4`) or converted `.flyx`; for Qwen3.5 / Qwen3.8, Qwen3, and Llama targets of its width and vocabulary (a drafter made for another model is refused) | A Llama decoder layer over the normed embedding and the normed hidden side by side, the hidden `fc` of the target's taps (each normed by its own `fc_norm`) for the first step and its own normed output after; its own head, scored exactly on the 128 candidates a rank-128 head picks from the whole vocabulary. It drafts step by step as MTP does (and takes MTP's place under `auto` / `all`: one hidden ring), chains and, by default (`--no-draft-trees`), EAGLE-2's best-first trees of 15 beside them: a beam of the best 4 nodes a step for 10 steps (`--backend-opt eagle4_tree=BEAMxSTEPS`), every candidate a step scored a node, the round's chain of 7 (kept in the beam) and the best 8 of the rest by their path's log-probability verified (CUDA backend; cpu_ref drafts chains). Its window (4096) is applied by position. Its linears follow `draft_weights`, NVFP4 beside a BF16 target under `auto` (about 1% fewer tokens a round than BF16, rounds 7% cheaper); the candidate head and its own head stay BF16. Several sessions' rounds in one batch draft their EAGLE-4 drafts a sequence at a time (MTP's run in lockstep, DFlash2's grouped); only their verify passes are batched. Under `--kv-method auto` its trees run only where the target's switch leaves the whole context exact, its chains alone otherwise. `--drafter eagle4` selects it exclusively. |
| **DV** (bifill-dv's canvas drafter) | A DV `.flyx` file: a Llama with bifill-dv's canvas and selector written in (`flyx_convert --dv`) | Only when named (`--drafter dv` or `all`). A canvas pass of the target's weights with its own adapter over 32 slots after the anchor, then a selector that grows a tree of the file's width (31) or of 15, verified exactly (rounds of up to 64 rows). Its warm-up, its trees of 15 and of 31, plain decoding and lookup rounds are chosen every round by tokens per ms at the context; `--width N` pins it. CUDA backend only; see [DV](design/dv.md). |
| **Prompt Lookup** | Built-in (enabled by default) | Verifies 15 drafts copied from matching 6-token n-grams in the current sequence with zero drafter forward pass. Big speedup on agent code edits/repeats; backs off for 32 tokens after poor rounds (`--no-lookup` disables). |

Verified drafters: `drafters.json` in the Hub's [`Enwove/flyx-drafters`](https://huggingface.co/Enwove/flyx-drafters) lists the drafters checked with flyx (every output plain decoding's, and faster), by the base model they were made for: `{"version": 1, "drafters": [{"base": "Qwen/Qwen3-8B", "repo": "deepseek-ai/dspark_qwen3_8b_block7", "kind": "dspark"}, ...]}` (`kind`: `--drafter`'s word). The server reads it once a process (from the Hugging Face cache when offline) and keeps the kinds the build runs. A drafter made for the model itself (the Hub repository it is, or a `.flyx` file's `source.repo`, which `flyx_convert --repo` writes) is offered as such; one made for a base model the model names (its model card's `base_model`, a GGUF's `general.source.repo_url` and `general.base_model.<i>.repo_url`, a `.flyx` file's `source.base_models`) is offered as inferred, with how the Hub says the model was made from it (`quantized`, `finetune`, ...). The dashboard lists them beside the default drafter, each the fastest on this machine's own runs first (every request that ran to its end drafting with it, by model: its output tokens over its decode time, kept in `<config>/flyx/drafter_speeds.json`, written every 30 seconds and at exit added to what the file has; N/A for one not run here yet, flyx gives no figures of its own), and a load notes them. A path's load and a listing of local files read the list's copy in the Hugging Face cache only; a Hub model's load and a repository's page ask the Hub for it (once a process; offline, the cache's copy). The list is `main` of that repository, which the enwove maintainers keep: a drafter goes in once checked with flyx. Today: DFlash2 for Qwen3.8-27B and Qwen3.5-4B, DSpark for Qwen3-8B, and DSpine for Qwen3-8B and Qwen3.8-27B.

Control flags:
- `--drafter mtp|dflash2|dflash|dspark|dspine|eagle4|dv|all|none`
- `--drafter-file PATH_OR_HF` (repeatable)
- `--width N` (pin model drafter width; lookup remains active unless `--no-lookup`)
- `--draft-trees` / `--no-draft-trees` (best-first dynamic trees beside chains: opt-in for DFlash2, on by default for EAGLE-4, on by default for DeepSpec's DSpark on a Qwen3 or Llama target; an `inkling2_loop` target verifies chains only: its relative-position attention and ShortConvs read a round's rows in order, so the CUDA backend refuses tree rounds there)
- `--no-spec` (plain decoding, disables all drafters and lookup)
- `--cycle I,J,...` (testing: a fixed cycle of round indices instead of the controller's choice; `-v` lists the rounds)
- **Elastic VRAM loading (`serve`, `run`, `chat`):** Before uploading weights and on VRAM exhaustion, `flyx` steps down optional VRAM consumers in order, the first four keeping the target's output tokens: (1) moving GPU-resident vision weights to `disk` (no resident projector weights, multimodal preserved), (2) compacting drafter weights to `nvfp4`, (3) skipping heavy DFlash2 while keeping DSpark beside it (`--drafter all`) or keeping or falling back to MTP, (4) skipping remaining neural drafters (MTP, DSpark, DV) while keeping zero-VRAM Prompt Lookup, and (5) when the weights fit but a session does not, halving prefill passes of an automatic size down to 256 rows: slower prompts, whose rows keep their bits except in a last pass of 16 rows or fewer, which computes as a round ([numerics](design/numerics.md), K), so such a prompt's tokens may differ. Steps 3 and 4 apply only to drafters the load chose (`--drafter auto` or `all`): a drafter named with `--drafter mtp\|dflash2\|dspark\|dv` is never skipped, nor does step 5 run beside it (plain decoding, fitting with longer passes, could give a prompt other tokens), and a load that cannot keep it fails with the reason. Every adjustment and the active configuration are logged clearly to stderr and reported in `/v1/load`, `/v1/telemetry`, and the dashboard without overwriting saved settings in `serve.json`. `flyx bench` keeps loading strict so benchmark runs never change configuration silently.

---

Windowed DFlash2 layers keep FP16 context keys and values in a ring even beside an FP8 target. Use `--backend-opt dflash_ring=0` for the flat control. DSpark has independent per-layer rings. CUDA's `held_precision()` classifies the bound drafter linears by their actual storage. The dashboard and telemetry report the resulting precision even when packed companion weights already came from disk in Q8_0.

## 4. Converting Hugging Face Checkpoints to `.flyx`

GGUF files run directly without conversion. Convert HF Safetensors (BF16 or ModelOpt NVFP4) or standalone DFlash2 / DFlash / DSpark / DSpine checkpoints to `.flyx` using `python/convert`:

```sh
# Target + DFlash2 combined into one .flyx (BF16 or ModelOpt NVFP4)
hf download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B
hf download vllm-project/Qwen3.5-4B-DFlash2 --local-dir ~/models/Qwen3.5-4B-DFlash2
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3.5-4B --draft ~/models/Qwen3.5-4B-DFlash2 \
    --repo Qwen/Qwen3.5-4B -o ~/models/qwen35-4b.flyx

# ModelOpt NVFP4 checkpoint
hf download AxionML/Qwen3.5-4B-NVFP4 --local-dir ~/models/Qwen3.5-4B-NVFP4
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3.5-4B-NVFP4 --draft ~/models/Qwen3.5-4B-DFlash2 \
    -o ~/models/qwen35-4b-nvfp4.flyx

# Standalone DFlash2 drafter file (reusable alongside any compatible GGUF)
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/models/Qwen3.8-27B-DFlash2
PYTHONPATH=python/convert python -m flyx_convert \
    --draft ~/models/Qwen3.8-27B-DFlash2 -o ~/models/qwen38-27b-dflash2.flyx

# DFlash (z-lab's first block drafters): beside a Qwen3 model the same way
hf download z-lab/Qwen3-8B-DFlash-b16 --local-dir ~/models/Qwen3-8B-DFlash-b16
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3-8B --draft ~/models/Qwen3-8B-DFlash-b16 -o ~/models/qwen3-8b-dflash.flyx

# DSpine: beside a Qwen3.5/3.8 or a Qwen3 model (in place of DFlash2), or as a drafter file of its own
hf download tencent/Qwen3-8B-DSpine-b16 --local-dir ~/models/Qwen3-8B-DSpine-b16
PYTHONPATH=python/convert python -m flyx_convert \
    --model ~/models/Qwen3-8B --draft ~/models/Qwen3-8B-DSpine-b16 -o ~/models/qwen3-8b-dspine.flyx
hf download tencent/Qwen3.8-27B-DSpine-b16 --local-dir ~/models/Qwen3.8-27B-DSpine-b16
PYTHONPATH=python/convert python -m flyx_convert \
    --draft ~/models/Qwen3.8-27B-DSpine-b16 -o ~/models/qwen38-27b-dspine.flyx
```

```sh
# MiniCPM5-1B (Llama without q/k norms), BF16
hf download openbmb/MiniCPM5-1B --local-dir ~/models/MiniCPM5-1B
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B -o ~/models/minicpm5-1b.flyx

# MiniCPM5-1B with a DV drafter, and with its V-LoRA: bifill-dv's checkpoints on Hugging Face, panmz/bifill-dv,
# a private repository: without access to it these files cannot be made
hf download panmz/bifill-dv canvas/dva_res2.pt selector/sel_orig_w8.pt \
    canvas/dva_cd_b2h.pt selector/sel_b2h_w8.pt v_lora/lora_g_b2h.pt --local-dir ~/models/bifill-dv
B=~/models/bifill-dv
python -I python/convert/export_dv.py --canvas $B/canvas/dva_res2.pt --selector $B/selector/sel_orig_w8.pt \
    --tree beam:31:8:cov -o ~/models/dv-res2-export
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B \
    --dv ~/models/dv-res2-export -o ~/models/minicpm5-1b-dv-res2.flyx
python -I python/convert/export_dv.py --canvas $B/canvas/dva_cd_b2h.pt --selector $B/selector/sel_b2h_w8.pt \
    --vlora $B/v_lora/lora_g_b2h.pt --tree beam:31:8:cov -o ~/models/dv-b2h-export
PYTHONPATH=python/convert python -m flyx_convert --model ~/models/MiniCPM5-1B \
    --dv ~/models/dv-b2h-export -o ~/models/minicpm5-1b-dv-b2h.flyx
```

A target's own adapters (a **V-LoRA**: LoRA beside the target's linears, e.g. bifill-dv's rank-64 `lora_g_b2h`) run unmerged on every pass, their sums in float32 on the kernels of the linears they sit beside (a bf16 merge loses about 70% of the delta). `export_dv.py --vlora` exports them with or without a DV; `flyx_convert --lora EXPORT` writes a V-LoRA alone. A file with adapters is `.flyx` format 2. Adapters on GGML block weights, on an attention with an output gate, or on an MTP layer are refused. Which checkpoints make which DV file, and what DV is held to: [DV](design/dv.md).

> **WSL2 note:** Store models on the Linux filesystem (`~/models`), not `/mnt/c`.

---

## 5. Long-Context KV Tiering and 2-bit StrataFlux

`flyx` budgets device and host memory before allocating. `-v` prints the exact memory bill at startup.

### Memory Tier Hierarchy

| Tier | Controlled By | What Lives Here | Performance Characteristic |
|---|---|---|---|
| **Tier 1: GPU VRAM** | `--kv fp8\|16`, `--kv-gpu N` | Model weights, 3 GDN round slots, prefix of KV cache up to VRAM budget, hot StrataFlux refined-pair cache (`strata_hot=2048`), StrataFlux copies (as many layers as fit) | Full GPU speed; exact reads for all GPU-resident KV |
| **Tier 2: Pinned Host RAM** | `--kv-ram GB\|auto`, `--host-cache GB` | Overflow KV positions beyond `--kv-gpu`, overflow StrataFlux layer copies (`--kv-copy-gpu`), parked session checkpoints (`--host-cache`, restored in ~0.2 s) | Staged ahead over PCIe concurrently with preceding layers (`stage_dma=1`, `stage_ahead=1`) |
| **Tier 3: Local Disk** | `--kv-disk DIR[:GB]` | Incremental prompt-prefix snapshots as conversations grow + spilled parked checkpoints | Prevents full re-prefill when switching sessions or handling Claude Code title/permission requests |

### KV Read Methods (`--kv-method`)

| Mode | Behavior | Quality vs Speed Trade-off |
|---|---|---|
| `auto` *(default, 2-bit)* | Exact attention while context ≤ `kv_gpu`; switches automatically to 2-bit StrataFlux attention once KV spills to host RAM (`strata_from`). Falls back to `exact` if StrataFlux copies don't fit alongside KV, or where its kernels do not take the model's attention (MiniCPM5-1B's 8 query heads per KV head). | Exact while KV fits on GPU; fast at long contexts without scanning full host KV every step. |
| `exact` | Reads every KV position exactly at every step (including host-resident KV over PCIe). | Zero approximation; decode latency scales linearly with host-resident KV length (~0.67 ms/step per 1K host tokens on 27B over PCIe 5 x16). |
| `strata[:2\|3\|4]` | Always scans a 2-, 3-, or 4-bit Walsh-Hadamard summary copy and refines the top attended key pairs (`strata_denom`, `strata_least=32`, `strata_most=256`) from exact FP8/FP16 KV (cached on GPU via `strata_hot=2048`). | 2-bit is default (smallest footprint, fastest 1-warp scan); `:3` or `:4` uses more memory for tighter approximation. |

### Recommended GPU Sizing Recipes (Qwen3.8-27B)

| GPU VRAM | Host RAM | Recommended Quantization & Flags | Typical Capability |
|---|---|---|---|
| **16 GB** (RTX 5070 Ti / 5080 / 4080S) | 32 GB | `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` (10.9 GB) or `UD-IQ2_S` (8.4 GB); `--ctx max --sessions 1 --kv-disk ./flyx-cache:40` | The planner places excess KV in host RAM. Disk-backed snapshots allow inactive sessions to be swapped out |
| **24 GB** (RTX 3090 / 4090) | 32–64 GB | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` (17.7 GB) or `UD-IQ3_XXS`; `--ctx max` | Fast GPU-resident context + high-speed 2-bit StrataFlux overflow to host RAM |
| **32 GB** (RTX 5090) | 64+ GB | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` + `--drafter-file z-lab/Qwen3.8-27B-DFlash2 --ctx max` | The planner caps the request at the trained 262,144 tokens and the available memory budget |

> **8–10 GB GPUs (e.g., RTX 3080 10 GB, RTX 4060 / 5060 8 GB):** Cannot fit 27B weights alongside session recurrent state; use smaller models such as `Qwen/Qwen3.5-4B` (BF16/NVFP4/GGUF). On tight 8 GB desktop GPUs (where Windows WDDM reserves VRAM for desktop composition), `flyx` queries NVML free memory and automatically trims speed-only auxiliary copies (e.g., duplicate W12/FP8 head copies) when needed to fit prefill workspace and sessions, and the head's copies before any of the context's K/V would go to host memory where the whole context then fits on the GPU (an explicit `--kv-gpu` keeps them).
>
> **GPU tuning tables:** Pre-tuned dispatch tables ship under `backends/cuda/variants/` for RTX 3060, RTX 3080, RTX 3090, RTX 4060 Ti, RTX 4070 Ti SUPER, RTX 4080 SUPER, RTX 4090, L4, L40S, RTX 5060 Ti, RTX 5070, RTX 5070 Ti, RTX 5080, RTX 5090, A100 PCIe 40 GB, and H100 80GB HBM3 (other GPUs inherit their architecture table or defaults).
