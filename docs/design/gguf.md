# GGUF Support

`ModelFile` identifies `.flyx` and GGUF files by magic bytes regardless of filename extension.

## Layer Responsibilities

| Component | Owns | Does Not Own |
|---|---|---|
| `core/modelfile/gguf_format.cpp` | GGUF container: header, metadata, tokenizer tables, tensor offsets and validation | Model architecture semantics; tensor rewriting |
| `models/gguf.h`, `models/<arch>/` | `llama.cpp` metadata keys, tensor names, and architecture layout flags | Data movement or dequantization |
| `models/drafters/` | A drafter GGUF's keys and tensor names (`general.architecture=dflash`), mapped from the names `python/convert` gives the same drafter | A target architecture's keys or layout |
| `ops/dtype/ggml.h` | Exact float64 definition of element `i` in every supported GGML block type | GPU kernel layout |
| `third_party/ggml/ggml-common.h` | Upstream GGML block structs and codebooks (unchanged) | Local modifications |
| `backends/<name>/` | Device upload, packing, and kernel execution according to layout flags | GGUF container parsing |

---

## Container Reader (`core/modelfile/gguf_format.cpp`)

Produces the same internal representation exposed for `.flyx` files:
- `header["arch"]`: `general.architecture`.
- `header["gguf"]`: all metadata keys except tokenizer tables.
- `header["tokenizer"]` + blobs (`tokenizer.vocab`, `tokenizer.merges`, `tokenizer.added`): `gpt2` byte-level BPE tables (`qwen35` and `qwen2` pre-tokenizers, NFC or no normalization).
- `header["generation"]["eos_ids"]`: file EOS tokens plus `<|im_end|>` and `<|endoftext|>` when present in the vocabulary.
- `chat_template.jinja`: embedded chat template.
- **Tensors:** exposed under GGUF names with row-major shapes (reversing GGUF's fastest-dimension-first order) and types mapped to `backend_abi.h`: `F32`, `F16`, `BF16`, `F64`, `Q2_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `MXFP4` (type 39), and GGML `NVFP4` (type 40). Every block tensor row must be a whole number of blocks.

Refused cleanly at open time: unsupported GGUF version/endianness, a split file with a part missing or its split metadata inconsistent, incomplete models, unsupported tensor types, quantization version ≠ 2, or any length/offset overflow, truncation, overlap, or duplicate name.

---

## Architecture Mapping (`models/qwen35/`, `models/llama/`, `models/inkling2/`)

All architectures read standard `llama.cpp` keys (`<arch>.block_count`, `embedding_length`, `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`, `attention.key_length`, `rope.dimension_count`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`; for Qwen3.5 also `ssm.group_count`, `ssm.time_step_rank`, `ssm.state_size`, `ssm.inner_size`, `ssm.conv_kernel`, `full_attention_interval` or `attention.recurrent_layers`, and `nextn_predict_layers`; for Inkling-2 `inkling2_loop` also loop prelude/core/epilogue counts, AttnRes / loop-control metadata, ShortConv, relative-position-bias attention, sliding-window SWA layers, and sparse MoE expert routing / SiTU parameters). A missing `output.weight` ties the output head to `token_embd.weight`. Scaled RoPE and vision towers are refused.

Sidecar drafters attach via `--drafter-file`: standalone MTP GGUFs (`MTP/mtp-*.gguf`), DFlash2 checkpoints (HF directory, `.flyx`, or a drafter GGUF, below), or DSpark GGUFs (`dspark-*.gguf`), sharing the target GGUF's packed embedding and output head.

### DFlash2 Drafter GGUF (`models/drafters/`, `general.architecture=dflash`)

A file `llama.cpp` reads as a DFlash2 drafter (e.g. `z-lab/Qwen3.8-27B-DFlash2-GGUF`) is a drafter alone: opened as a model it is refused, beside a model of any supported architecture it is the model's DFlash2 drafter. The blocks are the ones a `.flyx` drafter file gives; only the names differ.

| Read | Keys / Tensors |
|---|---|
| Dimensions | `dflash.block_count`, `embedding_length` (the target's width), `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`, `attention.key_length`, `rope.freq_base`, `attention.layer_norm_rms_epsilon` |
| Drafting | `dflash.block_size`, `conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k`, `target_layers` (the target's hidden states the drafter reads: `1..layers-1` tap the residual stream after that layer, `layers` taps post-`output_norm`), optional `final_logit_softcapping` and `selector_softcap_mode`, `attention.sliding_window` with `attention.sliding_window_pattern`, and `tokenizer.ggml.mask_token_id` (or `dflash.mask_token_id`) |
| Top-level tensors | `fc.weight`, `enc.output_norm.weight`, `output_norm.weight`, `selector_hidden.weight`, `selector_predecessor.weight`, `selector_successor.weight` |
| Per layer (`blk.<i>.`) | The dense layer's names (`attn_norm`, `attn_q`/`k`/`v`/`output`, `attn_q_norm`, `attn_k_norm`, `ffn_norm`, `ffn_gate`/`up`/`down`) plus `attn_conv_base`, `attn_conv_proj.weight`, `ffn_conv_base`, `ffn_conv_proj.weight` |

Its matrices stay as stored: GGML block types packed (mixed types in one stacked projection run as one GEMM per run of a type), BF16 as BF16 (quantized at load beside a quantized target, as a `.flyx` drafter's are). The file's own tokenizer is not read. Files with `dflash.has_markov_context = true` are claimed as DSpark drafters (`models/drafters/dspark.cpp`) rather than DFlash2.

Refused, with the file named: a width or vocabulary other than the target's (whether or not the drafter is chosen); and, when it is chosen to draft, causal layers, a sliding-window pattern that is not uniform, `dflash.sample_from_anchor`, and `dflash.has_confidence_head`.

### Qwen3.5 Layout Flags (Zero Weight Rewriting in `models/`)

| GGUF Storage Convention | Description Flag | Backend Handling |
|---|---|---|
| Norm weights stored as `1 + w` (except GDN output norm) | `flyx_norm.offset = 0` | Adds configured offset directly |
| Interleaved per-head gated Q rows (head `q` followed by head `gate`) | `flyx_attn_block.gate = FLYX_GATE_PER_HEAD` | Uploads/indexes rows into kernel layout |
| Value heads tiled over key heads (`v % nk`) | `head_order = FLYX_GDN_HEADS_TILED` | Maps value heads to key heads identically across all per-v-head tensors |
| Decay stored as `ssm_a = -exp(A_log)` | `flyx_gdn_block.decay = FLYX_GDN_DECAY_NEG_EXP` | Consumes decay directly (CUDA takes log as `A_log`) |

---

## Packed Device Execution (`backends/cuda/`)

- **GGML block weights:** Kept packed on GPU as split payload + scale planes in kernel read order (`kernels/gemm_blocks*.cu`). Kernels decode weights in registers (`d * s * q + min`) and multiply by **8-bit quantized activations** (scale per 32, matching `llama.cpp` precision): `dp4a` or integer tensor cores in 1..16-row rounds, and integer tensor cores with shape-fixed K runs in prefill. `--activations exact` switches to binary16 activations (`linear_a16h`); `--backend-opt blocks=bf16` dequantizes blocks to bf16 at load (comparison/debug only).
- **GGML NVFP4 (type 40):** Repacked at load (`ggml_nvfp4_planes`) into standard NVFP4 planes with row scale 1 and executed via the NVFP4 path (bf16 activations in rounds, NVFP4 activations in prefill on `sm_120a` unless `general.tensor_extra.prec_a4` or `--activations exact` disables W4A4).
- **Host-resident embeddings:** When untied from the head, `embed_blocks` reads packed embedding rows directly from host memory over PCIe (`--backend-opt embed=gpu` uploads to VRAM).
- **Reference backend (`cpu_ref`):** Decodes block weights to float64 via `ops/dtype/ggml.h` and applies identical input rounding (`linear_a8` / `linear_a16h` / `linear_a4`).

---

## Verification

`tests/core/modelfile/make_gguf.py` generates fixtures using the `gguf` Python package (tiny models + random blocks for every GGML block type + per-type tiny Qwen3).

| Test | Verifies |
|---|---|
| `ggml_blocks` | Every decoded value of random blocks for every block type vs Python `gguf` decoder |
| `gguf_reader` | Tokenizer tables, EOS IDs, chat template, tensor views, malformed-file refusal |
| `gguf_models` | GGUF model description matches `.flyx` description weight-for-weight via layout flags; a drafter GGUF matches its `.flyx` drafter file block-for-block |
| `dflash2_models` | Drafter GGUFs beside the tiny Qwen3 and Llama (block, taps, context nodes); drafter files refused for what they say or for the model they were made for |
| `gguf_qwen2_nfc`, `gguf_qwen2_none` | Qwen2 pre-tokenizer regex splits vs HuggingFace `tokenizers` |
| `gguf_cpu_ref`, `gguf_cuda`, `gguf_cuda_unfused`, `gguf_cuda_pdl`, `gguf_cuda_no_pdl` | End-to-end golden outputs, MTP/DFlash2 sidecars (drafter GGUFs in BF16, Q8_0 and mixed block types; beside Qwen3.5, Qwen3 and Llama), packed CUDA vs `cpu_ref` across all block types + GGML NVFP4 |
| `cuda_ops` (`test_blocks`) | Per-type bit-exact block decode, GEMMs vs float64, row-count bit invariance |
| `golden_*_qwen3_gguf` | Tiny Qwen3 GGUF against `transformers` |
