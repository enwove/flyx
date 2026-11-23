# GGUF Support

`ModelFile` identifies `.flyx` and GGUF files by magic bytes regardless of filename extension.

## Layer Responsibilities

| Component | Owns | Does Not Own |
|---|---|---|
| `core/modelfile/gguf_format.cpp` | GGUF container: header, metadata, tokenizer tables, tensor offsets and validation | Model architecture semantics; tensor rewriting |
| `models/gguf.h`, `models/<arch>/` | `llama.cpp` metadata keys, tensor names, and architecture layout flags | Data movement or dequantization |
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
- **Tensors:** exposed under GGUF names with row-major shapes (reversing GGUF's fastest-dimension-first order) and types mapped to `backend_abi.h`: `F32`, `F16`, `BF16`, `F64`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, and GGML `NVFP4` (type 40). Every block tensor row must be a whole number of blocks.

Refused cleanly at open time: unsupported GGUF version/endianness, a split file with a part missing or its split metadata inconsistent, incomplete models, unsupported tensor types (e.g., `MXFP4`), quantization version ≠ 2, or any length/offset overflow, truncation, overlap, or duplicate name.

---

## Architecture Mapping (`models/qwen35/`, `models/qwen3/`)

Both architectures read standard `llama.cpp` keys (`<arch>.block_count`, `embedding_length`, `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`, `attention.key_length`, `rope.dimension_count`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`; for Qwen3.5 also `ssm.group_count`, `ssm.time_step_rank`, `ssm.state_size`, `ssm.inner_size`, `ssm.conv_kernel`, `full_attention_interval` or `attention.recurrent_layers`, and `nextn_predict_layers`). A missing `output.weight` ties the output head to `token_embd.weight`. Scaled RoPE, sliding-window target attention, MoE experts, and scaled embeddings are refused.

Sidecar drafters attach via `--drafter-file`: standalone MTP GGUFs (`MTP/mtp-*.gguf`) or DFlash2 checkpoints (HF directory or `.flyx`), sharing the target GGUF's packed embedding and output head.

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
| `gguf_models` | GGUF model description matches `.flyx` description weight-for-weight via layout flags |
| `gguf_qwen2_nfc`, `gguf_qwen2_none` | Qwen2 pre-tokenizer regex splits vs HuggingFace `tokenizers` |
| `gguf_cpu_ref`, `gguf_cuda`, `gguf_cuda_unfused`, `gguf_cuda_pdl`, `gguf_cuda_no_pdl` | End-to-end golden outputs, MTP/DFlash2 sidecars, packed CUDA vs `cpu_ref` across all block types + GGML NVFP4 |
| `cuda_ops` (`test_blocks`) | Per-type bit-exact block decode, GEMMs vs float64, row-count bit invariance |
| `golden_*_qwen3_gguf` | Tiny Qwen3 GGUF against `transformers` |
