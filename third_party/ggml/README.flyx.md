# ggml's quantization header

`ggml-common.h` is the header of ggml (the tensor library of llama.cpp) that
defines the block layouts of its quantized tensor types and the constant
tables their decoding needs (the IQ codebooks, sign tables, the non-linear
4-bit values). GGUF files store weights in these types, so reading such a
file as it is means reading these layouts with these tables.

    https://github.com/ggml-org/llama.cpp  commit f7591a6 (2026-11-07)
    ggml/src/ggml-common.h
    sha256 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c

under the MIT license (`LICENSE`, "Copyright (c) 2023-2026 The ggml authors").
The file is byte-for-byte upstream. Nothing else of ggml is here: the
decoders, the kernels and the GGUF reader are ours (decision 0029).

The header is written to be included by every backend of ggml and picks its
declarations by macro. flyx includes it in two places:

| where | macros | what it takes |
|---|---|---|
| `ops/dtype/ggml.h` (host, the float64 reference) | `GGML_COMMON_DECL_CPP`, `GGML_COMMON_IMPL_CPP` | block structs, tables (`iq1s_grid` as 8 bytes a row) |
| `backends/cuda/kernels/gemm_blocks.cu` (device) | `GGML_COMMON_DECL_CUDA`, `GGML_COMMON_IMPL_CUDA` | tables as `__device__` constants (`iq1s_grid_gpu`) |

To update: copy the file from a newer llama.cpp commit, note the commit and
hash here, and run `ctest -R "gguf|cuda_ops"` (every encoding is checked
against the `gguf` Python package's own decoder).
