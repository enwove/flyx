# XGrammar (C++ core)

Grammar-constrained decoding: structural tags (tool calls, reasoning blocks),
JSON schema, EBNF; a matcher that accepts tokens, rolls back and fills the
bitmask of the tokens a grammar allows next. flyx uses it to keep tool
calls well formed (decision 0020, `core/grammar/`).

Copied unchanged from

    https://github.com/mlc-ai/xgrammar  commit 8f9a127 (2026-11-06)

under the Apache License 2.0 (`LICENSE`, `NOTICE`).

| here | upstream |
|---|---|
| `include/xgrammar/` | `include/xgrammar/` |
| `cpp/` | `cpp/` without `cpp/tvm_ffi/` (the Python bindings) |
| `3rdparty/picojson/picojson.h` | `3rdparty/picojson/picojson.h` (BSD-2-Clause, in the header) |
| `3rdparty/dlpack/` | dmlc/dlpack commit bbd2f4d (the submodule's pin): `include/dlpack/dlpack.h`, `LICENSE` (Apache-2.0) |

Not copied: the Python package, TVM-FFI bindings, cpptrace (built with
`XGRAMMAR_ENABLE_CPPTRACE=0`), GPU kernels, web code, tests. The structural
tags for each model family are built by flyx's chat formats
(`models/formats/`), not by xgrammar's Python `get_model_structural_tag`.

To update: copy the same directories from a newer commit, note it here, and
run the grammar tests (`ctest -R "grammar|golden|cuda_ops"`).
